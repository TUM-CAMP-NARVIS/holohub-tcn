/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tcn_stream_synchronizer.hpp"

#include <algorithm>
#include <utility>

#include <holoscan/core/conditions/gxf/multi_message_available.hpp>

namespace tcn::ops {

using holoscan::ConditionType;
using holoscan::MultiMessageAvailableCondition;

void TcnStreamSynchronizerOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(stream_names_, "streams", "Streams",
             "Input/output port names, in order. One port pair per stream.",
             std::vector<std::string>{});
  spec.param(optional_streams_, "optional_streams", "Optional Streams",
             "Subset of `streams` that may be absent from a group. Everything else is required: a "
             "group is published only when every required stream has a matching timestamp.",
             std::vector<std::string>{});
  spec.param(capacities_, "capacities", "Capacities",
             "Ring-buffer capacity per stream, parallel to `streams`. Capacity is PER STREAM "
             "deliberately: it must exceed that stream's worst-case skew relative to the slowest "
             "stream, so a fast stream (depth) needs several times more slots than a slow one "
             "(masks). Missing entries fall back to `default_capacity`.",
             std::vector<int64_t>{});
  spec.param(reference_stream_, "reference_stream", "Reference Stream",
             "Stream iterated oldest-first when searching for a group. Should be the one expected "
             "to LAG most (the masks), because a group can only be found once its slowest member "
             "has arrived. Defaults to the first entry of `streams`.",
             std::string{});
  spec.param(match_policy_, "match_policy", "Match Policy",
             "'exact' (default) or 'window'. Exact is correct for streams fed by one shared-memory "
             "segment, where every camera shares one acquisition timestamp by construction.",
             std::string{"exact"});
  spec.param(window_ns_, "window_ns", "Window (ns)",
             "Tolerance for match_policy 'window'. Ignored for 'exact'.", int64_t{0});
  spec.param(default_capacity_, "default_capacity", "Default Capacity",
             "Capacity for streams without an entry in `capacities`.", int64_t{8});
  spec.param(verbose_, "verbose", "Verbose", "Log every published group.", false);

  // Ports are created from the configured stream list. Input and output namespaces are separate, so
  // a stream's input and output may share its name.
  //
  // The list is read from args() rather than from the `streams` Parameter, because setup() runs
  // BEFORE parameter values are applied -- `stream_names_.get()` is empty here, and reading it
  // produced an operator with no ports at all ("does not have any input port but 'masks' was
  // specified in the port_map"). args() is populated by the constructor, so it is the only source
  // available this early. Consequence worth knowing: the stream list must be passed as a
  // CONSTRUCTOR argument and cannot be supplied from YAML via from_config, unlike every other
  // parameter here.
  std::vector<std::string> names;
  for (auto& a : args()) {
    if (a.name() != "streams") continue;
    try {
      names = std::any_cast<std::vector<std::string>>(a.value());
    } catch (const std::bad_any_cast&) {
      throw std::runtime_error(
          "TcnStreamSynchronizerOp: `streams` must be a list of strings passed to the constructor");
    }
    break;
  }
  if (names.empty()) {
    throw std::runtime_error(
        "TcnStreamSynchronizerOp: `streams` must be given as a CONSTRUCTOR argument (a non-empty "
        "list of port names). It cannot come from YAML, because ports are created in setup(), "
        "which runs before parameters are applied.");
  }
  configured_streams_ = names;

  // CRITICAL -- the conditions. Holoscan's default per-port MessageAvailableCondition would fire
  // compute() only when EVERY port has a message, which defeats buffering entirely: this operator
  // would degenerate into the rendezvous behaviour of the implementation it replaces, and a stream
  // running two frames behind could never be matched. So every input's own condition is removed
  // (kNone) and replaced by ONE multi-message condition over all of them with
  // SamplingMode::kSumOfAll and min_sum = 1 -- i.e. "run when ANY port has data". Each compute()
  // then ingests whatever arrived and publishes only if a complete group exists.
  std::vector<std::string> input_ports;
  input_ports.reserve(names.size());
  for (const auto& name : names) {
    spec.input<holoscan::gxf::Entity>(name).condition(ConditionType::kNone);
    spec.output<holoscan::gxf::Entity>(name);
    input_ports.push_back(name);
  }
  spec.multi_port_condition(
      ConditionType::kMultiMessageAvailable, input_ports,
      holoscan::ArgList{
          holoscan::Arg("sampling_mode", MultiMessageAvailableCondition::SamplingMode::kSumOfAll),
          holoscan::Arg("min_sum", static_cast<size_t>(1))});
}

void TcnStreamSynchronizerOp::start() {
  // Captured in setup() from args(); see the note there on why the Parameter cannot be used.
  const auto& names = configured_streams_;
  if (names.empty()) {
    throw std::runtime_error("TcnStreamSynchronizerOp: `streams` is empty; nothing to synchronise");
  }

  const auto& opt = optional_streams_.get();
  const auto& caps = capacities_.get();

  // An optional_streams entry that names no configured stream is almost certainly a typo, and it
  // would silently make a stream REQUIRED that was meant to be optional -- which shows up as
  // "nothing is ever published", the hardest symptom to trace. Refuse instead.
  for (const auto& o : opt) {
    if (std::find(names.begin(), names.end(), o) == names.end()) {
      throw std::runtime_error("TcnStreamSynchronizerOp: optional_streams entry '" + o +
                               "' is not in `streams`");
    }
  }
  if (!caps.empty() && caps.size() != names.size()) {
    throw std::runtime_error(
        "TcnStreamSynchronizerOp: `capacities` has " + std::to_string(caps.size()) +
        " entries but `streams` has " + std::to_string(names.size()) +
        "; give one capacity per stream or none at all");
  }

  streams_.clear();
  streams_.reserve(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    Stream s;
    s.name = names[i];
    s.required = std::find(opt.begin(), opt.end(), names[i]) == opt.end();
    int64_t cap = caps.empty() ? default_capacity_.get() : caps[i];
    if (cap < 1) cap = 1;
    s.buffer = std::make_unique<sync::RingBuffer<holoscan::gxf::Entity>>(
        static_cast<std::size_t>(cap));
    streams_.push_back(std::move(s));
  }

  reference_ = reference_stream_.get().empty() ? names.front() : reference_stream_.get();
  if (std::find(names.begin(), names.end(), reference_) == names.end()) {
    throw std::runtime_error("TcnStreamSynchronizerOp: reference_stream '" + reference_ +
                             "' is not in `streams`");
  }
  // A reference stream that is OPTIONAL cannot drive matching: the search iterates its timestamps,
  // so if it is absent nothing is ever attempted.
  for (const auto& s : streams_) {
    if (s.name == reference_ && !s.required) {
      throw std::runtime_error("TcnStreamSynchronizerOp: reference_stream '" + reference_ +
                               "' is listed as optional; the reference must be required");
    }
  }

  const std::string& mp = match_policy_.get();
  if (mp == "exact") {
    policy_ = std::make_unique<sync::ExactMatch>();
  } else if (mp == "window") {
    if (window_ns_.get() <= 0) {
      throw std::runtime_error(
          "TcnStreamSynchronizerOp: match_policy 'window' needs window_ns > 0 (got " +
          std::to_string(window_ns_.get()) + "); use 'exact' for a zero tolerance");
    }
    policy_ = std::make_unique<sync::WindowMatch>(window_ns_.get());
  } else {
    throw std::runtime_error("TcnStreamSynchronizerOp: unknown match_policy '" + mp +
                             "'; expected 'exact' or 'window'");
  }

  std::string summary;
  for (const auto& s : streams_) {
    summary += " " + s.name + "(" + (s.required ? "req" : "opt") + ",cap=" +
               std::to_string(s.buffer->capacity()) + ")";
  }
  HOLOSCAN_LOG_INFO("TcnStreamSynchronizerOp: policy={} reference='{}' streams:{}",
                    policy_->name(), reference_, summary);
}

void TcnStreamSynchronizerOp::stop() {
  HOLOSCAN_LOG_INFO(
      "TcnStreamSynchronizerOp: published {} groups, {} ticks without a match, {} forced drops",
      published_, unmatched_ticks_, forced_drops_);
  for (const auto& s : streams_) {
    HOLOSCAN_LOG_INFO("  {}: received={} unstamped={} evicted={} non-monotonic={} occupancy={}/{}",
                      s.name, s.received, s.unstamped, s.buffer->evicted(),
                      s.buffer->rejected_non_monotonic(), s.buffer->size(),
                      s.buffer->capacity());
  }
  streams_.clear();     // releases every buffered entity
}

void TcnStreamSynchronizerOp::drain_inputs(holoscan::InputContext& op_input) {
  for (auto& s : streams_) {
    // Ports have no condition of their own, so "nothing here this tick" is the normal case.
    auto maybe = op_input.receive<holoscan::gxf::Entity>(s.name.c_str());
    if (!maybe) continue;

    const auto ts = op_input.get_acquisition_timestamp(s.name.c_str());
    if (!ts.has_value()) {
      // Cannot group an unstamped message. Dropping it is the only honest option -- keeping it
      // would need an invented timestamp -- but silence here would look exactly like starvation,
      // so it is counted and reported.
      if (s.unstamped == 0) {
        HOLOSCAN_LOG_ERROR(
            "TcnStreamSynchronizerOp: '{}' delivered a message with NO acquisition timestamp; it "
            "cannot be grouped and is dropped. Upstream must stamp it: tcn_shm_subscriber attaches "
            "a gxf::Timestamp, and every Python stage must forward it via emit(acq_timestamp=...).",
            s.name);
      }
      ++s.unstamped;
      continue;
    }

    ++s.received;
    if (!s.buffer->push(*ts, std::move(maybe.value()))) {
      HOLOSCAN_LOG_WARN(
          "TcnStreamSynchronizerOp: '{}' rejected a non-advancing timestamp {} (last was newer); "
          "the stream is not monotonic and matching assumes it is",
          s.name, *ts);
    }
  }
}

void TcnStreamSynchronizerOp::log_starvation() {
  // Rate-limited: this fires every tick while starved, and starvation can persist for a long time.
  if (unmatched_ticks_ - last_logged_unmatched_ < 120) return;
  last_logged_unmatched_ = unmatched_ticks_;

  std::string blocking, spans;
  for (const auto& s : streams_) {
    if (s.required && s.buffer->empty()) {
      blocking += " " + s.name;
    }
    if (!s.buffer->empty()) {
      spans += " " + s.name + "=[" + std::to_string(s.buffer->begin()->timestamp) + ".." +
               std::to_string(std::prev(s.buffer->end())->timestamp) + "]x" +
               std::to_string(s.buffer->size());
    } else {
      spans += " " + s.name + "=empty";
    }
  }
  HOLOSCAN_LOG_WARN(
      "TcnStreamSynchronizerOp: {} ticks without a complete group. Empty required streams:{} "
      "Buffered spans:{}. If spans do not overlap, the streams are not carrying the same "
      "acquisition timestamps.",
      unmatched_ticks_, blocking.empty() ? std::string(" none") : blocking, spans);
}

void TcnStreamSynchronizerOp::force_progress() {
  // Only intervene once a buffer is actually full: until then, waiting is the correct behaviour --
  // the lagging stream may simply not have caught up yet.
  Stream* fullest = nullptr;
  for (auto& s : streams_) {
    if (s.buffer->full() && (fullest == nullptr || s.buffer->size() > fullest->buffer->size())) {
      fullest = &s;
    }
  }
  if (fullest == nullptr) return;

  const int64_t dropped = fullest->buffer->begin()->timestamp;
  fullest->buffer->discard_older_than(dropped + 1);   // drop exactly the oldest
  ++forced_drops_;
  HOLOSCAN_LOG_WARN(
      "TcnStreamSynchronizerOp: '{}' is full and nothing matches; dropping its oldest frame "
      "(ts={}) to make progress. This LOSES a frame -- the alternative is wedging the graph. "
      "Total forced drops: {}",
      fullest->name, dropped, forced_drops_);
}

void TcnStreamSynchronizerOp::compute(holoscan::InputContext& op_input,
                                      holoscan::OutputContext& op_output,
                                      holoscan::ExecutionContext&) {
  drain_inputs(op_input);

  // The matcher works on timestamps only, so it stays independent of the payload type and is
  // unit-tested on the host without Holoscan.
  std::vector<sync::StreamTimestamps> view;
  view.reserve(streams_.size());
  for (const auto& s : streams_) {
    sync::StreamTimestamps st;
    st.name = s.name;
    st.required = s.required;
    st.timestamps.reserve(s.buffer->size());
    for (auto it = s.buffer->begin(); it != s.buffer->end(); ++it) {
      st.timestamps.push_back(it->timestamp);
    }
    view.push_back(std::move(st));
  }

  sync::MatchResult match;
  if (!sync::find_oldest_group(view, *policy_, reference_, &match)) {
    ++unmatched_ticks_;
    log_starvation();
    force_progress();
    return;
  }

  // Take the chosen members OUT before discarding, so the discard cannot free a member of the group
  // being published.
  std::vector<std::pair<std::string, holoscan::gxf::Entity>> group;
  group.reserve(match.chosen.size());
  for (auto& s : streams_) {
    const auto it = match.chosen.find(s.name);
    if (it == match.chosen.end()) continue;          // optional stream with no match this group
    auto taken = s.buffer->take(it->second);
    if (!taken) {
      // Would mean the matcher and the buffers disagree -- impossible unless the buffer was mutated
      // between building `view` and here. Loud, because it indicates a logic error, not bad input.
      HOLOSCAN_LOG_ERROR("TcnStreamSynchronizerOp: '{}' lost timestamp {} between matching and "
                         "publishing; skipping this group", s.name, it->second);
      return;
    }
    group.emplace_back(s.name, std::move(taken->value));
  }

  // Every port of the group carries the SAME acquisition timestamp, so downstream sees one coherent
  // frame time. Under 'exact' this is what they already had; under 'window' it is the group's oldest
  // member, matching the discard boundary.
  for (auto& [name, entity] : group) {
    op_output.emit(entity, name.c_str(), match.group_timestamp);
  }

  // Everything strictly older than the published group can never be part of a future group, because
  // groups are published oldest-first.
  std::size_t discarded = 0;
  for (auto& s : streams_) discarded += s.buffer->discard_older_than(match.group_timestamp);

  ++published_;
  unmatched_ticks_ = 0;
  last_logged_unmatched_ = 0;
  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("TcnStreamSynchronizerOp: published group ts={} ({} ports, {} stale dropped)",
                      match.group_timestamp, group.size(), discarded);
  }
}

}  // namespace tcn::ops
