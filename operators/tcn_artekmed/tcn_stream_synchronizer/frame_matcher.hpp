/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TCN_STREAM_SYNCHRONIZER_FRAME_MATCHER_HPP
#define TCN_STREAM_SYNCHRONIZER_FRAME_MATCHER_HPP

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tcn::sync {

/**
 * @brief Whether a candidate timestamp from one stream corresponds to a reference timestamp.
 *
 * Kept behind an interface because the two cases have very different confidence: exact matching is
 * provably right for our source, and windowed matching is where order-dependence and tie-break bugs
 * live. Separating them keeps the reasoning about each honest.
 */
class MatchPolicy {
 public:
  virtual ~MatchPolicy() = default;
  virtual bool matches(int64_t reference, int64_t candidate) const = 0;
  virtual const char* name() const = 0;
};

/**
 * Equality. Correct for streams fed by ONE shared-memory segment: `tcn_shm_subscriber` stamps one
 * acquisition time per composite buffer and every camera in it shares that value, so corresponding
 * entities carry bit-identical timestamps. The default, and the only policy whose behaviour needs
 * no tie-break rule.
 */
class ExactMatch : public MatchPolicy {
 public:
  bool matches(int64_t reference, int64_t candidate) const override {
    return reference == candidate;
  }
  const char* name() const override { return "exact"; }
};

/**
 * Absolute tolerance, for genuinely independent sources. Note this makes matching depend on a
 * tie-break when several candidates fall inside the window -- see `find_oldest_group`, which
 * resolves nearest-to-reference first and older-wins on a tie, so the result cannot depend on
 * container order.
 */
class WindowMatch : public MatchPolicy {
 public:
  explicit WindowMatch(int64_t tolerance_ns) : tol_(tolerance_ns < 0 ? -tolerance_ns : tolerance_ns) {}
  bool matches(int64_t reference, int64_t candidate) const override {
    const int64_t d = reference > candidate ? reference - candidate : candidate - reference;
    return d <= tol_;
  }
  const char* name() const override { return "window"; }
  int64_t tolerance_ns() const { return tol_; }

 private:
  int64_t tol_;
};

/// One stream's contribution to a match attempt: its ascending timestamps and whether it is required.
struct StreamTimestamps {
  std::string          name;
  bool                 required = true;
  std::vector<int64_t> timestamps;   // MUST be ascending (RingBuffer guarantees this)
};

/// The chosen group.
struct MatchResult {
  /// Discard boundary: everything strictly older than this may be dropped in every stream.
  /// The minimum of the chosen timestamps, so a member of the published group is never discarded.
  int64_t group_timestamp = 0;
  /// stream name -> the timestamp selected from it. Optional streams appear only when matched.
  std::map<std::string, int64_t> chosen;
};

/**
 * @brief Find the OLDEST complete group, or nothing.
 *
 * A group is complete when every REQUIRED stream has a timestamp matching the reference under
 * `policy`. Optional streams join when they match and are simply absent otherwise -- they never
 * block a group and never hold one back.
 *
 * Iterating the reference stream oldest-first and returning the first complete group is what makes
 * this publish in order and bound latency. Which stream is the reference matters: it should be the
 * one expected to lag most (the masks, ~2 frames behind the source as measured on the harness),
 * because a group can only be discovered once its slowest member has arrived.
 *
 * Deliberately the straightforward O(streams x items x items) search. Streams number in the single
 * digits and buffers in the low double digits, so this is far cheaper than a single H2D copy, and an
 * index would be harder to reason about than the thing it replaced.
 *
 * @param streams  per-stream ascending timestamps; exactly one must be named `reference`
 * @param policy   match predicate
 * @param reference name of the stream to iterate oldest-first
 * @return the oldest complete group, or nullopt when none exists yet
 */
inline bool find_oldest_group(const std::vector<StreamTimestamps>& streams,
                              const MatchPolicy& policy,
                              const std::string& reference,
                              MatchResult* out) {
  const StreamTimestamps* ref = nullptr;
  for (const auto& s : streams) {
    if (s.name == reference) { ref = &s; break; }
  }
  if (ref == nullptr || ref->timestamps.empty()) return false;

  for (const int64_t candidate_ref : ref->timestamps) {     // ascending -> oldest group wins
    MatchResult result;
    result.chosen[ref->name] = candidate_ref;
    bool complete = true;

    for (const auto& s : streams) {
      if (s.name == ref->name) continue;

      // Best candidate under the policy: nearest to the reference, older winning a tie. Without a
      // deterministic rule a windowed match would depend on iteration order.
      bool     found = false;
      int64_t  best = 0;
      int64_t  best_dist = std::numeric_limits<int64_t>::max();
      for (const int64_t t : s.timestamps) {
        if (!policy.matches(candidate_ref, t)) continue;
        const int64_t d = candidate_ref > t ? candidate_ref - t : t - candidate_ref;
        if (!found || d < best_dist || (d == best_dist && t < best)) {
          found = true; best = t; best_dist = d;
        }
      }

      if (found) {
        result.chosen[s.name] = best;
      } else if (s.required) {
        complete = false;
        break;                    // this reference timestamp cannot form a group; try the next
      }
    }

    if (!complete) continue;

    int64_t oldest = std::numeric_limits<int64_t>::max();
    for (const auto& kv : result.chosen) {
      if (kv.second < oldest) oldest = kv.second;
    }
    result.group_timestamp = oldest;
    if (out != nullptr) *out = std::move(result);
    return true;
  }
  return false;
}

}  // namespace tcn::sync

#endif  // TCN_STREAM_SYNCHRONIZER_FRAME_MATCHER_HPP
