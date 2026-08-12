/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TCN_STREAM_SYNCHRONIZER_HPP
#define TCN_STREAM_SYNCHRONIZER_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "frame_matcher.hpp"
#include "ring_buffer.hpp"

namespace tcn::ops {

/**
 * @brief Groups entities arriving on several ports at different rates into frame-groups that share
 *        an acquisition timestamp.
 *
 * Motivation: the mask path runs at ~5 fps and about two frames behind the source, while the depth
 * path is independent and faster. Applying a mask to whatever depth frame happens to be current
 * mis-registers it by a varying amount. This operator buffers each stream and emits only complete,
 * timestamp-consistent groups. See
 * `applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-11-temporal-sync-design.md`.
 *
 * Requires that upstream stamps its messages: `tcn_shm_subscriber` attaches a
 * `nvidia::gxf::Timestamp`, and every Python stage on the mask path forwards it via
 * `emit(..., acq_timestamp=...)`. A stream whose messages carry no timestamp cannot be grouped, and
 * this operator says so loudly rather than guessing.
 *
 * Replay caveat: timestamps must ADVANCE for a buffer to accept them, since a timestamp is a
 * frame's identity and two frames cannot be the same frame. A looping source that restarts at its
 * first frame therefore has its second and later passes rejected, counted as `non-monotonic`.
 * `tcn_dataset_replayer` handles this by shifting each pass forward by one dataset span
 * (`_planning.loop_span_ns`); a non-advancing count against a looping source means some other
 * source is replaying raw timestamps.
 *
 * Rule: of all complete groups currently buffered, publish the OLDEST, then discard everything
 * strictly older than it in every buffer. Discarding relative to the published group -- rather than
 * to each stream's own head -- is what stops the lagging mask stream from having its future partners
 * thrown away before it catches up.
 */
class TcnStreamSynchronizerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnStreamSynchronizerOp)

  TcnStreamSynchronizerOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  struct Stream {
    std::string                                     name;
    bool                                            required = true;
    std::unique_ptr<sync::RingBuffer<holoscan::gxf::Entity>> buffer;
    std::size_t                                     received = 0;
    std::size_t                                     unstamped = 0;
  };

  /// Ingest whatever is available on each port. Ports carry no per-port condition (see setup), so
  /// an empty port is normal and not an error.
  void drain_inputs(holoscan::InputContext& op_input);

  /// Report, once per rate-limited interval, which streams are preventing a match.
  void log_starvation();

  /// Last resort when a buffer is full and still unmatched: drop the oldest of the fullest stream so
  /// the graph makes progress instead of wedging. Loses a frame, and says so.
  void force_progress();

  holoscan::Parameter<std::vector<std::string>> stream_names_;
  holoscan::Parameter<std::vector<std::string>> optional_streams_;
  holoscan::Parameter<std::vector<int64_t>>     capacities_;
  holoscan::Parameter<std::string>              reference_stream_;
  holoscan::Parameter<std::string>              match_policy_;
  holoscan::Parameter<int64_t>                  window_ns_;
  holoscan::Parameter<int64_t>                  default_capacity_;
  holoscan::Parameter<bool>                     verbose_;

  /// Port names captured in setup() from args(), because parameters are not yet applied there.
  std::vector<std::string>             configured_streams_;
  std::vector<Stream>                  streams_;
  std::string                          reference_;
  std::unique_ptr<sync::MatchPolicy>   policy_;

  std::size_t published_ = 0;
  std::size_t unmatched_ticks_ = 0;
  std::size_t forced_drops_ = 0;
  std::size_t last_logged_unmatched_ = 0;
};

}  // namespace tcn::ops

#endif  // TCN_STREAM_SYNCHRONIZER_HPP
