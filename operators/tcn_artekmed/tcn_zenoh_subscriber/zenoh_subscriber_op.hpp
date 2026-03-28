/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

// Forward-declare zenoh types to avoid pulling the full header into every consumer.
namespace zenoh {
class Session;
template <class Handler>
class Subscriber;
}  // namespace zenoh

namespace tcn::ops {

/// A single received Zenoh sample: CDR type name + raw payload bytes.
struct ZenohSample {
    std::string type_name;
    std::vector<uint8_t> payload;
};

/**
 * @brief Holoscan source operator that subscribes to a Zenoh topic.
 *
 * Receives CDR-encoded samples from Zenoh, queues them, and emits raw
 * payload bytes on the "output" port.  The CDR type name is attached as
 * metadata ("CdrTypeName") for downstream decoders.
 *
 * Uses AsynchronousCondition to integrate with Holoscan's event-based
 * scheduler — same pattern as TcnShmSubscriberOp.
 *
 * Outputs:
 *   - output: raw CDR payload bytes (std::vector<uint8_t>)
 */
class TcnZenohSubscriberOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnZenohSubscriberOp)

    TcnZenohSubscriberOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void initialize() override;
    void start() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;
    void stop() override;

    /// Set the Zenoh session (must be called before start).
    void set_session(std::shared_ptr<zenoh::Session> session) {
        session_ = std::move(session);
    }

 private:
    // Parameters
    holoscan::Parameter<std::string> topic_;
    holoscan::Parameter<std::shared_ptr<holoscan::AsynchronousCondition>> async_condition_;

    // Zenoh state
    std::shared_ptr<zenoh::Session> session_;
    std::unique_ptr<zenoh::Subscriber<void>> subscriber_;

    // Thread-safe sample queue (bounded for back-pressure)
    static constexpr size_t kMaxQueuedSamples = 4;
    std::mutex queue_mutex_;
    std::queue<ZenohSample> sample_queue_;
    std::atomic<uint64_t> samples_dropped_{0};
};

}  // namespace tcn::ops
