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
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

// Forward-declare zenoh types
namespace zenoh {
class Session;
class Publisher;
}  // namespace zenoh

namespace tcn::ops {

/**
 * @brief Holoscan sink operator that publishes CDR-encoded payloads to Zenoh.
 *
 * Takes raw bytes (e.g., CDR-encoded messages) on the "input" port and
 * publishes them to a Zenoh topic.  The CDR type name is read from input
 * metadata ("CdrTypeName") and attached to the Zenoh sample as an attachment.
 *
 * This is the publishing counterpart to TcnZenohSubscriberOp: the attachment
 * convention (type name as string) is symmetric.
 *
 * Inputs:
 *   - input: raw payload bytes (std::vector<uint8_t>)
 *
 * Parameters:
 *   - topic: Zenoh key expression to publish on
 */
class TcnZenohPublisherOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnZenohPublisherOp)

    TcnZenohPublisherOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
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
    holoscan::Parameter<std::string> topic_;

    std::shared_ptr<zenoh::Session> session_;
    std::unique_ptr<zenoh::Publisher> publisher_;

    std::atomic<uint64_t> messages_published_{0};
};

}  // namespace tcn::ops
