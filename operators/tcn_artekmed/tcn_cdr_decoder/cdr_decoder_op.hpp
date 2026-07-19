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

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace tcn::ops {

/**
 * @brief Holoscan operator that decodes CDR-encoded messages using the type registry.
 *
 * Takes raw CDR bytes and a type name string, looks up the deserializer in the
 * CdrTypeRegistry, and emits decoded payload bytes + metadata.
 *
 * Supports all types registered in the CdrTypeRegistry (VideoStreamMessage,
 * StreamDescriptorMessage, Pose6DMessage, CameraInfoMessage, etc.).
 *
 * Inputs:
 *   - input: raw CDR payload bytes (std::vector<uint8_t>)
 *   - type_name: CDR type name string (e.g. "tcnart_msgs::msg::VideoStreamMessage")
 *
 * Outputs:
 *   - output: decoded payload bytes (std::vector<uint8_t>)
 *
 * Metadata propagated via Holoscan message metadata:
 *   - All key-value pairs from the decoded message (frame_id, stamp, dimensions, etc.)
 */
class TcnCdrDecoderOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnCdrDecoderOp)

    TcnCdrDecoderOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;

 private:
    holoscan::Parameter<std::string> source_name_;
    holoscan::Parameter<int32_t> stream_index_;
};

}  // namespace tcn::ops
