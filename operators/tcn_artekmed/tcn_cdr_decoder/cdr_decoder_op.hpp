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
 * @brief Holoscan operator that decodes CDR-encoded VideoStreamMessage payloads.
 *
 * Takes raw CDR bytes from a Zenoh subscriber (or any source), deserializes
 * the VideoStreamMessage using FastCDR, and emits the raw image bytes on
 * the "output" port.
 *
 * The CDR type name is read from input metadata ("CdrTypeName") and used to
 * verify the expected message type.
 *
 * Inputs:
 *   - input: raw CDR payload bytes (std::vector<uint8_t>)
 *
 * Outputs:
 *   - output: decoded image bytes (std::vector<uint8_t>)
 *
 * Metadata propagated:
 *   - StreamSource, StreamIndex, SemanticType (if configured)
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
    // Optional metadata to attach to output
    holoscan::Parameter<std::string> source_name_;
    holoscan::Parameter<int32_t> stream_index_;
};

}  // namespace tcn::ops
