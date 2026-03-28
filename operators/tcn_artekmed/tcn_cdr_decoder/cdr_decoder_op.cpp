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

#include "cdr_decoder_op.hpp"
#include "cdr_serde.hpp"

#include <tcnart_msgs/msg/VideoStream.h>

namespace tcn::ops {

void TcnCdrDecoderOp::setup(holoscan::OperatorSpec& spec) {
    spec.input<std::vector<uint8_t>>("input");
    spec.input<std::string>("type_name");
    spec.output<std::vector<uint8_t>>("output");

    spec.param(source_name_, "source_name",
               "Source Name",
               "Stream source identifier",
               std::string{});
    spec.param(stream_index_, "stream_index",
               "Stream Index",
               "Stream index",
               static_cast<int32_t>(0));
}

void TcnCdrDecoderOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context) {

    auto payload = op_input.receive<std::vector<uint8_t>>("input").value();

    // Read CDR type name from upstream port
    auto type_name_opt = op_input.receive<std::string>("type_name");
    std::string type_name = type_name_opt.has_value() ? type_name_opt.value() : "";

    if (type_name.empty()) {
        HOLOSCAN_LOG_WARN("CdrDecoderOp: no type_name received, skipping");
        return;
    }

    // Deserialize VideoStreamMessage
    tcn::cdr::CdrBufferReader reader;
    tcnart_msgs::msg::VideoStream video_msg;

    if (!reader.read(payload, video_msg)) {
        HOLOSCAN_LOG_ERROR("CdrDecoderOp: failed to deserialize VideoStreamMessage");
        return;
    }

    // Extract image bytes from the deserialized message
    auto& image_data = video_msg.image();
    std::vector<uint8_t> image_bytes(image_data.begin(), image_data.end());

    op_output.emit(std::move(image_bytes), "output");
}

}  // namespace tcn::ops
