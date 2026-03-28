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
    spec.output<std::vector<uint8_t>>("output");

    spec.param(source_name_, "source_name",
               "Source Name",
               "Stream source identifier for metadata",
               std::string{});
    spec.param(stream_index_, "stream_index",
               "Stream Index",
               "Stream index for metadata",
               static_cast<int32_t>(0));
}

void TcnCdrDecoderOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context) {

    auto payload = op_input.receive<std::vector<uint8_t>>("input").value();

    // Read CDR type name from upstream metadata
    auto in_metadata = context.get_input_metadata("input");
    std::string type_name;
    if (in_metadata) {
        type_name = in_metadata->get<std::string>("CdrTypeName", "");
    }

    if (type_name.empty()) {
        HOLOSCAN_LOG_WARN("CdrDecoderOp: no CdrTypeName in metadata, skipping");
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

    // Propagate metadata
    auto out_metadata = context.get_output_metadata("output");
    if (out_metadata) {
        out_metadata->set("CdrTypeName", type_name);
        if (!source_name_.get().empty()) {
            out_metadata->set("StreamSource", source_name_.get());
        }
        out_metadata->set("StreamIndex", stream_index_.get());
    }

    op_output.emit(std::move(image_bytes), "output");
}

}  // namespace tcn::ops
