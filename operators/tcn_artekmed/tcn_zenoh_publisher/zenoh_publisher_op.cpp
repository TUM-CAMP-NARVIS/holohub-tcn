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

#include "zenoh_publisher_op.hpp"

#define ZENOHCXX_ZENOHC 1
#include <zenoh.hxx>

namespace tcn::ops {

void TcnZenohPublisherOp::setup(holoscan::OperatorSpec& spec) {
    spec.input<std::vector<uint8_t>>("input");
    spec.input<std::string>("type_name").condition(
        holoscan::ConditionType::kNone);

    spec.param(topic_, "topic",
               "Topic",
               "Zenoh key expression to publish on",
               std::string{});
}

void TcnZenohPublisherOp::start() {
    if (!session_) {
        HOLOSCAN_LOG_ERROR("TcnZenohPublisherOp: no session set.");
        return;
    }

    auto topic = topic_.get();
    HOLOSCAN_LOG_INFO("Declaring Zenoh publisher on: {}", topic);

    auto key_expr = zenoh::KeyExpr(topic);
    auto pub = session_->declare_publisher(key_expr);
    publisher_ = std::make_unique<zenoh::Publisher>(std::move(pub));

    HOLOSCAN_LOG_INFO("Zenoh publisher active on: {}", topic);
}

void TcnZenohPublisherOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext&,
    holoscan::ExecutionContext& context) {

    auto payload = op_input.receive<std::vector<uint8_t>>("input").value();

    if (payload.empty()) {
        return;
    }

    // Read CDR type name from optional input port (if connected)
    std::string type_name;
    auto type_name_opt = op_input.receive<std::string>("type_name");
    if (type_name_opt) {
        type_name = type_name_opt.value();
    }

    // Publish with CDR encoding and type name as attachment
    zenoh::Bytes zbytes(std::string(
        reinterpret_cast<const char*>(payload.data()), payload.size()));

    auto options = zenoh::Publisher::PutOptions::create_default();
    options.encoding = zenoh::Encoding::Predefined::application_cdr();
    if (!type_name.empty()) {
        options.attachment = type_name;
    }

    publisher_->put(std::move(zbytes), std::move(options));

    auto count = messages_published_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count % 100 == 0) {
        HOLOSCAN_LOG_DEBUG("TcnZenohPublisherOp: published {} messages on {}",
                           count, topic_.get());
    }
}

void TcnZenohPublisherOp::stop() {
    publisher_.reset();

    auto published = messages_published_.load(std::memory_order_relaxed);
    HOLOSCAN_LOG_INFO("TcnZenohPublisherOp stopped ({} messages published on {})",
                      published, topic_.get());
}

}  // namespace tcn::ops
