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

#include "zenoh_subscriber_op.hpp"

#define ZENOHCXX_ZENOHC 1
#include <zenoh.hxx>

namespace tcn::ops {

void TcnZenohSubscriberOp::setup(holoscan::OperatorSpec& spec) {
    spec.output<std::vector<uint8_t>>("output");
    spec.output<std::string>("type_name");

    spec.param(topic_, "topic",
               "Topic",
               "Zenoh key expression to subscribe to",
               std::string{});
    spec.param(async_condition_, "async_condition",
               "Async Condition",
               "AsynchronousCondition for scheduling");
}

void TcnZenohSubscriberOp::initialize() {
    if (async_condition_.has_value()) {
        add_arg(async_condition_.get());
    }
    Operator::initialize();
}

void TcnZenohSubscriberOp::start() {
    if (!session_) {
        HOLOSCAN_LOG_ERROR("TcnZenohSubscriberOp: no session set.");
        return;
    }

    auto topic = topic_.get();
    HOLOSCAN_LOG_INFO("Subscribing to Zenoh topic: {}", topic);

    if (async_condition_.get()) {
        async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    }

    // Declare Zenoh subscriber with callback + on_drop.
    // The callback runs on Zenoh's internal thread — must be fast.
    auto key_expr = zenoh::KeyExpr(topic);
    auto sub = session_->declare_subscriber(
        key_expr,
        [this](const zenoh::Sample& sample) {
            // Check if we're shutting down
            if (async_condition_.get() &&
                async_condition_.get()->event_state() ==
                    holoscan::AsynchronousEventState::EVENT_NEVER) {
                return;
            }

            ZenohSample zs;

            // Extract type name from attachment (if present)
            auto attachment = sample.get_attachment();
            if (attachment.has_value()) {
                zs.type_name = attachment.value().get().as_string();
            }

            // Extract payload bytes (zenoh-cpp 1.3.4 Bytes API → string → vector)
            auto payload_str = sample.get_payload().as_string();
            zs.payload.assign(payload_str.begin(), payload_str.end());

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                // Back-pressure: drop oldest if queue full
                while (sample_queue_.size() >= kMaxQueuedSamples) {
                    sample_queue_.pop();
                    samples_dropped_.fetch_add(1, std::memory_order_relaxed);
                }
                sample_queue_.push(std::move(zs));
            }

            // Wake the Holoscan scheduler
            if (async_condition_.get() &&
                async_condition_.get()->event_state() ==
                    holoscan::AsynchronousEventState::EVENT_WAITING) {
                async_condition_.get()->event_state(
                    holoscan::AsynchronousEventState::EVENT_DONE);
            }
        },
        []() {});  // on_drop (no-op)

    subscriber_ = std::make_unique<zenoh::Subscriber<void>>(std::move(sub));
    HOLOSCAN_LOG_INFO("Zenoh subscriber active on: {}", topic);
}

void TcnZenohSubscriberOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context) {

    ZenohSample sample;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (sample_queue_.empty()) {
            HOLOSCAN_LOG_WARN("compute() called with empty sample queue");
            if (async_condition_.get()) {
                async_condition_.get()->event_state(
                    holoscan::AsynchronousEventState::EVENT_WAITING);
            }
            return;
        }
        sample = std::move(sample_queue_.front());
        sample_queue_.pop();
    }

    // Emit CDR type name for downstream decoders
    op_output.emit(sample.type_name, "type_name");

    // Emit raw CDR payload bytes
    op_output.emit(sample.payload, "output");

    // Reset async condition for next sample
    if (async_condition_.get()) {
        async_condition_.get()->event_state(
            holoscan::AsynchronousEventState::EVENT_WAITING);
    }
}

void TcnZenohSubscriberOp::stop() {
    if (async_condition_.get()) {
        async_condition_.get()->event_state(
            holoscan::AsynchronousEventState::EVENT_NEVER);
    }

    // Undeclare subscriber (stops callbacks)
    subscriber_.reset();

    // Drain queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!sample_queue_.empty()) {
            sample_queue_.pop();
        }
    }

    auto dropped = samples_dropped_.load(std::memory_order_relaxed);
    if (dropped > 0) {
        HOLOSCAN_LOG_WARN("TcnZenohSubscriberOp: {} samples dropped (back-pressure)", dropped);
    }

    HOLOSCAN_LOG_INFO("TcnZenohSubscriberOp stopped");
}

}  // namespace tcn::ops
