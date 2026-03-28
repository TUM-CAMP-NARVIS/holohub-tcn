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

#include "zenoh_receiver_op.hpp"

#define ZENOHCXX_ZENOHC 1
#include <zenoh.hxx>

#include <gxf/std/tensor.hpp>

#include <holoscan/utils/cuda_macros.hpp>

#include "../tcn_cdr_serde/cdr_type_registry.hpp"

namespace tcn::ops {

// ---------------------------------------------------------------------------
// Discovery: resolve Zenoh streams via RPC + descriptor GET
// ---------------------------------------------------------------------------

std::vector<ZenohStreamConfig> TcnZenohReceiverOp::discover_streams(
    zenoh::Session& session,
    const std::string& topic_prefix,
    const std::string& capture_node,
    const std::vector<std::string>& stream_types) {

    std::vector<ZenohStreamConfig> configs;

    // Phase 1: Discover camera sensors via RPC GET
    // Key: {topic_prefix}/{capture_node}/rpc/sensor/*/describe
    std::string discover_topic =
        topic_prefix + "/" + capture_node + "/rpc/sensor/*/describe";
    HOLOSCAN_LOG_INFO("Discovering cameras on: {}", discover_topic);

    auto key_expr = zenoh::KeyExpr(discover_topic);
    auto replies = session.get(key_expr, "", zenoh::channels::FifoChannel(16));

    // Collect sensor names from replies
    std::vector<std::string> sensor_names;
    while (true) {
        auto reply_opt = replies.try_recv();
        if (!reply_opt.has_value()) break;
        auto& reply = reply_opt.value();

        if (reply.is_ok()) {
            auto& sample = reply.get_ok();
            auto key_str = sample.get_keyexpr().as_string_view();

            // Extract sensor name from key:
            // {prefix}/{capture_node}/rpc/sensor/{sensor_name}/describe
            std::string key(key_str);
            auto rpc_pos = key.find("/rpc/sensor/");
            if (rpc_pos != std::string::npos) {
                auto name_start = rpc_pos + std::string("/rpc/sensor/").size();
                auto name_end = key.find("/describe", name_start);
                if (name_end != std::string::npos) {
                    sensor_names.push_back(key.substr(name_start, name_end - name_start));
                }
            }
        }
    }

    HOLOSCAN_LOG_INFO("Discovered {} sensors", sensor_names.size());

    // Phase 2: For each sensor, fetch stream descriptors
    int32_t stream_index = 0;
    for (const auto& sensor : sensor_names) {
        for (const auto& stype : stream_types) {
            // Key: {topic_prefix}/{sensor}/cfg/dsc/{stype}_image_bitstream
            std::string desc_topic =
                topic_prefix + "/" + sensor + "/cfg/dsc/" + stype + "_image_bitstream";

            auto desc_key = zenoh::KeyExpr(desc_topic);
            auto desc_replies = session.get(desc_key, "", zenoh::channels::FifoChannel(4));

            auto desc_reply_opt = desc_replies.try_recv();
            if (!desc_reply_opt.has_value() || !desc_reply_opt.value().is_ok()) {
                HOLOSCAN_LOG_WARN("No descriptor for {}/{} — skipping", sensor, stype);
                continue;
            }

            auto& desc_sample = desc_reply_opt.value().get_ok();

            // CDR-decode the StreamDescriptorMessage
            auto payload_str = desc_sample.get_payload().as_string();
            std::vector<uint8_t> desc_bytes(payload_str.begin(), payload_str.end());

            auto& registry = tcn::cdr::CdrTypeRegistry::instance();
            tcn::cdr::DecodedMessage decoded;
            std::string desc_type = "tcnart_msgs::msg::StreamDescriptorMessage";

            ZenohStreamConfig cfg;
            cfg.sensor_name = sensor;
            cfg.name = sensor + "_" + stype;
            cfg.stream_index = stream_index++;

            if (registry.decode(desc_type, desc_bytes, decoded)) {
                // Extract stream topic from metadata
                auto it = decoded.metadata.find("stream_topic");
                if (it != decoded.metadata.end()) {
                    cfg.topic = it->second;
                } else {
                    cfg.topic = desc_topic;  // Fallback to descriptor topic
                }

                // Extract image dimensions
                auto w_it = decoded.metadata.find("image_width");
                if (w_it != decoded.metadata.end()) cfg.image_width = std::stoi(w_it->second);
                auto h_it = decoded.metadata.find("image_height");
                if (h_it != decoded.metadata.end()) cfg.image_height = std::stoi(h_it->second);
                auto s_it = decoded.metadata.find("image_step");
                if (s_it != decoded.metadata.end()) cfg.image_step = std::stoi(s_it->second);
                auto f_it = decoded.metadata.find("image_format");
                if (f_it != decoded.metadata.end()) cfg.image_format = std::stoi(f_it->second);
                auto c_it = decoded.metadata.find("image_compression");
                if (c_it != decoded.metadata.end()) cfg.image_compression = std::stoi(c_it->second);
                auto r_it = decoded.metadata.find("frame_rate");
                if (r_it != decoded.metadata.end()) cfg.frame_rate = std::stof(r_it->second);
            } else {
                HOLOSCAN_LOG_WARN("Failed to decode descriptor for {} — using topic as-is",
                                  cfg.name);
                cfg.topic = desc_topic;
            }

            HOLOSCAN_LOG_INFO("Stream '{}': topic={} ({}x{}, compression={})",
                              cfg.name, cfg.topic, cfg.image_width, cfg.image_height,
                              cfg.image_compression);
            configs.push_back(std::move(cfg));
        }
    }

    return configs;
}

// ---------------------------------------------------------------------------
// Operator lifecycle
// ---------------------------------------------------------------------------

void TcnZenohReceiverOp::setup(holoscan::OperatorSpec& spec) {
    spec.param(async_condition_, "async_condition",
               "Async Condition",
               "AsynchronousCondition for event-driven scheduling");
    spec.param(allocator_, "allocator",
               "Allocator",
               "GPU memory allocator");
    spec.param(cuda_stream_pool_, "cuda_stream_pool",
               "CUDA Stream Pool",
               "Pool for CUDA streams",
               holoscan::ParameterFlag::kOptional);

    // Register dynamic output ports — one per discovered stream
    for (const auto& cfg : stream_configs_) {
        spec.output<holoscan::gxf::Entity>(cfg.name);
    }
}

void TcnZenohReceiverOp::initialize() {
    if (async_condition_.has_value()) {
        add_arg(async_condition_.get());
    }
    Operator::initialize();
}

void TcnZenohReceiverOp::start() {
    if (!session_) {
        HOLOSCAN_LOG_ERROR("TcnZenohReceiverOp: no Zenoh session set");
        return;
    }

    if (stream_configs_.empty()) {
        HOLOSCAN_LOG_WARN("TcnZenohReceiverOp: no streams configured");
        return;
    }

    // Create CUDA stream for host-to-device uploads
    HOLOSCAN_CUDA_CALL(cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking));

    // Set initial async state
    if (async_condition_.get()) {
        async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    }

    // Create per-stream state and subscribe
    for (const auto& cfg : stream_configs_) {
        auto state = std::make_unique<StreamState>();
        state->config = cfg;

        HOLOSCAN_LOG_INFO("Subscribing to stream '{}' on topic: {}", cfg.name, cfg.topic);

        auto key_expr = zenoh::KeyExpr(cfg.topic);

        // Raw pointer for the lambda capture (StreamState outlives subscriber)
        auto* state_ptr = state.get();
        auto* cond_ptr = async_condition_.has_value() ? async_condition_.get().get() : nullptr;

        state->subscriber = std::make_unique<zenoh::Subscriber<void>>(
            session_->declare_subscriber(
                key_expr,
                [state_ptr, cond_ptr](const zenoh::Sample& sample) {
                    // Check shutdown
                    if (cond_ptr &&
                        cond_ptr->event_state() ==
                            holoscan::AsynchronousEventState::EVENT_NEVER) {
                        return;
                    }

                    ReceivedSample rs;

                    // Extract CDR type name from attachment
                    auto attachment = sample.get_attachment();
                    if (attachment.has_value()) {
                        rs.type_name = attachment.value().get().as_string();
                    }

                    // Extract payload bytes
                    auto payload_str = sample.get_payload().as_string();
                    rs.payload.assign(payload_str.begin(), payload_str.end());

                    {
                        std::lock_guard<std::mutex> lock(state_ptr->queue_mutex);
                        while (state_ptr->sample_queue.size() >=
                               StreamState::kMaxQueuedSamples) {
                            state_ptr->sample_queue.pop();
                            state_ptr->samples_dropped.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        state_ptr->sample_queue.push(std::move(rs));
                    }

                    // Wake scheduler
                    if (cond_ptr &&
                        cond_ptr->event_state() ==
                            holoscan::AsynchronousEventState::EVENT_WAITING) {
                        cond_ptr->event_state(
                            holoscan::AsynchronousEventState::EVENT_DONE);
                    }
                },
                []() {}));  // on_drop (no-op)

        stream_states_.push_back(std::move(state));
    }

    HOLOSCAN_LOG_INFO("TcnZenohReceiverOp started with {} streams", stream_states_.size());
}

void TcnZenohReceiverOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context) {

    auto& registry = tcn::cdr::CdrTypeRegistry::instance();

    // Get GXF allocator handle from Holoscan allocator
    auto gxf_alloc = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_.get()->gxf_cid()).value();

    for (auto& state : stream_states_) {
        ReceivedSample sample;
        {
            std::lock_guard<std::mutex> lock(state->queue_mutex);
            if (state->sample_queue.empty()) continue;
            sample = std::move(state->sample_queue.front());
            state->sample_queue.pop();
        }

        const auto& cfg = state->config;

        // Determine the bytes to upload (raw or CDR-decoded)
        const uint8_t* frame_data = nullptr;
        size_t frame_size = 0;
        int32_t height = cfg.image_height;
        int32_t width = cfg.image_width;

        tcn::cdr::DecodedMessage decoded;

        if (!sample.type_name.empty() && registry.has_type(sample.type_name)) {
            if (!registry.decode(sample.type_name, sample.payload, decoded)) {
                HOLOSCAN_LOG_WARN("Stream '{}': CDR decode failed for type '{}'",
                                  cfg.name, sample.type_name);
                continue;
            }

            if (decoded.payload.empty()) {
                // Metadata-only message — emit empty entity
                auto out_entity = holoscan::gxf::Entity::New(&context);
                op_output.emit(out_entity, cfg.name.c_str());
                continue;
            }

            // Override dimensions from decoded metadata if available
            auto h_it = decoded.metadata.find("image_height");
            auto w_it = decoded.metadata.find("image_width");
            if (h_it != decoded.metadata.end()) height = std::stoi(h_it->second);
            if (w_it != decoded.metadata.end()) width = std::stoi(w_it->second);

            frame_data = decoded.payload.data();
            frame_size = decoded.payload.size();
        } else {
            HOLOSCAN_LOG_DEBUG("Stream '{}': unknown CDR type '{}', emitting raw",
                               cfg.name, sample.type_name);
            frame_data = sample.payload.data();
            frame_size = sample.payload.size();
        }

        if (frame_size == 0) continue;

        // Compute tensor shape: HxWxC if dimensions known, otherwise 1D
        nvidia::gxf::Shape shape;
        if (height > 0 && width > 0) {
            int32_t channels = static_cast<int32_t>(
                static_cast<int64_t>(frame_size) / (height * width));
            if (channels < 1) channels = 1;
            shape = nvidia::gxf::Shape({height, width, channels});
        } else {
            shape = nvidia::gxf::Shape({static_cast<int32_t>(frame_size)});
        }

        auto strides = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(uint8_t));

        // Allocate GPU tensor and upload
        auto out_entity = holoscan::gxf::Entity::New(&context);
        auto out_tensor = static_cast<nvidia::gxf::Entity&>(out_entity)
                              .add<nvidia::gxf::Tensor>("");
        if (!out_tensor) {
            HOLOSCAN_LOG_ERROR("Stream '{}': failed to add tensor to entity", cfg.name);
            continue;
        }

        out_tensor.value()->reshapeCustom(
            shape, nvidia::gxf::PrimitiveType::kUnsigned8,
            sizeof(uint8_t), strides,
            nvidia::gxf::MemoryStorageType::kDevice, gxf_alloc);

        HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
            out_tensor.value()->pointer(),
            frame_data, frame_size,
            cudaMemcpyHostToDevice, upload_stream_));
        HOLOSCAN_CUDA_CALL(cudaStreamSynchronize(upload_stream_));

        op_output.emit(out_entity, cfg.name.c_str());
        state->frames_emitted.fetch_add(1, std::memory_order_relaxed);
    }

    // Reset async condition for next wake
    if (async_condition_.get()) {
        async_condition_.get()->event_state(
            holoscan::AsynchronousEventState::EVENT_WAITING);
    }
}

void TcnZenohReceiverOp::stop() {
    // Signal shutdown to callbacks
    if (async_condition_.get()) {
        async_condition_.get()->event_state(
            holoscan::AsynchronousEventState::EVENT_NEVER);
    }

    // Undeclare subscribers and drain queues
    for (auto& state : stream_states_) {
        state->subscriber.reset();
        {
            std::lock_guard<std::mutex> lock(state->queue_mutex);
            while (!state->sample_queue.empty()) {
                state->sample_queue.pop();
            }
        }
        auto dropped = state->samples_dropped.load(std::memory_order_relaxed);
        auto emitted = state->frames_emitted.load(std::memory_order_relaxed);
        HOLOSCAN_LOG_INFO("Stream '{}': {} frames emitted, {} samples dropped",
                          state->config.name, emitted, dropped);
    }
    stream_states_.clear();

    if (upload_stream_) {
        cudaStreamSynchronize(upload_stream_);
        cudaStreamDestroy(upload_stream_);
        upload_stream_ = nullptr;
    }

    HOLOSCAN_LOG_INFO("TcnZenohReceiverOp stopped");
}

}  // namespace tcn::ops
