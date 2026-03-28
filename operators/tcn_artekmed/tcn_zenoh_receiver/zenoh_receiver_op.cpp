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

#include "../tcn_cdr_serde/cdr_serde.hpp"
#include "../tcn_cdr_serde/cdr_type_registry.hpp"

// tcn_schema generated message types for RPC discovery
#include <pcpd_msgs/rpc/ServiceController.h>
#include <pcpd_msgs/msg/CameraSensor.h>
#include <tcnart_msgs/rpc/Requests.h>
#include <tcnart_msgs/msg/StreamDescriptor.h>

namespace tcn::ops {

// ---------------------------------------------------------------------------
// Discovery: resolve Zenoh streams via RPC + descriptor GET
//
// Protocol (matches artekmed pcp_shm_zenoh_receiver):
//   Phase 1: GET {prefix}/{capture_node}/rpc/sensor/*/describe
//            Send NullRequest CDR payload, receive DeviceContextReply per sensor
//   Phase 2: For each sensor with enabled streams:
//            GET {prefix}/{sensor}/cfg/dsc/{type}_image_bitstream
//            Receive StreamDescriptorMessage with actual data topic + metadata
//   Phase 3: subscribe_streams() uses the stream_topic from each descriptor
// ---------------------------------------------------------------------------

std::vector<ZenohStreamConfig> TcnZenohReceiverOp::discover_streams(
    zenoh::Session& session,
    const std::string& topic_prefix,
    const std::string& capture_node,
    const std::vector<std::string>& stream_types) {

    std::vector<ZenohStreamConfig> configs;

    // -----------------------------------------------------------------------
    // Phase 1: Discover camera sensors via RPC GET
    // -----------------------------------------------------------------------
    std::string discover_topic =
        topic_prefix + "/" + capture_node + "/rpc/sensor/*/describe";
    HOLOSCAN_LOG_INFO("Phase 1 — discovering sensors: {}", discover_topic);

    // CDR-encode NullRequest payload (matches Python/C++ reference)
    tcnart_msgs::rpc::NullRequest null_req;
    tcn::cdr::CdrBufferWriter writer;
    auto null_bytes = writer.write(null_req);

    // Configure GET options: query ALL queriables, 5s timeout, CDR encoding
    zenoh::Session::GetOptions get_opts = zenoh::Session::GetOptions::create_default();
    get_opts.timeout_ms = 5000;
    get_opts.target = Z_QUERY_TARGET_ALL;
    get_opts.consolidation = zenoh::QueryConsolidation(Z_CONSOLIDATION_MODE_MONOTONIC);
    get_opts.payload = zenoh::Bytes(std::move(null_bytes));
    get_opts.encoding = zenoh::Encoding::Predefined::application_cdr();

    auto replies = session.get(
        zenoh::KeyExpr(discover_topic), "",
        zenoh::channels::FifoChannel(16),
        std::move(get_opts));

    // Collect sensor info from DeviceContextReply messages (blocking recv)
    struct SensorInfo {
        std::string name;
        bool color_enabled = false;
        bool depth_enabled = false;
        bool infrared_enabled = false;
    };
    std::vector<SensorInfo> sensors;

    while (true) {
        auto result = replies.recv();
        if (std::holds_alternative<zenoh::channels::RecvError>(result)) {
            break;  // Z_DISCONNECTED — all replies received
        }

        auto& reply = std::get<zenoh::Reply>(result);
        if (!reply.is_ok()) {
            HOLOSCAN_LOG_WARN("Received error reply during sensor discovery");
            continue;
        }

        auto& sample = reply.get_ok();
        auto payload_vec = sample.get_payload().as_vector();

        // Decode DeviceContextReply to get CameraSensor with enabled streams
        pcpd_msgs::rpc::DeviceContextReply ctx_reply;
        tcn::cdr::CdrBufferReader reader;
        if (reader.read(payload_vec.data(), payload_vec.size(), ctx_reply)) {
            const auto& cam = ctx_reply.value();
            SensorInfo info;
            info.name = cam.name();
            info.color_enabled = cam.color_enabled();
            info.depth_enabled = cam.depth_enabled();
            info.infrared_enabled = cam.infrared_enabled();
            HOLOSCAN_LOG_INFO("  Sensor '{}': color={}, depth={}, infrared={} (type: {})",
                              info.name, info.color_enabled, info.depth_enabled,
                              info.infrared_enabled, ctx_reply.sensor_type());
            sensors.push_back(std::move(info));
        } else {
            // Fallback: extract sensor name from the reply key expression
            auto key_str = std::string(sample.get_keyexpr().as_string_view());
            auto rpc_pos = key_str.find("/rpc/sensor/");
            if (rpc_pos != std::string::npos) {
                auto name_start = rpc_pos + std::string("/rpc/sensor/").size();
                auto name_end = key_str.find("/describe", name_start);
                if (name_end != std::string::npos) {
                    SensorInfo info;
                    info.name = key_str.substr(name_start, name_end - name_start);
                    info.color_enabled = true;
                    info.depth_enabled = true;
                    HOLOSCAN_LOG_WARN("  Failed to decode DeviceContextReply, using key: '{}'",
                                      info.name);
                    sensors.push_back(std::move(info));
                }
            }
        }
    }

    HOLOSCAN_LOG_INFO("Discovered {} sensors", sensors.size());
    if (sensors.empty()) {
        HOLOSCAN_LOG_ERROR("No sensors responded to discovery query on '{}'", discover_topic);
        return configs;
    }

    // -----------------------------------------------------------------------
    // Phase 2: For each enabled stream, fetch its StreamDescriptor
    // -----------------------------------------------------------------------
    HOLOSCAN_LOG_INFO("Phase 2 — resolving stream descriptors");

    // Build stream_types filter set (empty = accept all enabled)
    std::set<std::string> type_filter(stream_types.begin(), stream_types.end());

    int32_t stream_index = 0;
    for (const auto& sensor : sensors) {
        // Map stream type → enabled flag
        struct StreamEntry {
            std::string type;
            bool enabled;
        };
        std::vector<StreamEntry> entries = {
            {"color", sensor.color_enabled},
            {"depth", sensor.depth_enabled},
            {"infrared", sensor.infrared_enabled},
        };

        for (const auto& entry : entries) {
            if (!entry.enabled) continue;
            if (!type_filter.empty() && type_filter.count(entry.type) == 0) continue;

            // Descriptor key: {prefix}/{sensor}/cfg/dsc/{type}_image_bitstream
            std::string desc_topic =
                topic_prefix + "/" + sensor.name + "/cfg/dsc/" + entry.type + "_image_bitstream";

            HOLOSCAN_LOG_INFO("  Fetching descriptor: {}", desc_topic);

            zenoh::Session::GetOptions desc_opts = zenoh::Session::GetOptions::create_default();
            desc_opts.timeout_ms = 5000;
            desc_opts.target = Z_QUERY_TARGET_ALL;
            desc_opts.consolidation = zenoh::QueryConsolidation(Z_CONSOLIDATION_MODE_MONOTONIC);

            auto desc_replies = session.get(
                zenoh::KeyExpr(desc_topic), "",
                zenoh::channels::FifoChannel(4),
                std::move(desc_opts));

            auto desc_result = desc_replies.recv();
            if (std::holds_alternative<zenoh::channels::RecvError>(desc_result)) {
                HOLOSCAN_LOG_WARN("  No descriptor reply for {}/{} — skipping",
                                  sensor.name, entry.type);
                continue;
            }

            auto& desc_reply = std::get<zenoh::Reply>(desc_result);
            if (!desc_reply.is_ok()) {
                HOLOSCAN_LOG_WARN("  Error reply for descriptor {}/{}", sensor.name, entry.type);
                continue;
            }

            auto& desc_sample = desc_reply.get_ok();
            auto desc_bytes = desc_sample.get_payload().as_vector();

            // CDR-decode StreamDescriptorMessage directly (no registry indirection)
            tcnart_msgs::msg::StreamDescriptorMessage desc_msg;
            tcn::cdr::CdrBufferReader desc_reader;

            ZenohStreamConfig cfg;
            cfg.sensor_name = sensor.name;
            cfg.name = sensor.name + "_" + entry.type;
            cfg.stream_index = stream_index++;

            if (desc_reader.read(desc_bytes.data(), desc_bytes.size(), desc_msg)) {
                cfg.topic = desc_msg.stream_topic();
                cfg.image_width = static_cast<int32_t>(desc_msg.image_width());
                cfg.image_height = static_cast<int32_t>(desc_msg.image_height());
                cfg.image_step = static_cast<int32_t>(desc_msg.image_step());
                cfg.image_format = static_cast<int32_t>(desc_msg.image_format());
                cfg.image_compression = static_cast<int32_t>(desc_msg.image_compression());
                cfg.frame_rate = static_cast<float>(desc_msg.frame_rate());
            } else {
                HOLOSCAN_LOG_WARN("  Failed to decode StreamDescriptorMessage for {}",
                                  cfg.name);
                // Fallback: use registry-based decode
                auto& registry = tcn::cdr::CdrTypeRegistry::instance();
                tcn::cdr::DecodedMessage decoded;
                if (registry.decode("tcnart_msgs::msg::StreamDescriptorMessage",
                                    desc_bytes, decoded)) {
                    auto it = decoded.metadata.find("stream_topic");
                    cfg.topic = (it != decoded.metadata.end()) ? it->second : desc_topic;
                    auto w = decoded.metadata.find("image_width");
                    if (w != decoded.metadata.end()) cfg.image_width = std::stoi(w->second);
                    auto h = decoded.metadata.find("image_height");
                    if (h != decoded.metadata.end()) cfg.image_height = std::stoi(h->second);
                    auto s = decoded.metadata.find("image_step");
                    if (s != decoded.metadata.end()) cfg.image_step = std::stoi(s->second);
                    auto f = decoded.metadata.find("image_format");
                    if (f != decoded.metadata.end()) cfg.image_format = std::stoi(f->second);
                    auto c = decoded.metadata.find("image_compression");
                    if (c != decoded.metadata.end()) cfg.image_compression = std::stoi(c->second);
                    auto r = decoded.metadata.find("frame_rate");
                    if (r != decoded.metadata.end()) cfg.frame_rate = std::stof(r->second);
                } else {
                    HOLOSCAN_LOG_ERROR("  Cannot decode descriptor for {} — skipping", cfg.name);
                    stream_index--;
                    continue;
                }
            }

            if (cfg.topic.empty()) {
                HOLOSCAN_LOG_WARN("  Descriptor for {} has no stream_topic — skipping", cfg.name);
                stream_index--;
                continue;
            }

            HOLOSCAN_LOG_INFO("  Stream '{}': topic={} ({}x{}, compression={}, fps={})",
                              cfg.name, cfg.topic, cfg.image_width, cfg.image_height,
                              cfg.image_compression, cfg.frame_rate);
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

                    // Extract payload bytes (use as_vector for binary CDR data)
                    rs.payload = sample.get_payload().as_vector();

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
