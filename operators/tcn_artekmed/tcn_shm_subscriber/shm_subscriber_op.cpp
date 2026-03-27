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

#include "shm_subscriber_op.hpp"

#include <cmath>
#include <cstring>

#include <holoscan/utils/cuda_macros.hpp>

#include "../common/utils.h"

namespace tcn::ops {

void TcnShmSubscriberOp::setup(holoscan::OperatorSpec& spec) {
    spec.output<holoscan::gxf::Entity>("color_outputs");
    spec.output<holoscan::gxf::Entity>("depth_outputs");

    spec.param(stream_name_, "stream_name",
               "Stream Name",
               "SHM stream name to subscribe to",
               std::string{});
    spec.param(cycle_time_ms_, "cycle_time_ms",
               "Cycle Time (ms)",
               "Wait time between frame polls in milliseconds",
               int32_t{1});
    spec.param(allocator_, "allocator",
               "Allocator",
               "Memory allocator for output tensors");
    spec.param(async_condition_, "async_condition",
               "Async Condition",
               "AsynchronousCondition for scheduling");
}

void TcnShmSubscriberOp::initialize() {
    // Register the AsynchronousCondition as a scheduling condition
    if (async_condition_.has_value()) {
        add_arg(async_condition_.get());
    }
    Operator::initialize();
}

void TcnShmSubscriberOp::start() {
    if (!receiver_) {
        HOLOSCAN_LOG_ERROR("ShmSubscriberOp: no receiver set. "
                           "Set receiver_ before starting the operator.");
        return;
    }

    // Subscribe to the frame data stream
    auto name = stream_name_.get();
    if (!receiver_->subscribe(name)) {
        HOLOSCAN_LOG_ERROR("Failed to subscribe to stream: {}", name);
        return;
    }

    should_stop_.store(false);

    // Set async condition to waiting state
    if (async_condition_.get()) {
        async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    }

    // Launch background receiver thread
    receiver_thread_ = std::thread(&TcnShmSubscriberOp::receiver_mainloop, this);

    HOLOSCAN_LOG_INFO("ShmSubscriberOp started for stream: {}", name);
}

void TcnShmSubscriberOp::stop() {
    should_stop_.store(true);

    // Signal the async condition to stop
    if (async_condition_.get()) {
        async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
    }

    // Wait for receiver thread to finish
    if (receiver_thread_.joinable()) {
        receiver_thread_.join();
    }

    // Clean up receiver
    if (receiver_) {
        receiver_->teardown();
    }

    HOLOSCAN_LOG_INFO("ShmSubscriberOp stopped");
}

void TcnShmSubscriberOp::receiver_mainloop() {
    while (!should_stop_.load()) {
        bool ok = receiver_->receive_frame(
            [this](const tcn::shm::ShmSerializedStreamHeader& header,
                   artekmed::shm::ShmBufferDescriptor::Reader descriptor) -> bool {
                return on_receive(header, descriptor);
            },
            cycle_time_ms_.get());

        if (!ok && !should_stop_.load()) {
            HOLOSCAN_LOG_WARN("Could not receive frame.");
        }
    }
}

bool TcnShmSubscriberOp::on_receive(
    const tcn::shm::ShmSerializedStreamHeader& header,
    artekmed::shm::ShmBufferDescriptor::Reader descriptor) {

    // Check if we should stop
    if (async_condition_.get() &&
        async_condition_.get()->event_state() == holoscan::AsynchronousEventState::EVENT_NEVER) {
        return false;
    }

    ShmFrameData frame;
    frame.timestamp = header.timestamp;

    // Iterate over ports in the buffer descriptor
    for (auto port : descriptor.getPorts()) {
        auto port_name = std::string(port.getName().cStr());
        auto port_data = port.getData();
        auto port_type = port_data.getPortType();
        auto metadata = port_data.getMetadata();
        auto stream_header = metadata.getHeader();

        int32_t dimX = stream_header.getDimX();
        int32_t dimY = stream_header.getDimY();
        int32_t bitsPerElement = stream_header.getBitsPerElement();
        int32_t bytesPerPixel = bitsPerElement / 8;

        auto raw_data = port_data.getData();
        auto data_ptr = reinterpret_cast<const uint8_t*>(raw_data.begin());
        auto data_size = raw_data.size();

        if (port_type == artekmed::schema::CameraPortType::COLORIMAGE) {
            frame.color_data[port_name].assign(data_ptr, data_ptr + data_size);
            frame.color_dims[port_name] = {dimX, dimY, bytesPerPixel};
        } else if (port_type == artekmed::schema::CameraPortType::DEPTHIMAGE) {
            frame.depth_data[port_name].assign(data_ptr, data_ptr + data_size);
            frame.depth_dims[port_name] = {dimX, dimY, 1};
        }
    }

    if (!frame.color_data.empty() || !frame.depth_data.empty()) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            frame_queue_.push(std::move(frame));
        }

        // Notify the Holoscan scheduler
        if (async_condition_.get() &&
            async_condition_.get()->event_state() == holoscan::AsynchronousEventState::EVENT_WAITING) {
            async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        }
        return true;
    }

    return false;
}

void TcnShmSubscriberOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context) {

    // Pop a frame from the queue
    ShmFrameData frame;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (frame_queue_.empty()) {
            HOLOSCAN_LOG_WARN("compute() called with empty frame queue");
            if (async_condition_.get()) {
                async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
            }
            return;
        }
        frame = std::move(frame_queue_.front());
        frame_queue_.pop();
    }

    HOLOSCAN_LOG_DEBUG("Processing frame with timestamp {}", frame.timestamp);

    // Get allocator handle for tensor allocation
    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_->gxf_cid());
    if (!allocator_handle) {
        HOLOSCAN_LOG_ERROR("Failed to get allocator handle");
        return;
    }

    // Get CUDA stream
    cudaStream_t cuda_stream = nullptr;  // Use default stream

    // Create color output entity
    auto color_entity = nvidia::gxf::Entity::New(context.context());
    if (!color_entity) {
        HOLOSCAN_LOG_ERROR("Failed to create color entity");
        return;
    }

    for (auto& [port_name, pixels] : frame.color_data) {
        auto& dims = frame.color_dims[port_name];
        nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
        if (!tcn::allocate_named_tensor<uint8_t>(
                allocator_handle.value(), cuda_stream, color_entity.value(),
                nvidia::gxf::Shape{{dims.height, dims.width, dims.channels}},
                nvidia::gxf::MemoryStorageType::kDevice,
                port_name, tensor)) {
            HOLOSCAN_LOG_ERROR("Failed to allocate color tensor for {}", port_name);
            continue;
        }

        // Copy pixel data to GPU
        auto maybe_data = tensor->data<uint8_t>();
        if (maybe_data) {
            HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
                maybe_data.value(), pixels.data(), pixels.size(),
                cudaMemcpyHostToDevice, cuda_stream));
        }
    }

    // Create depth output entity
    auto depth_entity = nvidia::gxf::Entity::New(context.context());
    if (!depth_entity) {
        HOLOSCAN_LOG_ERROR("Failed to create depth entity");
        return;
    }

    for (auto& [port_name, pixels] : frame.depth_data) {
        auto& dims = frame.depth_dims[port_name];
        nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
        if (!tcn::allocate_named_tensor<uint16_t>(
                allocator_handle.value(), cuda_stream, depth_entity.value(),
                nvidia::gxf::Shape{{dims.height, dims.width, dims.channels}},
                nvidia::gxf::MemoryStorageType::kDevice,
                port_name, tensor)) {
            HOLOSCAN_LOG_ERROR("Failed to allocate depth tensor for {}", port_name);
            continue;
        }

        auto maybe_data = tensor->data<uint16_t>();
        if (maybe_data) {
            HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
                maybe_data.value(), pixels.data(), pixels.size(),
                cudaMemcpyHostToDevice, cuda_stream));
        }
    }

    // Synchronize before emitting
    HOLOSCAN_CUDA_CALL(cudaStreamSynchronize(cuda_stream));

    // Emit outputs
    op_output.emit(color_entity.value(), "color_outputs");
    op_output.emit(depth_entity.value(), "depth_outputs");

    // Reset async condition to wait for next frame
    if (async_condition_.get()) {
        async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    }
}

}  // namespace tcn::ops
