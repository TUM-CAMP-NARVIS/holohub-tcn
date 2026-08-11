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

#include <holoscan/utils/cuda_macros.hpp>

#include "gxf/std/timestamp.hpp"  // nvidia::gxf::Timestamp -- acquisition time for frame grouping

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

    // Create a dedicated CUDA stream for SHM→GPU async copies
    HOLOSCAN_CUDA_CALL(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));

    should_stop_.store(false);

    // Set async condition to waiting state
    if (async_condition_.get()) {
        async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    }

    // Launch background receiver thread
    receiver_thread_ = std::thread(&TcnShmSubscriberOp::receiver_mainloop, this);

    HOLOSCAN_LOG_INFO("ShmSubscriberOp started for stream: {} (zero-copy)", name);
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

    // Drain the queue to release any held SHM segments
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!frame_queue_.empty()) {
            frame_queue_.pop();
        }
    }

    // Destroy CUDA stream
    if (copy_stream_) {
        cudaStreamSynchronize(copy_stream_);
        cudaStreamDestroy(copy_stream_);
        copy_stream_ = nullptr;
    }

    // Clean up receiver
    if (receiver_) {
        receiver_->teardown();
    }

    HOLOSCAN_LOG_INFO("ShmSubscriberOp stopped");
}

void TcnShmSubscriberOp::receiver_mainloop() {
    last_stats_log_ = std::chrono::steady_clock::now();

    while (!should_stop_.load()) {
        auto frame = receiver_->receive_frame_zero_copy(cycle_time_ms_.get());
        if (!frame.has_value()) {
            continue;
        }

        // Check if we should stop before queuing
        if (async_condition_.get() &&
            async_condition_.get()->event_state() == holoscan::AsynchronousEventState::EVENT_NEVER) {
            break;
        }

        frames_received_.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            // Back-pressure: if queue is full, drop oldest frame to keep
            // at most kMaxQueuedFrames in-flight (releases the SHM segment).
            while (frame_queue_.size() >= kMaxQueuedFrames) {
                frame_queue_.pop();
                frames_skipped_.fetch_add(1, std::memory_order_relaxed);
            }
            frame_queue_.push(std::move(*frame));
        }

        // Periodic stats log (every 5 seconds)
        auto now = std::chrono::steady_clock::now();
        if (now - last_stats_log_ >= std::chrono::seconds(5)) {
            auto skipped = frames_skipped_.exchange(0, std::memory_order_relaxed);
            auto received = frames_received_.exchange(0, std::memory_order_relaxed);
            if (skipped > 0) {
                HOLOSCAN_LOG_WARN("SHM receiver: {}/{} frames skipped in last 5s",
                                  skipped, received);
            } else {
                HOLOSCAN_LOG_INFO("SHM receiver: {}/{} frames processed in last 5s",
                                  received, received);
            }
            last_stats_log_ = now;
        }

        // Notify the Holoscan scheduler
        if (async_condition_.get() &&
            async_condition_.get()->event_state() == holoscan::AsynchronousEventState::EVENT_WAITING) {
            async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        }
    }
}

void TcnShmSubscriberOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context) {

    // Pop a zero-copy frame from the queue
    tcn::shm::ShmZeroCopyFrame frame;
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

    HOLOSCAN_LOG_DEBUG("Processing frame ts={} (zero-copy, {} ports)",
                       frame.timestamp, frame.ports.size());

    // Capture the acquisition timestamp BEFORE `frame` is cleared further down (`frame = {}` runs
    // before the emits, to release the SHM segment as early as possible), otherwise this reads 0.
    //
    // Units are nanoseconds, but the EPOCH is the publisher's std::chrono::steady_clock
    // (tcn_shm_zenoh_sender: user_header.timestamp = now_ns), NOT the GXF global clock and not wall
    // time. So these values are only meaningful RELATIVE TO EACH OTHER: they are safe to compare
    // and match across streams from the same publisher, and must never be compared against a
    // locally computed "now" or against a second publisher's timestamps.
    //
    // One timestamp per composite buffer, shared by every camera port in it -- the cameras in a
    // segment are synchronised at capture and ShmPortView carries no per-port time. So this is a
    // capture-GROUP identity, which is the granularity tcn_temporal_sync matches on.
    const int64_t acq_timestamp_ns = static_cast<int64_t>(frame.timestamp);

    // Get allocator handle for tensor allocation
    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_->gxf_cid());
    if (!allocator_handle) {
        HOLOSCAN_LOG_ERROR("Failed to get allocator handle");
        return;
    }

    // Create output entities
    auto color_entity = nvidia::gxf::Entity::New(context.context());
    if (!color_entity) {
        HOLOSCAN_LOG_ERROR("Failed to create color entity");
        return;
    }
    auto depth_entity = nvidia::gxf::Entity::New(context.context());
    if (!depth_entity) {
        HOLOSCAN_LOG_ERROR("Failed to create depth entity");
        return;
    }

    // For each port, allocate a GPU tensor and kick off an async copy
    // directly from the SHM pointer to GPU device memory.
    for (const auto& pv : frame.ports) {
        if (pv.is_color) {
            nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
            if (!tcn::allocate_named_tensor<uint8_t>(
                    allocator_handle.value(), copy_stream_, color_entity.value(),
                    nvidia::gxf::Shape{{pv.height, pv.width, pv.channels}},
                    nvidia::gxf::MemoryStorageType::kDevice,
                    pv.name, tensor)) {
                HOLOSCAN_LOG_ERROR("Failed to allocate color tensor for {}", pv.name);
                continue;
            }

            auto maybe_data = tensor->data<uint8_t>();
            if (maybe_data) {
                HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
                    maybe_data.value(), pv.data_ptr, pv.data_size,
                    cudaMemcpyHostToDevice, copy_stream_));
            }
        } else {
            nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
            if (!tcn::allocate_named_tensor<uint16_t>(
                    allocator_handle.value(), copy_stream_, depth_entity.value(),
                    nvidia::gxf::Shape{{pv.height, pv.width, pv.channels}},
                    nvidia::gxf::MemoryStorageType::kDevice,
                    pv.name, tensor)) {
                HOLOSCAN_LOG_ERROR("Failed to allocate depth tensor for {}", pv.name);
                continue;
            }

            auto maybe_data = tensor->data<uint16_t>();
            if (maybe_data) {
                HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
                    maybe_data.value(), pv.data_ptr, pv.data_size,
                    cudaMemcpyHostToDevice, copy_stream_));
            }
        }
    }

    // Wait for all async copies to complete before releasing the SHM segment.
    // This is the critical safety barrier: SHM data_ptrs are valid only while
    // frame.shm_handle is alive, and the CUDA DMA engine reads from those
    // addresses asynchronously.
    HOLOSCAN_CUDA_CALL(cudaStreamSynchronize(copy_stream_));

    // Release SHM segment — frame goes out of scope at function end, but
    // we explicitly clear it here to make the release point obvious and to
    // ensure it happens before the emit (defense in depth).
    frame = {};

    // Stamp both entities with the acquisition time so downstream operators can group frames that
    // belong together (tcn_temporal_sync). Holoscan already reads this back on the receive side --
    // InputContext::get_acquisition_timestamp() searches a received entity for ANY component of
    // type nvidia::gxf::Timestamp, so the component NAME is not significant; only the type is.
    // Nothing in Holoscan CREATES one, which is why this has to be done here: without it the
    // frame's identity is lost at the source and there is nothing downstream to match on.
    const int64_t pub_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    auto stamp = [&](nvidia::gxf::Entity& e, const char* what) {
        auto ts = e.add<nvidia::gxf::Timestamp>("timestamp");
        if (!ts) {
            // Non-fatal: the frame is still valid data, but anything downstream that groups by
            // acquisition time will not see this one. Warn rather than drop the frame.
            HOLOSCAN_LOG_WARN("Failed to add Timestamp to {} entity (acq={})", what,
                              acq_timestamp_ns);
            return;
        }
        ts.value()->acqtime = acq_timestamp_ns;
        ts.value()->pubtime = pub_timestamp_ns;
    };
    stamp(color_entity.value(), "color");
    stamp(depth_entity.value(), "depth");

    // Emit outputs
    op_output.emit(color_entity.value(), "color_outputs");
    op_output.emit(depth_entity.value(), "depth_outputs");

    // Reset async condition to wait for next frame
    if (async_condition_.get()) {
        async_condition_.get()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    }
}

}  // namespace tcn::ops
