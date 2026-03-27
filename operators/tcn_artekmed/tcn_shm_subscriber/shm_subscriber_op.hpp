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

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <cuda_runtime.h>

#include "shm_synchronized_buffer_receiver.hpp"

namespace tcn::ops {

/// Data received from a single frame: color and depth tensors keyed by port name.
struct ShmFrameData {
    uint64_t timestamp;
    std::unordered_map<std::string, std::vector<uint8_t>> color_data;  // port_name -> RGBA/BGRA pixels
    std::unordered_map<std::string, std::vector<uint8_t>> depth_data;  // port_name -> uint16 depth
    // Dimensions per port
    struct PortDims {
        int32_t width;
        int32_t height;
        int32_t channels;
    };
    std::unordered_map<std::string, PortDims> color_dims;
    std::unordered_map<std::string, PortDims> depth_dims;
};

/**
 * @brief Holoscan operator that subscribes to SHM camera streams via iceoryx2.
 *
 * C++ port of Python ShmSubscriberOp. Wraps ShmSynchronizedBufferReceiver in a
 * background thread, using an AsynchronousCondition to wake the Holoscan scheduler
 * when new data arrives.
 *
 * Outputs:
 *   - color_outputs: Entity with named tensors (one per camera, RGBA/BGRA uint8)
 *   - depth_outputs: Entity with named tensors (one per camera, uint16 depth)
 */
class TcnShmSubscriberOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnShmSubscriberOp)

    TcnShmSubscriberOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void initialize() override;
    void start() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;
    void stop() override;

    /// Set the SHM receiver instance (must be called before start).
    void set_receiver(std::shared_ptr<tcn::shm::ShmSynchronizedBufferReceiver> receiver) {
        receiver_ = std::move(receiver);
    }

 private:
    void receiver_mainloop();
    bool on_receive(const tcn::shm::ShmSerializedStreamHeader& header,
                    artekmed::shm::ShmBufferDescriptor::Reader descriptor);

    // Parameters
    holoscan::Parameter<std::string> stream_name_;
    holoscan::Parameter<int32_t> cycle_time_ms_;
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
    holoscan::Parameter<std::shared_ptr<holoscan::AsynchronousCondition>> async_condition_;

    // Receiver
    std::shared_ptr<tcn::shm::ShmSynchronizedBufferReceiver> receiver_;

    // Threading
    std::thread receiver_thread_;
    std::atomic<bool> should_stop_{false};

    // Thread-safe frame queue
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<ShmFrameData> frame_queue_;
};

}  // namespace tcn::ops
