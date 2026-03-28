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

#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>
#include <cuda_runtime.h>

#include "iox2/iceoryx2.hpp"

#include "../tcn_shm_serde/shm_types.hpp"
#include "../tcn_shm_serde/shm_serde.hpp"

namespace tcn::ops {

/**
 * @brief Holoscan operator that publishes decoded video frames to iceoryx2 SHM.
 *
 * Takes decoded frame tensors from the Holoscan dataflow and publishes them
 * to iceoryx2 shared memory for consumption by other local processes.
 *
 * The operator builds a Cap'n Proto ShmBufferDescriptor containing per-port
 * metadata and raw frame data, then publishes it as a Slice<uint8_t> with an
 * ShmSerializedStreamHeader user header.
 *
 * Inputs:
 *   - frame_input: Entity with named tensors (color RGBA/BGRA uint8, depth uint16)
 *
 * Parameters:
 *   - stream_name: SHM service name prefix (e.g. "camera_streams")
 *   - input_tensor_names: comma-separated list of tensor names to publish
 *   - cuda_stream_pool: optional CudaStreamPool for GPU->CPU copy
 */
class TcnShmZenohSenderOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnShmZenohSenderOp)

    TcnShmZenohSenderOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void initialize() override;
    void start() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;
    void stop() override;

 private:
    // iceoryx2 publisher state (PIMPL to avoid exposing complex template types)
    struct PublisherState;
    std::unique_ptr<PublisherState> pub_state_;

    // iceoryx2 node — must outlive pub_state_
    std::unique_ptr<iox2::Node<iox2::ServiceType::Ipc>> node_;

    // Parameters
    holoscan::Parameter<std::string> stream_name_;
    holoscan::Parameter<std::vector<std::string>> input_tensor_names_;

    // Dedicated CUDA stream for GPU->CPU copies
    cudaStream_t copy_stream_ = nullptr;

    // CPU staging buffer for GPU->CPU frame copies
    std::vector<uint8_t> staging_buffer_;

    // Frame counter for logging
    uint64_t frames_published_ = 0;
};

}  // namespace tcn::ops
