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

#include <cuda_runtime.h>
#include "../common/utils.h"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_depthimage_max_distance.cuh"

#include <gxf/std/tensor.hpp>

using std::string_literals::operator""s;

namespace tcn::ops {

__global__ void elementwise_max_u16_kernel(const uint16_t* __restrict__ input,
                                           uint16_t* __restrict__ max_buf,
                                           int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    uint16_t cur = input[idx];
    uint16_t prev = max_buf[idx];
    max_buf[idx] = (cur > prev) ? cur : prev;
}

void TcnDepthImageMaxDistanceOp::setup(holoscan::OperatorSpec& spec) {
    HOLOSCAN_LOG_DEBUG("TcnDepthImageMaxDistanceOp::setup");

    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
    spec.param(in_tensor_name_, "in_tensor_name", "Input Tensor Name", "", ""s);
    spec.param(out_tensor_name_, "out_tensor_name", "Output Tensor Name", "", ""s);
}

void TcnDepthImageMaxDistanceOp::compute(holoscan::InputContext& op_input,
                                         holoscan::OutputContext& op_output,
                                         holoscan::ExecutionContext& context) {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to read input entity");
    }

    auto input_tensor = maybe_entity.value().get<holoscan::Tensor>(in_tensor_name_.get().c_str());
    if (!input_tensor) {
        return;
    }

    cudaStream_t cuda_stream = op_input.receive_cuda_stream("input", true, false);

    const auto& shape = input_tensor->shape();
    std::vector<int32_t> shape_dims;
    int num_elements = 1;
    for (int i = 0; i < input_tensor->ndim(); ++i) {
        num_elements *= static_cast<int>(shape[i]);
        shape_dims.push_back(static_cast<int32_t>(shape[i]));
    }
    nvidia::gxf::Shape gxf_shape(shape_dims);

    auto* input_ptr = static_cast<const uint16_t*>(input_tensor->data());

    auto allocator =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

    // Lazy init of the max accumulation buffer
    if (!max_buffer_) {
        if (!tcn::allocate_tensor<uint16_t>(
                allocator.value(), cuda_stream, gxf_shape,
                nvidia::gxf::MemoryStorageType::kDevice,
                max_buffer_, true)) {
            throw std::runtime_error("Failed to allocate max_buffer.");
        }
    }

    auto maybe_max_data = max_buffer_->data<uint16_t>();
    if (!maybe_max_data) {
        HOLOSCAN_LOG_ERROR("Failed to access max_buffer data");
        return;
    }
    uint16_t* max_ptr = maybe_max_data.value();

    // Launch element-wise max kernel
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    elementwise_max_u16_kernel<<<blocks, threads, 0, cuda_stream>>>(input_ptr, max_ptr, num_elements);

    // Emit the max buffer as output — wrap in a new entity with a named tensor
    auto maybe_out_entity = nvidia::gxf::Entity::New(context.context());
    if (!maybe_out_entity) {
        throw std::runtime_error("Failed to allocate output entity.");
    }
    nvidia::gxf::Entity out_entity = std::move(maybe_out_entity.value());

    nvidia::gxf::Handle<nvidia::gxf::Tensor> out_gxf_tensor = nullptr;
    if (!tcn::allocate_named_tensor<uint16_t>(
            allocator.value(), cuda_stream, out_entity,
            gxf_shape,
            nvidia::gxf::MemoryStorageType::kDevice,
            out_tensor_name_.get(), out_gxf_tensor)) {
        throw std::runtime_error("Failed to allocate output tensor.");
    }

    auto maybe_out_data = out_gxf_tensor->data<uint16_t>();
    if (!maybe_out_data) {
        HOLOSCAN_LOG_ERROR("Failed to access output tensor data");
        return;
    }

    // Copy max_buffer to output
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
        maybe_out_data.value(), max_ptr,
        num_elements * sizeof(uint16_t),
        cudaMemcpyDeviceToDevice, cuda_stream));

    auto out_message = holoscan::gxf::Entity(std::move(out_entity));
    op_output.emit(out_message, "output");
}

}  // namespace tcn::ops
