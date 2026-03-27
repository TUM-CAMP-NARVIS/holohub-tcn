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
#include "tcn_convert_bgra_to_rgba.cuh"

#include <gxf/std/tensor.hpp>

using std::string_literals::operator""s;

namespace tcn::ops {

__global__ void bgra_to_rgba_kernel(const uint8_t* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    int num_pixels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    const int offset = idx * 4;
    output[offset + 0] = input[offset + 2];  // R <- B
    output[offset + 1] = input[offset + 1];  // G <- G
    output[offset + 2] = input[offset + 0];  // B <- R
    output[offset + 3] = input[offset + 3];  // A <- A
}

void TcnConvertBgraToRgbaOp::setup(holoscan::OperatorSpec& spec) {
    HOLOSCAN_LOG_DEBUG("TcnConvertBgraToRgbaOp::setup");

    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
    spec.param(in_tensor_name_, "in_tensor_name", "Input Tensor Name", "", ""s);
    spec.param(out_tensor_name_, "out_tensor_name", "Output Tensor Name", "", ""s);
}

void TcnConvertBgraToRgbaOp::compute(holoscan::InputContext& op_input,
                                     holoscan::OutputContext& op_output,
                                     holoscan::ExecutionContext& context) {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to read input entity");
    }

    auto input_tensor = maybe_entity.value().get<holoscan::Tensor>(in_tensor_name_.get().c_str());
    cudaStream_t cuda_stream = op_input.receive_cuda_stream("input", true, false);

    const auto& shape = input_tensor->shape();
    if (input_tensor->ndim() != 3 || shape[2] < 4) {
        HOLOSCAN_LOG_WARN("TcnConvertBgraToRgbaOp: expected HxWx4 tensor, got ndim={} last_dim={}",
                          input_tensor->ndim(), input_tensor->ndim() >= 3 ? shape[2] : -1);
        return;
    }

    const int H = static_cast<int>(shape[0]);
    const int W = static_cast<int>(shape[1]);
    const int num_pixels = H * W;

    auto* input_ptr = static_cast<const uint8_t*>(input_tensor->data());

    // Allocate output entity + tensor
    auto allocator =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

    auto maybe_out_entity = nvidia::gxf::Entity::New(context.context());
    if (!maybe_out_entity) {
        throw std::runtime_error("Failed to allocate output entity.");
    }
    nvidia::gxf::Entity out_entity = std::move(maybe_out_entity.value());

    nvidia::gxf::Handle<nvidia::gxf::Tensor> out_gxf_tensor = nullptr;
    if (!tcn::allocate_named_tensor<uint8_t>(
            allocator.value(), cuda_stream, out_entity,
            nvidia::gxf::Shape{{H, W, 4}},
            nvidia::gxf::MemoryStorageType::kDevice,
            out_tensor_name_.get(), out_gxf_tensor)) {
        throw std::runtime_error("Failed to allocate output tensor.");
    }

    auto maybe_out_data = out_gxf_tensor->data<uint8_t>();
    if (!maybe_out_data) {
        HOLOSCAN_LOG_ERROR("Failed to access output tensor data");
        return;
    }
    uint8_t* output_ptr = maybe_out_data.value();

    // Launch kernel
    const int threads = 256;
    const int blocks = (num_pixels + threads - 1) / threads;
    bgra_to_rgba_kernel<<<blocks, threads, 0, cuda_stream>>>(input_ptr, output_ptr, num_pixels);

    auto out_message = holoscan::gxf::Entity(std::move(out_entity));
    op_output.emit(out_message, "output");
}

}  // namespace tcn::ops
