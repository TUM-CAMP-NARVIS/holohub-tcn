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
#include <optional>
#include "../common/utils.h"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_depthimage_apply_mask.cuh"

#include <gxf/std/tensor.hpp>

using std::string_literals::operator""s;

namespace tcn::ops {

__global__ void apply_mask_kernel(const uint16_t* __restrict__ depth,
                                  const uint8_t* __restrict__ mask,
                                  uint16_t* __restrict__ out,
                                  bool invert,
                                  int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    if (invert) {
        // Keep pixels where mask is 0 (background)
        out[idx] = (mask[idx] == 0) ? depth[idx] : 0;
    } else {
        // Keep pixels where mask is non-zero (foreground)
        out[idx] = (mask[idx] != 0) ? depth[idx] : 0;
    }
}

void TcnDepthImageApplyMaskOp::setup(holoscan::OperatorSpec& spec) {
    HOLOSCAN_LOG_DEBUG("TcnDepthImageApplyMaskOp::setup");

    spec.input<holoscan::gxf::Entity>("depth_image");
    spec.input<holoscan::gxf::Entity>("mask_image");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
    spec.param(invert_mask_, "invert_mask", "Invert Mask", "If true, keep background instead of foreground.", false);
    spec.param(out_tensor_name_, "out_tensor_name", "Output Tensor Name", "", ""s);
}

void TcnDepthImageApplyMaskOp::compute(holoscan::InputContext& op_input,
                                       holoscan::OutputContext& op_output,
                                       holoscan::ExecutionContext& context) {
    auto maybe_depth_entity = op_input.receive<holoscan::gxf::Entity>("depth_image");
    auto maybe_mask_entity = op_input.receive<holoscan::gxf::Entity>("mask_image");

    if (!maybe_depth_entity || !maybe_mask_entity) {
        return;
    }

    auto depth_tensor = maybe_depth_entity.value().get<holoscan::Tensor>("");
    auto mask_tensor = maybe_mask_entity.value().get<holoscan::Tensor>("");

    if (!depth_tensor || !mask_tensor) {
        return;
    }

    // Synchronize input streams
    cudaStream_t cuda_stream = op_input.receive_cuda_stream("depth_image", false, false);
    op_input.receive_cuda_stream("mask_image", false, false);

    const auto& shape = depth_tensor->shape();
    std::vector<int32_t> shape_dims;
    int num_elements = 1;
    for (int i = 0; i < depth_tensor->ndim(); ++i) {
        num_elements *= static_cast<int>(shape[i]);
        shape_dims.push_back(static_cast<int32_t>(shape[i]));
    }
    nvidia::gxf::Shape gxf_shape(shape_dims);

    auto* depth_ptr = static_cast<const uint16_t*>(depth_tensor->data());
    auto* mask_ptr = static_cast<const uint8_t*>(mask_tensor->data());

    auto allocator =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

    // Lazy pre-allocation of output buffer
    if (buffer_num_elements_ != num_elements) {
        buffer_num_elements_ = num_elements;
        if (!tcn::allocate_tensor<uint16_t>(
                allocator.value(), cuda_stream, gxf_shape,
                nvidia::gxf::MemoryStorageType::kDevice,
                out_buffer_, true)) {
            throw std::runtime_error("Failed to allocate output buffer.");
        }
    }

    auto maybe_out_data = out_buffer_->data<uint16_t>();
    if (!maybe_out_data) {
        HOLOSCAN_LOG_ERROR("Failed to access output buffer data");
        return;
    }
    uint16_t* out_ptr = maybe_out_data.value();

    // Launch masking kernel
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    apply_mask_kernel<<<blocks, threads, 0, cuda_stream>>>(
        depth_ptr, mask_ptr, out_ptr, invert_mask_.get(), num_elements);

    // Emit
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

    auto maybe_emit_data = out_gxf_tensor->data<uint16_t>();
    if (!maybe_emit_data) {
        HOLOSCAN_LOG_ERROR("Failed to access emit tensor data");
        return;
    }

    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
        maybe_emit_data.value(), out_ptr,
        num_elements * sizeof(uint16_t),
        cudaMemcpyDeviceToDevice, cuda_stream));

    auto out_message = holoscan::gxf::Entity(std::move(out_entity));
    op_output.emit(out_message, "output");
}

}  // namespace tcn::ops
