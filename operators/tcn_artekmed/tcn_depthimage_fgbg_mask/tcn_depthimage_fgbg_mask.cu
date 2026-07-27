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
#include "tcn_depthimage_fgbg_mask.cuh"

#include <gxf/std/tensor.hpp>

using std::string_literals::operator""s;

namespace tcn::ops {

__global__ void fgbg_mask_kernel(const uint16_t* __restrict__ depth,
                                 const uint16_t* __restrict__ bg,
                                 uint8_t* __restrict__ fg_out,
                                 uint8_t* __restrict__ bg_out,
                                 float sensitivity,
                                 int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    uint16_t d = depth[idx];
    uint16_t b = bg[idx];

    // depth is in mm, convert to meters-ish scale for error calc
    float err = ((static_cast<float>(d) / 1000.0f) + 11.0f) * sensitivity;
    float fg_val = static_cast<float>(d) + err;
    float bg_val = static_cast<float>(b) - err;

    fg_out[idx] = (bg_val > fg_val) ? 1 : 0;
    bg_out[idx] = (bg_val <= fg_val) ? 1 : 0;
}

void TcnDepthImageFgbgMaskOp::setup(holoscan::OperatorSpec& spec) {
    HOLOSCAN_LOG_DEBUG("TcnDepthImageFgbgMaskOp::setup");

    spec.input<holoscan::gxf::Entity>("depth_image");
    spec.input<holoscan::gxf::Entity>("background_image");

    auto& fg_port = spec.output<holoscan::gxf::Entity>("foreground_mask");
    auto& bg_port = spec.output<holoscan::gxf::Entity>("background_mask");

    spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
    spec.param(sensitivity_, "sensitivity", "Sensitivity", "Error sensitivity factor.", 1.0f);
    spec.param(enable_foreground_, "enable_foreground", "Enable Foreground", "Enable foreground mask output.", true);
    spec.param(enable_background_, "enable_background", "Enable Background", "Enable background mask output.", false);
}

void TcnDepthImageFgbgMaskOp::compute(holoscan::InputContext& op_input,
                                      holoscan::OutputContext& op_output,
                                      holoscan::ExecutionContext& context) {
    auto maybe_depth_entity = op_input.receive<holoscan::gxf::Entity>("depth_image");
    auto maybe_bg_entity = op_input.receive<holoscan::gxf::Entity>("background_image");

    if (!maybe_depth_entity || !maybe_bg_entity) {
        return;
    }

    auto depth_tensor = maybe_depth_entity.value().get<holoscan::Tensor>("");
    auto bg_tensor = maybe_bg_entity.value().get<holoscan::Tensor>("");

    if (!depth_tensor || !bg_tensor) {
        return;
    }

    // Synchronize input streams
    cudaStream_t cuda_stream = op_input.receive_cuda_stream("depth_image", false, false);
    op_input.receive_cuda_stream("background_image", false, false);

    const auto& shape = depth_tensor->shape();
    std::vector<int32_t> shape_dims;
    int num_elements = 1;
    for (int i = 0; i < depth_tensor->ndim(); ++i) {
        num_elements *= static_cast<int>(shape[i]);
        shape_dims.push_back(static_cast<int32_t>(shape[i]));
    }
    nvidia::gxf::Shape gxf_shape(shape_dims);

    auto* depth_ptr = static_cast<const uint16_t*>(depth_tensor->data());
    auto* bg_ptr = static_cast<const uint16_t*>(bg_tensor->data());

    auto allocator =
        nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

    // Lazy pre-allocation of output buffers
    if (buffer_num_elements_ != num_elements) {
        buffer_num_elements_ = num_elements;

        if (!tcn::allocate_tensor<uint8_t>(
                allocator.value(), cuda_stream, gxf_shape,
                nvidia::gxf::MemoryStorageType::kDevice,
                fg_buffer_, true)) {
            throw std::runtime_error("Failed to allocate fg_buffer.");
        }
        if (!tcn::allocate_tensor<uint8_t>(
                allocator.value(), cuda_stream, gxf_shape,
                nvidia::gxf::MemoryStorageType::kDevice,
                bg_buffer_, true)) {
            throw std::runtime_error("Failed to allocate bg_buffer.");
        }
    }

    auto maybe_fg_data = fg_buffer_->data<uint8_t>();
    auto maybe_bg_data = bg_buffer_->data<uint8_t>();
    if (!maybe_fg_data || !maybe_bg_data) {
        HOLOSCAN_LOG_ERROR("Failed to access fgbg buffer data");
        return;
    }
    uint8_t* fg_ptr = maybe_fg_data.value();
    uint8_t* bg_out_ptr = maybe_bg_data.value();

    // Launch fgbg mask kernel
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    fgbg_mask_kernel<<<blocks, threads, 0, cuda_stream>>>(
        depth_ptr, bg_ptr, fg_ptr, bg_out_ptr, sensitivity_.get(), num_elements);

    // Emit foreground mask
    if (enable_foreground_.get()) {
        auto maybe_fg_entity = nvidia::gxf::Entity::New(context.context());
        if (!maybe_fg_entity) {
            throw std::runtime_error("Failed to allocate foreground entity.");
        }
        nvidia::gxf::Entity fg_entity = std::move(maybe_fg_entity.value());

        nvidia::gxf::Handle<nvidia::gxf::Tensor> fg_gxf_tensor = nullptr;
        if (!tcn::allocate_named_tensor<uint8_t>(
                allocator.value(), cuda_stream, fg_entity, gxf_shape,
                nvidia::gxf::MemoryStorageType::kDevice, ""s, fg_gxf_tensor)) {
            throw std::runtime_error("Failed to allocate foreground output tensor.");
        }
        auto maybe_fg_emit = fg_gxf_tensor->data<uint8_t>();
        if (maybe_fg_emit) {
            HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
                maybe_fg_emit.value(), fg_ptr,
                num_elements * sizeof(uint8_t),
                cudaMemcpyDeviceToDevice, cuda_stream));
        }

        auto fg_message = holoscan::gxf::Entity(std::move(fg_entity));
        op_output.set_cuda_stream(cuda_stream, "foreground_mask");
        op_output.emit(fg_message, "foreground_mask");
    }

    // Emit background mask
    if (enable_background_.get()) {
        auto maybe_bg_entity = nvidia::gxf::Entity::New(context.context());
        if (!maybe_bg_entity) {
            throw std::runtime_error("Failed to allocate background entity.");
        }
        nvidia::gxf::Entity bg_entity = std::move(maybe_bg_entity.value());

        nvidia::gxf::Handle<nvidia::gxf::Tensor> bg_gxf_tensor = nullptr;
        if (!tcn::allocate_named_tensor<uint8_t>(
                allocator.value(), cuda_stream, bg_entity, gxf_shape,
                nvidia::gxf::MemoryStorageType::kDevice, ""s, bg_gxf_tensor)) {
            throw std::runtime_error("Failed to allocate background output tensor.");
        }
        auto maybe_bg_emit = bg_gxf_tensor->data<uint8_t>();
        if (maybe_bg_emit) {
            HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
                maybe_bg_emit.value(), bg_out_ptr,
                num_elements * sizeof(uint8_t),
                cudaMemcpyDeviceToDevice, cuda_stream));
        }

        auto bg_message = holoscan::gxf::Entity(std::move(bg_entity));
        op_output.emit(bg_message, "background_mask");
    }
}

}  // namespace tcn::ops
