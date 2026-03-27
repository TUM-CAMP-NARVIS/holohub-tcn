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

#include "xy_lookup_table_source_op.hpp"

#include <holoscan/utils/cuda_macros.hpp>
#include "../common/utils.h"

namespace tcn::ops {

void XYLookupTableSourceOp::setup(holoscan::OperatorSpec& spec) {
    spec.output<nvidia::gxf::Entity>("xy_table");
    spec.param(camera_name_, "camera_name", "Camera Name",
               "Name of the camera to generate XY lookup table for");
    spec.param(allocator_, "allocator", "Allocator",
               "GPU memory allocator");
}

void XYLookupTableSourceOp::initialize() {
    holoscan::Operator::initialize();

    if (!ctx_service_) {
        HOLOSCAN_LOG_ERROR("XYLookupTableSourceOp: DeviceContextService not set");
        return;
    }

    auto xy_table = ctx_service_->get_xy_table(camera_name_.get());
    if (!xy_table) {
        HOLOSCAN_LOG_ERROR("XYLookupTableSourceOp: Could not create XY Table for camera: {}",
                           camera_name_.get());
        return;
    }

    table_width_ = xy_table->width;
    table_height_ = xy_table->height;

    // Copy XY table data to GPU
    size_t data_bytes = xy_table->data.size() * sizeof(float);
    HOLOSCAN_CUDA_CALL(cudaStreamCreate(&cuda_stream_));
    HOLOSCAN_CUDA_CALL(cudaMalloc(&xy_table_device_, data_bytes));
    HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
        xy_table_device_, xy_table->data.data(), data_bytes,
        cudaMemcpyHostToDevice, cuda_stream_));
    HOLOSCAN_CUDA_CALL(cudaStreamSynchronize(cuda_stream_));

    HOLOSCAN_LOG_INFO("XYLookupTableSourceOp: Created XY table for {} ({}x{})",
                      camera_name_.get(), table_width_, table_height_);
}

void XYLookupTableSourceOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context) {
    if (!xy_table_device_) {
        HOLOSCAN_LOG_ERROR("XYLookupTableSourceOp: No XY table data for camera: {}",
                           camera_name_.get());
        return;
    }

    // Get allocator handle
    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_->gxf_cid());
    if (!allocator_handle) {
        HOLOSCAN_LOG_ERROR("Failed to get allocator handle");
        return;
    }

    // Create output entity with XY table tensor
    auto entity = nvidia::gxf::Entity::New(context.context());
    if (!entity) {
        HOLOSCAN_LOG_ERROR("Failed to create output entity");
        return;
    }

    nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
    if (!tcn::allocate_named_tensor<float>(
            allocator_handle.value(), cuda_stream_, entity.value(),
            nvidia::gxf::Shape{{static_cast<int32_t>(table_height_),
                                static_cast<int32_t>(table_width_), 2}},
            nvidia::gxf::MemoryStorageType::kDevice,
            "", tensor)) {
        HOLOSCAN_LOG_ERROR("Failed to allocate XY table tensor");
        return;
    }

    // Copy cached GPU data to the allocated tensor
    auto maybe_data = tensor->data<float>();
    if (maybe_data) {
        size_t data_bytes = table_height_ * table_width_ * 2 * sizeof(float);
        HOLOSCAN_CUDA_CALL(cudaMemcpyAsync(
            maybe_data.value(), xy_table_device_, data_bytes,
            cudaMemcpyDeviceToDevice, cuda_stream_));
        HOLOSCAN_CUDA_CALL(cudaStreamSynchronize(cuda_stream_));
    }

    op_output.emit(entity.value(), "xy_table");
}

}  // namespace tcn::ops
