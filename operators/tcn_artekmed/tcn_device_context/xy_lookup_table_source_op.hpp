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
#include <vector>

#include <holoscan/holoscan.hpp>
#include <cuda_runtime.h>

#include "device_context_service.hpp"

namespace tcn::ops {

/**
 * @brief One-shot source operator that emits XY lookup tables for depth reprojection.
 *
 * C++ port of Python XYLookupTableSourceOp. Generates the XY lookup table
 * once during initialize() and emits it on each compute() call.
 * Typically used with CountCondition(1) for single emission.
 *
 * Parameters:
 *   - camera_name: Name of the camera to generate XY table for
 *   - allocator: GPU memory allocator
 *
 * Outputs:
 *   - xy_table: Tensor of shape (height, width, 2) with float32 XY coordinates
 */
class XYLookupTableSourceOp : public holoscan::Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(XYLookupTableSourceOp)

    XYLookupTableSourceOp() = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void initialize() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;

    /// Set the device context service (must be called before initialize).
    void set_device_context_service(std::shared_ptr<DeviceContextService> service) {
        ctx_service_ = std::move(service);
    }

 private:
    holoscan::Parameter<std::string> camera_name_;
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

    std::shared_ptr<DeviceContextService> ctx_service_;

    // XY table data cached on GPU after initialize()
    void* xy_table_device_{nullptr};
    uint32_t table_width_{0};
    uint32_t table_height_{0};
    cudaStream_t cuda_stream_{nullptr};
};

}  // namespace tcn::ops
