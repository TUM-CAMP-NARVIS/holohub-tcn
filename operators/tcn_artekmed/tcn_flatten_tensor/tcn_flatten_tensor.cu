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
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_flatten_tensor.cuh"

#include <gxf/std/tensor.hpp>

using std::string_literals::operator""s;

namespace tcn::ops {

void TcnFlattenTensorOp::setup(holoscan::OperatorSpec& spec) {
    HOLOSCAN_LOG_DEBUG("TcnFlattenTensorOp::setup");

    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(message_name_, "message_name", "Message Name", "Name of tensor in the message.", ""s);
    spec.param(allocator_,
               "allocator",
               "Allocator",
               "Allocator for output tensors.",
               holoscan::ParameterFlag::kOptional);
    spec.param(cuda_stream_pool_,
               "cuda_stream_pool",
               "Cuda Stream Pool",
               "Instance of gxf::CudaStreamPool.",
               holoscan::ParameterFlag::kOptional);
}

void TcnFlattenTensorOp::compute(holoscan::InputContext& op_input,
                                 holoscan::OutputContext& op_output,
                                 holoscan::ExecutionContext& context) {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to read input entity");
    }

    // Get the GXF tensor directly for type-agnostic reshape
    auto& in_entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());
    auto src_tensor = in_entity.get<nvidia::gxf::Tensor>(message_name_.get().c_str());
    if (!src_tensor) {
        HOLOSCAN_LOG_WARN("FlattenTensorOp: received empty tensor for: {}", message_name_.get());
        return;
    }

    cudaStream_t cuda_stream = op_input.receive_cuda_stream("input", true, false);

    const auto& src_shape = src_tensor.value()->shape();
    if (src_shape.rank() < 2) {
        HOLOSCAN_LOG_WARN("FlattenTensorOp: tensor has fewer than 2 dimensions");
        return;
    }

    // Flatten: [H, W, ...] -> [1, H*W, ...]
    std::vector<int32_t> new_dims;
    new_dims.push_back(1);
    new_dims.push_back(src_shape.dimension(0) * src_shape.dimension(1));
    for (uint32_t i = 2; i < src_shape.rank(); ++i) {
        new_dims.push_back(src_shape.dimension(i));
    }
    nvidia::gxf::Shape out_shape(new_dims);

    // Create output entity with zero-copy wrapMemory (same device pointer, different shape)
    auto out_entity = holoscan::gxf::Entity::New(&context);
    auto out_tensor = static_cast<nvidia::gxf::Entity&>(out_entity)
                          .add<nvidia::gxf::Tensor>(message_name_.get().c_str());
    if (!out_tensor) {
        throw std::runtime_error("Failed to add tensor to output entity.");
    }

    auto strides = nvidia::gxf::ComputeTrivialStrides(out_shape, src_tensor.value()->bytes_per_element());

    auto result = out_tensor.value()->wrapMemory(
        out_shape,
        src_tensor.value()->element_type(),
        src_tensor.value()->bytes_per_element(),
        strides,
        src_tensor.value()->storage_type(),
        src_tensor.value()->pointer(),
        // Keep the SOURCE entity alive for as long as the reshaped view is reachable. wrapMemory
        // does not take ownership; with a null release callback the output pointed at memory owned
        // solely by the input entity, so once compute() returned that allocation could be reused by
        // a later frame while a consumer was still reading the "flattened" view. Same defect as
        // tcn_stream_splitter had, and it is on tcn_shm_receiver's point-cloud path too.
        [keep_alive = std::make_shared<nvidia::gxf::Entity>(in_entity)](
            void*) mutable -> nvidia::gxf::Expected<void> {
          keep_alive.reset();
          return nvidia::gxf::Success;
        });
    if (!result) {
        throw std::runtime_error("Failed to wrap tensor memory with new shape.");
    }

    op_output.emit(out_entity, "output");
}

}  // namespace tcn::ops
