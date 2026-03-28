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

#include "tcn_stream_merger.hpp"

#include <cuda_runtime.h>
#include <gxf/std/tensor.hpp>
#include <optional>

#include "../common/utils.h"

namespace tcn::ops {

void TcnStreamMergerOp::setup(holoscan::OperatorSpec& spec) {
  HOLOSCAN_LOG_DEBUG("TcnStreamMergerOp::setup");

  spec.output<holoscan::gxf::Entity>("output");

  spec.param(input_port_names_,
             "input_port_names",
             "Input Port Names",
             "List of input port names to merge.");
  spec.param(input_message_name_,
             "input_message_name",
             "Input Message Name",
             "Name of the tensor to extract from each input message.");
  spec.param(output_message_name_,
             "output_message_name",
             "Output Message Name",
             "Name to use for the output tensor(s).");
  spec.param(fuse_buffers_,
             "fuse_buffers",
             "Fuse Buffers",
             "If true, concatenate all tensors along axis 1. If false, output separate tensors.",
             false);
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Allocator for output tensors (required for fuse mode).",
             holoscan::ParameterFlag::kOptional);

  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "Cuda Stream Pool",
             "Instance of gxf::CudaStreamPool.",
             holoscan::ParameterFlag::kOptional);

  // Register dynamic input ports — input_port_names_init_ was set in the
  // constructor, so it is available here before add_flow() checks ports.
  for (const auto& name : input_port_names_init_) {
    spec.input<holoscan::gxf::Entity>(name);
  }
}

std::string TcnStreamMergerOp::getCameraNameFromPortName(const std::string& port_name) const {
  std::smatch match;
  if (std::regex_match(port_name, match, portname_pattern_)) {
    return match[1].str();
  }
  return port_name;
}

void TcnStreamMergerOp::compute(holoscan::InputContext& op_input,
                                holoscan::OutputContext& op_output,
                                holoscan::ExecutionContext& context) {
  const auto& port_names = input_port_names_.get();
  const auto& in_msg_name = input_message_name_.get();
  const auto& out_msg_name = output_message_name_.get();
  const bool fuse = fuse_buffers_.get();

  // Collect all input tensors and their CUDA streams
  struct InputData {
    std::string port_name;
    nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
  };
  std::vector<InputData> inputs;
  std::vector<std::optional<cudaStream_t>> input_streams;

  // Keep input entities alive until we emit (tensors reference their memory)
  std::vector<holoscan::gxf::Entity> input_entities;

  for (const auto& port_name : port_names) {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>(port_name.c_str());
    if (!maybe_entity) {
      throw std::runtime_error(
          "TcnStreamMergerOp: failed to receive input from port '" + port_name + "'.");
    }
    auto entity = std::move(maybe_entity.value());

    cudaStream_t port_stream = op_input.receive_cuda_stream(port_name.c_str(), false, false);
    if (port_stream) {
      input_streams.push_back(port_stream);
    }

    auto tensor = static_cast<nvidia::gxf::Entity&>(entity)
                      .get<nvidia::gxf::Tensor>(in_msg_name.c_str());
    if (!tensor) {
      throw std::runtime_error(
          "TcnStreamMergerOp: input entity from port '" + port_name +
          "' missing tensor '" + in_msg_name + "'.");
    }
    inputs.push_back({port_name, tensor.value()});
    input_entities.push_back(std::move(entity));
  }

  // Synchronize input streams and allocate output stream
  cudaStream_t output_stream = 0;
  if (!input_streams.empty()) {
    auto maybe_stream = context.allocate_cuda_stream(name());
    if (!maybe_stream) {
      throw std::runtime_error("TcnStreamMergerOp: failed to allocate output CUDA stream.");
    }
    output_stream = maybe_stream.value();
    context.synchronize_streams(input_streams, output_stream);
  }

  // Create output entity
  auto out_entity = holoscan::gxf::Entity::New(&context);
  auto& out_gxf_entity = static_cast<nvidia::gxf::Entity&>(out_entity);

  if (fuse) {
    // Fuse mode: concatenate all tensors along axis 1
    if (inputs.empty()) {
      throw std::runtime_error("TcnStreamMergerOp: no inputs to fuse.");
    }

    auto& first = inputs[0].tensor;
    const auto& first_shape = first->shape();
    int32_t ndim = first_shape.rank();
    if (ndim < 2) {
      throw std::runtime_error(
          "TcnStreamMergerOp: tensors must have at least 2 dimensions for fusion.");
    }

    int64_t total_width = 0;
    for (const auto& input : inputs) {
      const auto& shape = input.tensor->shape();
      if (shape.rank() != ndim) {
        throw std::runtime_error(
            "TcnStreamMergerOp: all tensors must have same number of dimensions.");
      }
      for (int32_t d = 0; d < ndim; ++d) {
        if (d != 1 && shape.dimension(d) != first_shape.dimension(d)) {
          throw std::runtime_error(
              "TcnStreamMergerOp: tensor dimension mismatch at axis " + std::to_string(d) + ".");
        }
      }
      total_width += shape.dimension(1);
    }

    // Build output shape
    std::vector<int32_t> out_dims;
    for (int32_t d = 0; d < ndim; ++d) {
      out_dims.push_back(d == 1 ? static_cast<int32_t>(total_width)
                                : first_shape.dimension(d));
    }
    nvidia::gxf::Shape out_shape(out_dims);

    auto elem_type = first->element_type();
    size_t elem_size = first->bytes_per_element();

    // Allocate output tensor
    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(), allocator_->gxf_cid());

    // Allocate output tensor with the same element type as the inputs
    auto out_tensor_handle = out_gxf_entity.add<nvidia::gxf::Tensor>(out_msg_name.c_str());
    if (!out_tensor_handle) {
      throw std::runtime_error("TcnStreamMergerOp: failed to add fused output tensor.");
    }
    auto strides = nvidia::gxf::ComputeTrivialStrides(out_shape, elem_size);
    auto reshape_result = out_tensor_handle.value()->reshapeCustom(
        out_shape, elem_type, elem_size, strides,
        nvidia::gxf::MemoryStorageType::kDevice, allocator_handle.value());
    if (!reshape_result) {
      throw std::runtime_error("TcnStreamMergerOp: failed to allocate fused output tensor.");
    }

    auto* out_ptr = out_tensor_handle.value()->pointer();
    if (!out_ptr) {
      throw std::runtime_error("TcnStreamMergerOp: failed to access fused output tensor data.");
    }

    int64_t height = first_shape.dimension(0);
    int64_t depth = (ndim > 2) ? first_shape.dimension(2) : 1;
    size_t row_elem_offset = 0;

    for (const auto& input : inputs) {
      const auto& shape = input.tensor->shape();
      int64_t src_width = shape.dimension(1);
      size_t src_row_bytes = src_width * depth * elem_size;
      size_t dst_row_bytes = total_width * depth * elem_size;

      auto* src_ptr = input.tensor->pointer();
      if (!src_ptr) {
        throw std::runtime_error("TcnStreamMergerOp: failed to access input tensor data.");
      }

      for (int64_t row = 0; row < height; ++row) {
        size_t dst_offset = row * dst_row_bytes + row_elem_offset * depth * elem_size;
        size_t src_offset = row * src_row_bytes;
        cudaMemcpyAsync(out_ptr + dst_offset, src_ptr + src_offset,
                        src_row_bytes, cudaMemcpyDeviceToDevice, output_stream);
      }
      row_elem_offset += src_width;
    }

    HOLOSCAN_LOG_DEBUG("TcnStreamMergerOp: fused {} tensors into shape [{}x{}]",
                       inputs.size(), height, total_width);
  } else {
    // Separate mode: output each tensor with camera-prefixed name (zero-copy)
    for (const auto& input : inputs) {
      std::string camera_name = getCameraNameFromPortName(input.port_name);
      std::string tensor_name = camera_name + "_" + out_msg_name;

      auto out_tensor = out_gxf_entity.add<nvidia::gxf::Tensor>(tensor_name.c_str());
      if (!out_tensor) {
        throw std::runtime_error(
            "TcnStreamMergerOp: failed to add tensor '" + tensor_name + "' to output entity.");
      }

      // Compute strides from the tensor shape
      auto strides = nvidia::gxf::ComputeTrivialStrides(
          input.tensor->shape(), input.tensor->bytes_per_element());

      // Zero-copy: wrap the same device memory
      auto result = out_tensor.value()->wrapMemory(
          input.tensor->shape(),
          input.tensor->element_type(),
          input.tensor->bytes_per_element(),
          strides,
          input.tensor->storage_type(),
          input.tensor->pointer(),
          nullptr);
      if (!result) {
        throw std::runtime_error(
            "TcnStreamMergerOp: failed to wrap tensor memory for '" + tensor_name + "'.");
      }
    }
  }

  if (output_stream) {
    op_output.set_cuda_stream(output_stream, "output");
  }
  op_output.emit(out_entity, "output");
}

}  // namespace tcn::ops
