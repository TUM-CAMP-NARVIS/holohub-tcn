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

#include "tcn_stream_splitter.hpp"

#include <gxf/std/tensor.hpp>

namespace tcn::ops {

void TcnStreamSplitterOp::setup(holoscan::OperatorSpec& spec) {
  HOLOSCAN_LOG_DEBUG("TcnStreamSplitterOp::setup");

  spec.input<holoscan::gxf::Entity>("receivers");

  spec.param(channel_names_,
             "channel_names",
             "Channel Names",
             "List of channel names to split the input entity into separate outputs.");

  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "Cuda Stream Pool",
             "Instance of gxf::CudaStreamPool.",
             holoscan::ParameterFlag::kOptional);

  // Register dynamic output ports — channel_names_init_ was set in the
  // constructor, so it is available here even though the Parameter hasn't
  // been bound yet.
  for (const auto& name : channel_names_init_) {
    spec.output<holoscan::gxf::Entity>(name);
  }
}

void TcnStreamSplitterOp::compute(holoscan::InputContext& op_input,
                                  holoscan::OutputContext& op_output,
                                  holoscan::ExecutionContext& context) {
  auto maybe_input_entity = op_input.receive<holoscan::gxf::Entity>("receivers");
  if (!maybe_input_entity) {
    throw std::runtime_error("TcnStreamSplitterOp: failed to receive input entity.");
  }
  auto& input_entity = maybe_input_entity.value();

  // Synchronizes the real upstream producer stream to this op's internal stream
  // AND auto-configures all output ports to emit that (correctly-synced) stream.
  cudaStream_t _stream = op_input.receive_cuda_stream("receivers");

  for (const auto& channel_name : channel_names_.get()) {
    // Get the tensor for this channel from the input entity (as GXF tensor)
    auto src_tensor = static_cast<nvidia::gxf::Entity&>(input_entity)
                          .get<nvidia::gxf::Tensor>(channel_name.c_str());
    if (!src_tensor) {
      throw std::runtime_error(
          "TcnStreamSplitterOp: input entity missing tensor '" + channel_name + "'.");
    }

    // Create output entity with the tensor (zero-copy via wrapMemory)
    auto out_entity = holoscan::gxf::Entity::New(&context);
    auto out_tensor = static_cast<nvidia::gxf::Entity&>(out_entity)
                          .add<nvidia::gxf::Tensor>("");
    if (!out_tensor) {
      throw std::runtime_error(
          "TcnStreamSplitterOp: failed to add tensor to output entity for channel '" +
          channel_name + "'.");
    }

    // Compute strides from the source tensor shape
    auto strides = nvidia::gxf::ComputeTrivialStrides(
        src_tensor.value()->shape(), src_tensor.value()->bytes_per_element());

    // Wrap the same device memory (zero-copy), keeping the SOURCE entity alive for as long as the
    // wrapped memory is reachable.
    //
    // wrapMemory does not take ownership. Passing nullptr as the release callback left every output
    // tensor pointing at memory owned solely by `input_entity`: once compute() returned and that
    // last reference went away, the allocation returned to its pool and a later frame could reuse it
    // while a downstream consumer was still reading. Holding a reference in the release callback --
    // which GXF invokes when the wrapped buffer is actually freed -- ties the source's lifetime to
    // the outputs'.
    //
    // The symptom was a stale or recycled image in one consumer on some frames: nondeterministic,
    // and invisible while a single consumer read the tensor immediately after the split. It shows up
    // once a tensor is buffered (tcn_stream_synchronizer holds entities across ticks) or read by
    // more than one consumer, which is what the mask/depth join does.
    auto keep_alive =
        std::make_shared<nvidia::gxf::Entity>(static_cast<nvidia::gxf::Entity&>(input_entity));
    auto result = out_tensor.value()->wrapMemory(
        src_tensor.value()->shape(),
        src_tensor.value()->element_type(),
        src_tensor.value()->bytes_per_element(),
        strides,
        src_tensor.value()->storage_type(),
        src_tensor.value()->pointer(),
        [keep_alive](void*) mutable -> nvidia::gxf::Expected<void> {
          keep_alive.reset();      // the allocation's real owner frees it; we only held a reference
          return nvidia::gxf::Success;
        });
    if (!result) {
      throw std::runtime_error(
          "TcnStreamSplitterOp: failed to wrap tensor memory for channel '" +
          channel_name + "'.");
    }

    op_output.emit(out_entity, channel_name.c_str());
  }
}

}  // namespace tcn::ops
