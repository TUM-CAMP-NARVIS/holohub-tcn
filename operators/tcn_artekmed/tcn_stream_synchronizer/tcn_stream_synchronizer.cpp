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

#include "tcn_stream_synchronizer.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <cuda.h>
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/gxf/entity.hpp"

#include "gxf/core/entity.hpp"    // nvidia::gxf::Entity::Shared
#include "gxf/std/allocator.hpp"  // nvidia::gxf::Allocator, nvidia::gxf::MemoryStorageType
#include "gxf/std/tensor.hpp"     // nvidia::gxf::Tensor etc.
#include "gxf/std/timestamp.hpp"  // nvidia::gxf::Timestamp

#include "../common/utils.h"

namespace holoscan::ops {

void TcnStreamSynchronizerOp::setup(OperatorSpec& spec) {
  spec.param(num_streams_, "num_streams", "Number of Streams", "Number of input streams to synchronize.");
  HOLOSCAN_LOG_INFO("Synchronizer ports: {}", num_streams_.get());

  in_port_names.clear();
  for (int i = 0; i < num_streams_.get(); ++i) {
    in_port_names.push_back(std::string("input") + std::to_string(i));
    HOLOSCAN_LOG_INFO("Input port name: {}", in_port_names.back());
    spec.input<holoscan::gxf::Entity>(in_port_names[i]);
  }

  spec.output<holoscan::gxf::Entity>("output");

  spec.param(cuda_device_ordinal_,
             "cuda_device_ordinal",
             "CudaDeviceOrdinal",
             "Device to use for CUDA operations",
             ParameterFlag::kOptional);

  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(verbose_, "verbose", "Verbose", "Print detailed decoder information", false);

  cuda_stream_handler_.define_params(spec);
}

void TcnStreamSynchronizerOp::initialize() {
  Operator::initialize();

  // Initialize CUDA
  CudaCheck(cuInit(0));

  // Get the CUDA device
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;

  // Retain the primary context for the device
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));

  // Initialize state for synchronizer
}

void TcnStreamSynchronizerOp::compute(InputContext& op_input, OutputContext& op_output,
                                      ExecutionContext& context) {
  auto enter_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();

  std::map<std::string, nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>> received_frames{};
  for (int i = 0; i < num_streams_; ++i) {
    auto& port_name = in_port_names.at(i);

    // Get input tensor
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>(port_name.c_str());
    if (!maybe_entity) {
      //throw std::runtime_error("Failed to receive input entity");
      HOLOSCAN_LOG_WARN("Failed to receive input entity");
      continue;
    }

    // what happens if no videobuffer is present?
    auto maybe_buffer = holoscan::gxf::get_videobuffer(maybe_entity.value(), "");
    received_frames.emplace(port_name, maybe_buffer);
  }

  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("Operator received: {} frames", received_frames.size());
  }

  auto meta = metadata();

  // checks before processing go here

  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  auto output = nvidia::gxf::Entity::New(context.context());
  if (!output) {
    throw std::runtime_error("Failed to allocate message for output");
  }

  for (auto& [key, frame]: received_frames) {
    auto maybe_frame_dest = output.value().add<nvidia::gxf::VideoBuffer>(key.c_str());
    if (!maybe_frame_dest) {
      throw std::runtime_error("Failed to allocate videobuffer.");
    }
    nvidia::gxf::Handle<nvidia::gxf::Tensor> handle;
    frame->moveToTensor(handle);

    // for now we only support nv12 input
    maybe_frame_dest.value()->createFromTensor<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12>(handle, nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR);
  }

  // Log video buffer and decoder info for debugging
  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("---- Stream Synchronizer Debug Info ----");
    HOLOSCAN_LOG_INFO("------------------------------------------");
  }

  // clean up

  // Emit the single processed frame
  auto emit_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();

  // maybe set metadata ..

  auto output_result = gxf::Entity(std::move(output.value()));
  op_output.emit(output_result, "output");
  last_emit_timestamp_ = emit_timestamp;
}


void TcnStreamSynchronizerOp::stop() {
  // Cleanup resources in reverse order of creation
  // Release the primary context for the device if it was created by this operator
  if (cu_context_) {
    // Ensure the context is not active before releasing it
    CUcontext current_ctx;
    CUresult result = cuCtxGetCurrent(&current_ctx);
    if (result == CUDA_SUCCESS && current_ctx == cu_context_) {
      CudaCheck(cuCtxPopCurrent(nullptr));
    }

    CudaCheck(cuDevicePrimaryCtxRelease(cu_device_));
    cu_context_ = nullptr;
  }
}
}  // namespace holoscan::ops
