/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <any>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/utils.h"
#include "../cuda/tcn_labeled_pointcloud_kernel.cuh"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_labeled_pointcloud.cuh"

#include "../common/datatypes.hpp"

#include <gxf/std/tensor.hpp>

namespace tcn::ops {

namespace {

int64_t element_count(const std::shared_ptr<holoscan::Tensor>& t) {
  int64_t n = 1;
  for (const auto& d : t->shape()) { n *= d; }
  return n;
}

int bytes_per_element(const std::shared_ptr<holoscan::Tensor>& t) {
  return (t->dtype().bits + 7) / 8;
}

}  // namespace

std::string TcnLabeledPointcloudOp::port_name_for_class(int64_t cls) {
  return cls < 0 ? std::string("class_all") : ("class_" + std::to_string(cls));
}

void TcnLabeledPointcloudOp::setup(holoscan::OperatorSpec& spec) {
  using namespace std::string_literals;

  spec.input<holoscan::gxf::Entity>("positions");  // device, [H, W, 3], float32
  spec.input<holoscan::gxf::Entity>("labels");     // device, [H, W] or [H, W, 1], uint16

  // Ports must be created here, and setup() runs BEFORE parameter values are applied, so the class
  // list has to be read from args() -- the same constraint tcn_stream_synchronizer documents.
  configured_classes_.clear();
  for (const auto& arg : args()) {
    if (arg.name() != "classes") continue;
    try {
      configured_classes_ = std::any_cast<std::vector<int64_t>>(arg.value());
    } catch (const std::bad_any_cast&) {
      throw std::runtime_error("TcnLabeledPointcloudOp: `classes` must be a list of ints");
    }
  }
  if (configured_classes_.empty()) {
    // One port carrying every non-background class. Still a point cloud, just not split by class.
    configured_classes_.push_back(-1);
  }
  for (const auto& cls : configured_classes_) {
    spec.output<holoscan::gxf::Entity>(port_name_for_class(cls));
  }

  spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
  spec.param(in_positions_tensor_name_, "in_positions_tensor_name", "Positions Input Tensor Name",
             "", ""s);
  spec.param(in_labels_tensor_name_, "in_labels_tensor_name", "Labels Input Tensor Name", "", ""s);
  spec.param(out_positions_tensor_name_, "out_positions_tensor_name", "Positions Output Tensor "
             "Name", "", "positions"s);
  spec.param(out_labels_tensor_name_, "out_labels_tensor_name", "Labels Output Tensor Name", "",
             "labels"s);
  spec.param(classes_, "classes", "Classes", "Class ids to emit, one output port each.",
             std::vector<int64_t>{});
  spec.param(verbose_, "verbose", "Verbose", "Log the per-class point counts every frame.", false);
  spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
             "Device to use for CUDA operations", holoscan::ParameterFlag::kOptional);
  spec.param(cuda_stream_pool_, "cuda_stream_pool", "Cuda Stream Pool",
             "Instance of gxf::CudaStreamPool.", holoscan::ParameterFlag::kOptional);
}

void TcnLabeledPointcloudOp::initialize() {
  Operator::initialize();
}

void TcnLabeledPointcloudOp::start() {
  CudaCheck(cuInit(0));
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));

  for (const auto& cls : configured_classes_) {
    if (cls >= kPanopticNumClasses) {
      throw std::runtime_error("TcnLabeledPointcloudOp: class " + std::to_string(cls) +
                               " is outside [0, " + std::to_string(kPanopticNumClasses - 1) +
                               "]; the packed panoptic encoding gives a class id one byte");
    }
  }
  std::string summary;
  for (const auto& cls : configured_classes_) { summary += " " + port_name_for_class(cls); }
  HOLOSCAN_LOG_INFO("TcnLabeledPointcloudOp: emitting{}", summary);
}

void TcnLabeledPointcloudOp::stop() {
  if (selected_d_) { cudaFree(selected_d_); selected_d_ = nullptr; }
  if (indices_d_) { cudaFree(indices_d_); indices_d_ = nullptr; }
  scratch_count_ = 0;
  if (cu_context_ != nullptr) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
  }
}

void TcnLabeledPointcloudOp::ensure_scratch(int64_t count) {
  if (count <= scratch_count_) return;
  if (selected_d_) { cudaFree(selected_d_); selected_d_ = nullptr; }
  if (indices_d_) { cudaFree(indices_d_); indices_d_ = nullptr; }
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMalloc(reinterpret_cast<void**>(&selected_d_), count * sizeof(uint8_t)),
      "TcnLabeledPointcloudOp: failed to allocate the selection buffer");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMalloc(reinterpret_cast<void**>(&indices_d_), count * sizeof(int32_t)),
      "TcnLabeledPointcloudOp: failed to allocate the index buffer");
  scratch_count_ = count;
}

void TcnLabeledPointcloudOp::compute(holoscan::InputContext& op_input,
                                     holoscan::OutputContext& op_output,
                                     holoscan::ExecutionContext& context) {
  auto maybe_positions = op_input.receive<holoscan::gxf::Entity>("positions");
  if (!maybe_positions) {
    throw std::runtime_error("TcnLabeledPointcloudOp: failed to read the 'positions' input");
  }
  auto positions_t =
      maybe_positions.value().get<holoscan::Tensor>(in_positions_tensor_name_.get().c_str());
  if (!positions_t) {
    throw std::runtime_error("TcnLabeledPointcloudOp: no tensor named '" +
                             in_positions_tensor_name_.get() + "' on the 'positions' input");
  }
  cudaStream_t cuda_stream = op_input.receive_cuda_stream("positions", true, false);

  auto maybe_labels = op_input.receive<holoscan::gxf::Entity>("labels");
  if (!maybe_labels) {
    throw std::runtime_error("TcnLabeledPointcloudOp: failed to read the 'labels' input");
  }
  auto labels_t = maybe_labels.value().get<holoscan::Tensor>(in_labels_tensor_name_.get().c_str());
  if (!labels_t) {
    throw std::runtime_error("TcnLabeledPointcloudOp: no tensor named '" +
                             in_labels_tensor_name_.get() + "' on the 'labels' input");
  }
  op_input.receive_cuda_stream("labels", true, false);

  if (bytes_per_element(positions_t) != 4) {
    throw std::runtime_error("TcnLabeledPointcloudOp: 'positions' must be float32");
  }
  if (bytes_per_element(labels_t) != 2) {
    throw std::runtime_error("TcnLabeledPointcloudOp: 'labels' must be uint16 (packed panoptic)");
  }

  const int64_t label_count = element_count(labels_t);
  const int64_t position_count = element_count(positions_t) / 3;
  // The two must index the same grid: positions come from the depth image and the labels were
  // sampled through that image's texcoords. A mismatch means they are from different cameras or
  // different frames, and every point would be given some other pixel's label.
  if (label_count != position_count) {
    throw std::runtime_error(
        "TcnLabeledPointcloudOp: 'positions' holds " + std::to_string(position_count) +
        " points but 'labels' holds " + std::to_string(label_count) +
        "; they must be the same depth grid");
  }
  if (element_count(positions_t) % 3 != 0) {
    throw std::runtime_error("TcnLabeledPointcloudOp: 'positions' must be [.., 3] xyz");
  }

  ensure_scratch(label_count);

  const auto* positions_d = static_cast<const float*>(positions_t->data());
  const auto* labels_d = static_cast<const uint16_t*>(labels_t->data());

  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  auto gxf_context = context.context();

  std::string counts;
  for (const auto& cls : configured_classes_) {
    launch_select_class(labels_d, label_count, static_cast<int>(cls), selected_d_, cuda_stream);

    // Stable stream compaction: copy_if over ascending indices preserves source order, so the point
    // order is a function of the input alone. An atomic append would be faster and would reorder
    // points run to run, which would make any byte-comparison gate useless.
    auto policy = thrust::cuda::par.on(cuda_stream);
    auto end = thrust::copy_if(policy,
                               thrust::counting_iterator<int32_t>(0),
                               thrust::counting_iterator<int32_t>(
                                   static_cast<int32_t>(label_count)),
                               selected_d_,
                               indices_d_,
                               [] __device__(uint8_t flag) { return flag != 0; });
    const int64_t n = static_cast<int64_t>(end - indices_d_);   // synchronises on `cuda_stream`

    // An empty class still emits, because a downstream merger needs every input every frame; the
    // single point it emits is NaN, which the rasteriser culls (see the kernel).
    const int64_t emit_n = std::max<int64_t>(n, 1);

    auto maybe_entity = nvidia::gxf::Entity::New(gxf_context);
    if (!maybe_entity) {
      throw std::runtime_error("TcnLabeledPointcloudOp: failed to allocate an output message");
    }
    auto out_entity = std::move(maybe_entity.value());

    nvidia::gxf::Handle<nvidia::gxf::Tensor> out_positions = nullptr;
    if (!tcn::allocate_named_tensor<float>(allocator.value(), cuda_stream, out_entity,
                                           nvidia::gxf::Shape{{1, static_cast<int32_t>(emit_n), 3}},
                                           nvidia::gxf::MemoryStorageType::kDevice,
                                           out_positions_tensor_name_.get(), out_positions)) {
      throw std::runtime_error("TcnLabeledPointcloudOp: failed to allocate the positions output");
    }
    nvidia::gxf::Handle<nvidia::gxf::Tensor> out_labels = nullptr;
    if (!tcn::allocate_named_tensor<uint16_t>(allocator.value(), cuda_stream, out_entity,
                                              nvidia::gxf::Shape{{1, static_cast<int32_t>(emit_n),
                                                                  1}},
                                              nvidia::gxf::MemoryStorageType::kDevice,
                                              out_labels_tensor_name_.get(), out_labels)) {
      throw std::runtime_error("TcnLabeledPointcloudOp: failed to allocate the labels output");
    }

    if (n > 0) {
      launch_gather_points(positions_d, labels_d, indices_d_, n,
                           out_positions->data<float>().value(),
                           out_labels->data<uint16_t>().value(), cuda_stream);
    } else {
      launch_write_empty_point(out_positions->data<float>().value(),
                               out_labels->data<uint16_t>().value(), cuda_stream);
    }

    auto message = holoscan::gxf::Entity(std::move(out_entity));
    op_output.emit(message, port_name_for_class(cls).c_str());
    if (verbose_.get()) { counts += " " + port_name_for_class(cls) + "=" + std::to_string(n); }
  }

  ++emitted_;
  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("TcnLabeledPointcloudOp: frame {} of {} source points:{}",
                      emitted_, label_count, counts);
  }
}

}  // namespace tcn::ops
