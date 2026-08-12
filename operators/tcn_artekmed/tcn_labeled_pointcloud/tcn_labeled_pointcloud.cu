/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
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

namespace {

/// Makes a device current for the duration of a scope and restores the previous one.
///
/// `start()` retains the primary context for `cuda_device_ordinal` but that does not make the device
/// current, and the current device is per-THREAD -- the scheduler runs compute() on whatever worker
/// thread is free. Without this, every cudaMalloc here landed on whichever device that thread had
/// current (device 0 by default) while the stream and the input tensors lived on the configured one,
/// which CUB reports as "invalid device ordinal" and the runtime as an illegal access. Invisible
/// while cuda_device_ordinal is 0, which is why it survived the first round of testing.
struct ScopedDevice {
  int previous = 0;
  explicit ScopedDevice(int device) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDevice(&previous), "failed to read the current device");
    if (previous != device) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(device), "failed to select the CUDA device");
    }
  }
  ~ScopedDevice() {
    if (previous != current()) { cudaSetDevice(previous); }   // best effort in a destructor
  }
  static int current() { int d = 0; cudaGetDevice(&d); return d; }
  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;
};

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
  ScopedDevice device_guard(cuda_device_ordinal_.get());
  if (selected_d_) { cudaFree(selected_d_); selected_d_ = nullptr; }
  if (indices_d_) { cudaFree(indices_d_); indices_d_ = nullptr; }
  if (counts_d_) { cudaFree(counts_d_); counts_d_ = nullptr; }
  if (counts_h_) { cudaFreeHost(counts_h_); counts_h_ = nullptr; }
  if (cub_temp_d_) { cudaFree(cub_temp_d_); cub_temp_d_ = nullptr; }
  cub_temp_bytes_ = 0;
  scratch_count_ = 0;
  if (cu_context_ != nullptr) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
  }
}

void TcnLabeledPointcloudOp::ensure_scratch(int64_t count) {
  if (count <= scratch_count_) return;
  ScopedDevice device_guard(cuda_device_ordinal_.get());
  const int64_t k = static_cast<int64_t>(configured_classes_.size());

  if (selected_d_) { cudaFree(selected_d_); selected_d_ = nullptr; }
  if (indices_d_) { cudaFree(indices_d_); indices_d_ = nullptr; }
  if (cub_temp_d_) { cudaFree(cub_temp_d_); cub_temp_d_ = nullptr; }

  // One selection buffer for all classes: the per-class select and its compaction are issued on the
  // same stream, so stream ordering already prevents the next class from overwriting flags the
  // previous compaction has not read yet.
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMalloc(reinterpret_cast<void**>(&selected_d_), count * sizeof(uint8_t)),
      "TcnLabeledPointcloudOp: failed to allocate the selection buffer");
  // Indices are per class, unlike the flags: they must all still be readable after the single
  // synchronisation, when the gathers are issued.
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMalloc(reinterpret_cast<void**>(&indices_d_), k * count * sizeof(int32_t)),
      "TcnLabeledPointcloudOp: failed to allocate the index buffer");

  if (counts_d_ == nullptr) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMalloc(reinterpret_cast<void**>(&counts_d_), k * sizeof(int32_t)),
        "TcnLabeledPointcloudOp: failed to allocate the count buffer");
    // Pinned, so the one count copy is a real DMA rather than a staged pageable transfer.
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaHostAlloc(reinterpret_cast<void**>(&counts_h_), k * sizeof(int32_t),
                      cudaHostAllocDefault),
        "TcnLabeledPointcloudOp: failed to allocate the pinned count buffer");
  }

  // Temp-storage size depends on the item count, so it is queried whenever the count grows. The
  // query itself launches nothing.
  std::size_t bytes = 0;
  auto status = cub::DeviceSelect::Flagged(
      nullptr, bytes, thrust::counting_iterator<int32_t>(0), selected_d_, indices_d_, counts_d_,
      static_cast<int>(count));
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("TcnLabeledPointcloudOp: cub::DeviceSelect::Flagged size "
                                         "query failed: ") + cudaGetErrorString(status));
  }
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&cub_temp_d_, bytes),
                                 "TcnLabeledPointcloudOp: failed to allocate CUB temp storage");
  cub_temp_bytes_ = bytes;
  scratch_count_ = count;
}

void TcnLabeledPointcloudOp::compute(holoscan::InputContext& op_input,
                                     holoscan::OutputContext& op_output,
                                     holoscan::ExecutionContext& context) {
  // Everything below -- the scratch allocations, the CUB calls and the kernels -- must run against
  // the configured device, not whatever this worker thread happened to have current.
  ScopedDevice device_guard(cuda_device_ordinal_.get());

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

  const int64_t k_classes = static_cast<int64_t>(configured_classes_.size());

  // --- issue phase: select and compact every class with the stream still running ----------------
  // cub::DeviceSelect::Flagged writes its result count to DEVICE memory, which is the whole point:
  // thrust::copy_if returns a host-side iterator, and reading it forces a synchronise per class.
  // It is also stable, so the compacted order is the source order and the output stays reproducible.
  for (int64_t k = 0; k < k_classes; ++k) {
    const auto cls = configured_classes_[static_cast<std::size_t>(k)];
    launch_select_class(labels_d, label_count, static_cast<int>(cls), selected_d_, cuda_stream);
    auto status = cub::DeviceSelect::Flagged(
        cub_temp_d_, cub_temp_bytes_, thrust::counting_iterator<int32_t>(0), selected_d_,
        indices_d_ + k * label_count, counts_d_ + k, static_cast<int>(label_count), cuda_stream);
    if (status != cudaSuccess) {
      int current = -1;
      cudaGetDevice(&current);
      throw std::runtime_error(
          std::string("TcnLabeledPointcloudOp: cub::DeviceSelect::Flagged failed: ") +
          cudaGetErrorString(status) + " (configured device " +
          std::to_string(cuda_device_ordinal_.get()) + ", current device " +
          std::to_string(current) + ", " + std::to_string(label_count) + " items)");
    }
  }

  // The ONE synchronisation. The counts have to reach the host because each output tensor is sized
  // to its point count before it can be allocated; what this avoids is paying for that round trip
  // once per class.
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpyAsync(counts_h_, counts_d_, k_classes * sizeof(int32_t), cudaMemcpyDeviceToHost,
                      cuda_stream),
      "TcnLabeledPointcloudOp: failed to copy the per-class counts back");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(cuda_stream),
                                 "TcnLabeledPointcloudOp: failed to wait for the per-class counts");

  // --- emit phase: allocate to the now-known sizes and gather ------------------------------------
  std::string counts;
  for (int64_t k = 0; k < k_classes; ++k) {
    const auto cls = configured_classes_[static_cast<std::size_t>(k)];
    const int64_t n = static_cast<int64_t>(counts_h_[k]);

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
      launch_gather_points(positions_d, labels_d, indices_d_ + k * label_count, n,
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
