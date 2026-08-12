/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../common/utils.h"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_instance_stats.cuh"

#include "../common/datatypes.hpp"

#include <gxf/std/tensor.hpp>

namespace tcn::ops {

namespace {

/// Makes a device current for a scope and restores the previous one. `start()` retains the primary
/// context for `cuda_device_ordinal`, but that does not make the device current, and the current
/// device is per-THREAD while the scheduler runs compute() on any free worker. Without this, the
/// scratch allocations land on whichever device that thread had current.
struct ScopedDevice {
  int previous = 0;
  explicit ScopedDevice(int device) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDevice(&previous), "failed to read the current device");
    if (previous != device) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaSetDevice(device), "failed to select the CUDA device");
    }
  }
  ~ScopedDevice() {
    int now = 0;
    cudaGetDevice(&now);
    if (now != previous) { cudaSetDevice(previous); }   // best effort in a destructor
  }
  ScopedDevice(const ScopedDevice&) = delete;
  ScopedDevice& operator=(const ScopedDevice&) = delete;
};

int64_t element_count(const std::shared_ptr<holoscan::Tensor>& t) {
  int64_t n = 1;
  for (const auto& d : t->shape()) { n *= d; }
  return n;
}

int bytes_per_element(const std::shared_ptr<holoscan::Tensor>& t) {
  return (t->dtype().bits + 7) / 8;
}

}  // namespace

void TcnInstanceStatsOp::setup(holoscan::OperatorSpec& spec) {
  using namespace std::string_literals;

  spec.input<holoscan::gxf::Entity>("positions");   // device, [H, W, 3], float32, world space
  spec.input<holoscan::gxf::Entity>("labels");      // device, [H, W] or [H, W, 1], uint16

  spec.output<holoscan::gxf::Entity>("instances");  // HOST, [K, kInstanceStatColumns] + [K] uint16

  spec.param(allocator_, "allocator", "Allocator", "Allocator used for the output tensors.");
  spec.param(in_positions_tensor_name_, "in_positions_tensor_name", "Positions Input Tensor Name",
             "", ""s);
  spec.param(in_labels_tensor_name_, "in_labels_tensor_name", "Labels Input Tensor Name", "", ""s);
  spec.param(out_rows_tensor_name_, "out_rows_tensor_name", "Rows Output Tensor Name", "", "rows"s);
  spec.param(out_labels_tensor_name_, "out_labels_tensor_name", "Labels Output Tensor Name", "",
             "labels"s);
  spec.param(camera_index_, "camera_index",
             "Camera Index",
             "Written into every row, so a consumer receiving several cameras on one ANY_SIZE port "
             "can tell which camera an observation came from.",
             static_cast<int64_t>(0));
  spec.param(sigma_k_, "sigma_k",
             "Sigma K",
             "Points further than this many standard deviations from the instance mean on any axis "
             "are excluded from the box and centroid.",
             2.5);
  spec.param(sigma_floor_m_, "sigma_floor_m",
             "Sigma Floor",
             "Lower bound on the per-axis sigma used for trimming, in metres. Without it a "
             "perfectly flat instance (sigma ~ 0 on one axis) rejects all of its own points.",
             0.01);
  spec.param(min_points_, "min_points",
             "Minimum Points",
             "Instances with fewer surviving points than this are dropped -- that is depth noise or "
             "mask fringe, not an object.",
             static_cast<int64_t>(64));
  spec.param(max_instances_, "max_instances",
             "Maximum Instances",
             "Output row capacity. Overflow is counted and reported, never silently truncated.",
             static_cast<int64_t>(64));
  spec.param(verbose_, "verbose", "Verbose", "Log the per-instance rows every frame.", false);
  spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
             "Device to use for CUDA operations", holoscan::ParameterFlag::kOptional);
  spec.param(cuda_stream_pool_, "cuda_stream_pool", "Cuda Stream Pool",
             "Instance of gxf::CudaStreamPool.", holoscan::ParameterFlag::kOptional);
}

void TcnInstanceStatsOp::initialize() { Operator::initialize(); }

void TcnInstanceStatsOp::start() {
  CudaCheck(cuInit(0));
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));

  if (sigma_k_.get() <= 0.0) {
    throw std::runtime_error("TcnInstanceStatsOp: sigma_k must be > 0 (got " +
                             std::to_string(sigma_k_.get()) + "); to disable trimming use a large "
                             "value rather than zero, which would reject every point");
  }
  if (max_instances_.get() < 1) {
    throw std::runtime_error("TcnInstanceStatsOp: max_instances must be >= 1");
  }

  ScopedDevice guard(cuda_device_ordinal_.get());
  allocate_scratch();
  HOLOSCAN_LOG_INFO("TcnInstanceStatsOp: camera_index={} sigma_k={} min_points={} max_instances={}",
                    camera_index_.get(), sigma_k_.get(), min_points_.get(), max_instances_.get());
}

void TcnInstanceStatsOp::allocate_scratch() {
  const size_t slots = static_cast<size_t>(kInstanceSlots);
  auto alloc = [](void** p, size_t bytes, const char* what) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(p, bytes),
                                   (std::string("TcnInstanceStatsOp: failed to allocate ") + what));
  };
  alloc(reinterpret_cast<void**>(&acc_.count1), slots * sizeof(uint32_t), "count1");
  alloc(reinterpret_cast<void**>(&acc_.sum1), slots * 3 * sizeof(float), "sum1");
  alloc(reinterpret_cast<void**>(&acc_.sqsum1), slots * 3 * sizeof(float), "sqsum1");
  alloc(reinterpret_cast<void**>(&acc_.mean), slots * 3 * sizeof(float), "mean");
  alloc(reinterpret_cast<void**>(&acc_.sigma), slots * 3 * sizeof(float), "sigma");
  alloc(reinterpret_cast<void**>(&acc_.count2), slots * sizeof(uint32_t), "count2");
  alloc(reinterpret_cast<void**>(&acc_.sum2), slots * 3 * sizeof(float), "sum2");
  alloc(reinterpret_cast<void**>(&acc_.minEnc), slots * 3 * sizeof(int32_t), "minEnc");
  alloc(reinterpret_cast<void**>(&acc_.maxEnc), slots * 3 * sizeof(int32_t), "maxEnc");

  const size_t max_rows = static_cast<size_t>(max_instances_.get());
  alloc(reinterpret_cast<void**>(&rows_d_), max_rows * kInstanceStatColumns * sizeof(float), "rows");
  alloc(reinterpret_cast<void**>(&row_labels_d_), max_rows * sizeof(uint16_t), "row labels");
  alloc(reinterpret_cast<void**>(&row_count_d_), sizeof(uint32_t), "row count");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaHostAlloc(reinterpret_cast<void**>(&row_count_h_), sizeof(uint32_t), cudaHostAllocDefault),
      "TcnInstanceStatsOp: failed to allocate the pinned row count");
}

void TcnInstanceStatsOp::free_scratch() {
  for (void* p : {reinterpret_cast<void*>(acc_.count1), reinterpret_cast<void*>(acc_.sum1),
                  reinterpret_cast<void*>(acc_.sqsum1), reinterpret_cast<void*>(acc_.mean),
                  reinterpret_cast<void*>(acc_.sigma), reinterpret_cast<void*>(acc_.count2),
                  reinterpret_cast<void*>(acc_.sum2), reinterpret_cast<void*>(acc_.minEnc),
                  reinterpret_cast<void*>(acc_.maxEnc), reinterpret_cast<void*>(rows_d_),
                  reinterpret_cast<void*>(row_labels_d_), reinterpret_cast<void*>(row_count_d_)}) {
    if (p) cudaFree(p);
  }
  acc_ = InstanceAccumulators{};
  rows_d_ = nullptr;
  row_labels_d_ = nullptr;
  row_count_d_ = nullptr;
  if (row_count_h_) { cudaFreeHost(row_count_h_); row_count_h_ = nullptr; }
}

void TcnInstanceStatsOp::stop() {
  ScopedDevice guard(cuda_device_ordinal_.get());
  free_scratch();
  if (overflow_frames_ > 0) {
    HOLOSCAN_LOG_ERROR("TcnInstanceStatsOp[cam {}]: {} of {} frames exceeded max_instances={}; "
                       "those frames reported only the first {} instances",
                       camera_index_.get(), overflow_frames_, emitted_, max_instances_.get(),
                       max_instances_.get());
  }
  if (cu_context_ != nullptr) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
  }
}

void TcnInstanceStatsOp::compute(holoscan::InputContext& op_input,
                                 holoscan::OutputContext& op_output,
                                 holoscan::ExecutionContext& context) {
  ScopedDevice guard(cuda_device_ordinal_.get());

  auto maybe_positions = op_input.receive<holoscan::gxf::Entity>("positions");
  if (!maybe_positions) {
    throw std::runtime_error("TcnInstanceStatsOp: failed to read the 'positions' input");
  }
  auto positions_t =
      maybe_positions.value().get<holoscan::Tensor>(in_positions_tensor_name_.get().c_str());
  if (!positions_t) {
    throw std::runtime_error("TcnInstanceStatsOp: no tensor named '" +
                             in_positions_tensor_name_.get() + "' on the 'positions' input");
  }
  cudaStream_t cuda_stream = op_input.receive_cuda_stream("positions", true, false);

  auto maybe_labels = op_input.receive<holoscan::gxf::Entity>("labels");
  if (!maybe_labels) {
    throw std::runtime_error("TcnInstanceStatsOp: failed to read the 'labels' input");
  }
  auto labels_t = maybe_labels.value().get<holoscan::Tensor>(in_labels_tensor_name_.get().c_str());
  if (!labels_t) {
    throw std::runtime_error("TcnInstanceStatsOp: no tensor named '" +
                             in_labels_tensor_name_.get() + "' on the 'labels' input");
  }
  op_input.receive_cuda_stream("labels", true, false);

  if (bytes_per_element(positions_t) != 4) {
    throw std::runtime_error("TcnInstanceStatsOp: 'positions' must be float32");
  }
  if (bytes_per_element(labels_t) != 2) {
    throw std::runtime_error("TcnInstanceStatsOp: 'labels' must be uint16 (packed panoptic)");
  }
  const int64_t label_count = element_count(labels_t);
  const int64_t position_count = element_count(positions_t) / 3;
  if (element_count(positions_t) % 3 != 0) {
    throw std::runtime_error("TcnInstanceStatsOp: 'positions' must be [.., 3] xyz");
  }
  // The two must index the same grid: positions come from the depth image and the labels were
  // sampled through that image's texcoords. A mismatch means different cameras or different frames,
  // and every instance would be reduced over some other pixel's points.
  if (label_count != position_count) {
    throw std::runtime_error(
        "TcnInstanceStatsOp: 'positions' holds " + std::to_string(position_count) +
        " points but 'labels' holds " + std::to_string(label_count) +
        "; they must be the same depth grid");
  }

  const auto* positions_d = static_cast<const float*>(positions_t->data());
  const auto* labels_d = static_cast<const uint16_t*>(labels_t->data());

  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemsetAsync(row_count_d_, 0, sizeof(uint32_t), cuda_stream),
                                 "TcnInstanceStatsOp: failed to reset the row counter");
  launch_instance_reset(acc_, cuda_stream);
  launch_instance_pass1(positions_d, labels_d, label_count, acc_, cuda_stream);
  launch_instance_finalize1(acc_, cuda_stream);
  launch_instance_pass2(positions_d, labels_d, label_count,
                        static_cast<float>(sigma_k_.get()),
                        static_cast<float>(sigma_floor_m_.get()), acc_, cuda_stream);
  launch_instance_compact(acc_, static_cast<int>(camera_index_.get()),
                          static_cast<uint32_t>(std::max<int64_t>(min_points_.get(), 0)),
                          rows_d_, row_labels_d_, row_count_d_,
                          static_cast<uint32_t>(max_instances_.get()), cuda_stream);

  // The one synchronisation: the output tensors are sized to the instance count, so it has to reach
  // the host before they can be allocated.
  HOLOSCAN_CUDA_CALL_THROW_ERROR(
      cudaMemcpyAsync(row_count_h_, row_count_d_, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                      cuda_stream),
      "TcnInstanceStatsOp: failed to copy the instance count back");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(cuda_stream),
                                 "TcnInstanceStatsOp: failed to wait for the instance count");

  const uint32_t found = *row_count_h_;
  const uint32_t capacity = static_cast<uint32_t>(max_instances_.get());
  const uint32_t k = std::min(found, capacity);
  if (found > capacity) {
    ++overflow_frames_;
    HOLOSCAN_LOG_WARN("TcnInstanceStatsOp[cam {}]: {} instances exceed max_instances={}; reporting "
                      "the first {}. Raise max_instances or raise min_points.",
                      camera_index_.get(), found, capacity, capacity);
  }

  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  auto maybe_entity = nvidia::gxf::Entity::New(context.context());
  if (!maybe_entity) {
    throw std::runtime_error("TcnInstanceStatsOp: failed to allocate the output message");
  }
  auto out_entity = std::move(maybe_entity.value());

  // A frame with no instances still emits, with one all-zero row whose count is 0. An empty tensor
  // would give the Python consumer a shape it has to special-case, and a missing emit would starve a
  // downstream operator that expects one message per frame.
  const int32_t rows = static_cast<int32_t>(std::max<uint32_t>(k, 1u));

  // NOTE the `initialize_buffer` argument of allocate_named_tensor is deliberately NOT used here: it
  // zeroes via cudaMemsetAsync, which is not valid on a HOST pointer, so the buffer would be left
  // holding whatever the allocator last had there. That is not hypothetical -- it showed up as a
  // no-instance frame reporting a previous frame's label and count. Zero on the host instead.
  nvidia::gxf::Handle<nvidia::gxf::Tensor> out_rows = nullptr;
  if (!tcn::allocate_named_tensor<float>(allocator.value(), cuda_stream, out_entity,
                                         nvidia::gxf::Shape{{rows, kInstanceStatColumns}},
                                         nvidia::gxf::MemoryStorageType::kHost,
                                         out_rows_tensor_name_.get(), out_rows)) {
    throw std::runtime_error("TcnInstanceStatsOp: failed to allocate the rows output");
  }
  nvidia::gxf::Handle<nvidia::gxf::Tensor> out_labels = nullptr;
  if (!tcn::allocate_named_tensor<uint16_t>(allocator.value(), cuda_stream, out_entity,
                                            nvidia::gxf::Shape{{rows}},
                                            nvidia::gxf::MemoryStorageType::kHost,
                                            out_labels_tensor_name_.get(), out_labels)) {
    throw std::runtime_error("TcnInstanceStatsOp: failed to allocate the labels output");
  }
  std::memset(out_rows->data<float>().value(), 0,
              static_cast<size_t>(rows) * kInstanceStatColumns * sizeof(float));
  std::memset(out_labels->data<uint16_t>().value(), 0,
              static_cast<size_t>(rows) * sizeof(uint16_t));

  if (k > 0) {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpyAsync(out_rows->data<float>().value(), rows_d_,
                        static_cast<size_t>(k) * kInstanceStatColumns * sizeof(float),
                        cudaMemcpyDeviceToHost, cuda_stream),
        "TcnInstanceStatsOp: failed to copy the instance rows back");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpyAsync(out_labels->data<uint16_t>().value(), row_labels_d_,
                        static_cast<size_t>(k) * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                        cuda_stream),
        "TcnInstanceStatsOp: failed to copy the instance labels back");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamSynchronize(cuda_stream),
                                   "TcnInstanceStatsOp: failed to wait for the instance rows");
  }

  ++emitted_;
  if (verbose_.get()) {
    const float* r = out_rows->data<float>().value();
    const uint16_t* l = out_labels->data<uint16_t>().value();
    for (uint32_t i = 0; i < k; ++i) {
      const float* row = r + static_cast<size_t>(i) * kInstanceStatColumns;
      HOLOSCAN_LOG_INFO("TcnInstanceStatsOp[cam {}] frame {}: label {} (class {} inst {}) "
                        "n={} centroid=({:.3f},{:.3f},{:.3f}) extent=({:.3f},{:.3f},{:.3f})",
                        camera_index_.get(), emitted_, l[i], l[i] >> kPanopticClassShift,
                        l[i] & 0xFF, static_cast<int>(row[kColCount]),
                        row[kColCentroidX], row[kColCentroidY], row[kColCentroidZ],
                        row[kColMaxX] - row[kColMinX], row[kColMaxY] - row[kColMinY],
                        row[kColMaxZ] - row[kColMinZ]);
    }
  }

  // Forward the acquisition timestamp. This operator emits a FRESH entity, so frame identity does
  // not propagate by itself -- and without it a downstream consumer that groups by frame (the
  // cross-camera fusion does) has nothing to group on. Omitting this showed up immediately as
  // `acq=-1` in the tracker's output.
  const auto acq = op_input.get_acquisition_timestamp("positions");
  auto message = holoscan::gxf::Entity(std::move(out_entity));
  op_output.emit(message, "instances", acq.value_or(-1));
}

}  // namespace tcn::ops
