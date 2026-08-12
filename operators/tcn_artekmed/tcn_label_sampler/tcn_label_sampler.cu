/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/utils.h"
#include "../cuda/tcn_label_sampler_kernel.cuh"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_label_sampler.cuh"

#include "../common/datatypes.hpp"

#include <gxf/std/tensor.hpp>

namespace tcn::ops {

namespace {

/// Number of elements in a tensor whose trailing dimensions are 1, i.e. [H,W] and [H,W,1] agree.
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

void TcnLabelSamplerOp::setup(holoscan::OperatorSpec& spec) {
  using namespace std::string_literals;
  HOLOSCAN_LOG_DEBUG("TcnLabelSamplerOp::setup");

  spec.input<holoscan::gxf::Entity>("labels");     // device, [Hc, Wc] or [Hc, Wc, 1], uint16
  spec.input<holoscan::gxf::Entity>("texcoords");  // device, [Hd, Wd, 2], float32

  spec.output<holoscan::gxf::Entity>("labels_out");  // device, [Hd, Wd, 1], uint16
  spec.output<holoscan::gxf::Entity>("mask_out");    // device, [Hd, Wd, 1], uint8

  spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");

  spec.param(in_labels_tensor_name_, "in_labels_tensor_name", "Labels Input Tensor Name", "", ""s);
  spec.param(in_texcoord_tensor_name_, "in_texcoord_tensor_name", "Texcoord Input Tensor Name", "",
             ""s);
  spec.param(out_labels_tensor_name_, "out_labels_tensor_name", "Labels Output Tensor Name", "",
             ""s);
  spec.param(out_mask_tensor_name_, "out_mask_tensor_name", "Mask Output Tensor Name", "", ""s);

  spec.param(select_classes_,
             "select_classes",
             "Selected Classes",
             "Class ids that mask_out marks. Empty selects every non-background class.",
             std::vector<int64_t>{});
  spec.param(unlabeled_value_,
             "unlabeled_value",
             "Unlabeled Value",
             "Label written where a depth pixel has no colour correspondence.",
             static_cast<int64_t>(0));

  spec.param(cuda_device_ordinal_,
             "cuda_device_ordinal",
             "CudaDeviceOrdinal",
             "Device to use for CUDA operations",
             holoscan::ParameterFlag::kOptional);
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "Cuda Stream Pool",
             "Instance of gxf::CudaStreamPool.",
             holoscan::ParameterFlag::kOptional);
}

void TcnLabelSamplerOp::initialize() {
  HOLOSCAN_LOG_DEBUG("TcnLabelSamplerOp::initialize");
  Operator::initialize();
}

void TcnLabelSamplerOp::start() {
  CudaCheck(cuInit(0));
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));

  // Reject an out-of-range class id at startup rather than indexing a 256-entry table with it.
  // The packed encoding gives a class the high byte of a uint16, so this bound is structural.
  for (const auto& c : select_classes_.get()) {
    if (c < 0 || c >= kPanopticNumClasses) {
      throw std::runtime_error("TcnLabelSamplerOp: select_classes entry " + std::to_string(c) +
                               " is outside [0, " + std::to_string(kPanopticNumClasses - 1) +
                               "]; the packed panoptic encoding gives a class id one byte");
    }
  }
  const auto unlabeled = unlabeled_value_.get();
  if (unlabeled < 0 || unlabeled > 0xFFFF) {
    throw std::runtime_error("TcnLabelSamplerOp: unlabeled_value " + std::to_string(unlabeled) +
                             " does not fit the uint16 label type");
  }
}

void TcnLabelSamplerOp::stop() {
  ScopedDevice device_guard(cuda_device_ordinal_.get());
  if (class_select_d_ != nullptr) {
    cudaFree(class_select_d_);
    class_select_d_ = nullptr;
  }
  if (cu_context_ != nullptr) {
    cuDevicePrimaryCtxRelease(cu_device_);
    cu_context_ = nullptr;
  }
}

const uint8_t* TcnLabelSamplerOp::class_select_device() {
  ScopedDevice device_guard(cuda_device_ordinal_.get());
  const auto& selected = select_classes_.get();
  if (selected.empty()) {
    return nullptr;   // kernel contract: nullptr means "every non-background class"
  }
  if (class_select_d_ == nullptr) {
    std::vector<uint8_t> host(kPanopticNumClasses, 0);
    for (const auto& c : selected) { host[static_cast<std::size_t>(c)] = 1; }
    // CudaCheck (common/utils.h) wraps the DRIVER api and rejects a cudaError_t; these are runtime
    // api calls, so they use holoscan's runtime-api macro.
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMalloc(reinterpret_cast<void**>(&class_select_d_), host.size()),
        "TcnLabelSamplerOp: failed to allocate the class-selection table");
    HOLOSCAN_CUDA_CALL_THROW_ERROR(
        cudaMemcpy(class_select_d_, host.data(), host.size(), cudaMemcpyHostToDevice),
        "TcnLabelSamplerOp: failed to upload the class-selection table");
  }
  return class_select_d_;
}

void TcnLabelSamplerOp::compute(holoscan::InputContext& op_input,
                                holoscan::OutputContext& op_output,
                                holoscan::ExecutionContext& context) {
  auto maybe_labels_entity = op_input.receive<holoscan::gxf::Entity>("labels");
  if (!maybe_labels_entity) {
    throw std::runtime_error("TcnLabelSamplerOp: failed to read the 'labels' input entity");
  }
  auto labels_t =
      maybe_labels_entity.value().get<holoscan::Tensor>(in_labels_tensor_name_.get().c_str());
  if (!labels_t) {
    throw std::runtime_error("TcnLabelSamplerOp: no tensor named '" +
                             in_labels_tensor_name_.get() + "' on the 'labels' input");
  }
  cudaStream_t cuda_stream = op_input.receive_cuda_stream("labels", true, false);

  auto maybe_texcoords_entity = op_input.receive<holoscan::gxf::Entity>("texcoords");
  if (!maybe_texcoords_entity) {
    throw std::runtime_error("TcnLabelSamplerOp: failed to read the 'texcoords' input entity");
  }
  auto texcoord_t =
      maybe_texcoords_entity.value().get<holoscan::Tensor>(in_texcoord_tensor_name_.get().c_str());
  if (!texcoord_t) {
    throw std::runtime_error("TcnLabelSamplerOp: no tensor named '" +
                             in_texcoord_tensor_name_.get() + "' on the 'texcoords' input");
  }
  op_input.receive_cuda_stream("texcoords", true, false);

  // Sampling an image of the wrong element width reads neighbouring labels as if they were one, or
  // walks off the buffer. Both produce plausible-looking output, so check rather than trust.
  if (bytes_per_element(labels_t) != 2) {
    throw std::runtime_error(
        "TcnLabelSamplerOp: 'labels' must be uint16 (packed panoptic), got " +
        std::to_string(bytes_per_element(labels_t) * 8) + "-bit elements");
  }
  if (bytes_per_element(texcoord_t) != 4) {
    throw std::runtime_error("TcnLabelSamplerOp: 'texcoords' must be float32, got " +
                             std::to_string(bytes_per_element(texcoord_t) * 8) + "-bit elements");
  }

  const auto& label_shape = labels_t->shape();
  if (label_shape.size() < 2) {
    throw std::runtime_error("TcnLabelSamplerOp: 'labels' needs at least 2 dimensions");
  }
  const int label_h = static_cast<int>(label_shape[0]);
  const int label_w = static_cast<int>(label_shape[1]);
  if (element_count(labels_t) != static_cast<int64_t>(label_h) * label_w) {
    throw std::runtime_error(
        "TcnLabelSamplerOp: 'labels' must be single-channel [H,W] or [H,W,1]; got " +
        std::to_string(element_count(labels_t)) + " elements for a " + std::to_string(label_h) +
        "x" + std::to_string(label_w) + " image");
  }

  const auto& uv_shape = texcoord_t->shape();
  if (uv_shape.size() < 3 || uv_shape[2] != 2) {
    throw std::runtime_error(
        "TcnLabelSamplerOp: 'texcoords' must be [H,W,2]; the third dimension is the uv pair");
  }
  const int H = static_cast<int>(uv_shape[0]);
  const int W = static_cast<int>(uv_shape[1]);

  auto gxf_context = context.context();
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  // One entity per output port: the two results have different dtypes and different consumers
  // (`tcn_depthimage_apply_mask` wants the mask alone, as the single tensor of its entity).
  auto maybe_labels_out_entity = nvidia::gxf::Entity::New(gxf_context);
  if (!maybe_labels_out_entity) {
    throw std::runtime_error("TcnLabelSamplerOp: failed to allocate the labels output message");
  }
  auto labels_out_entity = std::move(maybe_labels_out_entity.value());
  nvidia::gxf::Handle<nvidia::gxf::Tensor> labels_out_buffer = nullptr;
  if (!tcn::allocate_named_tensor<uint16_t>(allocator.value(),
                                            cuda_stream,
                                            labels_out_entity,
                                            nvidia::gxf::Shape{{H, W, 1}},
                                            nvidia::gxf::MemoryStorageType::kDevice,
                                            out_labels_tensor_name_.get(),
                                            labels_out_buffer)) {
    throw std::runtime_error("TcnLabelSamplerOp: failed to allocate the labels output tensor");
  }

  auto maybe_mask_out_entity = nvidia::gxf::Entity::New(gxf_context);
  if (!maybe_mask_out_entity) {
    throw std::runtime_error("TcnLabelSamplerOp: failed to allocate the mask output message");
  }
  auto mask_out_entity = std::move(maybe_mask_out_entity.value());
  nvidia::gxf::Handle<nvidia::gxf::Tensor> mask_out_buffer = nullptr;
  if (!tcn::allocate_named_tensor<uint8_t>(allocator.value(),
                                           cuda_stream,
                                           mask_out_entity,
                                           nvidia::gxf::Shape{{H, W, 1}},
                                           nvidia::gxf::MemoryStorageType::kDevice,
                                           out_mask_tensor_name_.get(),
                                           mask_out_buffer)) {
    throw std::runtime_error("TcnLabelSamplerOp: failed to allocate the mask output tensor");
  }

  LabelSamplerParams params{};
  params.labels = static_cast<const uint16_t*>(labels_t->data());
  params.uv = reinterpret_cast<const float2*>(texcoord_t->data());
  params.labelsOut = labels_out_buffer->data<uint16_t>().value();
  params.maskOut = mask_out_buffer->data<uint8_t>().value();
  params.classSelect = class_select_device();
  params.width = W;
  params.height = H;
  params.labelWidth = label_w;
  params.labelHeight = label_h;
  params.unlabeled = static_cast<uint16_t>(unlabeled_value_.get());

  const dim3 block(16, 16);
  const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  label_sampler_nearest_kernel<<<grid, block, 0, cuda_stream>>>(params);

  // Fresh entities: forward frame identity (see the collection README on acquisition timestamps).
  const int64_t acq = op_input.get_acquisition_timestamp("labels").value_or(-1);
  auto labels_message = holoscan::gxf::Entity(std::move(labels_out_entity));
  op_output.emit(labels_message, "labels_out", acq);
  auto mask_message = holoscan::gxf::Entity(std::move(mask_out_entity));
  op_output.emit(mask_message, "mask_out", acq);
}

}  // namespace tcn::ops
