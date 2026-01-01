#include <cuda.h>
#include <cuda_runtime.h>
#include "../common/utils.h"

#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_depthimage_weights.cuh"
#include "tcn_depthimage_weights_kernel.cuh"

#include "../common/datatypes.hpp"

#include <gxf/std/tensor.hpp>

namespace tcn::ops {

void TcnDepthImageWeightsOp::setup(holoscan::OperatorSpec& spec) {
  using holoscan::Arg;
  HOLOSCAN_LOG_DEBUG("TcnDepthImageWeightsOp::setup");

  // Inputs
  spec.input<holoscan::gxf::Entity>("depth_image");  // device, [H, W], uint16
  spec.input<holoscan::gxf::Entity>("xy_table").condition(holoscan::ConditionType::kNone);     // device, [H, W, 2], float32

  // Outputs (planar)
  spec.output<holoscan::gxf::Entity>("output");

  // Configurable params (optional)
  spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");

  spec.param(depth_units_per_meter_, "depth_units_per_meter", "depth_units_per_meter", "", 1000.f);
  spec.param(angle_reject_limit_, "angle_reject_limit", "angle_reject_limit", "", 3.1415926535f / 9.f);
  spec.param(angle_reject_envelope_, "angle_reject_envelope", "angle_reject_envelope", "", 1.f);
  spec.param(offset_envelope_, "offset_envelope", "offset_envelope", "", 1.f);
  spec.param(depth_near_limit_, "depth_near_limit", "depth_near_limit", "", 0.1f);
  spec.param(depth_far_limit_, "depth_far_limit", "depth_far_limit", "", 8.f);

  spec.param(in_tensor_name_, "in_tensor_name", "Input Tensor Name", "", ""s);
  spec.param(out_tensor_name_, "out_tensor_name", "Output Tensor Name", "", ""s);

  spec.param(cuda_device_ordinal_,
             "cuda_device_ordinal",
             "CudaDeviceOrdinal",
             "Device to use for CUDA operations",
             holoscan::ParameterFlag::kOptional);
}


void TcnDepthImageWeightsOp::initialize() {
  HOLOSCAN_LOG_DEBUG("TcnDepthImageWeightsOp::initialize");

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void TcnDepthImageWeightsOp::start() {
  // Initialize CUDA
  CudaCheck(cuInit(0));

  // Get the CUDA device
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;

  // Retain the primary context for the device
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));
}


void TcnDepthImageWeightsOp::compute(holoscan::InputContext& op_input,
                                            holoscan::OutputContext& op_output,
                                            holoscan::ExecutionContext& context) {


  // Receive tensors
  auto maybe_depth_t_entity = op_input.receive<holoscan::gxf::Entity>("depth_image");
  if (!maybe_depth_t_entity) {
    throw std::runtime_error("Failed to read input entity");
  }

  auto depth_t = maybe_depth_t_entity.value().get<holoscan::Tensor>(in_tensor_name_.get().c_str());
  cudaStream_t cuda_stream = op_input.receive_cuda_stream("depth_image", true, false);

  auto maybe_xy_t_entity = op_input.receive<holoscan::gxf::Entity>("xy_table");
  if (maybe_xy_t_entity) {
    HOLOSCAN_LOG_INFO("received xy table input");
    xylookup_table_tensor_ = maybe_xy_t_entity.value().get<holoscan::Tensor>(in_tensor_name_.get().c_str());
  }
  if (!xylookup_table_tensor_) {
    HOLOSCAN_LOG_DEBUG("missing xy lookup table");
    return;
  }

  const auto& depth_shape = depth_t->shape();  // [H,W]
  const int H = static_cast<int>(depth_shape[0]);
  const int W = static_cast<int>(depth_shape[1]);

  assert(xylookup_table_tensor_->shape()[0] == depth_t->shape()[0]);
  assert(xylookup_table_tensor_->shape()[1] == depth_t->shape()[1]);
  assert(xylookup_table_tensor_->shape()[2] == 2);
  
  // Map tensors to raw device pointers
  auto* depth_ptr = static_cast<uint16_t*>(depth_t->data());
  auto* xy_ptr = static_cast<float*>(xylookup_table_tensor_->data());

  // Allocate Holoscan outputs (device)
  auto gxf_context = context.context();

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  float* out_ptr = nullptr;
  nvidia::gxf::Entity weights_buffer_entity;
  nvidia::gxf::Handle<nvidia::gxf::Tensor> weights_buffer = nullptr;

  auto maybe_weights_buffer_entity = nvidia::gxf::Entity::New(gxf_context);
  if (!maybe_weights_buffer_entity) {
    throw std::runtime_error("Failed to allocate message for output weights_buffer tensor.");
  }
  weights_buffer_entity = std::move(maybe_weights_buffer_entity.value());

  if (!tcn::allocate_named_tensor<float>(allocator.value(),
                                         cuda_stream,
                                         weights_buffer_entity,
                                         nvidia::gxf::Shape{{H, W, 1}},
                                         nvidia::gxf::MemoryStorageType::kDevice,
                                         out_tensor_name_.get(),
                                         weights_buffer
                                         )) {
    throw std::runtime_error("Failed to allocate message for weights_buffer.");
                                         }
  if (auto maybe_weights_buffer_data = weights_buffer->data<float>()) {
    out_ptr = maybe_weights_buffer_data.value();
  } else {
    HOLOSCAN_LOG_ERROR("error access weights_buffer tensor data");
  }

  // Launch kernel
  CameraQualityWeightParams cqwp{};
  cqwp.angleRejectEnvelope = angle_reject_envelope_.get();
  cqwp.angleRejectLimit = angle_reject_limit_.get();
  cqwp.offsetEnvelope = offset_envelope_.get();
  cqwp.depthNearLimit = depth_near_limit_.get();
  cqwp.depthFarLimit = depth_far_limit_.get();

  WeightsParams params{};
  params.depth = depth_ptr;
  params.xy = reinterpret_cast<const float2*>(xy_ptr);
  params.computeWeightsOutput = out_ptr;

  params.depthUnitsPerMeter = depth_units_per_meter_.get();
  params.quality_weight_params = cqwp;

  params.width = W;
  params.height = H;

  const dim3 block(16, 16);
  const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  compute_weights_u16_kernel<<<grid, block, 0, cuda_stream>>>(params);

  auto weights_buffer_message = holoscan::gxf::Entity(std::move(weights_buffer_entity));
  // op_output.set_cuda_stream(cuda_stream, "output");
  op_output.emit(weights_buffer_message, "output");

}
}  // namespace tcn::ops