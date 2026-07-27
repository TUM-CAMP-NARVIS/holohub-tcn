#include <cuda.h>
#include <cuda_runtime.h>
#include "../common/utils.h"

#include "../cuda/tcn_texture_sampler_kernel.cuh"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_texture_sampler.cuh"

#include "../common/datatypes.hpp"

#include <gxf/std/tensor.hpp>

namespace tcn::ops {

void TcnTextureSamplerOp::setup(holoscan::OperatorSpec& spec) {
  using holoscan::Arg;
  HOLOSCAN_LOG_DEBUG("TcnTextureSamplerOp::setup");

  // Inputs
  spec.input<holoscan::gxf::Entity>("color_image");  // device, [H, W, 3], uint8
  spec.input<holoscan::gxf::Entity>("texcoords");     // device, [H, W, 2], float32

  // Outputs (planar)
  spec.output<holoscan::gxf::Entity>("output");

  // Configurable params (optional)
  spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");

  spec.param(in_color_tensor_name_, "in_color_tensor_name", "Color Input Tensor Name", "", ""s);
  spec.param(in_texcoord_tensor_name_, "in_texcoord_tensor_name", "Texcoord Input Tensor Name", "", ""s);
  spec.param(out_tensor_name_, "out_tensor_name", "Output Tensor Name", "", ""s);

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


void TcnTextureSamplerOp::initialize() {
  HOLOSCAN_LOG_DEBUG("TcnTextureSamplerOp::initialize");

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void TcnTextureSamplerOp::start() {
  // Initialize CUDA
  CudaCheck(cuInit(0));

  // Get the CUDA device
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;

  // Retain the primary context for the device
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));
}


void TcnTextureSamplerOp::compute(holoscan::InputContext& op_input,
                                            holoscan::OutputContext& op_output,
                                            holoscan::ExecutionContext& context) {


  // Receive tensors
  auto maybe_color_t_entity = op_input.receive<holoscan::gxf::Entity>("color_image");
  if (!maybe_color_t_entity) {
    throw std::runtime_error("Failed to read input entity");
  }

  auto color_t = maybe_color_t_entity.value().get<holoscan::Tensor>(in_color_tensor_name_.get().c_str());
  cudaStream_t cuda_stream = op_input.receive_cuda_stream("color_image", true, false);

  auto maybe_texcoords_entity = op_input.receive<holoscan::gxf::Entity>("texcoords");
  if (!maybe_texcoords_entity) {
    throw std::runtime_error("Failed to read input entity");
  }
  auto texcoord_t = maybe_texcoords_entity.value().get<holoscan::Tensor>(in_texcoord_tensor_name_.get().c_str());
  op_input.receive_cuda_stream("texcoords", true, false);

  const auto& color_shape = color_t->shape();
  assert(color_shape[2] == 4); // for now we require a 4 component input.

  const auto& uv_shape = texcoord_t->shape();  // [H,W]
  const int H = static_cast<int>(uv_shape[0]);
  const int W = static_cast<int>(uv_shape[1]);

  // Map tensors to raw device pointers
  auto* color_ptr = static_cast<uint8_t*>(color_t->data());
  auto* uv_ptr = static_cast<float*>(texcoord_t->data());

  // Allocate Holoscan outputs (device)
  auto gxf_context = context.context();

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  uint8_t* out_ptr = nullptr;
  nvidia::gxf::Entity out_buffer_entity;
  nvidia::gxf::Handle<nvidia::gxf::Tensor> out_buffer = nullptr;

  auto maybe_out_buffer_entity = nvidia::gxf::Entity::New(gxf_context);
  if (!maybe_out_buffer_entity) {
    throw std::runtime_error("Failed to allocate message for output color tensor.");
  }
  out_buffer_entity = std::move(maybe_out_buffer_entity.value());

  if (!tcn::allocate_named_tensor<uint8_t>(allocator.value(),
                                         cuda_stream,
                                         out_buffer_entity,
                                         nvidia::gxf::Shape{{H, W, 4}},
                                         nvidia::gxf::MemoryStorageType::kDevice,
                                         out_tensor_name_.get(),
                                         out_buffer
                                         )) {
    throw std::runtime_error("Failed to allocate message for out_buffer.");
                                         }
  if (auto maybe_out_buffer_data = out_buffer->data<uint8_t>()) {
    out_ptr = maybe_out_buffer_data.value();
  } else {
    HOLOSCAN_LOG_ERROR("error access out_buffer tensor data");
  }

  // Launch kernel
  TextureSamplerParams params{};
  params.color = color_ptr;
  params.uv = reinterpret_cast<const float2*>(uv_ptr);
  params.colorOutput = reinterpret_cast<uchar4*>(out_ptr);

  params.width = W;
  params.height = H;
  params.colorWidth = color_shape[1];
  params.colorHeight = color_shape[0];

  // int minGridSize;
  // int blockSize;
  //
  // // Let CUDA decide the best block size for this specific kernel
  // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
  //                                    texture_sampler_rgba_kernel, 0, 0);
  //
  // int numPixels = H * W;
  // int gridSize = (numPixels + blockSize - 1) / blockSize;

  const dim3 block(16, 16);
  const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  texture_sampler_rgba_kernel<<<grid, block, 0, cuda_stream>>>(params);

  auto out_buffer_message = holoscan::gxf::Entity(std::move(out_buffer_entity));
  op_output.emit(out_buffer_message, "output");

}
}  // namespace tcn::ops