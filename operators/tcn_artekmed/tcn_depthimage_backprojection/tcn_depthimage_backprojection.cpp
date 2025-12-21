#include "tcn_depthimage_backprojection.hpp"

#include <cuda_runtime.h>
#include <holoscan/core/resources/cuda_stream_pool.hpp>

namespace tcn::ops {

void TcnDepthImageBackprojectionOp::setup(holoscan::OperatorSpec& spec) {
  using holoscan::Arg;

  // Inputs
  spec.input<holoscan::Tensor>("depth_image");     // device, [H, W], uint16
  spec.input<holoscan::Tensor>("xy_table");        // device, [H, W, 2], float32
  spec.input<holoscan::Tensor>("color_params");    // host/device, raw bytes of CameraParameters
  spec.input<holoscan::Tensor>("color_to_depth");  // host/device, [4,4], float32
  spec.input<holoscan::Tensor>("depth_extrinsics");// host/device, [4,4], float32

  // Outputs (planar)
  spec.output<holoscan::Tensor>("positions");  // device, [H, W, 3], float32
  spec.output<holoscan::Tensor>("texcoords");  // device, [H, W, 2], float32

  // Configurable params (optional)
  spec.param(depth_units_per_meter_, "depth_units_per_meter", 1000.0f,
             "Depth units per meter", "Scaling from depth units to meters");
  spec.param(near_limit_m_, "near_limit_m", 0.1f,
             "Near depth limit (m)", "Discard depths below this");
  spec.param(far_limit_m_, "far_limit_m", 10.0f,
             "Far depth limit (m)", "Discard depths above this");
  spec.param(color_image_width_, "color_image_width", 1920, "Color image width", "");
  spec.param(color_image_height_, "color_image_height", 1080, "Color image height", "");
  spec.param(sync_device_, "sync_device", 1, "CUDA sync strategy", "0=NoSync, 1=DeviceSync");
}

void TcnDepthImageBackprojectionOp::compute(holoscan::InputContext& op_input,
                               holoscan::OutputContext& op_output,
                               holoscan::ExecutionContext& context) {
  // Get CUDA stream from Holoscan
  cudaStream_t stream = nullptr;
  if (auto gxf_stream = context.get<::holoscan::CudaStream>()) {
    stream = gxf_stream->stream();
  }
  if (!stream) stream = 0;

  // Receive tensors
  auto depth_t = op_input.receive<holoscan::Tensor>("depth_image").value();
  auto xy_t    = op_input.receive<holoscan::Tensor>("xy_table").value();
  auto cp_t    = op_input.receive<holoscan::Tensor>("color_params").value();
  auto c2d_t   = op_input.receive<holoscan::Tensor>("color_to_depth").value();
  auto de_t    = op_input.receive<holoscan::Tensor>("depth_extrinsics").value();

  const auto& depth_shape = depth_t->shape();  // [H,W]
  const int H = static_cast<int>(depth_shape[0]);
  const int W = static_cast<int>(depth_shape[1]);

  // Map tensors to raw device pointers
  auto* depth_ptr = static_cast<uint16_t*>(depth_t->data());
  auto* xy_ptr    = static_cast<float*>(xy_t->data());

  // Camera parameters
  CameraParameters color_params{};
  std::memcpy(&color_params, cp_t->data(), sizeof(color_params));

  // Matrices (float32[4,4])
  Eigen::Matrix4f color_to_depth = Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor>>(
      static_cast<float*>(c2d_t->data()));
  Eigen::Matrix4f depth_extrinsics = Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor}}(
      static_cast<float*>(de_t->data()));

  // Allocate Holoscan outputs (device)
  auto positions = holoscan::Tensor::create(
      holoscan::make_shape({H, W, 3}), holoscan::MemoryStorageType::kDevice, holoscan::Tensor::DataType::kFloat32);
  auto texcoords = holoscan::Tensor::create(
      holoscan::make_shape({H, W, 2}), holoscan::MemoryStorageType::kDevice, holoscan::Tensor::DataType::kFloat32);

  auto* pos_ptr = static_cast<float*>(positions->data());
  auto* tex_ptr = static_cast<float*>(texcoords->data());

  // Launch kernel
  BackProjectionParams params{};
  params.depth = depth_ptr;
  params.xy = reinterpret_cast<const float2*>(xy_ptr);
  params.positions = pos_ptr;
  params.texcoords = tex_ptr;
  params.width = W;
  params.height = H;
  params.depth_units_per_meter = depth_units_per_meter_;
  params.near_limit_m = near_limit_m_;
  params.far_limit_m = far_limit_m_;
  params.color_params = color_params;
  params.color_to_depth = float4x4Cast(color_to_depth);
  params.depth_extrinsics = float4x4Cast(depth_extrinsics);

  const dim3 block(16, 16);
  const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  backprojection_u16_kernel<<<grid, block, 0, stream>>>(params);

  if (sync_device_) {
    cudaStreamSynchronize(stream);
  }

  op_output.emit(positions, "positions");
  op_output.emit(texcoords, "texcoords");
}

} // namespace tcn::ops