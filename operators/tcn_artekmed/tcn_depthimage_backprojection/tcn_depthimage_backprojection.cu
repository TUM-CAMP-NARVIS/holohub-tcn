#include "gxf/multimedia/camera.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_depthimage_backprojection.cuh"
#include "tcn_depthimage_backprojection_kernel.cuh"

#include <cuda_runtime.h>
#include "../common/datatypes.hpp"

#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/Core>
#include <gxf/std/tensor.hpp>
#include "holoscan/pose_tree/math/pose3.hpp"
#include "holoscan/pose_tree/math/so3.hpp"

namespace tcn::ops {

void TcnDepthImageBackprojectionOp::setup(holoscan::OperatorSpec& spec) {
  using holoscan::Arg;

  register_converter<Eigen::Vector3f>();
  register_converter<Eigen::Quaternionf>();
  register_converter<RigidTransform>();
  register_converter<CameraParameters>();

  register_converter<nvidia::gxf::Vector2f>();
  register_converter<nvidia::gxf::Vector2u>();
  register_converter<nvidia::gxf::DistortionType>();
  register_converter<nvidia::gxf::CameraModel>();

  // Inputs
  spec.input<holoscan::gxf::Entity>("depth_image");  // device, [H, W], uint16
  spec.input<holoscan::gxf::Entity>("xy_table");     // device, [H, W, 2], float32

  // Outputs (planar)
  spec.output<holoscan::Tensor>("positions");  // device, [H, W, 3], float32
  spec.output<holoscan::Tensor>("texcoords");  // device, [H, W, 2], float32

  // Configurable params (optional)
  spec.param(allocator_, "allocator", "Allocator", "Allocator used to allocate tensor output.");
  spec.param(depth_units_per_meter_,
             "depth_units_per_meter",
             "Depth units per meter",
             "Scaling from depth units to meters",
             1000.0f);
  spec.param(
      near_limit_m_, "near_limit_m", "Near depth limit (m)", "Discard depths below this", 0.1f);
  spec.param(
      far_limit_m_, "far_limit_m", "Far depth limit (m)", "Discard depths above this", 10.0f);
  spec.param(color_image_width_, "color_image_width", "Color image width", "", 1920);
  spec.param(color_image_height_, "color_image_height", "Color image height", "", 1080);
  spec.param(cuda_stream_pool_,
             "cuda_stream_pool",
             "CUDA Stream Pool",
             "Instance of gxf::CudaStreamPool.",
             holoscan::ParameterFlag::kOptional);
  spec.param(depth_extrinsics_,
             "depth_extrinsics",
             "Depth Camera Extrinsics",
             "Camera Pose of the Depth Sensor.");
  spec.param(color_to_depth_,
              "color_to_depth",
              "Color to Depth Sensor Transform",
              "Tranform from Camera to Depth Sensor");
  spec.param(color_params_,
             "color_params",
             "Color Camera Parameters",
             "Intrinsic and Distortion Parameters");
}

void TcnDepthImageBackprojectionOp::compute(holoscan::InputContext& op_input,
                                            holoscan::OutputContext& op_output,
                                            holoscan::ExecutionContext& context) {
  // Get CUDA stream from Holoscan
  cudaStream_t stream = nullptr;
  if (cuda_stream_pool_.try_get()) {
    auto maybe_stream = cuda_stream_pool_->get()->allocateStream();
    if (maybe_stream) {
      stream = maybe_stream.value()->stream().value();
    }
  }
  if (!stream)
    stream = nullptr;

  // Receive tensors
  auto maybe_depth_t_entity = op_input.receive<holoscan::gxf::Entity>("depth_image");
  if (!maybe_depth_t_entity) {
    throw std::runtime_error("Failed to read depth_image entity");
  }
  auto depth_t = maybe_depth_t_entity.value().get<holoscan::Tensor>("");

  auto maybe_xy_t_entity = op_input.receive<holoscan::gxf::Entity>("xy_table");
  if (!maybe_xy_t_entity) {
    throw std::runtime_error("Failed to read xy_table entity");
  }
  auto xy_t = maybe_xy_t_entity.value().get<holoscan::Tensor>("");

  auto& cp_t = color_params_.get();
  auto& c2d_t = color_to_depth_.get();
  auto& de_t = depth_extrinsics_.get();

  const auto& depth_shape = depth_t->shape();  // [H,W]
  const int H = static_cast<int>(depth_shape[0]);
  const int W = static_cast<int>(depth_shape[1]);

  // Map tensors to raw device pointers
  auto* depth_ptr = static_cast<uint16_t*>(depth_t->data());
  auto* xy_ptr = static_cast<float*>(xy_t->data());

  Eigen::Matrix4f color_to_depth;
  c2d_t.toMatrix4f(color_to_depth);

  Eigen::Matrix4f depth_extrinsics;
  de_t.toMatrix4f(depth_extrinsics);

  // Camera parameters
  CameraParameters color_params;
  color_params.cx = cp_t.principal_point.x;
  color_params.cy = cp_t.principal_point.y;
  color_params.fx = cp_t.focal_length.x;
  color_params.fy = cp_t.focal_length.y;
  assert(cp_t.distortion_coefficients.size() == 8);
  color_params.k1 = cp_t.distortion_coefficients[0];
  color_params.k2 = cp_t.distortion_coefficients[1];
  color_params.p1 = cp_t.distortion_coefficients[2];
  color_params.p2 = cp_t.distortion_coefficients[3];
  color_params.k3 = cp_t.distortion_coefficients[4];
  color_params.k4 = cp_t.distortion_coefficients[5];
  color_params.k5 = cp_t.distortion_coefficients[6];
  color_params.k6 = cp_t.distortion_coefficients[7];
  color_params.codx = 0;
  color_params.cody = 0;
  color_params.is_distorted = true;

  // Matrices (float32[4,4])

  // Allocate Holoscan outputs (device)
  auto gxf_context = context.context();

  auto positions_entity = nvidia::gxf::Entity::New(gxf_context);
  if (!positions_entity) {
    throw std::runtime_error("Failed to allocate message for output positions tensor.");
  }

  auto texcoords_entity = nvidia::gxf::Entity::New(gxf_context);
  if (!texcoords_entity) {
    throw std::runtime_error("Failed to allocate message for output texcoords tensor.");
  }

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_context, allocator_->gxf_cid());

  auto maybe_positions = positions_entity.value().add<nvidia::gxf::Tensor>();
  if (!maybe_positions) {
    throw std::runtime_error("Failed to allocate message for positions.");
  }
  auto& positions = maybe_positions.value();
  auto positions_storage_type = nvidia::gxf::MemoryStorageType::kDevice;
  auto positions_shape = nvidia::gxf::Shape{{H, W, 3}};
  const auto positions_dtype = nvidia::gxf::PrimitiveType::kFloat32;
  const uint64_t positions_bytes_per_element = nvidia::gxf::PrimitiveTypeSize(positions_dtype);
  auto positions_strides =
      nvidia::gxf::ComputeTrivialStrides(positions_shape, positions_bytes_per_element);

  auto positions_result = positions->reshapeCustom(positions_shape,
                                                   positions_dtype,
                                                   positions_bytes_per_element,
                                                   positions_strides,
                                                   positions_storage_type,
                                                   allocator.value());
  if (!positions_result) {
    HOLOSCAN_LOG_ERROR("failed to generate positions tensor");
  }

  auto maybe_texcoords = texcoords_entity.value().add<nvidia::gxf::Tensor>();
  if (!maybe_texcoords) {
    throw std::runtime_error("Failed to allocate message for texcoords.");
  }
  auto& texcoords = maybe_texcoords.value();
  auto texcoords_storage_type = nvidia::gxf::MemoryStorageType::kDevice;
  auto texcoords_shape = nvidia::gxf::Shape{{H, W, 2}};
  const auto texcoords_dtype = nvidia::gxf::PrimitiveType::kFloat32;
  const uint64_t texcoords_bytes_per_element = nvidia::gxf::PrimitiveTypeSize(texcoords_dtype);
  auto texcoords_strides =
      nvidia::gxf::ComputeTrivialStrides(texcoords_shape, texcoords_bytes_per_element);

  auto texcoords_result = texcoords->reshapeCustom(texcoords_shape,
                                                   texcoords_dtype,
                                                   texcoords_bytes_per_element,
                                                   texcoords_strides,
                                                   texcoords_storage_type,
                                                   allocator.value());
  if (!texcoords_result) {
    HOLOSCAN_LOG_ERROR("failed to generate positions tensor");
  }

  float* pos_ptr = nullptr;
  float* tex_ptr = nullptr;

  if (auto maybe_positions_data = positions->data<float>()) {
    pos_ptr = maybe_positions_data.value();
  } else {
    HOLOSCAN_LOG_ERROR("error access positions tensor data");
  }
  if (auto maybe_texcoords_data = texcoords->data<float>()) {
    tex_ptr = maybe_texcoords_data.value();
  } else {
    HOLOSCAN_LOG_ERROR("error access texcoord tensor data");
  }

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

  auto positions_message = holoscan::gxf::Entity(std::move(positions_entity.value()));
  op_output.emit(positions_message, "positions");

  auto texcoords_message = holoscan::gxf::Entity(std::move(texcoords_entity.value()));
  op_output.emit(texcoords_message, "texcoords");
}

}  // namespace tcn::ops