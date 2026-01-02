#include "../cuda/tcn_depthimage_backprojection_kernel.cuh"
#include "gxf/multimedia/camera.hpp"
#include "holoscan/utils/cuda_macros.hpp"
#include "tcn_depthimage_backprojection.cuh"

#include <cuda_runtime.h>
#include "../common/datatypes.hpp"
#include "../common/utils.h"

#include <Eigen/src/Geometry/Quaternion.h>
#include <Eigen/Core>
#include <gxf/std/tensor.hpp>

namespace tcn::ops {

void TcnDepthImageBackprojectionOp::setup(holoscan::OperatorSpec& spec) {
  using holoscan::Arg;

  HOLOSCAN_LOG_INFO("TcnDepthimageBackprojectionOp::setup");

  // Inputs
  spec.input<holoscan::gxf::Entity>("depth_image");  // device, [H, W], uint16
  spec.input<holoscan::gxf::Entity>("xy_table").condition(holoscan::ConditionType::kNone);     // device, [H, W, 2], float32

  // Outputs (planar)
  spec.output<holoscan::gxf::Entity>("positions");
  spec.output<holoscan::gxf::Entity>("texcoords");
  spec.output<holoscan::gxf::Entity>("depth_float");

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
  spec.param(in_tensor_name_, "in_tensor_name", "Input Tensor Name", "", ""s);
  spec.param(out_tensor_name_, "out_tensor_name", "Output Tensor Name", "", ""s);

  spec.param(enable_positions_, "enable_positions", "Enable Positions Output", "", true);
  spec.param(enable_texcoords_, "enable_texcoords", "Enable Texcoords Output", "", true);
  spec.param(enable_depth_float_, "enable_depth_float", "Enable DepthFloat Output", "", false);

  spec.param(cuda_device_ordinal_,
             "cuda_device_ordinal",
             "CudaDeviceOrdinal",
             "Device to use for CUDA operations",
             holoscan::ParameterFlag::kOptional);
  // cuda_stream_handler_.define_params(spec);
}


void TcnDepthImageBackprojectionOp::initialize() {
  HOLOSCAN_LOG_DEBUG("TcnDepthimageBackprojectionOp::initialize");

  // register type converters (args)
  register_converter<Eigen::Vector3f>();
  register_converter<Eigen::Quaternionf>();
  register_converter<RigidTransform>();
  register_converter<CameraParameters>();

  register_converter<nvidia::gxf::Vector2f>();
  register_converter<nvidia::gxf::Vector2u>();
  register_converter<nvidia::gxf::DistortionType>();
  register_converter<nvidia::gxf::CameraModel>();


  positions_output_enabled_ = enable_conditional_port("positions", true);
  texcoords_output_enabled_ = enable_conditional_port("texcoords", true);
  depth_float_output_enabled_ = enable_conditional_port("depth_float", true);

  if (texcoords_output_enabled_ && !positions_output_enabled_) {
    throw std::runtime_error("positions output must be enabled for texture-coordinates output");
  }

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void TcnDepthImageBackprojectionOp::start() {
  // Initialize CUDA
  CudaCheck(cuInit(0));

  // Get the CUDA device
  CUdevice cu_device;
  CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal_.get()));
  cu_device_ = cu_device;

  // Retain the primary context for the device
  CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));
}

void TcnDepthImageBackprojectionOp::compute(holoscan::InputContext& op_input,
                                            holoscan::OutputContext& op_output,
                                            holoscan::ExecutionContext& context) {
  // Receive tensors
  auto maybe_depth_t_entity = op_input.receive<holoscan::gxf::Entity>("depth_image");
  if (!maybe_depth_t_entity) {
    throw std::runtime_error("Failed to read depth_image entity");
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

  auto& cp_t = color_params_.get();
  auto& c2d_t = color_to_depth_.get();
  auto& de_t = depth_extrinsics_.get();

  const auto& depth_shape = depth_t->shape();  // [H,W]
  const int H = static_cast<int>(depth_shape[0]);
  const int W = static_cast<int>(depth_shape[1]);

  assert(xylookup_table_tensor_->shape()[0] == depth_t->shape()[0]);
  assert(xylookup_table_tensor_->shape()[1] == depth_t->shape()[1]);
  assert(xylookup_table_tensor_->shape()[2] == 2);

  // Map tensors to raw device pointers
  auto* depth_ptr = static_cast<uint16_t*>(depth_t->data());
  auto* xy_ptr = static_cast<float*>(xylookup_table_tensor_->data());

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

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_context, allocator_->gxf_cid());

  float* pos_ptr = nullptr;
  float* tex_ptr = nullptr;
  float* dmf_ptr = nullptr;

  nvidia::gxf::Entity positions_entity;
  nvidia::gxf::Entity texcoords_entity;
  nvidia::gxf::Entity depth_float_entity;

  nvidia::gxf::Handle<nvidia::gxf::Tensor> positions = nullptr;
  nvidia::gxf::Handle<nvidia::gxf::Tensor> texcoords = nullptr;
  nvidia::gxf::Handle<nvidia::gxf::Tensor> depth_float = nullptr;

  if (positions_output_enabled_) {
    auto maybe_positions_entity = nvidia::gxf::Entity::New(gxf_context);
    if (!maybe_positions_entity) {
      throw std::runtime_error("Failed to allocate message for output positions tensor.");
    }
    positions_entity = std::move(maybe_positions_entity.value());

    if (!tcn::allocate_named_tensor<float>(allocator.value(),
                                           cuda_stream,
                                           positions_entity,
                                           nvidia::gxf::Shape{{1, H*W, 3}},
                                           nvidia::gxf::MemoryStorageType::kDevice,
                                           out_tensor_name_.get(),
                                           positions
                                           )) {
      throw std::runtime_error("Failed to allocate message for positions.");
    }
    if (auto maybe_positions_data = positions->data<float>()) {
      pos_ptr = maybe_positions_data.value();
    } else {
      HOLOSCAN_LOG_ERROR("error access positions tensor data");
    }
  }

  if (texcoords_output_enabled_) {
    auto maybe_texcoords_entity = nvidia::gxf::Entity::New(gxf_context);
    if (!maybe_texcoords_entity) {
      throw std::runtime_error("Failed to allocate message for output texcoords tensor.");
    }
    texcoords_entity = std::move(maybe_texcoords_entity.value());
    if (!tcn::allocate_named_tensor<float>(allocator.value(),
                                               cuda_stream,
                                               texcoords_entity,
                                               nvidia::gxf::Shape{{1, H*W, 2}},
                                               nvidia::gxf::MemoryStorageType::kDevice,
                                               out_tensor_name_.get(),
                                               texcoords
                                               )) {
      throw std::runtime_error("Failed to allocate message for texcoords.");
    }
    if (auto maybe_texcoords_data = texcoords->data<float>()) {
      tex_ptr = maybe_texcoords_data.value();
    } else {
      HOLOSCAN_LOG_ERROR("error access texcoord tensor data");
    }
  }

  if (depth_float_output_enabled_) {
    auto maybe_depth_float_entity = nvidia::gxf::Entity::New(gxf_context);
    if (!maybe_depth_float_entity) {
      throw std::runtime_error("Failed to allocate message for output depth_float tensor.");
    }
    depth_float_entity = std::move(maybe_depth_float_entity.value());
    if (!tcn::allocate_named_tensor<float>(allocator.value(),
                                               cuda_stream,
                                               depth_float_entity,
                                               nvidia::gxf::Shape{{H, W, 1}},
                                               nvidia::gxf::MemoryStorageType::kDevice,
                                               out_tensor_name_.get(),
                                               depth_float
                                               )) {
      throw std::runtime_error("Failed to allocate message for texcoords.");
    }
    if (auto maybe_depth_float_data = depth_float->data<float>()) {
      dmf_ptr = maybe_depth_float_data.value();
    } else {
      HOLOSCAN_LOG_ERROR("error access depth_float tensor data");
    }
  }

  // Launch kernel
  BackProjectionParams params{};
  params.depth = depth_ptr;
  params.xy = reinterpret_cast<const float2*>(xy_ptr);
  params.positions = pos_ptr;
  params.texcoords = tex_ptr;
  params.depth_float = dmf_ptr;
  params.width = W;
  params.height = H;
  params.depth_units_per_meter = depth_units_per_meter_;
  params.near_limit_m = near_limit_m_;
  params.far_limit_m = far_limit_m_;
  params.color_params = color_params;
  params.color_to_depth = float4x4Cast(color_to_depth);
  params.depth_extrinsics = float4x4Cast(depth_extrinsics);
  params.positions_enabled = positions_output_enabled_;
  params.texcoords_enabled = texcoords_output_enabled_;
  params.depth_float_enabled = depth_float_output_enabled_;

  const dim3 block(16, 16);
  const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  backprojection_u16_kernel<<<grid, block, 0, cuda_stream>>>(params);

  if (positions_output_enabled_) {
    auto positions_message = holoscan::gxf::Entity(std::move(positions_entity));
    // op_output.set_cuda_stream(cuda_stream, "positions");
    op_output.emit(positions_message, "positions");
  }

  if (texcoords_output_enabled_) {
    auto texcoords_message = holoscan::gxf::Entity(std::move(texcoords_entity));
    // op_output.set_cuda_stream(cuda_stream, "texcoords");
    op_output.emit(texcoords_message, "texcoords");
  }

  if (depth_float_output_enabled_) {
    auto depth_float_message = holoscan::gxf::Entity(std::move(depth_float_entity));
    // op_output.set_cuda_stream(cuda_stream, "depth_float");
    op_output.emit(depth_float_message, "depth_float");
  }
}


bool TcnDepthImageBackprojectionOp::enable_conditional_port(const std::string& port_name,
                                        bool set_none_condition_on_disabled) {
  bool enable_port = false;

  // Check if the boolean argument with the name "enable_(port_name)" is present.
  const std::string enable_port_name = std::string("enable_") + port_name;
  auto enable_port_arg =
      std::find_if(args().begin(), args().end(), [&enable_port_name](const auto& arg) {
        return (arg.name() == enable_port_name);
      });

  // If present ...
  if (enable_port_arg != args().end()) {
    // ... and with a value ...
    if (enable_port_arg->has_value()) {
      // ...try extracting a boolean value through YAML::Node or generic bool cast
      std::any& any_arg = enable_port_arg->value();
      if (enable_port_arg->arg_type().element_type() == holoscan::ArgElementType::kYAMLNode) {
        auto& arg_value = std::any_cast<YAML::Node&>(any_arg);
        bool parse_ok = YAML::convert<bool>::decode(arg_value, enable_port);
        if (!parse_ok) {
          HOLOSCAN_LOG_ERROR("Could not parse YAML parameter '{}' as a 'bool' type",
                             enable_port_name);
          enable_port = false;
        }
      } else {
        try {
          enable_port = std::any_cast<bool>(any_arg);
        } catch (const std::bad_any_cast& e) {
          HOLOSCAN_LOG_ERROR(
              "Could not cast parameter '{}' as 'bool': {}", enable_port_name, e.what());
        }
      }
    }
    // If the "enable_(port_name)" argument is present, we remove it so that it won't
    // be passed on further, since we only care about "(port_name)" afterwards.
    args().erase(enable_port_arg);
  }

  // Disable the '(port_name)' argument based on the value of "enable_(port_name)"
  // if (!enable_port) {
  //   add_arg(holoscan::Arg(port_name) = static_cast<holoscan::IOSpec*>(nullptr));
  // }

  // If 'set_none_condition_on_disabled' is true and the port (named by 'port_name') is disabled,
  // insert ConditionType::kNone condition so that its default condition
  // (DownstreamMessageAffordableCondition) is not added during Operator::initialize().
  if (!enable_port && set_none_condition_on_disabled) {
    spec()->outputs()[port_name]->condition(holoscan::ConditionType::kNone);
  }

  return enable_port;
}
}  // namespace tcn::ops