#pragma once

#include <holoscan/holoscan.hpp>
#include "holoscan/utils/cuda_stream_handler.hpp"
#include "../common/datatypes.hpp"

namespace tcn::ops {

class TcnDepthImageBackprojectionOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnDepthImageBackprojectionOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:

  bool enable_conditional_port(const std::string& name,
                                 bool set_none_condition_on_disabled = false);


  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_{nullptr};
  holoscan::Parameter<int> cuda_device_ordinal_;

  holoscan::Parameter<float> depth_units_per_meter_;
  holoscan::Parameter<float> near_limit_m_;
  holoscan::Parameter<float> far_limit_m_;
  holoscan::Parameter<int> color_image_width_;
  holoscan::Parameter<int> color_image_height_;
  holoscan::Parameter<nvidia::gxf::CameraModel> color_params_;
  holoscan::Parameter<RigidTransform> depth_extrinsics_;
  holoscan::Parameter<RigidTransform> color_to_depth_;
  holoscan::Parameter<std::string> in_tensor_name_;
  holoscan::Parameter<std::string> out_tensor_name_;
  holoscan::Parameter<bool> enable_positions_;
  holoscan::Parameter<bool> enable_texcoords_;
  holoscan::Parameter<bool> enable_depth_float_;

  std::shared_ptr<holoscan::Tensor> xylookup_table_tensor_;

  bool positions_output_enabled_ = false;
  bool texcoords_output_enabled_ = false;
  bool depth_float_output_enabled_ = false;

  holoscan::CudaStreamHandler cuda_stream_handler_;
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};

};

} // namespace tcn::ops