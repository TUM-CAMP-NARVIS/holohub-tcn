#pragma once

#include <holoscan/holoscan.hpp>
#include <Eigen/Core>

#include "../common/datatypes.hpp"
#include "tcn_depthimage_backprojection_kernel.cuh"

namespace tcn::ops {

class TcnDepthImageBackprojectionOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnDepthImageBackprojectionOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  float depth_units_per_meter_ = 1000.0f;   // e.g., mm -> meters
  float near_limit_m_ = 0.1f;
  float far_limit_m_ = 10.0f;
  int color_image_width_ = 1920;
  int color_image_height_ = 1080;
  int sync_device_ = 1; // 0 = no sync, 1 = cudaStreamSynchronize
};

} // namespace tcn::ops