#pragma once

#include <holoscan/holoscan.hpp>
// #include "holoscan/utils/cuda_stream_handler.hpp"

namespace tcn::ops {

class TcnDepthImageWeightsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnDepthImageWeightsOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:

  void buildPersistenceMap(nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator, cudaStream_t stream);

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_{nullptr};
  holoscan::Parameter<int> cuda_device_ordinal_;

  holoscan::Parameter<float> depth_units_per_meter_;

  holoscan::Parameter<float> angle_reject_limit_;
  holoscan::Parameter<float> angle_reject_envelope_;
  holoscan::Parameter<float> offset_envelope_;
  holoscan::Parameter<float> depth_near_limit_;
  holoscan::Parameter<float> depth_far_limit_;

  holoscan::Parameter<std::string> in_tensor_name_;
  holoscan::Parameter<std::string> out_tensor_name_;
  holoscan::Parameter<std::shared_ptr<holoscan::CudaStreamPool>> cuda_stream_pool_;

  std::shared_ptr<holoscan::Tensor> xylookup_table_tensor_;

  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};


};

} // namespace tcn::ops