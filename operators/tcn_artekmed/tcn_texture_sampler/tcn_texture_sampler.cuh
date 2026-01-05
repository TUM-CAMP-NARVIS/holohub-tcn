#pragma once

#include <holoscan/holoscan.hpp>
// #include "holoscan/utils/cuda_stream_handler.hpp"

namespace tcn::ops {

class TcnTextureSamplerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnTextureSamplerOp)

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

  holoscan::Parameter<std::string> in_color_tensor_name_;
  holoscan::Parameter<std::string> in_texcoord_tensor_name_;
  holoscan::Parameter<std::string> out_tensor_name_;

  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};


};

} // namespace tcn::ops