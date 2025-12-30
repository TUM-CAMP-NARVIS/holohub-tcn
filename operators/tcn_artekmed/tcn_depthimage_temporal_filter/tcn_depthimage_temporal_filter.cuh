#pragma once

#include <holoscan/holoscan.hpp>
// #include "holoscan/utils/cuda_stream_handler.hpp"

namespace tcn::ops {

class TcnDepthImageTemporalFilterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnDepthImageTemporalFilterOp)

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

  holoscan::Parameter<uint8_t> persistence_;
  holoscan::Parameter<uint16_t> delta_;
  holoscan::Parameter<float> alpha_;
  holoscan::Parameter<std::string> in_tensor_name_;
  holoscan::Parameter<std::string> out_tensor_name_;

  std::shared_ptr<nvidia::gxf::Tensor> buffer_last_frame_;
  std::shared_ptr<nvidia::gxf::Tensor> buffer_history_;
  std::shared_ptr<nvidia::gxf::Tensor> buffer_persistence_map_;

  // internal state
  int current_frame_index_{0};

  // holoscan::CudaStreamHandler cuda_stream_handler_;
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};


};

} // namespace tcn::ops