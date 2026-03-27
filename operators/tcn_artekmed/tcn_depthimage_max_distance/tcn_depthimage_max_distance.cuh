/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <holoscan/holoscan.hpp>

namespace tcn::ops {

class TcnDepthImageMaxDistanceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnDepthImageMaxDistanceOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_{nullptr};
  holoscan::Parameter<std::string> in_tensor_name_;
  holoscan::Parameter<std::string> out_tensor_name_;

  std::shared_ptr<nvidia::gxf::Tensor> max_buffer_;
};

}  // namespace tcn::ops
