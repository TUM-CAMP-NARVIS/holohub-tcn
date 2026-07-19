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

#ifndef TCN_STREAM_SYNCHRONIZER_TCN_STREAM_SYNCHRONIZER_HPP
#define TCN_STREAM_SYNCHRONIZER_TCN_STREAM_SYNCHRONIZER_HPP

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

namespace tcn::ops {

/**
 * @brief Operator to synchronize multiple video streams.
 *
 */
class TcnStreamSynchronizerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnStreamSynchronizerOp)

  TcnStreamSynchronizerOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  std::vector<std::string> in_port_names;

  holoscan::Parameter<int> cuda_device_ordinal_;
  holoscan::Parameter<int> width_;
  holoscan::Parameter<int> height_;
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  holoscan::Parameter<int> num_streams_;
  holoscan::Parameter<bool> verbose_;

  holoscan::CudaStreamHandler cuda_stream_handler_;

  // CUDA
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};

  uint64_t last_emit_timestamp_ = 0;
};

}  // namespace tcn::ops

#endif /* TCN_STREAM_SYNCHRONIZER_TCN_STREAM_SYNCHRONIZER_HPP */
