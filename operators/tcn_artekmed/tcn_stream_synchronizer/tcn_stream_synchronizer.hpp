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

namespace holoscan::ops {

/**
 * @brief Operator to synchronize multiple video streams.
 *
 */
class TcnStreamSynchronizerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnStreamSynchronizerOp)

  TcnStreamSynchronizerOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  std::vector<std::string> in_port_names;

  Parameter<int> cuda_device_ordinal_;
  Parameter<int> width_;
  Parameter<int> height_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<int> num_streams_;
  Parameter<bool> verbose_;

  CudaStreamHandler cuda_stream_handler_;

  // CUDA
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};

  uint64_t last_emit_timestamp_ = 0;
};

}  // namespace holoscan::ops

#endif /* TCN_STREAM_SYNCHRONIZER_TCN_STREAM_SYNCHRONIZER_HPP */
