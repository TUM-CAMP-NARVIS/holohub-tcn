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

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace tcn::ops {

/**
 * @brief Routes a multi-camera entity (dict of camera_name -> tensor) to separate named outputs.
 *
 * Each output channel gets its own CUDA stream allocated from the stream pool.
 * The input entity is expected to contain named tensors matching the configured channel names.
 */
class TcnStreamSplitterOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnStreamSplitterOp)

  TcnStreamSplitterOp() = default;

  void set_channel_names_init(std::vector<std::string> names) {
    channel_names_init_ = std::move(names);
  }

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 protected:
  std::vector<std::string> channel_names_init_;

 private:
  holoscan::Parameter<std::vector<std::string>> channel_names_;
};

}  // namespace tcn::ops
