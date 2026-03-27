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

#include <regex>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace tcn::ops {

/**
 * @brief Merges multiple camera stream inputs back into a single output entity.
 *
 * Supports two modes:
 * - Fuse mode: concatenates all input tensors along axis 1 into a single tensor.
 * - Separate mode: outputs each tensor with a camera-prefixed name (e.g. "camera0_depth").
 *
 * Synchronizes CUDA streams from all input ports before producing output.
 */
class TcnStreamMergerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnStreamMergerOp)

  TcnStreamMergerOp() = default;

  void set_input_port_names_init(std::vector<std::string> names) {
    input_port_names_init_ = std::move(names);
  }

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 protected:
  std::vector<std::string> input_port_names_init_;

 private:
  // Extract camera name from port name (e.g. "camera0_depth" -> "camera0")
  std::string getCameraNameFromPortName(const std::string& port_name) const;

  holoscan::Parameter<std::vector<std::string>> input_port_names_;
  holoscan::Parameter<std::string> input_message_name_;
  holoscan::Parameter<std::string> output_message_name_;
  holoscan::Parameter<bool> fuse_buffers_;
  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

  std::regex portname_pattern_{R"(^(camera[0-9]+)_.*$)"};
};

}  // namespace tcn::ops
