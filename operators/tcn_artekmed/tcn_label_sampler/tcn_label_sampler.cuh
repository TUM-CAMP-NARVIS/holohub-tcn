/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace tcn::ops {

/**
 * @brief Gives every depth pixel the panoptic label of the scene point it observes.
 *
 * Consumes the `texcoords` output of `tcn_depthimage_backprojection` -- normalised colour-image
 * coordinates per depth pixel, computed through the full geometry (unproject with the depth
 * intrinsics, transform into colour-camera space, project with the colour intrinsics and
 * distortion). Sampling the panoptic map through them is the only correct mask/depth
 * correspondence: the two sensors differ in resolution, intrinsics, distortion and optical centre,
 * so rescaling a mask onto the depth grid is spatially wrong in a depth-dependent way. See
 * `applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-12-mask-depth-join-design.md`.
 *
 * Emits both the sampled labels and a binary mask on the depth grid. The mask is what makes
 * `tcn_depthimage_apply_mask` usable -- that operator needs a single unnamed uint8 mask with the
 * same element count as the depth image, which a colour-resolution panoptic map can never be.
 *
 * Sampling is nearest-neighbour and out-of-frustum texcoords are rejected rather than clamped; see
 * the kernel for why neither is negotiable for label data.
 */
class TcnLabelSamplerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnLabelSamplerOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  /// Upload the class-selection lookup table once. Returns nullptr when every non-background class
  /// is selected, which the kernel reads as "no filtering" instead of a table of all-ones.
  const uint8_t* class_select_device();

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_{nullptr};
  holoscan::Parameter<int> cuda_device_ordinal_;

  holoscan::Parameter<std::string> in_labels_tensor_name_;
  holoscan::Parameter<std::string> in_texcoord_tensor_name_;
  holoscan::Parameter<std::string> out_labels_tensor_name_;
  holoscan::Parameter<std::string> out_mask_tensor_name_;
  holoscan::Parameter<std::vector<int64_t>> select_classes_;
  holoscan::Parameter<int64_t> unlabeled_value_;
  holoscan::Parameter<std::shared_ptr<holoscan::CudaStreamPool>> cuda_stream_pool_;

  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};

  uint8_t* class_select_d_ = nullptr;     ///< owned; freed in stop()
};

}  // namespace tcn::ops
