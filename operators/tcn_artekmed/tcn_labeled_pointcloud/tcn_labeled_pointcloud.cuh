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
 * @brief Turns a labeled depth grid into one compacted point cloud per class.
 *
 * Consumes `positions` from `tcn_depthimage_backprojection` (world space, one point per depth
 * pixel) and the aligned `labels_out` from `tcn_label_sampler`, and emits, for each configured
 * class, only the points carrying that class -- each entity holding both the positions and the
 * packed labels, so the instance id survives into the data product.
 *
 * One output port per class rather than one cloud with a colour buffer, because HolovizOp colours
 * `POINTS_3D` per InputSpec and not per vertex: a port per class is what lets the fused view show
 * classes in different colours using the existing viewer. Each cloud is shaped `[1, N, 3]`, which is
 * both what `tcn_stream_merger` concatenates along (dimension 1) and what HolovizOp expects, so a
 * fused cloud needs no flatten step. See
 * `applications/tcn_artekmed/tcn_shm_vlm_inference/docs/specs/2026-08-12-mask-depth-join-design.md`.
 *
 * Compaction is order-preserving (a stable scan over the source order, not atomics), so the same
 * input yields the same point order every run and a byte-comparison gate is meaningful.
 */
class TcnLabeledPointcloudOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnLabeledPointcloudOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

  /// Port carrying the points of class `cls`. Also used by the app to wire the fused view, so the
  /// naming convention lives in one place.
  static std::string port_name_for_class(int64_t cls);

 private:
  /// Grow the device scratch buffers to hold `count` source points. Sized once, then reused.
  void ensure_scratch(int64_t count);

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_{nullptr};
  holoscan::Parameter<int> cuda_device_ordinal_;
  holoscan::Parameter<std::string> in_positions_tensor_name_;
  holoscan::Parameter<std::string> in_labels_tensor_name_;
  holoscan::Parameter<std::string> out_positions_tensor_name_;
  holoscan::Parameter<std::string> out_labels_tensor_name_;
  holoscan::Parameter<std::vector<int64_t>> classes_;
  holoscan::Parameter<bool> verbose_;
  holoscan::Parameter<std::shared_ptr<holoscan::CudaStreamPool>> cuda_stream_pool_;

  /// Classes captured in setup() from args(), because parameters are not applied yet at that point.
  std::vector<int64_t> configured_classes_;

  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};

  int64_t scratch_count_ = 0;
  uint8_t* selected_d_ = nullptr;     ///< [count] 0/1 per source point
  int32_t* indices_d_ = nullptr;      ///< [count] compacted source indices
  std::size_t emitted_ = 0;
};

}  // namespace tcn::ops
