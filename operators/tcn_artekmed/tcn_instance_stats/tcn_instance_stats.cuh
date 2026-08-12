/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <string>

#include <holoscan/holoscan.hpp>

#include "../cuda/tcn_instance_stats_kernel.cuh"

namespace tcn::ops {

/**
 * @brief Reduces a labeled point grid to one row per panoptic instance: count, centroid, box, spread.
 *
 * Consumes the same pair as `tcn_labeled_pointcloud` -- `positions` from
 * `tcn_depthimage_backprojection` (world space) and the aligned `labels_out` from
 * `tcn_label_sampler` -- so it hangs in parallel with the point-cloud path rather than downstream of
 * it, and adding it changes nothing that already exists.
 *
 * This is a reduction BY KEY. The masks have already segmented the points, so no clustering is
 * needed to find instances; every point states which instance it belongs to. What clustering would
 * buy is rejecting a mask that covers two physical surfaces, and that is approximated here by a
 * per-axis sigma trim -- see the README for where that is not enough.
 *
 * Output rows are emitted in HOST memory: the consumers are Python (cross-camera fusion and
 * tracking) and the data is a handful of rows, so a device tensor would only force them to copy it
 * back themselves.
 */
class TcnInstanceStatsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TcnInstanceStatsOp)

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  void allocate_scratch();
  void free_scratch();

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_{nullptr};
  holoscan::Parameter<int> cuda_device_ordinal_;
  holoscan::Parameter<std::string> in_positions_tensor_name_;
  holoscan::Parameter<std::string> in_labels_tensor_name_;
  holoscan::Parameter<std::string> out_rows_tensor_name_;
  holoscan::Parameter<std::string> out_labels_tensor_name_;
  holoscan::Parameter<int64_t> camera_index_;
  holoscan::Parameter<double> trim_percentile_;
  holoscan::Parameter<double> trim_margin_;
  holoscan::Parameter<double> min_range_m_;
  holoscan::Parameter<int64_t> min_points_;
  holoscan::Parameter<int64_t> max_instances_;
  holoscan::Parameter<bool> verbose_;
  holoscan::Parameter<std::shared_ptr<holoscan::CudaStreamPool>> cuda_stream_pool_;

  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_{};

  InstanceAccumulators acc_{};
  float* rows_d_ = nullptr;
  uint16_t* row_labels_d_ = nullptr;
  uint32_t* row_count_d_ = nullptr;
  uint32_t* row_count_h_ = nullptr;      ///< pinned, for the single count read-back
  std::size_t emitted_ = 0;
  std::size_t overflow_frames_ = 0;
};

}  // namespace tcn::ops
