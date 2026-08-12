/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

#include "macros.hpp"

namespace tcn::doc::TcnInstanceStatsOp {

PYDOC(TcnInstanceStatsOp, R"doc(
Reduces a labeled point grid to one row per panoptic instance: point count, centroid, axis-aligned
bounding box and per-axis spread.

Consumes the same pair as `tcn_labeled_pointcloud` -- `positions` from
`tcn_depthimage_backprojection` (world space) and the aligned `labels_out` from `tcn_label_sampler` --
so it runs in parallel with the point-cloud path rather than downstream of it.

This is a reduction by key: the masks have already segmented the points, so no clustering is needed to
find instances. A per-axis sigma trim excludes outliers from the box and centroid; see the operator's
README for the case it does not cover (a mask covering two physical surfaces).

**==Named Inputs==**

    positions : device float32 `[H, W, 3]` -- world-space points
    labels : device uint16 `[H, W]` or `[H, W, 1]` -- packed `(class << 8) | instance`

**==Named Outputs==**

    instances : an entity with two HOST tensors --
        `<out_rows_tensor_name>` float32 `[K, 14]`, columns
            camera_index, count, centroid xyz, bbox_min xyz, bbox_max xyz, sigma xyz
        `<out_labels_tensor_name>` uint16 `[K]`, the packed label of each row

Host memory because the consumers are Python and the data is a handful of rows. A frame with no
instances emits one all-zero row whose count is 0, so a downstream operator expecting one message per
frame never starves.

Parameters
----------
fragment : holoscan.core.Fragment (constructor positional only)
    The fragment (or subgraph) that the operator belongs to.
allocator : holoscan.core.Allocator
    Allocator for the output tensors.
cuda_device_ordinal : int, optional
    Device used for CUDA operations. Default is 0.
in_positions_tensor_name : str, optional
    Tensor name to read from the `positions` entity. Default is "" (the unnamed tensor).
in_labels_tensor_name : str, optional
    Tensor name to read from the `labels` entity. Default is "" (the unnamed tensor).
out_rows_tensor_name : str, optional
    Tensor name for the statistics rows. Default is "rows".
out_labels_tensor_name : str, optional
    Tensor name for the per-row packed labels. Default is "labels".
camera_index : int, optional
    Written into every row, so a consumer receiving several cameras on one ANY_SIZE port can tell
    which camera an observation came from. Default is 0.
sigma_k : float, optional
    Points further than this many standard deviations from the instance mean on any axis are excluded
    from the box and centroid. Default is 2.5.
sigma_floor_m : float, optional
    Lower bound on the per-axis sigma used for trimming, in metres. Without it a perfectly flat
    instance rejects all of its own points. Default is 0.01.
min_points : int, optional
    Instances with fewer surviving points are dropped. Default is 64.
max_instances : int, optional
    Output row capacity. Overflow is warned about per frame and counted at shutdown, never silently
    truncated. Default is 64.
verbose : bool, optional
    Log every instance row each frame. Default is False.
name : str, optional
    The name of the operator. Default is "tcn_instance_stats".
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace tcn::doc::TcnInstanceStatsOp
