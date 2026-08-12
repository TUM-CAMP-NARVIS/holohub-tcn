/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 TUM CAMP / NARVIS. All rights reserved.
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

#include "macros.hpp"

namespace tcn::doc::TcnLabeledPointcloudOp {

PYDOC(TcnLabeledPointcloudOp, R"doc(
Turns a labeled depth grid into one compacted point cloud per class.

Consumes `positions` from `tcn_depthimage_backprojection` (world space, one point per depth pixel)
and the aligned `labels_out` from `tcn_label_sampler`, and emits, per configured class, only the
points carrying that class. Each output entity holds both the positions and the packed labels, so
the instance id survives into the data product rather than being collapsed to the class.

One output port per class rather than one cloud with a colour buffer, because `HolovizOp` colours
`POINTS_3D` per InputSpec and not per vertex: a port per class is what lets a fused view show
classes in different colours through the existing viewer. Fuse each class across cameras with
`tcn_stream_merger` (it concatenates along dimension 1, so per-frame point counts may differ) and
hand the result straight to `HolovizOp` as `POINTS_3D`. No `tcn_flatten_tensor` is needed, unlike
`tcn_shm_receiver`'s unlabeled clouds: that operator maps `[H, W, ...]` to `[1, H*W, ...]`, i.e. to
exactly the `[1, N, 3]` this operator already emits, so flattening a fused cloud is a no-op.

Compaction preserves source order (a stable scan, not an atomic append), so identical input yields
an identical point order and a byte-comparison gate is meaningful.

Background points are never emitted: label 0 is background in the packed encoding, and it is also
what `tcn_label_sampler` writes for a depth pixel with no colour correspondence.

**==Named Inputs==**

    positions : device float32 `[H, W, 3]` -- world-space points, from backprojection
    labels : device uint16 `[H, W]` or `[H, W, 1]` -- packed panoptic labels for the same grid

**==Named Outputs==**

    class_<id> : one per entry in `classes`, each an entity with
        `<out_positions_tensor_name>` device float32 `[1, N, 3]` and
        `<out_labels_tensor_name>` device uint16 `[1, N, 1]`

A class with no points in a frame still emits, because a downstream merger needs every input every
frame; it emits a single NaN point, which the rasteriser culls. `N` therefore never reaches 0.

Parameters
----------
fragment : holoscan.core.Fragment (constructor positional only)
    The fragment (or subgraph) that the operator belongs to.
allocator : holoscan.core.Allocator
    Allocator for the output tensors.
classes : list of int, optional
    Class ids to emit, one output port `class_<id>` each. Ids are the high byte of the packed label,
    so each must be < 256. MUST be a constructor argument, not `from_config()`: ports are created in
    `setup()`, which Holoscan runs before parameter values are applied. Default is empty, which
    emits a single `class_all` port carrying every non-background class.
cuda_device_ordinal : int, optional
    Device used for CUDA operations. Default is 0.
in_positions_tensor_name : str, optional
    Tensor name to read from the `positions` entity. Default is "" (the unnamed tensor).
in_labels_tensor_name : str, optional
    Tensor name to read from the `labels` entity. Default is "" (the unnamed tensor).
out_positions_tensor_name : str, optional
    Tensor name for the positions in each output entity. Default is "positions"; it must match the
    downstream merger's `input_message_name`.
out_labels_tensor_name : str, optional
    Tensor name for the labels in each output entity. Default is "labels".
verbose : bool, optional
    Log the per-class point counts every frame. Default is False.
name : str, optional
    The name of the operator. Default is "tcn_labeled_pointcloud".
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Creates one output port per entry in `classes`, read from the constructor arguments because
parameter values are not applied yet at this point.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace tcn::doc::TcnLabeledPointcloudOp
