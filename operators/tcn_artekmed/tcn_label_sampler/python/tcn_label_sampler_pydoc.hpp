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

namespace tcn::doc::TcnLabelSamplerOp {

PYDOC(TcnLabelSamplerOp, R"doc(
Gives every depth pixel the panoptic label of the scene point it observes.

Samples a packed panoptic label map (`(class_id << 8) | instance_id`, uint16, 0 = background) through
the `texcoords` output of `tcn_depthimage_backprojection` -- normalised colour-image coordinates per
depth pixel, computed through the full geometry. This is the only correct mask/depth correspondence:
the depth and colour sensors differ in resolution, intrinsics, distortion and optical centre, so
rescaling a mask onto the depth grid is spatially wrong in a depth-dependent way that no aggregate
test detects.

Sampling is nearest-neighbour, because interpolating packed label ids produces ids denoting none of
the classes involved. Texcoords outside [0, 1] are rejected rather than clamped: edge extension is
reasonable for colour and would smear a border object across everything beside it in a label image.
A non-finite texcoord (which `tcn_depthimage_backprojection` writes where depth is invalid or outside
the near/far limits) is rejected the same way. Both cases write `unlabeled_value` and mask 0.

**==Named Inputs==**

    labels : device uint16 `[Hc, Wc]` or `[Hc, Wc, 1]` -- the panoptic map, at colour resolution
    texcoords : device float32 `[Hd, Wd, 2]` -- from `tcn_depthimage_backprojection`

**==Named Outputs==**

    labels_out : device uint16 `[Hd, Wd, 1]` -- packed label per depth pixel
    mask_out : device uint8 `[Hd, Wd, 1]` -- 255 where a selected class was hit, else 0

`mask_out` is what makes `tcn_depthimage_apply_mask` connectable: that operator needs a single
unnamed uint8 mask with the same element count as the depth image, which a colour-resolution panoptic
map can never be. Leave `out_mask_tensor_name` empty for that consumer, since it reads the entity's
unnamed tensor.

Parameters
----------
fragment : holoscan.core.Fragment (constructor positional only)
    The fragment (or subgraph) that the operator belongs to.
allocator : holoscan.core.Allocator
    Allocator for the two output tensors.
cuda_device_ordinal : int, optional
    Device used for CUDA operations. Default is 0.
in_labels_tensor_name : str, optional
    Tensor name to read from the `labels` entity. Default is "" (the unnamed tensor).
in_texcoord_tensor_name : str, optional
    Tensor name to read from the `texcoords` entity. Default is "" (the unnamed tensor).
out_labels_tensor_name : str, optional
    Tensor name for `labels_out`. Default is "" (unnamed).
out_mask_tensor_name : str, optional
    Tensor name for `mask_out`. Default is "" (unnamed), which is what
    `tcn_depthimage_apply_mask` requires.
select_classes : list of int, optional
    Class ids that `mask_out` marks; ids are the high byte of the packed label, so each must be in
    [0, 255] and an out-of-range entry raises at `start()`. Default is empty, which selects every
    non-background class.
unlabeled_value : int, optional
    Label written where a depth pixel has no colour correspondence. Must fit uint16. Default is 0.
    A non-zero value never leaks into `mask_out`: rejection is tracked separately from the sampled
    label, so a value that collides with a real class still masks as unselected.
name : str, optional
    The name of the operator. Default is "tcn_label_sampler".
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace tcn::doc::TcnLabelSamplerOp
