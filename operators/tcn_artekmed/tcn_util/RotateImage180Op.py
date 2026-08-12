# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from .rotate import rotate180


class RotateImage180Op(Operator):
    """Rotate an image or map 180 degrees on the GPU.

    Supports any layout whose first two axes are spatial: colour `(H, W, 3|4)` and single-channel
    `(H, W)` depth or label maps. The rotation is `arr[::-1, ::-1]` (see `rotate.rotate180`), so it is
    exact, its own inverse, and costs one copy to make the result contiguous.

    For rotating individual cameras out of a multi-camera entity, call `rotate180` directly instead of
    inserting this operator per camera -- that is what `tcn_langsam` does for upside-down cameras.
    """

    def __init__(self, *args, in_tensor_name="", out_tensor_name="", **kwargs):
        self.in_tensor_name = str(in_tensor_name)
        self.out_tensor_name = str(out_tensor_name)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("input")
        tensor = None if msg is None else msg.get(self.in_tensor_name)
        if tensor is None:
            # Loudly, rather than returning: a silent skip drops the frame AND starves whatever is
            # downstream, and the cause (a tensor-name mismatch) is invisible in the log.
            available = sorted(msg.keys()) if msg is not None else []
            raise ValueError(
                f"RotateImage180Op: no tensor named {self.in_tensor_name!r} on 'input'; "
                f"available: {available}. Set in_tensor_name to one of those.")

        rotated = cp.ascontiguousarray(rotate180(cp.asarray(tensor)))
        op_output.emit({self.out_tensor_name: hs.as_tensor(rotated)}, "output")
