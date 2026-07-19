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


class RotateImage180Op(Operator):
    """
    Efficiently rotate an image by 180 degrees using CuPy on the GPU.

    Supports multiple image formats:
    - Color: RGBA (HxWx4), BGRA (HxWx4), RGB (HxWx3), BGR (HxWx3)
    - Depth: UINT16 (HxW), Float32 (HxW)

    The rotation is performed using array slicing which is very efficient on GPU.
    Equivalent to: image[::-1, ::-1] (flip both height and width dimensions)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("input")
        tensor = msg.get("")
        if tensor is None:
            # Try alternative key if empty string doesn't work
            # Check all available keys in the message
            return

        # Convert to CuPy array (no-op if already on GPU)
        image = cp.asarray(tensor)

        # Rotate 180 degrees using efficient array slicing
        # [::-1, ::-1] flips both height and width axes
        # Works for any dimensionality: (H, W), (H, W, C), etc.
        rotated = image[::-1, ::-1]

        # Ensure the output is contiguous in memory for better performance
        rotated = cp.ascontiguousarray(rotated)

        # Emit the rotated image
        op_output.emit({"": hs.as_tensor(rotated)}, "output")
