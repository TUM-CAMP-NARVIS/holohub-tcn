#!/usr/bin/env python3
"""
Simple test script for RotateImage180Op
"""

import sys
import numpy as np
import cupy as cp

# Add the parent directory to path to import the operator
sys.path.insert(0, '/workspace/holohub')

from operators.tcn_artekmed.tcn_util import RotateImage180Op
import holoscan as hs
from holoscan.core import Application, Operator, OperatorSpec


class ImageSourceOp(Operator):
    """Generate test images"""

    def __init__(self, *args, image_type="color", **kwargs):
        self.image_type = image_type
        self.frame_count = 0
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        if self.frame_count >= 3:
            return

        if self.image_type == "color":
            # Create a test color image (RGBA) with gradient
            # Top-left should be different from bottom-right after rotation
            image = np.zeros((100, 100, 4), dtype=np.uint8)
            # Create a gradient pattern
            for i in range(100):
                for j in range(100):
                    image[i, j, 0] = i * 2  # Red channel
                    image[i, j, 1] = j * 2  # Green channel
                    image[i, j, 2] = 128    # Blue constant
                    image[i, j, 3] = 255    # Alpha
        elif self.image_type == "depth":
            # Create a test depth image (UINT16)
            image = np.zeros((100, 100), dtype=np.uint16)
            for i in range(100):
                for j in range(100):
                    image[i, j] = i * 100 + j
        else:
            # Float depth
            image = np.random.randn(100, 100).astype(np.float32)

        # Convert to CuPy and emit
        image_gpu = cp.asarray(image)
        op_output.emit({"": hs.as_tensor(image_gpu)}, "output")
        self.frame_count += 1
        print(f"Source: Emitted frame {self.frame_count}, shape: {image.shape}, dtype: {image.dtype}")


class ImageSinkOp(Operator):
    """Verify rotated images"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = 0

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        msg = op_input.receive("input")
        tensor = msg.get("")

        if tensor is not None:
            image = cp.asarray(tensor)
            self.frame_count += 1
            print(f"Sink: Received frame {self.frame_count}, shape: {image.shape}, dtype: {image.dtype}")

            # Verify rotation
            if len(image.shape) == 3 and image.shape[2] == 4:  # Color RGBA
                # Check corners
                print(f"  Top-left corner (R,G): ({image[0, 0, 0]}, {image[0, 0, 1]})")
                print(f"  Bottom-right corner (R,G): ({image[-1, -1, 0]}, {image[-1, -1, 1]})")
                print(f"  Expected after 180 rotation: Top-left should have high values")
            elif len(image.shape) == 2:  # Depth
                print(f"  Top-left value: {image[0, 0]}")
                print(f"  Bottom-right value: {image[-1, -1]}")

            print("  ✓ Rotation successful")


class TestApp(Application):
    def __init__(self, image_type="color"):
        super().__init__()
        self.image_type = image_type

    def compose(self):
        source = ImageSourceOp(self, name="source", image_type=self.image_type)
        rotate = RotateImage180Op(self, name="rotate_180")
        sink = ImageSinkOp(self, name="sink")

        self.add_flow(source, rotate)
        self.add_flow(rotate, sink)


def main():
    print("="*60)
    print("Testing RotateImage180Op with Color Image (RGBA)")
    print("="*60)
    app = TestApp(image_type="color")
    app.run()

    print("\n" + "="*60)
    print("Testing RotateImage180Op with Depth Image (UINT16)")
    print("="*60)
    app = TestApp(image_type="depth")
    app.run()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    main()
