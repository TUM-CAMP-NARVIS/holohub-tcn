# RotateImage180Op

Efficiently rotate images by 180 degrees on the GPU using CuPy.

## Features

- **GPU-accelerated**: Uses CuPy array slicing for maximum performance
- **Multiple formats supported**:
  - Color images: RGBA, BGRA, RGB, BGR (HxWxC format)
  - Depth images: UINT16, Float32 (HxW format)
- **Efficient**: Uses simple array slicing (`[::-1, ::-1]`) which is optimized on GPU
- **Memory-contiguous output**: Ensures output is contiguous for downstream operators

## Usage

### Basic Example

```python
from operators.tcn_artekmed.tcn_util import RotateImage180Op

# In your Holoscan application compose method:
rotate_op = RotateImage180Op(self, name="rotate_camera")
self.add_flow(camera_source, rotate_op, {("output", "input")})
self.add_flow(rotate_op, next_operator, {("output", "input")})
```

### With LangSAM Pipeline

To rotate camera images before LangSAM inference:

```python
from operators.tcn_artekmed.tcn_util import RotateImage180Op

# In langsam_fragment.py or your compose method:
rotate_camera = RotateImage180Op(self, name="rotate_camera_180")
col_conv = ConvertBgraToRgbaOp(self, name="color_converter_rgba")

# Insert rotation before color conversion
self.add_flow(input_source, rotate_camera, {("output", "input")})
self.add_flow(rotate_camera, col_conv, {("output", "input")})
```

### Configuration Example

For camera-specific rotation (e.g., if camera01 is upside-down):

```python
# In your application
camera_rotation_config = {
    "camera01": True,  # Rotate this camera
    "camera02": False,
    "camera03": False,
    "camera04": True,  # Rotate this camera too
}

# Create rotation operators conditionally
for camera_name, needs_rotation in camera_rotation_config.items():
    if needs_rotation:
        rotate_op = RotateImage180Op(self, name=f"rotate_{camera_name}")
        self.add_flow(camera_source, rotate_op)
        self.add_flow(rotate_op, next_operator)
    else:
        self.add_flow(camera_source, next_operator)
```

## Performance

- **Operation**: Array slicing `image[::-1, ::-1]`
- **Memory**: Single copy to ensure contiguous layout
- **Speed**: Near-instant on GPU (< 1ms for typical image sizes)

## Input/Output Format

- **Input Port**: `"input"` - Receives image tensor with empty string key `""`
- **Output Port**: `"output"` - Emits rotated tensor with empty string key `""`

## Supported Data Types

- `uint8` (color images)
- `uint16` (depth images)
- `float32` (depth images)
- Any other numeric type supported by CuPy

## Example: Full Pipeline Integration

```python
class LangSamProcessingSubgraph(Subgraph):
    def compose(self):
        # Rotation for upside-down camera
        rotate = RotateImage180Op(self, name="rotate_180")

        # Color conversion
        col_conv = ConvertBgraToRgbaOp(self, name="color_converter_rgba")

        # LangSAM inference
        langsam = LangSAM2Operator(self, name="langsam_inference", **args)

        # Connect pipeline
        self.add_flow(rotate, col_conv, {("output", "input")})
        self.add_flow(col_conv, langsam, {("output", "image")})

        # Expose ports
        self.add_input_interface_port("input", rotate, "input")
```

## Notes

- Rotation is lossless - no interpolation required for 180° rotation
- Works with any image dimensions
- Preserves all channels and data types
- Output is guaranteed to be C-contiguous in memory
