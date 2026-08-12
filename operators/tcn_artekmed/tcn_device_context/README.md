# tcn_device_context

Camera calibration for the pipeline: a fragment **service** holding per-camera intrinsics and
extrinsics, plus a source **operator** that turns intrinsics into the xy lookup table that
backprojection needs.

## `DeviceContextService`

A `holoscan::DefaultFragmentService`, registered on the fragment and looked up by operators that need
calibration.

```python
from holohub.tcn_device_context._tcn_device_context import DeviceContextService

ctx = DeviceContextService.create(device_contexts)   # dict from discover_shm()
self.register_service(ctx)
```

| method | returns |
|---|---|
| `has_camera(name)` | whether calibration exists — check before building a geometric path |
| `camera_names()` | all known cameras |
| `get_camera_name_from_port_name(port)` | `camera01_colorimage` → `camera01` |
| `get_depth_camera_model(name)` / `get_color_camera_model(name)` | `nvidia::gxf::CameraModel` incl. distortion and dimensions |
| `get_xy_table_intrinsics(name)` / `get_xy_table(name)` | intrinsics and table used by backprojection |
| `get_depth_extrinsics(name)` | depth camera → world |
| `get_color_to_depth(name)` / `get_color_to_depth_inv(name)` | the rigid transform between the two cameras; the `_inv` form is what backprojection's `depth_to_color` expects |

### Input dictionary shape

`create()` takes the "legacy" Python dictionary that `discover_shm()` produces, one entry per camera:

```python
{
  "camera01": {
    "calibration": {
      "depthCameraParameters": {"width", "height", "fovX", "fovY", "cX", "cY",
                                "distortionParams": {"k1","k2","tx","ty","k3","k4","k5","k6"}},
      "colorCameraParameters": {...},
      "cameraPose":           {"translation": {"x","y","z"}, "rotation": {"x","y","z","w"}},
      "color2depthTransform": {"translation": {...},          "rotation": {...}},
    },
    "depthUnitsPerMeter": 1000.0,
    "isValid": True,
    "frameRate": 30.0,
  },
}
```

Note the distortion order: Brown coefficients are read as **k1, k2, tx, ty, k3, k4, k5, k6** — the
tangential pair sits in the *middle*. A source that stores six radial coefficients followed by two
tangential ones must be reordered, not copied; getting it wrong distorts plausibly rather than
obviously. `tcn_dataset_replayer/_calibration.py` does this conversion for the dataset path and has
host tests covering exactly that ordering.

`depthUnitsPerMeter` is carried here but **not exposed by a getter**, so operators take it from
their own configuration instead. Keep the two consistent.

## `XYLookupTableSourceOp`

Emits the per-pixel ray-direction table for one camera, computed from its depth intrinsics.

| port | dir | shape |
|---|---|---|
| `xy_table` | out | device `[H, W, 2]` float32 |

| parameter | notes |
|---|---|
| `allocator` | |
| `camera_name` | which camera's intrinsics to use |

The table is constant, so the operator is built with a `CountCondition(count=1)` and emits **once**;
consumers declare their `xy_table` port with `ConditionType::kNone` so later ticks are not gated on
it. The service must be attached explicitly:

```python
xylt = XYLookupTableSourceOp(self, CountCondition(self, count=1),
                             allocator=pool, camera_name="camera01", name="xylt_camera01")
xylt.set_device_context_service(ctx)
self.add_flow(xylt, bp_op, {("xy_table", "xy_table")})
```
