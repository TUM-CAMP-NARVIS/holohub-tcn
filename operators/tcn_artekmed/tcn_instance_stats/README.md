# tcn_instance_stats

`TcnInstanceStatsOp` — reduces a labeled point grid to one row per panoptic instance: point count,
centroid, axis-aligned bounding box and per-axis spread.

First stage of the tracked-object path; see [`tcn_object_tracking`](../tcn_object_tracking/README.md)
for the rest.

## Ports

| port | dir | shape / dtype |
|---|---|---|
| `positions` | in | device float32 `[H, W, 3]` — world space, from `tcn_depthimage_backprojection` |
| `labels` | in | device uint16 `[H, W]`/`[H, W, 1]` — packed `(class << 8) \| instance`, from `tcn_label_sampler` |
| `instances` | out | **host** float32 `[K, 14]` + uint16 `[K]` — one row per instance present |

The inputs are **exactly the pair `tcn_labeled_pointcloud` consumes**, so this operator runs in
parallel with the point-cloud path rather than downstream of it: adding it changes nothing that
exists.

Row columns (positional — a tensor has no field names, so
`cuda/tcn_instance_stats_kernel.cuh::InstanceStatColumn` and `tcn_object_tracking/ops.py` must change
together):

```
0     camera_index      (from the camera_index parameter)
1     count             points surviving the sigma trim
2-4   centroid xyz
5-7   bbox_min xyz
8-10  bbox_max xyz
11-13 sigma xyz         PRE-trim spread -- see below
```

Output is in **host** memory: the consumers are Python and the payload is a handful of rows, so a
device tensor would only force them to copy it back.

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | |
| `camera_index` | `0` | Stamped into every row, so a consumer receiving several cameras on one `ANY_SIZE` port can tell them apart. |
| `sigma_k` | `2.5` | Points further than this many standard deviations from the instance mean **on any axis** are excluded from the box and centroid. |
| `sigma_floor_m` | `0.01` | Lower bound on the per-axis sigma used for trimming. Without it a perfectly flat instance (sigma ≈ 0 on one axis, e.g. a wall patch) rejects all of its own points. |
| `min_points` | `64` | Instances with fewer surviving points are dropped. At depth-camera resolution 64 is permissive — a real scene produced 100-point "people"; several hundred is more realistic. |
| `max_instances` | `64` | Row capacity. Overflow is warned about per frame and counted at shutdown, never silently truncated. |
| `in_*` / `out_*_tensor_name` | see pydoc | |
| `verbose` | `false` | Log every row each frame. |

## Why this is a reduction, not clustering

The masks have already segmented the points: every point states which instance it belongs to. So
finding instances is a **reduction by key**, and the accumulator table is sized by the *label space*
(65536 slots, 4.5 MB) rather than by the instances present — which is what lets the whole thing run
without first discovering how many instances there are.

Two passes:

1. count, sum and sum-of-squares per label (`atomicAdd`)
2. count, sum, min and max over only the points within `sigma_k · sigma` of the mean, using
   `atomicMin`/`atomicMax` on **ordered-int-encoded** floats (a monotonic float→int mapping, so
   integer atomics order floats correctly — negatives are where a naive encoding breaks, and there is
   a test for exactly that)

The reported `sigma` is the **pre-trim** value, deliberately: it is the signal that a mask covered two
surfaces. The trim hides that from the box, and it should not also hide it from the log.

## What the sigma trim does and does not fix

It rejects depth noise and thin mask fringes. It does **not** reject a mask that bleeds onto a distant
surface if that bleed is a substantial fraction of the points — the mean itself moves. The symptom is
an implausibly large extent with a large pre-trim sigma. The escalation, if the data calls for it, is
voxel connected components: keep only the largest connected blob per instance.

## Tests

```bash
PYTHONPATH=<build>/python/lib python3 tests/test_instance_stats.py   # 8 cases
```

Exact box and centroid for a planted cuboid, negative coordinates, background exclusion, a planted
far blob that inflates the untrimmed box and leaves the trimmed one alone, `min_points`,
`camera_index` stamping, and the empty-frame placeholder row.
