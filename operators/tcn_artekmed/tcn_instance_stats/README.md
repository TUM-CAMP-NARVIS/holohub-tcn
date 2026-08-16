# tcn_instance_stats

`TcnInstanceStatsOp` — reduces a labeled point grid to one row per panoptic instance: point count,
centroid, axis-aligned bounding box, per-axis spread, and a yaw-oriented box.

First stage of the tracked-object path; see [`tcn_object_tracking`](../tcn_object_tracking/README.md)
for the rest.

## Ports

| port | dir | shape / dtype |
|---|---|---|
| `positions` | in | device float32 `[H, W, 3]` — world space, from `tcn_depthimage_backprojection` |
| `labels` | in | device uint16 `[H, W]`/`[H, W, 1]` — packed `(class << 8) \| instance`, from `tcn_label_sampler` |
| `instances` | out | **host** float32 `[K, 21]` + uint16 `[K]` — one row per instance present |

The inputs are **exactly the pair `tcn_labeled_pointcloud` consumes**, so this operator runs in
parallel with the point-cloud path rather than downstream of it: adding it changes nothing that
exists.

Row columns (positional — a tensor has no field names, so
`cuda/tcn_instance_stats_kernel.cuh::InstanceStatColumn` and `tcn_object_tracking/ops.py` must change
together):

```
0     camera_index      (from the camera_index parameter)
1     count             points surviving the trim
2-4   centroid xyz
5-7   bbox_min xyz
8-10  bbox_max xyz
11-13 sigma xyz         PRE-trim spread -- see below
14    yaw               rotation about the vertical axis, radians; 0 = no usable orientation
15-17 oriented_extent   (along yaw, across yaw, vertical)
18-20 oriented_center   world-space centre of the oriented box
```

`oriented_center` is a separate field rather than reusing the centroid: the centroid is a mean and the
box centre is a midpoint of extremes, and for a partially observed object those differ by tens of
centimetres.

Output is in **host** memory: the consumers are Python and the payload is a handful of rows, so a
device tensor would only force them to copy it back.

## Parameters

| parameter | default | notes |
|---|---|---|
| `allocator` | — | |
| `camera_index` | `0` | Stamped into every row, so a consumer receiving several cameras on one `ANY_SIZE` port can tell them apart. |
| `trim_percentile` | `0.02` | Fraction of points discarded from **each end of each axis** before the box, centroid and orientation are computed. This is the outlier rejection; see below for why it replaced sigma clipping. |
| `trim_margin` | `0.05` | Fraction of the axis range added back outside the percentile bound, so a point marginally outside it is kept. Without it the box is systematically a little small. |
| `min_range_m` | `0.01` | Lower bound on an axis range before trimming applies to it. Without it a perfectly flat instance (a wall patch) rejects all of its own points. |
| `up_axis` | `1` | Index of the vertical world axis. A property of the **calibration**, not a convention — for the artekmed exports it is `y` (1). |
| `min_anisotropy` | `1.5` | Ratio the two ground-plane eigenvalues must differ by before the yaw is trusted. See below. |
| `min_points` | `64` | Instances with fewer surviving points are dropped. At depth-camera resolution 64 is permissive — a real scene produced 100-point "people"; several hundred is more realistic. |
| `max_instances` | `64` | Row capacity. Overflow is warned about per frame and counted at shutdown, never silently truncated. |
| `in_*` / `out_*_tensor_name` | see pydoc | |
| `verbose` | `false` | Log every row each frame. |

## Why this is a reduction, not clustering

The masks have already segmented the points: every point states which instance it belongs to. So
finding instances is a **reduction by key**, and the accumulator table is sized by the *label space*
(65536 slots, 4.5 MB) rather than by the instances present — which is what lets the whole thing run
without first discovering how many instances there are.

Six passes, all of them `atomicAdd`/`atomicMin`/`atomicMax` over the same table — no per-instance
loop and no sort:

1. **raw moments and range** — count, sum, sum-of-squares, min and max per label
2. **untrimmed sigma** — reported as-is, see below
3. **histogram** — 64 bins per axis per label, spanning that label's raw range
4. **bounds** — walk in from both ends of each histogram until `trim_percentile` of the points have
   been passed; that bin edge, widened by `trim_margin`, is the accepted interval
5. **trimmed reduction** — count, sum, min and max over only the points inside the interval on all
   three axes, plus the ground-plane cross-moments `sum(uu)`, `sum(vv)`, `sum(uv)` needed for the yaw
6. **yaw, then oriented extents** — see the next section

`atomicMin`/`atomicMax` run on **ordered-int-encoded** floats (a monotonic float→int mapping, so
integer atomics order floats correctly — negatives are where a naive encoding breaks, and there is a
test for exactly that).

### Zeroing the table is not free — and used to be the whole cost

Sizing by the label space buys "no pre-pass", but it has to be paid for once per frame per camera:
the histogram alone is `65536 × 3 × 64 × 4 B` = **48 MiB**. The original `reset_kernel` was one
thread per *slot*, each looping over its own 192 bins, so adjacent threads wrote 768 bytes apart:
50 MB at **45.7 GB/s**, about a fifteenth of the card. That made zeroing the third most expensive
kernel in the whole application — 1101 µs median, 2.56 s over a 103 s profile, 5.6% of GPU 0
(nsys, 2026-08-16) — for a fill that computes nothing.

`launch_instance_reset` now uses `cudaMemsetAsync` for everything that resets to plain zero (`0.f`
is all-zero bytes, so the float arrays qualify) and keeps a kernel only for the min/max sentinels,
which are `0x7FFFFFFF` / `0x80000000` and therefore not byte-fillable — flat-indexed, so
consecutive threads write consecutive words.

| | before | after |
|---|---|---|
| histogram fill | 45.7 GB/s | **563 GB/s** |
| whole reset | 1101 µs median | **116.4 µs** (89.3 hist + 18.2 small + 8.9 sentinels) |

The 48 MiB memset now dominates what is left. Shrinking `kInstanceSlots` to `(max_class+1) << 8`
would cut it ~40× and shrink the other table-sized kernels with it, but those total only ~0.12 s
against the 2.4 s this recovered — so it is not worth the new failure mode (a class id above the
bound) today.

The reported `sigma` is the **pre-trim** value, deliberately: it is the signal that a mask covered two
surfaces. The trim hides that from the box, and it should not also hide it from the log.

## The yaw-oriented box

A world-axis AABB is inflated for anything standing at an angle to the world axes: a 1.2 × 0.2 m plate
at 30° needs a 1.14 × 0.77 m axis-aligned box, nearly four times its true footprint. So the box is also
reported rotated about the vertical:

1. the 2×2 ground-plane covariance is eigen-decomposed, giving
   `yaw = ½·atan2(2·c_uv, c_uu − c_vv)` — the direction of the footprint's principal axis
2. every surviving point is projected onto that frame and the min/max recorded, giving the extents

Rotation is **only about the vertical axis**. Objects in a room stand upright, so a full 3D orientation
would spend two of its three degrees of freedom fitting noise in the one direction already known.

`min_anisotropy` is what stops that box from spinning. A near-circular footprint — a standing person
seen from above — has no meaningful orientation, and the principal axis of near-equal eigenvalues is
decided by noise, so it swings freely frame to frame. Below the threshold the yaw is reported as 0 and
the oriented box degenerates to the axis-aligned one. This is why a person's box is usually unrotated
while a bed's is not, and it is deliberate.

Orientation is computed from the **trimmed** points, so a bleed blob cannot drag the principal axis
onto itself — there is a test for exactly that.

## What the trim does and does not fix

`trim_percentile` has a **breakdown point**: it discards that fraction from each end by count, so it
removes outliers up to that fraction *however far away they are*. This is why it replaced iterative
sigma clipping, whose breakdown point is effectively zero — a blob holding 23% of an instance's points
3 m away inflates sigma to ~1.2 m, so a ±2σ window already contains the blob and iteration cannot
escape it, being at a fixed point from the first pass. Both behaviours are pinned by tests.

It does **not** reject an outlier population larger than `trim_percentile`, by construction. The
symptom is an implausibly large extent with a large pre-trim sigma. Raising the percentile is the wrong
answer — it starts eating the object. That escalation is now implemented: see below.

## The connected-component filter — escaping the breakdown point

`component_filter: true` keeps only the largest connected component of each instance, running
**before** everything above and rewriting nothing but its own scratch: losing pixels get label 0, so
the six passes never learn it exists.

Why connectivity rather than another distance rule: every distance rule here has a breakdown point.
The percentile trim removes outliers up to `trim_percentile` and no further. Sigma clipping is
strictly worse — for a tabletop with 20% of its points bled onto the floor 0.5 m below,
`sigma_y = 0.20 m`, so the ±2σ window `[-0.50, +0.30]` *contains the blob it is meant to reject*. A
real bleed population is routinely above both. Connectivity does not care how many the outliers are,
only that they are not attached.

**On the image grid, not a voxel grid.** These points come from a depth image, so they are an
*organised* cloud: 3D adjacency is 2D pixel adjacency plus a depth-continuity test. That makes this a
2D labelling over `H*W` rather than a voxel grid to allocate, size and hash — and it is the more
faithful test, because bleed lands on a background surface and so is separated from its object by
exactly the discontinuity `component_max_gap_m` looks for.

Four-connectivity, deliberately: a diagonal-only link is a single-pixel bridge, and mask fringe is
where those occur. Convergence is `kComponentRounds` rounds of propagate + pointer-jump with **no
host synchronisation** — pointer jumping shortens chains geometrically, and under-convergence would
only split a component further, never merge two.

### `component_min_fraction` is the parameter that matters

`1.0` — "keep strictly the largest" — is a **trap**. Masks have holes (occlusion, invalid depth),
holes are label 0, and a hole breaks grid adjacency, so a real object routinely arrives as several
islands. Measured on the 4-camera `k4a_capture` replay:

| setting | objects | person | computer | monitor | table | chair |
|---|---|---|---|---|---|---|
| filter off | 18 | 9 | 1 | 1 | 2 | 5 |
| on, fraction **1.0** | 14 | 9 | **0** | **0** | 2 | **3** |
| on, fraction **0.10** | **19** | 9 | 1 | 1 | 3 | 5 |

At `1.0` the result was *identical* for gaps 0.05, 0.10 and 0.20 — the gap making no difference is
what identified holes, rather than depth steps, as the cause. `0.10` is the default for that reason.

### What it buys

Median box volume **−30%** across 468 matched instances (439 shrank, 2 grew) for only −6.1% of
points — outliers, not erosion. The worst bleeding table went `4.47 × 0.79 × 4.79 m` →
`2.96 × 0.47 × 3.58 m`.

### Cost

**~443 µs per camera-frame** at 640×576 (nsys, 20 rounds): propagate 20 × 9.4 µs, compress 40 ×
4.4 µs, then init 11.5, count 16.9, best 45.2, select 5.2. That is on top of the existing six passes'
~633 µs, and less than half of the ~985 µs the reset rewrite above gave back.

## Tests

```bash
PYTHONPATH=<build>/python/lib:$PYTHONPATH python3 tests/test_instance_stats.py   # 18 cases
```

Runs the real operator on synthetic point grids with hand-computed expectations. Exact box and centroid
for a planted cuboid, negative coordinates, background exclusion, a planted far blob that inflates the
untrimmed box and leaves the trimmed one alone, the masking effect that defeats sigma clipping, the
documented breakdown point above `trim_percentile`, `min_points`, `camera_index` stamping, and the
empty-frame placeholder row.

For the oriented box: a plate planted at a known yaw must report that yaw and its true extents while
its AABB is inflated; the oriented centre must land on the plate centre (a sign error in the inverse
rotation is invisible in the extents); a square footprint must report no yaw, and must report one when
`min_anisotropy` is 0 — so the guard is being tested rather than a kernel that never rotates anything;
the oriented volume must never exceed the axis-aligned one across five yaws; `up_axis` must select the
vertical; and the yaw must be immune to an outlier blob outside the trim.
