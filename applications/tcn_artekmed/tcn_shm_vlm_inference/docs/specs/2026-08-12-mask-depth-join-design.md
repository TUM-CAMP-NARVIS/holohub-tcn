# Mask ↔ depth join via backprojection texcoords — design

**Status:** design
**Date:** 2026-08-12
**Depends on:** `2026-08-11-temporal-sync-design.md` (frame grouping), `2026-08-10-replay-harness-design.md` (deterministic source)

## Goal

Give every depth pixel the panoptic label of the scene point it observes, so the unprojected depth
map (point cloud) can be segmented by the LangSAM masks.

## Why this is not a resize

The masks are produced at colour resolution and the depth map is a different sensor: 1920×1080 vs
640×576 on the dataset export, with different intrinsics, different distortion, and a rigid offset
between the two optical centres. Rescaling a mask to the depth grid therefore *looks* plausible and
is spatially wrong — a systematic, depth-dependent error that no test on aggregate statistics
catches. The only correct correspondence goes through the geometry:

```
depth pixel (x,y) + depth d  ──xy_table──►  3D point in depth-camera space
                             ──depth_to_color──►  point in colour-camera space
                             ──project (distortion-aware)──►  colour pixel (u,v)
                             ──sample panoptic map──►  label for this depth pixel
```

`tcn_depthimage_backprojection` already computes exactly this and emits it as `texcoords`
(normalised uv per depth pixel). The join is therefore a *sampling* problem, not a registration
problem — the registration is already solved and unused.

## Why the existing sampler cannot be reused

`tcn_texture_sampler` samples an RGBA uint8 image **bilinearly**. A panoptic map is packed
`uint16 = (class << 8) | instance`. Averaging four of those ids produces an id that denotes neither
of the classes involved — at every object boundary it invents a label. Label images admit exactly
one filter: nearest neighbour.

The existing sampler also *clamps* uv into [0,1], which is right for colour (edge extension) and
wrong for labels: a depth pixel whose 3D point falls outside the colour frustum has no label, and
must not silently inherit the border one.

## Components

### 1. `tcn_depthimage_backprojection`: two fixes

**Texcoords are unreachable without positions.** The texcoord write is nested inside
`if (positions_enabled && positions)`, so `enable_positions=false, enable_texcoords=true` allocates
the output and emits it untouched. Latent today (`tcn_shm_receiver` enables both), and exactly the
configuration this join wants if the point cloud is not needed. The two outputs become independent.

**Invalid depth is indistinguishable from uv (0,0).** A pixel failing the near/far/NaN test keeps
`out_uv = {0,0}`, which is a legitimate texcoord — the colour image's top-left corner. The kernel
will instead write `NaN` for "no correspondence".

This is behaviour-preserving for the colour path: `clampf` is `fminf(fmaxf(u,0),1)`, and IEEE
`fmaxf(NaN,0)` returns `0`, so a NaN texcoord clamps to 0 and samples pixel (0,0) — bit-identical to
today. The difference is only that a *label* sampler can now detect the case.

### 2. New operator `tcn_label_sampler`

Nearest-neighbour sampling of an integer label image through texcoords.

| Port | Direction | Shape / dtype |
|---|---|---|
| `labels` | in | device `[Hc, Wc]` or `[Hc, Wc, 1]` uint16 — the panoptic map |
| `texcoords` | in | device `[Hd, Wd, 2]` float32 — from backprojection |
| `labels_out` | out | device `[Hd, Wd, 1]` uint16 — label per depth pixel |
| `mask_out` | out | device `[Hd, Wd, 1]` uint8 — 255 where the label's class is selected, else 0 |

Parameters: `in_labels_tensor_name`, `in_texcoord_tensor_name`, `out_labels_tensor_name`,
`out_mask_tensor_name`, `select_classes` (list of class ids; empty = every non-zero class),
`unlabeled_value` (default 0), `allocator`, `cuda_device_ordinal`, `cuda_stream_pool`.

Both outputs are always emitted. An `emit_mask` switch was specified and dropped during
implementation: the mask costs one uint8 buffer on the depth grid (368 KiB at 640×576), and making
it conditional buys nothing while adding a half-configured state in which a wired consumer starves.

`mask_out` exists so the join composes with the operator that already applies a mask to a depth
image: `tcn_depthimage_apply_mask` requires a single unnamed uint8 mask with the *same element
count* as the depth image, which is precisely what this produces — the resolution mismatch that made
it unusable is gone once the mask lives on the depth grid.

Semantics: `unlabeled_value` is written when the texcoord is NaN (no valid depth) or falls outside
[0,1] (outside the colour frustum). Nearest neighbour is `round(u * (W-1))`, matching the existing
sampler's `u * (W-1)` mapping so colour and label sampling address the same pixel grid.

The class-packing constants (`class = label >> 8`, 256 class ids) live in the sampler's kernel
header. They cannot be shared with `tcn_panoptic_map`, which never unpacks — it receives already
packed `values` from Python. The convention is therefore encoded in exactly two places,
`langsam_helpers.py` and that header, and both say so.

### 3. Calibration for the replay source

`compose()` sets `device_contexts = {}` when `source: dataset`, so there is no calibration, no
xy_table, and therefore no backprojection on the one source that is deterministic. Without this the
join can only ever be eyeballed on a live stream.

The export carries per-camera calibration (`calibration/cameraNN.json`) with exactly the fields the
device-context dict needs — depth and colour intrinsics with distortion, `camera_pose`,
`color2depth_transform` — under snake_case names instead of the SHM path's camelCase. A pure adapter
maps one to the other, so `DeviceContextService`, `XYLookupTableSourceOp`, backprojection and the
sampler all run unmodified on replayed frames.

Adapter lives in `operators/tcn_artekmed/tcn_dataset_replayer/_calibration.py` — pure, no
holoscan/cupy imports, host-testable next to `_planning.py`.

### 4. App wiring

Per camera, downstream of the synchroniser so the mask and the depth belong to the same frame:

```
temporal_sync ──depth──► splitter ──► <cam>_depth ──► backprojection ──texcoords──┐
temporal_sync ──masks──► splitter ──► <cam>_panoptic ─────────────────────────────┤
                                                                                  ▼
                                                                          label_sampler
                                                                          ├─ labels_out
                                                                          └─ mask_out
```

Gated on `mask_depth_join.enabled`, which requires `temporal_sync.enabled` (an ungrouped join is the
mis-registration this design exists to remove) and calibration for every camera (refuse rather than
silently skip a camera).

## Correctness gates

1. **Host tests** for the calibration adapter: field mapping, and a round trip through
   `DeviceContextService` asserting the intrinsics survive.
2. **Geometry gate**, replay source, no network: for a synthetic depth image of constant depth,
   every sampled label must equal the panoptic label at the analytically projected pixel. Run on
   the dataset's real calibration so distortion is exercised.
3. **Invalid gate**: depth 0 everywhere ⇒ `labels_out` is entirely `unlabeled_value` and `mask_out`
   entirely 0. Catches the (0,0)-corner bug this design fixes.
4. **Mask implies depth**: a pixel can only be selected if its depth was valid (an invalid sample
   yields a NaN texcoord, hence no label), so applying the mask must keep every selected pixel.
   Checked per frame by `JoinCheckOp` and reported as a per-camera verdict. This is what caught the
   splitter bug above.

Status: gates 1 and 3 pass (11/11 host, 5/5 operator cases); the join runs end to end on the
4-camera replay with 14–17% of depth pixels labeled and the mask-implies-depth invariant holding on
every camera. Gate 2 (analytic projection against a synthetic constant-depth image) is not yet
written — it is the one that would prove the correspondence is *correct* rather than merely present.
Note the mask counts vary by a few tenths of a percent between runs: the GDINO/SAM stage is not
bit-reproducible here, so the join's own determinism has to be gated with fixed input masks.

## Found during implementation

**`tcn_stream_splitter` wrapped memory without keeping its owner alive.** `wrapMemory` was called
with a null release callback, so every split output pointed at memory owned solely by the input
entity; once `compute()` returned, that allocation could be reused by a later frame while a
downstream consumer was still reading it. Invisible while one consumer reads a split tensor
immediately, which is all the colour path ever did. The join exposed it because the synchroniser
buffers entities across ticks and three operators read the same split depth image. Symptom: on ~1 of
8 frames, one camera's labels did not belong to the depth image they were applied to — caught by the
mask-implies-depth invariant below, which is why that check earns its place. Fixed by holding a
reference to the source entity in the release callback.

**The backprojection operator rejected texcoords-without-positions** in `initialize()`, matching the
kernel's old nesting. Removed; allocation and emission were already per-output.

**`emit_depth` was hardcoded false** on the dataset source, so depth entities were empty and the
earlier timestamp-grouping runs were grouping empty payloads (which does not invalidate that gate —
it tests grouping, not content). Now derived from whether the join is enabled.

## Labeled point clouds and the fused view

Each camera's join produces a point cloud rather than only a masked depth image: backprojection's
`positions` (world space, via `depth_extrinsics`) carry one point per depth pixel, and
`tcn_label_sampler`'s labels index the same grid, so the two are aligned by construction.
`tcn_labeled_pointcloud` compacts them into one entity per class — positions plus the *packed* label,
so the instance id survives into the data and not just the class.

Fusion follows `tcn_shm_receiver`: one `tcn_stream_merger` per class concatenating the cameras along
dimension 1 (per-frame point counts differ, which that axis allows), straight into a `HolovizOp`
`POINTS_3D` spec per class. One chain per class because Holoviz colours an InputSpec, not a vertex —
that is what makes classes separable in the fused view, and each class takes its base colour from the
same `build_panoptic_lut` the 2D overlay uses, so a class looks the same in both views.

No flatten step: `tcn_flatten_tensor` maps `[H, W, …]` to `[1, H*W, …]`, which for an already
`[1, N, 3]` cloud is an exact no-op.

A class with no points still emits one NaN point. An empty tensor would starve the merger, which
needs every input every frame, and a zero position would draw a stray point at the origin; a NaN
vertex is culled instead.

Compaction is order-preserving (a stable scan, not an atomic append), so identical input gives an
identical point order — an atomic append would reorder points run to run and make byte-comparison
gates useless.

The join gets its own `RMMAllocator`. The shared pool is a `BlockMemoryPool` with frame-sized blocks,
and the join adds dozens of small, variable-sized tensors per frame; each would consume a whole
block, exhausting the pool while wasting most of what it handed out.

## Explicitly out of scope

- Occlusion. A depth pixel visible to the depth camera but occluded in the colour view gets the
  occluder's label. Correct handling needs a depth test in colour space; the error is confined to
  grazing surfaces and is not addressed here.
- Sub-pixel or class-aware filtering (majority vote in a neighbourhood).
- Per-instance colouring in the view. Instance ids are carried in the data, but the fused view
  colours by class, since Holoviz has one colour per InputSpec. Per-point colour would mean adding a
  colour vertex buffer to `tcn_slang_renderer`'s pointcloud renderable.
- Publishing the labeled clouds (e.g. over zenoh).
