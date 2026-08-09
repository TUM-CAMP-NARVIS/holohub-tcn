# SAM batched decode across cameras (design)

Replace `SAM.predict_batch_gpu`'s per-camera decode loop with a single batched decoder call over
all boxes from all cameras.

Implements step 4a of [`../dataflow-and-pipelining-roadmap.md`](../dataflow-and-pipelining-roadmap.md).

## Motivation — what the measurement says

From the 2026-08-08/09 pipelining measurement (see
[`2026-08-07-langsam-pipelining-design.md`](./2026-08-07-langsam-pipelining-design.md) §Results):

- `sam` is the stage that **sets the pipelined period** (220.8 ms of a 221.9 ms period) and is the
  largest stage in the monolithic path on the busier GPU (89.9 ms of a 195.0 ms period).
- It issues **~940 `cudaLaunchKernel` calls per tick** and **29–43 `cudaStreamSynchronize` calls
  per tick**.
- Its inflation under pipelining was ~2:1 **non-API** time, i.e. GIL re-acquisition across those
  ~1000 Python-driven launches.

The decode loop is therefore the single item standing in front of both failure modes at once: it
supplies the launches that fill the one CUDA stream's queue *and* the GIL round-trips that made
pipelining lose. Reducing the loop from `num_images` iterations to one attacks both.

This step needs **no engine export and no C++** — it is a restructure of existing Python. That is
deliberate: it tells us how much of SAM's cost is loop structure versus the model itself, which is
the number that decides whether step 4b (exporting the mask decoder to TensorRT) is worth its cost.

## Why it is possible

`SAM2ImagePredictor._predict` is already batched over dim 0; it is merely *fed* one image at a
time. The hinge is in `sam2/modeling/sam/mask_decoder.py` (`predict_masks`, ~line 199):

```python
if repeat_image:
    src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
else:
    assert image_embeddings.shape[0] == tokens.shape[0]
    src = image_embeddings
```

Today the caller passes one image's embedding with `repeat_image=True` so it is broadcast across
that image's boxes. Instead we pass `image_embeddings` of shape `(M, C, 64, 64)` — where `M` is the
total box count across all cameras and row `j` is the embedding of box `j`'s source image — with
`repeat_image=False`. `high_res_feats` gathers identically. The transformer's rows are independent,
so this is the same computation, reassociated.

Three upstream facts were verified against the vendored SAM 2 checkout and make the batched path
safe:

| fact | consequence |
|---|---|
| `max_hole_area=0.0`, `max_sprinkle_area=0.0` (predictor defaults) | `postprocess_masks` reduces to `masks.float()` + one `F.interpolate`; the `get_connected_components` branches are dead. Fully batched over dim 0. |
| `dynamic_multimask_via_stability=False` (mask-decoder default) | not on the path; and `_dynamic_multimask_via_stability` is row-independent anyway, so it would batch correctly if ever enabled. |
| `postprocess_masks(masks, orig_hw)` takes ONE `orig_hw` | batching requires a uniform camera resolution. See §Fallback. |

## Design

### 1. Gather, one call, split

In `langsam_common.py`, `predict_batch_gpu` gains a batched path. Shape of it:

```
counts   = [b.shape[0] for b in xyxy]                  # host metadata on cupy arrays -> no sync
box_img  = repeat_interleave(arange(num_images), counts)   # (M,) int64, on device
boxes    = cat([as_tensor(b) for b in xyxy])               # (M,4)
unnorm   = transform_boxes(boxes, normalize=True, orig_hw=orig_hw0)
sparse, dense = sam_prompt_encoder(points=(unnorm.reshape(-1,2,2), box_labels), ...)
low_res, iou  = sam_mask_decoder(image_embed[box_img], ..., repeat_image=False,
                                high_res_features=[f[box_img] for f in high_res_feats])
masks    = postprocess_masks(low_res, orig_hw0) > mask_threshold
per_image = torch.split(masks, counts)                     # back to the per-camera contract
```

`image_embed[box_img]` is integer-array indexing on the GPU — no host round-trip, no sync.

The return contract is unchanged: a list of cupy `(K,H,W)` uint8 masks and a list of cupy `(K,)`
scores, one entry per input image, in input order. A short loop remains for the
`torch -> cupy` conversion (2 ops per camera); it is not the loop that costs anything.

### 2. This replicates `_predict`'s body inline

The batched path cannot call `p._predict`, because `_predict`'s signature commits to a single
`img_idx`. It therefore duplicates `_predict`'s sequence (prompt encode → mask decode →
postprocess → threshold) with the gather substituted.

That is a real maintenance cost: a SAM 2 upgrade that changes `_predict` will not change our copy.
Mitigation is to name the upstream function and the vendored commit in a comment directly above the
batched path, and to keep the loop path as the reference implementation rather than deleting it —
the toggle (§4) means the original stays exercised and A/B-able, so drift is detectable rather than
silent.

### 3. Correctness gate: near-identical, NOT bitwise

Batching changes GEMM shapes, so cuBLAS may pick different tile and split-k configurations, so
floating-point reductions associate differently. Under `bf16` autocast (8 mantissa bits) those
differences are small but real. Masks are then **thresholded** (`masks > mask_threshold`), so a
pixel whose logit sits within epsilon of the threshold can flip.

The gate is therefore **not** byte-identity. It is:

- per-mask IoU between loop and batched output **≥ 0.999**, and
- no change in the number of masks or their label assignment, and
- no visible difference in the rendered panoptic map.

Claiming bitwise identity here would be wrong, and a test asserting it would fail spuriously. This
differs from the pipelining refactor, which *was* required to be byte-identical because it moved no
arithmetic.

### 4. Opt-in, with the loop retained

`langsam_inference.sam_batched_decode: true|false`, **default `false`**, following the
`gdino_backend` / `sam_backend` / `pipelined` precedent: A/B-able in one config line, instantly
revertible, and the reference path stays live for the identity comparison.

### 5. Fallback

The batched path requires all cameras in a worker to share a resolution, because `postprocess_masks`
and `transform_boxes` each take a single `orig_hw`. `p._orig_hw` is host-side ints, so the check is
free. When `len(set(p._orig_hw)) > 1`, log once at WARNING and use the loop.

This is currently always satisfied (all five cameras are 2048×1536) — but `detect_batch` carefully
carries per-frame `(H0, W0)` through to pixel scaling, so the per-camera contract does hold up to
this point and silently breaking it would be a real regression. Grouping by resolution is possible
and deliberately out of scope.

## Expected effect

The decoder's total GPU pixel work is unchanged — the same `M` masks are upsampled to the same
resolution. The win is launches and Python:

- decoder-path launches drop by roughly the camera count (3 calls → 1 on the busier worker)
- the per-image `_prep_prompts` / `_predict` Python disappears

If `sam` on dev1 goes 89.9 → ~50 ms, the monolithic period goes ~195 → ~155 ms (≈6.4 fps, +26%).
That is an estimate from the launch-count ratio, not a measurement, and the honest downside case is
that the decoder was already GPU-bound at these batch sizes and little changes — which would itself
be the answer on whether to fund 4b.

## Testing

| test | where | gate |
|---|---|---|
| existing host suites | host, numpy | 8/8, 11/11, 7/7, 11/11 unchanged |
| `plan_batch_padding`-style pure helpers for `counts`/`box_img` construction | host | new; the index arithmetic is the part that can be got wrong without a GPU |
| `sam_batched_decode: false` unchanged | container | proves the toggle is inert |
| loop vs batched, same scene | container | **the gate**: per-mask IoU ≥ 0.999, same mask count and labels |
| `sam` NVTX wall + launch count per tick | container, nsys | vs 89.9 ms and ~940 launches on dev1 |
| period and fps | container, nsys | vs 195.0 ms / 5.13 fps (5-prompt config) |

Note the operators and `SAM` itself import torch/cupy and **cannot be host-tested**; only the pure
index arithmetic can. As with the pipelining work, correctness comes from the container A/B.

## Risks

- **Mask flips at the threshold.** Mitigated by the IoU gate rather than pretended away. If IoU
  comes in below 0.999, that is a finding about bf16 sensitivity, not a bug to paper over — report
  it before flipping the default.
- **Upstream drift** from the inlined `_predict` body (§2).
- **Memory.** One `(M,1,2048,1536)` float mask tensor for all cameras at once instead of one
  `(K,1,2048,1536)` per camera — for M=10, ~126 MB transiently. Should be fine next to the engines;
  worth watching if box counts spike.
- **Zero-box images.** `GdinoOp` only forwards detecting cameras, so `counts` should never contain
  0, but `repeat_interleave` and `split` both handle 0 correctly and the tests should cover it.

## Out of scope

- Step 4b: exporting the SAM 2 mask decoder to TensorRT (this step's result decides its value).
- Step 2: the panoptic CUDA kernel — a C++/CUDA operator matching `operators/tcn_artekmed/`.
- The unverified FP32 `sm86_xmma_gemm_f32f32_tf32f32` observation (12.3 ms/tick under bf16
  autocast); separate investigation.
