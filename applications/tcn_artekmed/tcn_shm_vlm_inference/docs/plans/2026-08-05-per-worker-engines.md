# Per-worker engine batches + `gpu_workers` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Each LangSAM worker runs an engine built for its own camera count, and the camera→GPU split lives in one global `gpu_workers` node that the app and both offline builders read.

**Architecture:** Batch is *derived* from a worker's camera count, never written down. Engine paths become `{batch}` templates resolved per worker; a path without the placeholder formats to itself, so single-engine configs keep working. Both builders gain `--from-config`, reading the same node and building exactly the distinct batches the split needs.

**Spec:** [`../specs/2026-08-05-per-worker-engines-design.md`](../specs/2026-08-05-per-worker-engines-design.md)

## Global Constraints

- Host tests: no pytest, numpy-only, plain `__main__` PASS/FAIL runner, run as `python3 tests/test_<name>.py` from `applications/tcn_artekmed/tcn_shm_vlm_inference/python`.
- `langsam_helpers.py` must stay importable without torch/cupy/holoscan/tensorrt.
- Commit style `feat(tcn_artekmed): ...` / `fix(...)`, ending with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Never `git add -A`; stage only the files a task names. The tree carries unrelated build/IDE artifacts.
- **Baselines:** GPU 0 total **172.8 ms**, GPU 1 **155.9 ms**, period **189.9 ms**, **5.27 fps**. GPU 0's `gdino` 94.9 / `sam` 72.8 are the numbers this work should move.

---

### Task P1: Pure helpers — topology, derived batch, path templates

**Files:**
- Modify: `python/langsam_helpers.py`
- Create: `python/tests/test_gpu_workers.py`

**Interfaces produced:**
- `resolve_workers(cfg, all_color_cameras)` — unchanged name/return, now also accepts the `gpu_workers` node shape.
- `worker_batch(worker) -> int`
- `distinct_batches(workers) -> list[int]`
- `worker_engine_path(template, batch) -> str`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_gpu_workers.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Host tests for the gpu_workers topology helpers (numpy-free)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langsam_helpers import (resolve_workers, worker_batch, distinct_batches,
                             worker_engine_path)

CFG = {"workers": [
    {"device": 0, "cameras": ["camera01_colorimage", "camera02_colorimage"]},
    {"device": 1, "cameras": ["camera03_colorimage", "camera04_colorimage",
                              "camera05_colorimage"]},
]}
ALL = ["camera0%d_colorimage" % i for i in range(1, 6)]


def test_resolve_workers_reads_the_node():
    w = resolve_workers(CFG, ALL)
    assert [x["device"] for x in w] == [0, 1]
    assert len(w[0]["cameras"]) == 2 and len(w[1]["cameras"]) == 3


def test_empty_config_falls_back_to_one_worker_on_device_0():
    for cfg in (None, {}, {"workers": []}):
        w = resolve_workers(cfg, ALL)
        assert w == [{"device": 0, "cameras": ALL}], cfg


def test_batch_is_derived_from_the_camera_count():
    w = resolve_workers(CFG, ALL)
    assert worker_batch(w[0]) == 2 and worker_batch(w[1]) == 3


def test_distinct_batches_are_sorted_and_deduplicated():
    assert distinct_batches(resolve_workers(CFG, ALL)) == [2, 3]
    same = [{"device": 0, "cameras": ["a", "b"]}, {"device": 1, "cameras": ["c", "d"]}]
    assert distinct_batches(same) == [2]


def test_engine_path_substitutes_batch():
    assert worker_engine_path("/m/gdino_b{batch}_tf32.engine", 2) == "/m/gdino_b2_tf32.engine"


def test_engine_path_without_placeholder_is_a_noop():
    """Lets a single-engine config keep working while only some batches exist."""
    p = "/m/gdino_b3_tf32.engine"
    assert worker_engine_path(p, 2) == p


def test_engine_path_rejects_a_still_templated_result():
    try:
        worker_engine_path("/m/gdino_{size}_b{batch}.engine", 2)
    except ValueError as e:
        assert "size" in str(e) or "{" in str(e)
        return
    raise AssertionError("expected ValueError for an unresolved placeholder")


def test_engine_path_accepts_none():
    assert worker_engine_path(None, 2) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except AssertionError as e:
            bad += 1; print("FAIL", fn.__name__, repr(e))
    print(f"{len(fns)-bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_gpu_workers.py
```
Expected: `ImportError: cannot import name 'worker_batch'`.

- [ ] **Step 3: Implement**

In `python/langsam_helpers.py`, replace the `resolve_workers` docstring's first line and append the three new helpers after it:

```python
def resolve_workers(cfg, all_color_cameras):
    """Resolve the per-GPU worker assignment from the `gpu_workers` node.

    A worker is ``{"device": int, "cameras": [port, ...]}``. Empty/missing ``workers`` ->
    a single worker on device 0 processing all color cameras. A worker's ENGINE BATCH is its
    camera count (see `worker_batch`) -- it is derived, never configured, so the two cannot
    disagree.
    """
    workers = (cfg or {}).get("workers") or []
    if not workers:
        return [{"device": 0, "cameras": list(all_color_cameras)}]
    return [
        {"device": int(w.get("device", 0)), "cameras": list(w.get("cameras") or [])}
        for w in workers
    ]


def worker_batch(worker):
    """The engine batch a worker needs: one slice per camera it owns."""
    return len(worker["cameras"])


def distinct_batches(workers):
    """Sorted, deduplicated engine batches a worker list requires -> what to build."""
    return sorted({worker_batch(w) for w in workers if worker_batch(w) > 0})


def worker_engine_path(template, batch):
    """Resolve an engine path template for one worker's batch.

    `{batch}` is substituted; a template WITHOUT it formats to itself, so a single-engine
    configuration keeps working while only some batches exist. Raises if the result still
    contains a placeholder -- that means a typo'd field name, which would otherwise surface
    much later as a confusing missing-file error.
    """
    if template is None:
        return None
    out = str(template).replace("{batch}", str(int(batch)))
    if "{" in out or "}" in out:
        raise ValueError(f"unresolved placeholder in engine path template: {template!r} "
                         f"-> {out!r} (only {{batch}} is substituted)")
    return out
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 tests/test_gpu_workers.py && python3 tests/test_prompt_remap.py && python3 tests/test_gdino_postprocess.py && python3 tests/test_langsam_multicam.py
```
Expected: `8/8 passed`, then `11/11`, `7/7`, `9/9`.

If `test_langsam_multicam.py` fails because it calls `resolve_workers` with the old key name, fix the TEST to use the `gpu_workers` node shape (the shape is `{"workers": [...]}` either way, so it should not) — do not change the helper's behaviour to accommodate it.

- [ ] **Step 5: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_helpers.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tests/test_gpu_workers.py
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): derive engine batch from the gpu_workers topology

worker_batch/distinct_batches make a worker's engine batch a derived property of its
camera list rather than a separately configured number, so the two cannot disagree.
worker_engine_path resolves {batch} per worker and raises on an unresolved
placeholder; a template without {batch} formats to itself, so single-engine configs
keep working and the migration is incremental.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task P2: Runtime + config migration

**Files:**
- Modify: `python/langsam_multicam_fragment.py`
- Modify: `python/tcn_shm_vlm_inference.yaml`

**Interfaces consumed:** `resolve_workers`, `worker_batch`, `worker_engine_path` (Task P1).

- [ ] **Step 1: Migrate the YAML**

Replace the whole `langsam_multicam:` block (its comment and its `workers:` list) with a new
top-level node placed immediately **before** `langsam_inference:`:

```yaml
# ---------------------------------------------------------------------------
# GPU topology -- the single source of truth for which cameras run on which GPU.
# Read by the application AND by the offline engine builders
# (docs/gdino_trt_export.py / docs/sam_trt_export.py, --from-config), so the engines that
# exist always match the split that runs.
#
# A worker's ENGINE BATCH is its camera count -- derived, never configured. GPU 0 gets the
# smaller share because it also drives Holoviz, the collector and colorize.
# Changing a split means rebuilding that batch; the app fails fast at startup with the command.
# ---------------------------------------------------------------------------
gpu_workers:
  workers:
    - device: 0
      cameras: ["camera01_colorimage", "camera02_colorimage"]
    - device: 1
      cameras: ["camera03_colorimage", "camera04_colorimage", "camera05_colorimage"]
```

Then change the two engine keys in `langsam_inference` to templates:

```yaml
  gdino_trt_engine: "/srv/models/active/groundingdino/gdino_swint_512x672_b{batch}_tf32.engine"
```
```yaml
  sam_trt_engine: "/srv/models/active/sam2/sam2.1_hiera_tiny_encoder_b{batch}_fp16.engine"
```

Leave `langsam_multicam_holoviz` alone — it is unrelated to the worker split.

- [ ] **Step 2: Read the new node**

In `langsam_multicam_fragment.py`'s `compose`, change

```python
        multicam_cfg = self._get("langsam_multicam") or {}
```
to
```python
        multicam_cfg = self._get("gpu_workers") or {}
```

(the variable feeds `resolve_workers`, whose accepted shape is unchanged).

- [ ] **Step 3: Resolve engine paths per worker**

Add `worker_batch` and `worker_engine_path` to the `from langsam_common import (` line's helper
re-exports if not already available there; they are defined in `langsam_helpers` and
`langsam_common` re-exports that module's names.

In `LangSamBatchOp.__init__`, immediately after `self.cameras = list(cameras)`, add:

```python
        # This worker's engines are built for exactly its camera count. Resolving the paths
        # here -- rather than passing one fixed path for every worker -- is what lets a
        # 2-camera worker stop paying 3-camera cost.
        self.batch = len(self.cameras)
        gdino_engine = worker_engine_path(langsam_cfg.get("gdino_trt_engine"), self.batch)
        sam_engine = worker_engine_path(langsam_cfg.get("sam_trt_engine"), self.batch)
        for kind, path in (("GDINO", gdino_engine), ("SAM", sam_engine)):
            backend = langsam_cfg.get("gdino_backend" if kind == "GDINO" else "sam_backend",
                                      "pytorch")
            if backend == "trt" and path and not os.path.exists(path):
                raise FileNotFoundError(
                    f"{kind} engine for batch {self.batch} not found: {path}\n"
                    f"This worker owns {self.batch} cameras, so it needs a batch-{self.batch} "
                    f"engine. Build it with --from-config, or for this batch alone:\n"
                    f"  GDINO (host then container): gdino_trt_export.py --stage export "
                    f"--batch {self.batch} ... ; --stage build --batch {self.batch} ...\n"
                    f"  SAM (container):             sam_trt_export.py --batch {self.batch} ...")
```

Add `import os` at the top of the file if it is not already imported.

Then use the resolved paths instead of the raw config values: change
`sam_trt_engine=langsam_cfg.get("sam_trt_engine")` to `sam_trt_engine=sam_engine`, and
`langsam_cfg["gdino_trt_engine"]` to `gdino_engine`.

- [ ] **Step 4: Verify**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/python && python3 -m py_compile langsam_multicam_fragment.py && echo "compile OK"
python3 - <<'PY'
import yaml
from langsam_helpers import resolve_workers, distinct_batches, worker_engine_path, worker_batch
c = yaml.safe_load(open("tcn_shm_vlm_inference.yaml"))
w = resolve_workers(c["gpu_workers"], [])
li = c["langsam_inference"]
print("workers        :", [(x["device"], worker_batch(x)) for x in w])
print("batches to build:", distinct_batches(w))
for x in w:
    b = worker_batch(x)
    print(f"  device {x['device']} b{b}: {worker_engine_path(li['gdino_trt_engine'], b).split('/')[-1]}"
          f" | {worker_engine_path(li['sam_trt_engine'], b).split('/')[-1]}")
assert "langsam_multicam" not in c, "old node still present"
PY
python3 tests/test_gpu_workers.py && python3 tests/test_langsam_multicam.py
```
Expected: `compile OK`; workers `[(0, 2), (1, 3)]`; batches `[2, 3]`; the four resolved
filenames carrying `_b2_` and `_b3_`; both suites pass.

- [ ] **Step 5: Commit**

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/python/langsam_multicam_fragment.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/python/tcn_shm_vlm_inference.yaml
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): per-worker engine paths from a global gpu_workers node

The camera->GPU split moves out of langsam_multicam into a top-level gpu_workers
node that the app and the offline builders both read, and each worker resolves its
own engine paths from its camera count via the {batch} templates. A 2-camera worker
therefore stops padding to 3 and paying 3-camera cost on both stages.

A missing engine for a worker's batch fails at construction with the build command,
rather than deep inside compute() on the first tick.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task P3: `--from-config` in both builders

**Files:**
- Modify: `docs/sam_trt_export.py`
- Modify: `docs/gdino_trt_export.py`
- Modify: `docs/sam_trt_export.md`, `docs/gdino_trt_export.md`

- [ ] **Step 1: Shared resolution in both tools**

Each tool already inserts `../python` on `sys.path`. In **both**, add near the other imports:

```python
from langsam_helpers import resolve_workers, distinct_batches      # noqa: E402
```

and add this helper to each:

```python
def batches_from_config(config_path):
    """Distinct engine batches the configured GPU split needs.

    Reading the same `gpu_workers` node the application reads is the point: the engines that
    get built cannot drift from the split that runs.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    workers = resolve_workers(cfg.get("gpu_workers"), [])
    batches = distinct_batches(workers)
    if not batches:
        raise SystemExit(f"no workers with cameras in {config_path}: nothing to build")
    print(f"gpu_workers -> {[(w['device'], len(w['cameras'])) for w in workers]} "
          f"-> building batches {batches}")
    return batches
```

- [ ] **Step 2: Wire it into `sam_trt_export.py`**

Add the argument next to `--batch`:

```python
    ap.add_argument("--from-config", default=None,
                    help="path to tcn_shm_vlm_inference.yaml; builds one engine per distinct "
                         "worker camera count in its gpu_workers node (mutually exclusive "
                         "with --batch)")
```

In `main`, replace the single-batch flow with a loop. Right after `args = ap.parse_args()`:

```python
    if args.from_config:
        batches = batches_from_config(args.from_config)
    else:
        batches = [int(args.batch)]
```

Then wrap everything from `tag = ...` to the final `print` in `for batch in batches:`, using
`batch` in place of `args.batch` throughout (the tag, `export_onnx`, `build_engine`,
`make_test_image`, `slice_gate`). Build the model and wrapper **once**, before the loop — they
do not depend on the batch — but recompute `images`/`batch_in`/`ref_fp32`/`ref_bf16` inside it,
since those are batch-shaped.

- [ ] **Step 3: Wire it into `gdino_trt_export.py`**

Same argument, and both stages iterate. In `main`, after computing `batches` the same way,
loop the per-batch body of whichever stage was selected: `stage_export` for each batch (each
writes its own `_b<N>_` ONNX) and `stage_build` for each batch. The prompts npz and parity ref
are batch-independent and must be written **once**, not per batch — check `stage_export` and
hoist those writes out of the loop if they are inside it.

- [ ] **Step 4: Verify both tools**

```bash
cd applications/tcn_artekmed/tcn_shm_vlm_inference/docs
python3 -m py_compile sam_trt_export.py gdino_trt_export.py && echo "compile OK"
~/develop/vision/GroundingDINO/.venv-gdino-export/bin/python gdino_trt_export.py --stage build --help 2>&1 | grep -A3 -- "--from-config"
```
Expected: `compile OK`, and the help text for `--from-config`.

Then confirm the batch resolution works against the real config without building anything:

```bash
python3 -c "
import sys; sys.path.insert(0,'../python')
import importlib.util as u
s=u.spec_from_file_location('t','sam_trt_export.py'); m=u.module_from_spec(s)
import unittest.mock as mock
with mock.patch.dict(sys.modules, {'tensorrt': mock.MagicMock(), 'torch': mock.MagicMock(), 'torch.nn': mock.MagicMock(), 'langsam_common': mock.MagicMock()}):
    s.loader.exec_module(m)
    print(m.batches_from_config('../python/tcn_shm_vlm_inference.yaml'))"
```
Expected: the worker list and `building batches [2, 3]`.

- [ ] **Step 5: Document and commit**

Add a `--from-config` section to both `.md` files: it reads the app's `gpu_workers` node and
builds one engine per distinct worker camera count, which is how the engines are kept in sync
with the split. Note that GDINO needs `--stage export` (host) and `--stage build` (container)
run with the same `--from-config`.

```bash
git add applications/tcn_artekmed/tcn_shm_vlm_inference/docs/sam_trt_export.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.py \
        applications/tcn_artekmed/tcn_shm_vlm_inference/docs/sam_trt_export.md \
        applications/tcn_artekmed/tcn_shm_vlm_inference/docs/gdino_trt_export.md
git commit -m "$(cat <<'EOF'
feat(tcn_artekmed): --from-config builds the batches the GPU split needs

Both builders read the same gpu_workers node the application reads and build one
engine per distinct worker camera count, so the engines that exist cannot drift from
the split that runs. --batch remains for building a single engine by hand.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task P4: Build, validate, measure

Verification only. Needs the host GroundingDINO venv and the running container.

- [ ] **Step 1: SAM engines (container, one command)**

```bash
python3 /workspace/holohub/applications/tcn_artekmed/tcn_shm_vlm_inference/docs/sam_trt_export.py \
  --from-config /workspace/holohub/applications/tcn_artekmed/tcn_shm_vlm_inference/python/tcn_shm_vlm_inference.yaml \
  --out /srv/models/active/sam2
```
Expected: `building batches [2, 3]`, then for each batch all three gates pass and the engine
installs. The b3 engine is rebuilt; that is fine and idempotent.

- [ ] **Step 2: GDINO engines (host export, then container build)**

```bash
# host, in the wingdzero checkout
PYTHONPATH=$PWD ~/develop/vision/GroundingDINO/.venv-gdino-export/bin/python gdino_trt_export.py \
  --stage export --prompts floor person --hw 512 672 \
  --from-config /home/ecku/develop/holoscan/holohub-tcn/applications/tcn_artekmed/tcn_shm_vlm_inference/python/tcn_shm_vlm_inference.yaml \
  --out /data/models/active/groundingdino --parity-image images/in/person.jpg
# container
python3 /workspace/holohub/.../docs/gdino_trt_export.py --stage build --hw 512 672 \
  --from-config /workspace/holohub/.../python/tcn_shm_vlm_inference.yaml \
  --out /srv/models/active/groundingdino
```
Expected: `_b2_` and `_b3_` ONNX and engines, each passing its slice-consistency gate.

**STOP if a batch-2 slice-consistency gate fails** — that would mean the batch-2 trace baked
something wrong; report rather than working around it.

- [ ] **Step 3: Run and measure**

Run the app, confirm masks unchanged, then profile and compare: GPU 0 total vs **172.8 ms**
(its `gdino` 94.9 and `sam` 72.8 are what should fall), GPU 1 vs **155.9 ms**, period vs
**189.9 ms**, fps vs **5.27**.

- [ ] **Step 4: Record**

Append `## Results` to the spec with the measured stage times, period and fps; update
`../optimization-playbook.md` §4 with the new arc entry and add the padding-cost lesson (a
fixed batch shared across unequal workers taxes the smaller one on every stage). Commit.

---

## Self-Review notes

- **Spec coverage:** §1 topology node → P1 Step 3 + P2 Step 1; §2 templates → P1 (`worker_engine_path`) + P2 Steps 1, 3; §3 runtime + fail-fast → P2 Step 3; §4 builders → P3. Testing table → P1 Steps 1-4, P2 Step 4, P4. Risks → P2 Step 3 (missing engine), P4 Step 2 (GDINO host re-export, stop condition).
- **Naming consistency:** `worker_batch`, `distinct_batches`, `worker_engine_path` used identically in P1 (defined), P2 (runtime) and P3 (builders). `gpu_workers` is the node name in the yaml, the fragment's `_get`, and both builders.
- **Ordering:** P1 → P2 and P1 → P3 (both need the helpers); P2 and P3 are independent of each other; P4 needs all three.
