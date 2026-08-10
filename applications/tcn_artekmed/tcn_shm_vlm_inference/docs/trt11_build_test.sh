#!/usr/bin/env bash
#
# Feasibility test: rebuild the Holoscan SDK against TensorRT 11 and re-gate the pipeline.
#
# WHY: TRT 10.9 depresses Grounding DINO confidence scores (0.487 vs PyTorch 0.871, |d|=0.384);
# TRT 11.2 reproduces PyTorch exactly (|d|=0.000, box IoU 1.0000). See
# docs/specs/2026-08-10-tensorrt-upgrade-assessment.md. That defect is the root cause of the
# FP16 rejection and of the corner-case detection misses we have carried as an open issue.
#
# holoinfer (libholoscan_infer) is the only library in the image that links TensorRT, and DA2/DA3
# use it via InferenceOp. Those switches are off for LangSAM testing but the feature is expected
# to stay available, so this script rebuilds the SDK -- which rebuilds holoinfer against TRT 11 --
# rather than side-loading a second TRT.
#
# The SDK checkout is NEVER left modified: the Dockerfile change lives as a patch in THIS repo
# (patches/holoscan-sdk-4.4.0-trt-major-param.patch) and is reverted on exit, including on error
# or Ctrl-C. The patch is a strict generalisation -- it derives the package-name major from the
# existing TENSORRT_CU*_VERSION arg, so the current 10.3/10.16 pins behave exactly as today.
#
# ONE APPROVAL, MANY STEPS: everything runs inside this script so you approve once, not per docker
# command. Re-runnable; use --stage to resume at a step.
#
# Usage:
#   ./trt11_build_test.sh                 # full sequence
#   ./trt11_build_test.sh --stage verify  # just one stage
#   ./trt11_build_test.sh --dry-run       # print the commands, run nothing
#   TRT_VERSION=11.1 ./trt11_build_test.sh
#
set -uo pipefail

# ---------------------------------------------------------------- configuration
SDK_DIR="${SDK_DIR:-/home/ecku/develop/holoscan/holoscan-sdk}"
SDK_EXPECTED_SHA="${SDK_EXPECTED_SHA:-e64af8f270896599ec4c7e53a25acef12dbb3934}"   # v4.4.0
HOLOHUB_DIR="${HOLOHUB_DIR:-/home/ecku/develop/holoscan/holohub-tcn}"
TRT_VERSION="${TRT_VERSION:-11.2}"      # apt-cache madison is grepped for this prefix
CUDA_MAJOR="${CUDA_MAJOR:-12}"          # 11.2.1.2 ships +cuda12.9 AND +cuda13.3; 12 is the smaller delta
APP_REL="applications/tcn_artekmed/tcn_shm_vlm_inference"

# Engines built here go to a SEPARATE tree. The live TRT 10.9 engines must stay loadable by the
# current container until the new one is proven -- engines are TRT-version-locked, so installing
# TRT 11 engines over them would break the working setup with a deserialization error.
ENGINE_SRC_HOST="${ENGINE_SRC_HOST:-/data/models}"
ENGINE_OUT_HOST="${ENGINE_OUT_HOST:-/data/models_trt11}"

DATASET_HOST="${DATASET_HOST:-/home/ecku/develop/artekmed/artekmed_test_data}"
TMP_HOST="${TMP_HOST:-/tmp/tcn}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="${SCRIPT_DIR}/patches/holoscan-sdk-4.4.0-trt-major-param.patch"
LOG_DIR="${LOG_DIR:-${TMP_HOST}/trt11}"
LOG="${LOG_DIR}/build-$(date +%Y%m%d-%H%M%S).log"

STAGE="all"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"
say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*" | tee -a "$LOG"; }
info() { printf '   %s\n' "$*" | tee -a "$LOG"; }
fail() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" | tee -a "$LOG"; exit 1; }
run()  {
  info "\$ $*"
  [[ $DRY_RUN -eq 1 ]] && return 0
  "$@" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------- patch lifecycle
PATCH_APPLIED=0
revert_patch() {
  [[ $PATCH_APPLIED -eq 1 ]] || return 0
  say "Reverting SDK patch (checkout must be left pristine)"
  ( cd "$SDK_DIR" && git apply -R "$PATCH" ) \
    && { PATCH_APPLIED=0; info "reverted"; } \
    || printf '\033[1;31mWARNING: could not revert %s -- check %s manually\033[0m\n' "$PATCH" "$SDK_DIR"
}
trap revert_patch EXIT INT TERM

stage_preflight() {
  say "Preflight"
  [[ -f "$PATCH" ]] || fail "patch not found: $PATCH"
  [[ -d "$SDK_DIR/.git" ]] || fail "not a git checkout: $SDK_DIR"
  local sha; sha="$(cd "$SDK_DIR" && git rev-parse HEAD)"
  info "SDK at $sha"
  [[ "$sha" == "$SDK_EXPECTED_SHA" ]] || \
    info "WARNING: SDK is not the commit this patch was generated against ($SDK_EXPECTED_SHA). The patch may not apply."
  ( cd "$SDK_DIR" && git diff --quiet -- Dockerfile ) || fail "$SDK_DIR/Dockerfile already modified -- refusing to patch on top"
  ( cd "$SDK_DIR" && git apply --check "$PATCH" ) || fail "patch does not apply cleanly to this SDK checkout"
  info "patch applies cleanly (checked, not applied)"
  command -v docker >/dev/null || fail "docker not found"
  local free; free=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc '0-9')
  info "free space on /var/lib/docker: ${free:-?} GiB (an SDK + holohub image build wants >100)"
  [[ -n "${free:-}" && "$free" -lt 60 ]] && info "WARNING: low disk space; SDK builds are large"
  info "TRT target: ${TRT_VERSION} on CUDA ${CUDA_MAJOR}"
  docker run --rm "nvcr.io/nvidia/clara-holoscan/holoscan:v4.4.0-dgpu" \
      bash -lc "apt-get update -qq >/dev/null 2>&1; apt-cache madison libnvinfer${TRT_VERSION%%.*} | grep -m2 '+cuda${CUDA_MAJOR}'" \
      2>/dev/null | tee -a "$LOG" \
    || info "NOTE: could not pre-check apt availability (not fatal; the build will tell us)"
}

stage_sdk() {
  say "Building the Holoscan SDK image against TensorRT ${TRT_VERSION}"
  ( cd "$SDK_DIR" && git apply "$PATCH" ) || fail "git apply failed"
  PATCH_APPLIED=1
  info "patch applied to $SDK_DIR/Dockerfile"

  local ARG="TENSORRT_CU${CUDA_MAJOR}_VERSION=${TRT_VERSION}"
  # build_image passes trailing args through to `docker build`, so --build-arg reaches the
  # tensorrt-dev stage. This stage is where a TRT-11 apt resolution failure would surface.
  run "$SDK_DIR/run" build_image --cuda "$CUDA_MAJOR" --build-arg "$ARG" \
    || fail "SDK builder image failed. If apt could not resolve TRT ${TRT_VERSION}, check the
   tensorrt-dev stage output above; if the ARG was rejected, the patch may need to also change
   the ARG default rather than relying on --build-arg."

  # This compiles the SDK -- including holoinfer -- against TRT 11. It is THE compatibility test:
  # holoinfer uses only enqueueV3/setTensorAddress, none of the binding API removed in TRT 11,
  # so it is expected to compile, but that is an inspection, not a proof.
  say "Compiling the SDK + building the runtime image (this is holoinfer's real TRT-11 test)"
  run "$SDK_DIR/run" build_run_image --cuda "$CUDA_MAJOR" --build-arg "$ARG" \
    || fail "SDK runtime image failed -- if the error is in modules/holoinfer, that is the
   compatibility answer we came for. Capture it; it is a genuine result, not a script bug."

  say "Runtime image tags"
  run docker images --format '{{.Repository}}:{{.Tag}}  {{.CreatedSince}}' \
    | grep -i holoscan | head -10
  info "Note the runtime image name above; pass it to the next stage as BASE_IMG=<name:tag>."
  revert_patch
}

stage_holohub() {
  say "Building the holohub image on the new base"
  [[ -n "${BASE_IMG:-}" ]] || fail "set BASE_IMG=<holoscan runtime image:tag> (see the 'sdk' stage output)"
  run "$HOLOHUB_DIR/holohub" build --base-img "$BASE_IMG" --cuda "$CUDA_MAJOR" tcn_shm_vlm_inference \
    || fail "holohub image build failed against base $BASE_IMG"
}

stage_verify() {
  say "Verifying the new image: TRT version, holoinfer soname, InferenceOp"
  [[ -n "${TEST_IMG:-}" ]] || fail "set TEST_IMG=<holohub image:tag> built by the 'holohub' stage"
  run docker run --rm --gpus all "$TEST_IMG" bash -lc '
    set -e
    echo "python tensorrt : $(python3 -c "import tensorrt as t;print(t.__version__)")"
    echo "libnvinfer      : $(ls /usr/lib/x86_64-linux-gnu/libnvinfer.so.* 2>/dev/null | tr "\n" " ")"
    echo -n "holoinfer links : "
    objdump -p $(ls /opt/nvidia/holoscan/lib/libholoscan_infer.so.4.4.0) | grep -oE "libnvinfer[a-z_]*\.so\.[0-9]+" | sort -u | tr "\n" " "; echo
    python3 -c "from holoscan.operators import InferenceOp; print(\"InferenceOp import OK\")"
  ' || fail "verification failed"
  info "EXPECT: python tensorrt 11.x, libnvinfer.so.11, holoinfer linking libnvinfer*.so.11."
  info "If holoinfer still shows .so.10 the SDK was not actually rebuilt -- the image is stale."
}

stage_engines() {
  say "Rebuilding GDINO + SAM engines against TensorRT ${TRT_VERSION}"
  [[ -n "${TEST_IMG:-}" ]] || fail "set TEST_IMG=<holohub image:tag>"
  mkdir -p "$ENGINE_OUT_HOST/active/groundingdino" "$ENGINE_OUT_HOST/active/sam2"
  # Seed the precision-neutral inputs the build stage reads (ONNX + prompts + parity ref).
  for f in gdino_swint_512x672_b2_tf32.onnx gdino_swint_512x672_b3_tf32.onnx \
           gdino_swint_512x672_parity_ref.npz gdino_swint_prompts.npz; do
    [[ -e "$ENGINE_OUT_HOST/active/groundingdino/$f" ]] || \
      cp -v "$ENGINE_SRC_HOST/active/groundingdino/$f" "$ENGINE_OUT_HOST/active/groundingdino/" 2>&1 | tee -a "$LOG"
  done
  run docker run --rm --gpus all \
      -v "$ENGINE_OUT_HOST:/srv/models" \
      -v "$HOLOHUB_DIR:/workspace/holohub" \
      "$TEST_IMG" bash -lc "
        cd /workspace/holohub/${APP_REL}/docs &&
        python3 gdino_trt_export.py --stage build --from-config ../python/tcn_shm_vlm_inference.yaml \
                --out /srv/models/active/groundingdino
      " || fail "GDINO engine build failed under TRT ${TRT_VERSION}"
  info "EXPECT 'pytorch fidelity [OK] ... |d|=0.000'. That single line is the whole point of the exercise."
  run docker run --rm --gpus all \
      -v "$ENGINE_OUT_HOST:/srv/models" -v "$HOLOHUB_DIR:/workspace/holohub" \
      "$TEST_IMG" bash -lc "
        cd /workspace/holohub/${APP_REL}/docs &&
        python3 sam_trt_export.py --from-config ../python/tcn_shm_vlm_inference.yaml \
                --out /srv/models/active/sam2
      " || info "WARNING: SAM encoder rebuild failed -- GDINO is the one that matters for the fidelity question."
}

stage_gate() {
  say "Harness run against the TRT ${TRT_VERSION} engines"
  [[ -n "${TEST_IMG:-}" ]] || fail "set TEST_IMG=<holohub image:tag>"
  local cfg="${TMP_HOST}/gates/trt11.yaml"
  mkdir -p "${TMP_HOST}/gates" "${TMP_HOST}/harness"
  python3 - "$HOLOHUB_DIR/${APP_REL}/python/tcn_shm_vlm_inference.yaml" "$cfg" <<'PY'
import re, sys, pathlib
s = pathlib.Path(sys.argv[1]).read_text()
s = re.sub(r'^(mask_dump_dir:).*$', r'\1 "/srv/tmp/harness/trt11"', s, flags=re.M)
s = re.sub(r'^(source:).*$', r'\1 "dataset"', s, flags=re.M)
pathlib.Path(sys.argv[2]).write_text(s)
print("wrote", sys.argv[2])
PY
  rm -rf "${TMP_HOST}/harness/trt11"
  run docker run --rm --gpus all --ipc=host --shm-size=16gb \
      -v "$ENGINE_OUT_HOST:/srv/models" -v "$TMP_HOST:/srv/tmp" \
      -v "$HOLOHUB_DIR:/workspace/holohub" \
      -v "$DATASET_HOST:/workspace/volumes/artekmed_test_data" \
      "$TEST_IMG" bash -lc "
        BIN=/workspace/holohub/build/tcn_shm_vlm_inference
        SRC=/workspace/holohub/${APP_REL}/python
        cd \$BIN && PYTHONPATH=\$BIN/python/lib:/workspace/holohub:\$SRC:\$PYTHONPATH \
        python3 \$SRC/tcn_shm_vlm_inference.py -c /srv/tmp/gates/trt11.yaml
      " || fail "harness run failed under TRT ${TRT_VERSION}"
  say "Comparing against the TRT 10.9 baseline"
  info "READ THIS AS A CHANGE, NOT A REGRESSION: TRT 10.9 -> 11.2 is the FIX landing, so masks are"
  info "EXPECTED to differ. Judge per-class INSTANCE COUNTS against what the scene contains."
  info "Do NOT conclude a regression from a low --iou-gate score against the old (wrong) output."
  run python3 "$HOLOHUB_DIR/${APP_REL}/docs/compare_mask_dumps.py" \
      "${TMP_HOST}/harness/g_bd_on_tf32" "${TMP_HOST}/harness/trt11" --top 10
}

case "$STAGE" in
  all)       stage_preflight && stage_sdk && info "Now re-run with --stage holohub after setting BASE_IMG" ;;
  preflight) stage_preflight ;;
  sdk)       stage_preflight && stage_sdk ;;
  holohub)   stage_holohub ;;
  verify)    stage_verify ;;
  engines)   stage_engines ;;
  gate)      stage_gate ;;
  *)         fail "unknown stage: $STAGE (preflight|sdk|holohub|verify|engines|gate|all)" ;;
esac

say "Done. Log: $LOG"
