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
#
# Stage order: preflight -> sdk -> verify -> engines -> gate. The `holohub` stage is OPTIONAL --
# the sdk stage's overlay is built on the existing HoloHub image, so it is already runnable.
#   ./trt11_build_test.sh --dry-run       # print the commands, run nothing
#   TRT_VERSION=11.1 ./trt11_build_test.sh
#
set -uo pipefail

# ---------------------------------------------------------------- configuration
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
ENV_FILE="${TCN_ALL_ENV_FILE:-${PROJECT_ROOT}/.tcn_all_env}"

# Load the project-local configuration before applying defaults. The file uses normal shell
# assignments, so values can be shared with other TCN scripts and can contain spaces when quoted.
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# These defaults follow the repository layout. Override them in .tcn_all_env for a different
# checkout layout or machine.
SDK_DIR="${SDK_DIR:-${PROJECT_ROOT}/../../holoscan-sdk}"
SDK_EXPECTED_SHA="${SDK_EXPECTED_SHA:-e64af8f270896599ec4c7e53a25acef12dbb3934}"   # v4.4.0
HOLOHUB_DIR="${HOLOHUB_DIR:-${PROJECT_ROOT}}"
TRT_VERSION="${TRT_VERSION:-11.2}"      # apt-cache madison is grepped for this prefix
CUDA_MAJOR="${CUDA_MAJOR:-13}"          # 11.2.1.2 ships +cuda12.9 AND +cuda13.3; 12 is the smaller delta
TRT_FULL="${TRT_FULL:-11.2.1.2}"        # exact apt/pip version for the packaging overlay
TRT_CUDA_MINOR="${TRT_CUDA_MINOR:-13.3}"  # the +cudaX.Y suffix the TRT packages carry
TRT11_BASE_IMG="${TRT11_BASE_IMG:-holoscan-trt11:4.4.0-cu${CUDA_MAJOR}}"

# Which engine batches to build. Empty = derive from the ACTIVE gpu_workers profile via
# --from-config. That is a trap when you run more than one camera topology: building with the
# 4-camera (2+2) profile active yields batch 2 ONLY, and switching to the live 5-camera (2/3)
# split then fails at startup with "GDINO engine for batch 3 not found". Set ENGINE_BATCHES to
# build every batch any topology needs, regardless of which profile happens to be active:
#   ENGINE_BATCHES="2 3" ./trt11_build_test.sh --stage engines
ENGINE_BATCHES="${ENGINE_BATCHES:-}"
APP_REL="applications/tcn_artekmed/tcn_all"

# Engines built here go to a SEPARATE tree. The live TRT 10.9 engines must stay loadable by the
# current container until the new one is proven -- engines are TRT-version-locked, so installing
# TRT 11 engines over them would break the working setup with a deserialization error.
ENGINE_SRC_HOST="${ENGINE_SRC_HOST:-/data/models}"
ENGINE_OUT_HOST="${ENGINE_OUT_HOST:-/data/models_trt11}"

DATASET_HOST="${DATASET_HOST:-${PROJECT_ROOT}/../../artekmed/artekmed_test_data}"
TMP_HOST="${TMP_HOST:-/tmp/tcn}"

# Resolve relative values from .tcn_all_env against the repository root, rather than against the
# caller's current working directory. This makes the script safe to invoke from any directory.
project_path() {
  if [[ "$1" == /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s/%s\n' "$PROJECT_ROOT" "$1"
  fi
}

SDK_DIR="$(project_path "$SDK_DIR")"
HOLOHUB_DIR="$(project_path "$HOLOHUB_DIR")"
ENGINE_SRC_HOST="$(project_path "$ENGINE_SRC_HOST")"
ENGINE_OUT_HOST="$(project_path "$ENGINE_OUT_HOST")"
DATASET_HOST="$(project_path "$DATASET_HOST")"
TMP_HOST="$(project_path "$TMP_HOST")"
LOG_DIR="$(project_path "${LOG_DIR:-${TMP_HOST}/trt11}")"

# HolovizOp needs a real display -- the app dies with "Failed to initialize glfw" otherwise, and
# the harness app builds Holoviz operators even in dataset mode. These mirror what `./holohub run`
# passes (read off the working container): DISPLAY, the X socket, and an xauth file.
DISPLAY_VAL="${DISPLAY:-:1}"
XAUTH_FILE="${XAUTH_FILE:-$(ls -t /tmp/.docker.xauth-* 2>/dev/null | head -1)}"
PATCH="${SCRIPT_DIR}/patches/holoscan-sdk-4.4.0-trt-major-param.patch"
# holoinfer does not compile against TRT 11 unmodified: BuilderFlag::kFP16 and
# kPREFER_PRECISION_CONSTRAINTS were REMOVED (they were deprecated in 10.12). Found by the
# 2026-08-10 build; two enum constants in one file, guarded the same way the file already guards
# another TRT deprecation. This is the "does holoinfer survive TRT 11" answer.
PATCH_HOLOINFER="${SCRIPT_DIR}/patches/holoscan-sdk-4.4.0-holoinfer-trt11.patch"
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
  # `git checkout` rather than `git apply -R`: the sdk stage makes TWO edits (the patch and the
  # ARG-default injection below), and preflight has already asserted Dockerfile was pristine.
  ( cd "$SDK_DIR" && git checkout -- Dockerfile modules/holoinfer ) \
    && { PATCH_APPLIED=0; info "reverted (git checkout -- Dockerfile)"; } \
    || printf '\033[1;31mWARNING: could not revert -- run: cd %s && git checkout -- Dockerfile modules/holoinfer\033[0m\n' "$SDK_DIR"
}
trap revert_patch EXIT INT TERM

stage_preflight() {
  say "Preflight"
  [[ -f "$PATCH" ]] || fail "patch not found: $PATCH"
  [[ -f "$PATCH_HOLOINFER" ]] || fail "patch not found: $PATCH_HOLOINFER"
  [[ -d "$SDK_DIR/.git" ]] || fail "not a git checkout: $SDK_DIR"
  local sha; sha="$(cd "$SDK_DIR" && git rev-parse HEAD)"
  info "SDK at $sha"
  [[ "$sha" == "$SDK_EXPECTED_SHA" ]] || \
    info "WARNING: SDK is not the commit this patch was generated against ($SDK_EXPECTED_SHA). The patch may not apply."
  ( cd "$SDK_DIR" && git diff --quiet -- Dockerfile modules/holoinfer ) \
    || fail "$SDK_DIR has local modifications to Dockerfile or modules/holoinfer -- refusing to patch on top"
  ( cd "$SDK_DIR" && git apply --check "$PATCH" "$PATCH_HOLOINFER" ) \
    || fail "patches do not apply cleanly to this SDK checkout"
  info "both patches apply cleanly (checked, not applied)"
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
  ( cd "$SDK_DIR" && git apply "$PATCH" "$PATCH_HOLOINFER" ) || fail "git apply failed"
  PATCH_APPLIED=1
  info "applied: TRT-major parameterisation + holoinfer TRT-11 guards"

  # Set the ARG DEFAULT rather than passing --build-arg. `run build_run_image` forwards its
  # arguments to an internal `build`, which hands them to `docker run` -- so --build-arg dies with
  # "unknown flag" there, and on `build_image` it silently failed to override (the first attempt
  # produced a TRT 10.3 image, the default). Editing the default sidesteps the CLI entirely.
  local ARGLINE="TENSORRT_CU${CUDA_MAJOR}_VERSION"
  # Replace the WHOLE line: the original carries a trailing comment ("TRT 10.3 is the last
  # version that supports CUDA 12 on sbsa 22.04") that would be stale and misleading against a
  # bumped value, and anyone inspecting the checkout mid-build should see this is temporary.
  run sed -i -E "s|^ARG ${ARGLINE}=.*|ARG ${ARGLINE}=${TRT_VERSION}  # TEMPORARY (trt11_build_test.sh) -- reverted on exit|" \
      "$SDK_DIR/Dockerfile" || fail "could not set ${ARGLINE} default"
  local got; got=$(grep -E "^ARG ${ARGLINE}=" "$SDK_DIR/Dockerfile")
  info "Dockerfile now has: ${got}"
  [[ "$got" == *"=${TRT_VERSION}"* ]] || fail "ARG default not applied: ${got}"

  # ./run must execute with CWD inside the SDK checkout: it derives image tags from `git rev-parse`
  # in the CURRENT directory, so running it from elsewhere tagged the image with holohub's sha.
  # CUDA major goes through the documented env var, avoiding option-ordering ambiguity.
  say "Building the SDK builder image (this is where TRT ${TRT_VERSION} apt resolution is proven)"
  ( cd "$SDK_DIR" && HOLOSCAN_CUDA_MAJOR_VERSION="$CUDA_MAJOR" ./run build_image ) 2>&1 | tee -a "$LOG"
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || fail "SDK builder image failed -- check the tensorrt-dev stage output above"

  # `./run build` compiles AND installs into install-cu12-x86_64/. This is holoinfer's real
  # TRT test: at 2026-08-10 it failed here on BuilderFlag::kFP16 / kPREFER_PRECISION_CONSTRAINTS,
  # which PATCH_HOLOINFER now guards; with that patch all 1080 targets build.
  say "Compiling + installing the SDK (holoinfer's real TRT-${TRT_VERSION} test)"
  ( cd "$SDK_DIR" && HOLOSCAN_CUDA_MAJOR_VERSION="$CUDA_MAJOR" ./run build ) 2>&1 | tee -a "$LOG"
  [[ "${PIPESTATUS[0]}" -eq 0 ]] || fail "SDK compile failed -- if the error is in modules/holoinfer,
   that is a NEW TRT-${TRT_VERSION} incompatibility beyond the two BuilderFlag enums already patched.
   Capture it; it is a genuine result, not a script bug."

  # Deliberately NOT `./run build_run_image`: it is broken upstream at v4.4.0 -- it passes
  # -f ${TOP}/runtime_docker/Dockerfile, but runtime_docker/ was deleted in the v4.0.0 release
  # commit and is absent from the repo. The compile+install above is unaffected, so we package
  # the resulting tree ourselves.
  say "Packaging the rebuilt SDK into a base image (upstream build_run_image is broken at v4.4.0)"
  local instdir="install-cu${CUDA_MAJOR}-x86_64"
  [[ -d "$SDK_DIR/$instdir" ]] || fail "$SDK_DIR/$instdir not found -- the compile stage did not install"
  local hi="$SDK_DIR/$instdir/lib/libholoscan_infer.so.4.4.0"
  if [[ -f "$hi" ]]; then
    info "holoinfer links: $(objdump -p "$hi" | grep -oE 'libnvinfer[a-z_]*\.so\.[0-9]+' | sort -u | tr '\n' ' ')"
  fi
  run docker build \
      -f "${SCRIPT_DIR}/patches/Dockerfile.holoscan-trt11" \
      --build-arg "TRT_VERSION=${TRT_FULL}" \
      --build-arg "TRT_CUDA_TAG=cuda${TRT_CUDA_MINOR}" \
      --build-arg "HOST_INSTALL_DIR=${instdir}" \
      -t "${TRT11_BASE_IMG}" \
      "$SDK_DIR" \
    || fail "packaging the TRT ${TRT_VERSION} base image failed"
  info "built base image: ${TRT11_BASE_IMG}"
  info "Next: ./trt11_build_test.sh --stage holohub   (BASE_IMG defaults to this image)"

  say "Confirming the builder image really has TRT ${TRT_VERSION} (a cached layer would hide it)"
  local bimg; bimg="holoscan-sdk-build-cu${CUDA_MAJOR}-x86_64:latest"
  run docker run --rm --entrypoint bash "$bimg" -lc \
    'dpkg -l | grep -E "^ii  libnvinfer[0-9]" | awk "{print \$2, \$3}"' \
    || info "WARNING: could not inspect $bimg"
  info "EXPECT libnvinfer${TRT_VERSION%%.*} at ${TRT_VERSION}.x. If it says 10.3, the ARG default did not take."

  # Filtered listing: an unfiltered `docker images` walks every image and dies on a corrupt
  # content blob in this host's local store ("blob not found"), which looks alarming and is
  # unrelated to this build.
  say "Image built"
  run docker images --filter "reference=${TRT11_BASE_IMG%%:*}" \
      --format '  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}'
  info "Use it as the HoloHub base image, e.g.:"
  info "  ./holohub run --base-img ${TRT11_BASE_IMG} ... tcn_all"
  info "or continue here: ./trt11_build_test.sh --stage verify   (TEST_IMG defaults to it)"

  revert_patch
}

stage_holohub() {
  say "Building the holohub image on the new base"
  BASE_IMG="${BASE_IMG:-$TRT11_BASE_IMG}"
  info "base image: $BASE_IMG"
  run "$HOLOHUB_DIR/holohub" build --base-img "$BASE_IMG" --cuda "$CUDA_MAJOR" tcn_all \
    || fail "holohub image build failed against base $BASE_IMG"
}

stage_verify() {
  say "Verifying the new image: TRT version, holoinfer soname, InferenceOp"
  # The sdk stage's overlay is built ON the HoloHub image, so it is directly runnable and is the
  # default test image; the separate `holohub` stage is only needed if you want a clean rebuild
  # of HoloHub on a bare Holoscan base.
  TEST_IMG="${TEST_IMG:-$TRT11_BASE_IMG}"
  info "test image: $TEST_IMG"
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
  TEST_IMG="${TEST_IMG:-$TRT11_BASE_IMG}"
  info "test image: $TEST_IMG"
  mkdir -p "$ENGINE_OUT_HOST/active/groundingdino" "$ENGINE_OUT_HOST/active/sam2"
  # Seed the precision-neutral inputs each build stage reads. Engines are NOT copied -- they are
  # what we are rebuilding, and a stale TRT 10 engine sitting in the output tree would be loaded
  # by the app in preference to noticing it was never rebuilt.
  for f in gdino_swint_512x672_b2_tf32.onnx gdino_swint_512x672_b3_tf32.onnx \
           gdino_swint_512x672_parity_ref.npz gdino_swint_prompts.npz; do
    [[ -e "$ENGINE_OUT_HOST/active/groundingdino/$f" ]] || \
      cp -v "$ENGINE_SRC_HOST/active/groundingdino/$f" "$ENGINE_OUT_HOST/active/groundingdino/" 2>&1 | tee -a "$LOG"
  done
  # SAM needs its model configs and checkpoint, not just an ONNX: sam_trt_export.py instantiates
  # the SAM 2 model from sam2.1_hiera_*.yaml + the .pt weights. Missing configs is what failed the
  # first run ("No such file: .../configs/sam2.1/sam2.1_hiera_t.yaml"). Symlink rather than copy --
  # the checkpoints are 1.6 GB and are read-only inputs.
  [[ -e "$ENGINE_OUT_HOST/active/sam2/configs" ]] || \
    ln -sfn "$ENGINE_SRC_HOST/active/sam2/configs" "$ENGINE_OUT_HOST/active/sam2/configs"
  for f in "$ENGINE_SRC_HOST"/active/sam2/*.pt; do
    [[ -e "$ENGINE_OUT_HOST/active/sam2/$(basename "$f")" ]] || \
      ln -sfn "$f" "$ENGINE_OUT_HOST/active/sam2/$(basename "$f")"
  done
  info "sam2 inputs: $(ls "$ENGINE_OUT_HOST/active/sam2" | tr '\n' ' ')"
  # One --batch N per requested batch, or a single --from-config when none were named.
  local _batch_args=""
  if [[ -n "$ENGINE_BATCHES" ]]; then
    for b in $ENGINE_BATCHES; do _batch_args+="--batch=$b "; done
    info "building explicit batches: $ENGINE_BATCHES"
  else
    _batch_args="--from-config=../python/tcn_all.yaml"
    info "building batches from the ACTIVE gpu_workers profile (set ENGINE_BATCHES to override)"
  fi
  run docker run --rm --gpus all \
      -v "$ENGINE_OUT_HOST:/srv/models" \
      -v "$ENGINE_SRC_HOST:$ENGINE_SRC_HOST:ro" \
      -v "$HOLOHUB_DIR:/workspace/holohub" \
      "$TEST_IMG" bash -lc "
        cd /workspace/holohub/${APP_REL}/docs &&
        for b in ${_batch_args}; do
          echo \"=== GDINO batch \$b ===\" &&
          python3 gdino_trt_export.py --stage build \$b --out /srv/models/active/groundingdino || exit 1
        done
      " || fail "GDINO engine build failed under TRT ${TRT_VERSION}"
  info "EXPECT 'pytorch fidelity [OK] ... |d|=0.000'. That single line is the whole point of the exercise."
  # ENGINE_SRC_HOST is mounted read-only at its own path because the sam2 configs/checkpoints
  # in the output tree are absolute symlinks back into it (they are 1.6 GB of read-only inputs,
  # not worth copying). Without this mount they dangle and the export fails on a missing yaml.
  run docker run --rm --gpus all \
      -v "$ENGINE_OUT_HOST:/srv/models" \
      -v "$ENGINE_SRC_HOST:$ENGINE_SRC_HOST:ro" \
      -v "$HOLOHUB_DIR:/workspace/holohub" \
      "$TEST_IMG" bash -lc "
        cd /workspace/holohub/${APP_REL}/docs &&
        for b in ${_batch_args}; do
          echo \"=== SAM batch \$b ===\" &&
          python3 sam_trt_export.py \$b --out /srv/models/active/sam2 || exit 1
        done
      " || info "WARNING: SAM encoder rebuild failed -- GDINO is the one that matters for the fidelity question."
}

stage_gate() {
  say "Harness run against the TRT ${TRT_VERSION} engines"
  TEST_IMG="${TEST_IMG:-$TRT11_BASE_IMG}"
  info "test image: $TEST_IMG"
  local cfg="${TMP_HOST}/gates/trt11.yaml"
  mkdir -p "${TMP_HOST}/gates" "${TMP_HOST}/harness"
  python3 - "$HOLOHUB_DIR/${APP_REL}/python/tcn_all.yaml" "$cfg" <<'PY'
import re, sys, pathlib
s = pathlib.Path(sys.argv[1]).read_text()
s = re.sub(r'^(mask_dump_dir:).*$', r'\1 "/srv/tmp/harness/trt11"', s, flags=re.M)
s = re.sub(r'^(source:).*$', r'\1 "dataset"', s, flags=re.M)
pathlib.Path(sys.argv[2]).write_text(s)
print("wrote", sys.argv[2])
PY
  rm -rf "${TMP_HOST}/harness/trt11"
  # ENGINE_SRC_HOST again mounted read-only at its own path: the sam2 configs/checkpoints in the
  # output tree are absolute symlinks into it, and the APP loads that yaml at startup too -- not
  # just the export tool. Missing it fails in LangSamBatchOp's SAM.build_model().
  local x11=()
  if [[ -n "$XAUTH_FILE" && -e "$XAUTH_FILE" ]]; then
    x11+=(-e "XAUTHORITY=$XAUTH_FILE" -v "$XAUTH_FILE:$XAUTH_FILE")
  else
    info "WARNING: no /tmp/.docker.xauth-* found; Holoviz may fail to open a window."
  fi
  [[ -d /tmp/.X11-unix ]] && x11+=(-v /tmp/.X11-unix:/tmp/.X11-unix)
  # --runtime=nvidia, NOT --gpus all: Holoviz needs Vulkan, and only the nvidia runtime injects
  # /etc/vulkan/icd.d/nvidia_icd.json plus the graphics libraries. With --gpus all the ICD is
  # absent and the app dies with "Failed to create the Vulkan instance" AFTER glfw succeeds.
  # /dev/nvidia-modeset mirrors what the working container gets.
  run docker run --rm --runtime=nvidia --ipc=host --shm-size=16gb \
      -e NVIDIA_VISIBLE_DEVICES=all \
      --device /dev/nvidia-modeset \
      -e "DISPLAY=$DISPLAY_VAL" -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
      "${x11[@]}" \
      -v "$ENGINE_OUT_HOST:/srv/models" \
      -v "$ENGINE_SRC_HOST:$ENGINE_SRC_HOST:ro" \
      -v "$TMP_HOST:/srv/tmp" \
      -v "$HOLOHUB_DIR:/workspace/holohub" \
      -v "$DATASET_HOST:/workspace/volumes/artekmed_test_data" \
      "$TEST_IMG" bash -lc "
        BIN=/workspace/holohub/build/tcn_all
        SRC=/workspace/holohub/${APP_REL}/python
        cd \$BIN && PYTHONPATH=\$BIN/python/lib:/workspace/holohub:\$SRC:\$PYTHONPATH \
        python3 \$SRC/tcn_all.py -c /srv/tmp/gates/trt11.yaml
      " || fail "harness run failed under TRT ${TRT_VERSION}"
  say "Comparing against the TRT 10.9 baseline"
  info "READ THIS AS A CHANGE, NOT A REGRESSION: TRT 10.9 -> 11.2 is the FIX landing, so masks are"
  info "EXPECTED to differ. Judge per-class INSTANCE COUNTS against what the scene contains."
  info "Do NOT conclude a regression from a low --iou-gate score against the old (wrong) output."
  run python3 "$HOLOHUB_DIR/${APP_REL}/docs/compare_mask_dumps.py" \
      "${TMP_HOST}/harness/g_bd_on_tf32" "${TMP_HOST}/harness/trt11" --top 10
}

case "$STAGE" in
  all)       stage_preflight && stage_sdk \
               && info "Next: --stage verify, then --stage engines, then --stage gate." \
               && info "The holohub stage is optional; building via your own run script with" \
               && info "--base-img ${TRT11_BASE_IMG} is equivalent and inherits X11/Vulkan/env." ;;
  preflight) stage_preflight ;;
  sdk)       stage_preflight && stage_sdk ;;
  holohub)   stage_holohub ;;
  verify)    stage_verify ;;
  engines)   stage_engines ;;
  gate)      stage_gate ;;
  *)         fail "unknown stage: $STAGE (preflight|sdk|holohub|verify|engines|gate|all)" ;;
esac

say "Done. Log: $LOG"
