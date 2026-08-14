#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${TCN_ALL_ENV_FILE:-${SCRIPT_DIR}/.tcn_all_env}"

# A local .tcn_all_env contains shell-style assignments, for example:
#   TCN_ALL_MOUNT_MODELS=0
#   TCN_ALL_MODELS_HOST=/data/models
# It is deliberately not committed; use tcn_all_example.env as a template.
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

TCN_ALL_MOUNT_TMP="${TCN_ALL_MOUNT_TMP:-1}"
TCN_ALL_TMP_HOST="${TCN_ALL_TMP_HOST:-/tmp/tcn}"
TCN_ALL_MOUNT_MODELS="${TCN_ALL_MOUNT_MODELS:-1}"
TCN_ALL_MODELS_HOST="${TCN_ALL_MODELS_HOST:-/data/models_trt11}"
TCN_ALL_MOUNT_TEST_DATA="${TCN_ALL_MOUNT_TEST_DATA:-1}"
TCN_ALL_TEST_DATA_HOST="${TCN_ALL_TEST_DATA_HOST:-/home/ecku/develop/artekmed/artekmed_test_data}"
TCN_ALL_MOUNT_ZENOH_CONFIG="${TCN_ALL_MOUNT_ZENOH_CONFIG:-1}"
TCN_ALL_ZENOH_CONFIG_HOST="${TCN_ALL_ZENOH_CONFIG_HOST:-/home/ecku/develop/holoscan/zenoh_config}"
TCN_ALL_BASE_IMAGE="${TCN_ALL_BASE_IMAGE:-holoscan-trt11:4.4.0-cu12}"

is_enabled() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        0|false|no|off) return 1 ;;
        *)
            echo "Invalid boolean value: ${1} (use 0/1, true/false, yes/no, or on/off)" >&2
            return 2
            ;;
    esac
}

require_directory() {
    local label="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "${label} directory does not exist: ${path}" >&2
        echo "Set the corresponding TCN_ALL_MOUNT_* variable to 0 or update the host path in ${ENV_FILE}." >&2
        exit 1
    fi
}

# The CLI receives --docker-opts as one shell-parsed string. Quote mount sources
# here so paths containing spaces remain a single Docker mount argument.
shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

DOCKER_OPTS="--ipc=host --shm-size=16gb --mount type=bind,src=/tmp/iceoryx2,dst=/tmp/iceoryx2"
ADD_VOLUME_ARGS=()

mkdir -p /tmp/iceoryx2

if is_enabled "${TCN_ALL_MOUNT_TMP}"; then
    mkdir -p "${TCN_ALL_TMP_HOST}"
    DOCKER_OPTS+=" --mount type=bind,src=$(shell_quote "${TCN_ALL_TMP_HOST}"),dst=/srv/tmp -e HOLOSCAN_ENABLE_PROFILE=1"
fi

if is_enabled "${TCN_ALL_MOUNT_MODELS}"; then
    require_directory "/srv/models host" "${TCN_ALL_MODELS_HOST}"
    DOCKER_OPTS+=" --mount type=bind,src=$(shell_quote "${TCN_ALL_MODELS_HOST}"),dst=/srv/models"
fi

if is_enabled "${TCN_ALL_MOUNT_TEST_DATA}"; then
    require_directory "artekmed_test_data host" "${TCN_ALL_TEST_DATA_HOST}"
    ADD_VOLUME_ARGS+=(--add-volume "${TCN_ALL_TEST_DATA_HOST}")
fi

if is_enabled "${TCN_ALL_MOUNT_ZENOH_CONFIG}"; then
    require_directory "zenoh_config host" "${TCN_ALL_ZENOH_CONFIG_HOST}"
    ADD_VOLUME_ARGS+=(--add-volume "${TCN_ALL_ZENOH_CONFIG_HOST}")
fi

./holohub run-container \
    --cuda 13 \
    --docker-opts="${DOCKER_OPTS}" \
    "${ADD_VOLUME_ARGS[@]}" \
    --base-img "${TCN_ALL_BASE_IMAGE}" \
    tcn_shm_receiver
