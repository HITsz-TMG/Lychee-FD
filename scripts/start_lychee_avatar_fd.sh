#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FD_DIR="${ROOT_DIR}"
source "${ROOT_DIR}/scripts/avatar_env.sh"
require_avatar_env FD_ENV
require_avatar_env MODEL_ROOT

if [[ "${FD_WRAPPER_IN_ENV:-0}" != "1" && "${CONDA_PREFIX:-}" != "${FD_ENV}" ]]; then
  exec conda run --no-capture-output -p "${FD_ENV}" \
    env FD_WRAPPER_IN_ENV=1 bash "${BASH_SOURCE[0]}" "$@"
fi

cd "${FD_DIR}"
export CUDA_VISIBLE_DEVICES="${FD_GPU:-2}"
export LYCHEEFD_CONDA_ENV_PATH="${FD_ENV}"
export STEPAUDIO2_SOURCE_DIR="${FD_DIR}/third_party/Step-Audio2"
export LYCHEEFD_VLLM_SOURCE_DIR="${FD_DIR}/third_party/vllm"
export LYCHEEFD_VLLM_SYNC_FLASH_ATTN="1"
export LYCHEEFD_VLLM_FORCE_SYNC_FLASH_ATTN="1"
export ALLOWED_MODEL_ROOT="${MODEL_ROOT}"
export AUTO_LOAD_DEFAULT="${AUTO_LOAD_DEFAULT:-1}"

export LYCHEEFD_REALTIME_STRICT_INFER_WINDOW="1"
export LYCHEEFD_STOKEN_DELAY_NUM="${LYCHEEFD_STOKEN_DELAY_NUM:-10}"
export LYCHEEFD_TTS_VOCODER_HOP_SIZE="${LYCHEEFD_TTS_VOCODER_HOP_SIZE:-10}"
export LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN="${LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN:-3}"

# Match the original Lychee-FD state machine.  Natural completion is signaled
# by <tts_end>; only S->L before <tts_end> is treated as a user interruption.
export LYCHEEFD_CONTROL_EARLY_EXIT_ENABLED="${LYCHEEFD_CONTROL_EARLY_EXIT_ENABLED:-1}"
export LYCHEEFD_CONTROL_EARLY_STATE_SSE="${LYCHEEFD_CONTROL_EARLY_STATE_SSE:-1}"
export LYCHEEFD_CONTROL_EARLY_TTS_ABORT="${LYCHEEFD_CONTROL_EARLY_TTS_ABORT:-1}"

export LYCHEEFD_T2W_MODEL_PATH="${LYCHEEFD_T2W_MODEL_PATH:-${MODEL_ROOT}/Step-Audio-2-mini/token2wav}"
export LYCHEEFD_T2W_REMOTE_ENABLED="1"
export LYCHEEFD_T2W_REMOTE_URL="${LYCHEEFD_T2W_REMOTE_URL:-http://127.0.0.1:8091}"
export LYCHEEFD_T2W_REMOTE_FALLBACK="0"

export LYCHEEFD_USE_VLLM="1"
export LYCHEEFD_VLLM_MAX_MODEL_LEN="${LYCHEEFD_VLLM_MAX_MODEL_LEN:-16384}"
export LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION="${LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

export LYCHEEFD_AVATAR_ENABLED="1"
export LYCHEEFD_AVATAR_URL="${LYCHEEFD_AVATAR_URL:-http://127.0.0.1:8092}"
export LYCHEEFD_AVATAR_IMAGE_PATH="${LYCHEEFD_AVATAR_IMAGE_PATH:-${ROOT_DIR}/assets/avatar/default.png}"
export LYCHEEFD_AVATAR_IDLE_VIDEO_PATH="${LYCHEEFD_AVATAR_IDLE_VIDEO_PATH:-}"
export LYCHEEFD_AVATAR_PROMPT="${LYCHEEFD_AVATAR_PROMPT:-A person is having a natural conversation with the user.}"
# Continuous idle inference must remain faster than playback.  On the current
# two-H100 LiveAct deployment, 18 FPS measures about RTF 1.17 and accumulates
# lag, while 13 FPS measures about RTF 0.82 with the same model/chunk shapes.
export LYCHEEFD_AVATAR_FPS="${LYCHEEFD_AVATAR_FPS:-13}"
export LYCHEEFD_AVATAR_PLAYBACK_BUFFER_SEC="${LYCHEEFD_AVATAR_PLAYBACK_BUFFER_SEC:-2.5}"
export LYCHEEFD_SERVER_PORT="${LYCHEEFD_SERVER_PORT:-7860}"
export FRONTEND_PORT="${FRONTEND_PORT:-8084}"

if [[ ! -x "${FD_DIR}/frontend/node_modules/.bin/vue-cli-service" ]]; then
  command -v npm >/dev/null 2>&1 || {
    echo "[lychee-fd][ERROR] npm is not available; install Node.js/npm first" >&2
    exit 1
  }
  echo "[lychee-fd] frontend dependencies missing; running npm ci..."
  (
    cd "${FD_DIR}/frontend"
    npm ci --no-audit --no-fund
  )
fi

[[ -x "${FD_DIR}/frontend/node_modules/.bin/vue-cli-service" ]] || {
  echo "[lychee-fd][ERROR] npm ci completed but vue-cli-service is still missing" >&2
  exit 1
}

exec ./scripts/start_frontend_dev.sh prod public
