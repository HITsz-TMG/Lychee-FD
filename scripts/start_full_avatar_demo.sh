#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FD_DIR="${ROOT_DIR}"
source "${ROOT_DIR}/scripts/avatar_env.sh"
require_avatar_env MODEL_ROOT
require_avatar_env LIVEACT_ENV
require_avatar_env FD_ENV
RUN_TAG="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${FULL_DEMO_LOG_DIR:-${ROOT_DIR}/runtime_logs/full_demo_${RUN_TAG}}"

AVATAR_GPU="${AVATAR_GPU:-0,1}"
FD_GPU="${FD_GPU:-2}"
TOKEN2WAV_GPU="${TOKEN2WAV_GPU:-3}"

AVATAR_PORT="${LIVEACT_AVATAR_PORT:-8092}"
TOKEN2WAV_PORT="${LYCHEEFD_T2W_SERVER_PORT:-8091}"
BACKEND_PORT="${LYCHEEFD_SERVER_PORT:-7860}"
FRONTEND_PORT="${FRONTEND_PORT:-8084}"

LIVEACT_CKPT_DIR="${LIVEACT_CKPT_DIR:-${MODEL_ROOT}/LiveAct}"
LIVEACT_WAV2VEC_DIR="${LIVEACT_WAV2VEC_DIR:-${MODEL_ROOT}/chinese-wav2vec2-base}"
FD_MODEL_DIR="${LYCHEEFD_MODEL_PATH:-${MODEL_ROOT}/lychee_full_duplex}"
TOKEN2WAV_DIR="${LYCHEEFD_T2W_MODEL_PATH:-${MODEL_ROOT}/Step-Audio-2-mini/token2wav}"
AVATAR_IMAGE="${LYCHEEFD_AVATAR_IMAGE_PATH:-${ROOT_DIR}/assets/avatar/default.png}"

AVATAR_PID=""
TOKEN2WAV_PID=""
FRONTEND_PID=""

die() {
  echo "[full-demo][ERROR] $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "required directory not found: $1"
}

wait_health() {
  local name="$1" url="$2" pid="$3" timeout_sec="$4"
  local start now
  start="$(date +%s)"
  echo "[full-demo] waiting for ${name}: ${url}"
  while true; do
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      echo "[full-demo] ${name} is ready"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[full-demo][ERROR] ${name} exited early; inspect ${LOG_DIR}" >&2
      return 1
    fi
    now="$(date +%s)"
    if (( now - start >= timeout_sec )); then
      echo "[full-demo][ERROR] timed out waiting for ${name}" >&2
      return 1
    fi
    sleep 2
  done
}

cleanup() {
  local pid
  echo "[full-demo] stopping services..."
  for pid in "${FRONTEND_PID}" "${TOKEN2WAV_PID}" "${AVATAR_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${FRONTEND_PID}" "${TOKEN2WAV_PID}" "${AVATAR_PID}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

require_dir "${FD_DIR}"
require_file "${LIVEACT_ENV}/bin/python"
require_file "${FD_ENV}/bin/python"
require_file "${FD_MODEL_DIR}/config.json"
require_file "${FD_MODEL_DIR}/model-00006-of-00006.safetensors"
require_file "${TOKEN2WAV_DIR}/speech_tokenizer_v2_25hz.onnx"
require_dir "${LIVEACT_CKPT_DIR}"
require_dir "${LIVEACT_WAV2VEC_DIR}"
require_file "${AVATAR_IMAGE}"

if [[ ! -x "${FD_DIR}/frontend/node_modules/.bin/vue-cli-service" ]]; then
  echo "[full-demo] frontend dependencies are missing; running npm ci..."
  (
    cd "${FD_DIR}/frontend"
    npm ci --no-audit --no-fund
  )
fi
require_file "${FD_DIR}/frontend/node_modules/.bin/vue-cli-service"

mkdir -p "${LOG_DIR}"

echo "[full-demo] logs=${LOG_DIR}"
echo "[full-demo] GPUs: LiveAct=${AVATAR_GPU}, Lychee-FD=${FD_GPU}, Token2Wav=${TOKEN2WAV_GPU}"

(
  cd "${ROOT_DIR}"
  export PATH="${LIVEACT_ENV}/bin:${PATH}"
  export LIVEACT_CKPT_DIR LIVEACT_WAV2VEC_DIR
  export LIVEACT_AVATAR_CUDA_VISIBLE_DEVICES="${AVATAR_GPU}"
  export LIVEACT_AVATAR_NPROC="2"
  export LIVEACT_AVATAR_PORT="${AVATAR_PORT}"
  export LIVEACT_AVATAR_SIZE="${LIVEACT_AVATAR_SIZE:-416*720}"
  exec ./liveact_avatar/start_avatar_server.sh
) >"${LOG_DIR}/avatar.log" 2>&1 &
AVATAR_PID=$!
wait_health "LiveAct avatar" "http://127.0.0.1:${AVATAR_PORT}/health" "${AVATAR_PID}" 900

(
  cd "${FD_DIR}"
  export PATH="${FD_ENV}/bin:${PATH}"
  export CUDA_VISIBLE_DEVICES="${TOKEN2WAV_GPU}"
  export LYCHEEFD_T2W_MODEL_PATH="${TOKEN2WAV_DIR}"
  export LYCHEEFD_T2W_SERVER_PORT="${TOKEN2WAV_PORT}"
  exec ./scripts/start_token2wav_server.sh
) >"${LOG_DIR}/token2wav.log" 2>&1 &
TOKEN2WAV_PID=$!
wait_health "Token2Wav" "http://127.0.0.1:${TOKEN2WAV_PORT}/health" "${TOKEN2WAV_PID}" 300

(
  cd "${FD_DIR}"
  export PATH="${FD_ENV}/bin:${PATH}"
  export CUDA_VISIBLE_DEVICES="${FD_GPU}"
  export LYCHEEFD_CONDA_ENV_PATH="${FD_ENV}"
  export STEPAUDIO2_SOURCE_DIR="${FD_DIR}/third_party/Step-Audio2"
  export LYCHEEFD_VLLM_SOURCE_DIR="${FD_DIR}/third_party/vllm"
  export LYCHEEFD_VLLM_SYNC_FLASH_ATTN="1"
  export LYCHEEFD_VLLM_FORCE_SYNC_FLASH_ATTN="1"
  export ALLOWED_MODEL_ROOT="${MODEL_ROOT}"
  export AUTO_LOAD_DEFAULT="1"
  export LYCHEEFD_REALTIME_STRICT_INFER_WINDOW="1"
  export LYCHEEFD_STOKEN_DELAY_NUM="10"
  export LYCHEEFD_TTS_VOCODER_HOP_SIZE="10"
  export LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN="3"
  export LYCHEEFD_T2W_MODEL_PATH="${TOKEN2WAV_DIR}"
  export LYCHEEFD_T2W_REMOTE_ENABLED="1"
  export LYCHEEFD_T2W_REMOTE_URL="http://127.0.0.1:${TOKEN2WAV_PORT}"
  export LYCHEEFD_T2W_REMOTE_FALLBACK="0"
  export LYCHEEFD_USE_VLLM="1"
  export LYCHEEFD_VLLM_MAX_MODEL_LEN="${LYCHEEFD_VLLM_MAX_MODEL_LEN:-16384}"
  export LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION="${LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  export LYCHEEFD_AVATAR_ENABLED="1"
  export LYCHEEFD_AVATAR_URL="http://127.0.0.1:${AVATAR_PORT}"
  export LYCHEEFD_AVATAR_IMAGE_PATH="${AVATAR_IMAGE}"
  export LYCHEEFD_SERVER_PORT="${BACKEND_PORT}"
  export FRONTEND_PORT="${FRONTEND_PORT}"
  exec ./scripts/start_frontend_dev.sh prod public
) >"${LOG_DIR}/lychee_fd.log" 2>&1 &
FRONTEND_PID=$!

wait_health "Lychee-FD frontend/controller" "http://127.0.0.1:${FRONTEND_PORT}/admin/status" "${FRONTEND_PID}" 300

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "[full-demo] all launchers are running"
echo "[full-demo] frontend: http://${HOST_IP:-127.0.0.1}:${FRONTEND_PORT}"
echo "[full-demo] backend model is auto-loading on GPU ${FD_GPU}; watch:"
echo "  tail -f ${LOG_DIR}/lychee_fd.log"
echo "[full-demo] press Ctrl+C to stop all services"

wait "${FRONTEND_PID}"
