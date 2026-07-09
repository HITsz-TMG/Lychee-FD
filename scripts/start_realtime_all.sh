#!/usr/bin/env bash
set -euo pipefail

# 一键启动后端 + 前端（默认等价于用户当前手动命令）
#
# 用法:
#   ./scripts/start_realtime_all.sh [backend_mode] [frontend_mode] [frontend_expose_mode] [backend_type]
#
# 示例:
#   ./start_realtime_all.sh
#   ./start_realtime_all.sh stable prod public vllm
#   ./start_realtime_all.sh aggressive dev private hf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_MODE="${1:-stable}"
FRONTEND_MODE="${2:-prod}"
FRONTEND_EXPOSE_MODE="${3:-public}"
BACKEND_TYPE="${4:-${LYCHEEFD_BACKEND:-vllm}}"

is_truthy() {
  local v
  v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "${v}" == "1" || "${v}" == "true" || "${v}" == "yes" || "${v}" == "on" ]]
}

# 你的当前实时参数，允许外部环境变量覆盖
export LYCHEEFD_TTS_VOCODER_HOP_SIZE="${LYCHEEFD_TTS_VOCODER_HOP_SIZE:-12}"
export LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN="${LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN:-3}"
export LYCHEEFD_USE_VLLM="${LYCHEEFD_USE_VLLM:-1}"
export LYCHEEFD_REALTIME_DEBUG="${LYCHEEFD_REALTIME_DEBUG:-0}"
export LYCHEEFD_REALTIME_TRUE_INCREMENTAL_AUDIO="1"

if is_truthy "${LYCHEEFD_REALTIME_DEBUG}"; then
  DEFAULT_STAGE_TIMING_LOG="1"
  DEFAULT_VLLM_TOKEN_TRACE="1"
else
  DEFAULT_STAGE_TIMING_LOG="0"
  DEFAULT_VLLM_TOKEN_TRACE="0"
fi

export LYCHEEFD_RUNTIME_LOG_ROOT="${LYCHEEFD_RUNTIME_LOG_ROOT:-${ROOT_DIR}/runtime_logs}"
export LYCHEEFD_RUNTIME_LOG_RUN_TAG="${LYCHEEFD_RUNTIME_LOG_RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
export LYCHEEFD_RUNTIME_LOG_DIR="${LYCHEEFD_RUNTIME_LOG_DIR:-${LYCHEEFD_RUNTIME_LOG_ROOT}/${LYCHEEFD_RUNTIME_LOG_RUN_TAG}}"
export LYCHEEFD_REALTIME_STAGE_TIMING_LOG="${LYCHEEFD_REALTIME_STAGE_TIMING_LOG:-${DEFAULT_STAGE_TIMING_LOG}}"
export LYCHEEFD_VLLM_TOKEN_TRACE="${LYCHEEFD_VLLM_TOKEN_TRACE:-${DEFAULT_VLLM_TOKEN_TRACE}}"
export LYCHEEFD_REALTIME_STAGE_TIMING_LOG_DIR="${LYCHEEFD_REALTIME_STAGE_TIMING_LOG_DIR:-${LYCHEEFD_RUNTIME_LOG_DIR}/realtime_stage_timing}"
export LYCHEEFD_VLLM_TOKEN_TRACE_PATH="${LYCHEEFD_VLLM_TOKEN_TRACE_PATH:-${LYCHEEFD_RUNTIME_LOG_DIR}/vllm_token_trace_${LYCHEEFD_RUNTIME_LOG_RUN_TAG}.jsonl}"

mkdir -p "${LYCHEEFD_RUNTIME_LOG_DIR}"
mkdir -p "${LYCHEEFD_REALTIME_STAGE_TIMING_LOG_DIR}"
mkdir -p "$(dirname "${LYCHEEFD_VLLM_TOKEN_TRACE_PATH}")"

BACKEND_LOG="${LYCHEEFD_BACKEND_LOG_PATH:-${LYCHEEFD_RUNTIME_LOG_DIR}/backend_launcher.log}"
FRONTEND_LOG="${LYCHEEFD_FRONTEND_LOG_PATH:-${LYCHEEFD_RUNTIME_LOG_DIR}/frontend_launcher.log}"
mkdir -p "$(dirname "${BACKEND_LOG}")" "$(dirname "${FRONTEND_LOG}")"

BACKEND_PID=""

cleanup() {
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "[launcher] stopping backend pid=${BACKEND_PID}"
    kill "${BACKEND_PID}" 2>/dev/null || true
    wait "${BACKEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[launcher] root=${ROOT_DIR}"
echo "[launcher] backend: ./scripts/start_backend.sh ${BACKEND_MODE} ${BACKEND_TYPE}"
echo "[launcher] frontend: ./scripts/start_frontend.sh ${FRONTEND_MODE} ${FRONTEND_EXPOSE_MODE}"
echo "[launcher] env: HOP_SIZE=${LYCHEEFD_TTS_VOCODER_HOP_SIZE} LOOKAHEAD=${LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN} USE_VLLM=${LYCHEEFD_USE_VLLM}"
echo "[launcher] env: TRUE_INCREMENTAL=${LYCHEEFD_REALTIME_TRUE_INCREMENTAL_AUDIO} DEBUG=${LYCHEEFD_REALTIME_DEBUG}"
echo "[launcher] env: STAGE_TIMING_LOG=${LYCHEEFD_REALTIME_STAGE_TIMING_LOG} VLLM_TOKEN_TRACE=${LYCHEEFD_VLLM_TOKEN_TRACE}"
echo "[launcher] env: RUNTIME_LOG_DIR=${LYCHEEFD_RUNTIME_LOG_DIR}"
echo "[launcher] env: STAGE_TIMING_LOG_DIR=${LYCHEEFD_REALTIME_STAGE_TIMING_LOG_DIR}"
echo "[launcher] env: TOKEN_TRACE_PATH=${LYCHEEFD_VLLM_TOKEN_TRACE_PATH}"
echo "[launcher] logs: backend=${BACKEND_LOG}, frontend=${FRONTEND_LOG}"

cd "${ROOT_DIR}"

"${ROOT_DIR}/scripts/start_backend.sh" "${BACKEND_MODE}" "${BACKEND_TYPE}" >>"${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!
echo "[launcher] backend started pid=${BACKEND_PID}"

sleep "${LYCHEEFD_LAUNCHER_BACKEND_WAIT_SECONDS:-2}"

set +e
"${ROOT_DIR}/scripts/start_frontend.sh" "${FRONTEND_MODE}" "${FRONTEND_EXPOSE_MODE}" | tee -a "${FRONTEND_LOG}"
FRONTEND_EXIT_CODE=$?
set -e

if [[ ${FRONTEND_EXIT_CODE} -ne 0 ]]; then
  echo "[launcher][ERROR] frontend exited with code=${FRONTEND_EXIT_CODE}"
  exit "${FRONTEND_EXIT_CODE}"
fi
