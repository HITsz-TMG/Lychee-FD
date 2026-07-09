#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/stepaudio_numba_cache}"
export LYCHEEFD_T2W_SERVER_HOST="${LYCHEEFD_T2W_SERVER_HOST:-127.0.0.1}"
export LYCHEEFD_T2W_SERVER_PORT="${LYCHEEFD_T2W_SERVER_PORT:-8091}"
export LYCHEEFD_T2W_MODEL_PATH="${LYCHEEFD_T2W_MODEL_PATH:-${ROOT_DIR}/models/Step-Audio-2-mini/token2wav}"
export STEPAUDIO2_SOURCE_DIR="${STEPAUDIO2_SOURCE_DIR:-${ROOT_DIR}/third_party/Step-Audio2}"
if [[ -d "${STEPAUDIO2_SOURCE_DIR}" ]]; then
  export PYTHONPATH="${STEPAUDIO2_SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[token2wav-server][ERROR] python not found; set PYTHON_BIN explicitly." >&2
  exit 1
fi

echo "[token2wav-server] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[token2wav-server] python=${PYTHON_BIN}"
echo "[token2wav-server] host=${LYCHEEFD_T2W_SERVER_HOST} port=${LYCHEEFD_T2W_SERVER_PORT}"
echo "[token2wav-server] model=${LYCHEEFD_T2W_MODEL_PATH}"
echo "[token2wav-server] stepaudio2_source=${STEPAUDIO2_SOURCE_DIR}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$PYTHON_BIN" -u -m lychee_fd.token2wav_server \
  --host "$LYCHEEFD_T2W_SERVER_HOST" \
  --port "$LYCHEEFD_T2W_SERVER_PORT" \
  --model-path "$LYCHEEFD_T2W_MODEL_PATH"
