#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/stepaudio_numba_cache}"
export LYCHEEFD_T2W_SERVER_HOST="${LYCHEEFD_T2W_SERVER_HOST:-127.0.0.1}"
export LYCHEEFD_T2W_SERVER_PORT="${LYCHEEFD_T2W_SERVER_PORT:-8091}"
if [[ -z "${LYCHEEFD_T2W_MODEL_PATH:-}" ]]; then
  if [[ -z "${MODEL_ROOT:-}" ]]; then
    echo "[token2wav-server][ERROR] set LYCHEEFD_T2W_MODEL_PATH or MODEL_ROOT" >&2
    exit 1
  fi
  export LYCHEEFD_T2W_MODEL_PATH="${MODEL_ROOT}/Step-Audio-2-mini/token2wav"
fi
export STEPAUDIO2_SOURCE_DIR="${STEPAUDIO2_SOURCE_DIR:-${ROOT_DIR}/third_party/Step-Audio2}"
if [[ -d "${STEPAUDIO2_SOURCE_DIR}" ]]; then
  export PYTHONPATH="${STEPAUDIO2_SOURCE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi

for required_file in speech_tokenizer_v2_25hz.onnx flow.pt hift.pt; do
  if [[ ! -f "${LYCHEEFD_T2W_MODEL_PATH}/${required_file}" ]]; then
    echo "[token2wav-server][ERROR] missing model file: ${LYCHEEFD_T2W_MODEL_PATH}/${required_file}" >&2
    echo "[token2wav-server][ERROR] set LYCHEEFD_T2W_MODEL_PATH to the complete token2wav directory" >&2
    exit 1
  fi
done

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
