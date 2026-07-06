#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SOULX_DUPLUG_PYTHON:-$(command -v python || command -v python3 || true)}"
HOST="${SOULX_DUPLUG_HOST:-127.0.0.1}"
PORT="${SOULX_DUPLUG_PORT:-18080}"
CONFIG_PATH="${SOULX_DUPLUG_CONFIG_PATH:-${ROOT_DIR}/config/config.yaml}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "[SoulX-Duplug][ERROR] python not found; set SOULX_DUPLUG_PYTHON." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SOULX_DUPLUG_CONFIG_PATH="${CONFIG_PATH}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT_DIR}"

echo "[SoulX-Duplug] python=${PYTHON_BIN}"
echo "[SoulX-Duplug] host=${HOST} port=${PORT}"
echo "[SoulX-Duplug] config=${SOULX_DUPLUG_CONFIG_PATH}"
echo "[SoulX-Duplug] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec "${PYTHON_BIN}" -m uvicorn server:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --workers 1
