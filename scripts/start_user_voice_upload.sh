#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_PATH="${USER_VOICE_CONDA_ENV_PATH:-}"
if [[ -n "${CONDA_ENV_PATH}" && -x "${CONDA_ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
fi
HOST="${USER_VOICE_UPLOAD_HOST:-0.0.0.0}"
PORT="${USER_VOICE_UPLOAD_PORT:-18092}"

export USER_VOICE_ASR_MODEL_DIR="${USER_VOICE_ASR_MODEL_DIR:-${ROOT_DIR}/models/paraformer-zh}"
export LYCHEEFD_CLONE_PROMPT_DIR="${LYCHEEFD_CLONE_PROMPT_DIR:-${ROOT_DIR}/frontend/public/clone_24k_mono}"
export USER_VOICE_TMP_DIR="${USER_VOICE_TMP_DIR:-/tmp/lychee_user_voice_uploads}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable; set PYTHON_BIN or USER_VOICE_CONDA_ENV_PATH." >&2
  exit 1
fi

echo "[user-voice] python=${PYTHON_BIN}"
echo "[user-voice] host=${HOST} port=${PORT}"
echo "[user-voice] asr_model=${USER_VOICE_ASR_MODEL_DIR}"
echo "[user-voice] clone_prompt_dir=${LYCHEEFD_CLONE_PROMPT_DIR}"
echo "[user-voice] tmp_dir=${USER_VOICE_TMP_DIR}"
echo "[user-voice] open: http://127.0.0.1:${PORT}/"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m uvicorn lychee_fd.user_voice_server:app --host "${HOST}" --port "${PORT}"
