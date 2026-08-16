#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FD_DIR="${ROOT_DIR}"
source "${ROOT_DIR}/scripts/avatar_env.sh"
require_avatar_env FD_ENV
if [[ -z "${LYCHEEFD_T2W_MODEL_PATH:-}" ]]; then
  require_avatar_env MODEL_ROOT
fi

if [[ "${FD_WRAPPER_IN_ENV:-0}" != "1" && "${CONDA_PREFIX:-}" != "${FD_ENV}" ]]; then
  exec conda run --no-capture-output -p "${FD_ENV}" \
    env FD_WRAPPER_IN_ENV=1 bash "${BASH_SOURCE[0]}" "$@"
fi

cd "${FD_DIR}"
export CUDA_VISIBLE_DEVICES="${TOKEN2WAV_GPU:-3}"
export LYCHEEFD_T2W_MODEL_PATH="${LYCHEEFD_T2W_MODEL_PATH:-${MODEL_ROOT}/Step-Audio-2-mini/token2wav}"
export LYCHEEFD_T2W_SERVER_HOST="${LYCHEEFD_T2W_SERVER_HOST:-127.0.0.1}"
export LYCHEEFD_T2W_SERVER_PORT="${LYCHEEFD_T2W_SERVER_PORT:-8091}"

exec ./scripts/start_token2wav_server.sh
