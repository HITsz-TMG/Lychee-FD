#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/avatar_env.sh"
require_avatar_env LIVEACT_ENV

if [[ -z "${LIVEACT_CKPT_DIR:-}" || -z "${LIVEACT_WAV2VEC_DIR:-}" ]]; then
  require_avatar_env MODEL_ROOT
fi

if [[ "${LIVEACT_WRAPPER_IN_ENV:-0}" != "1" && "${CONDA_PREFIX:-}" != "${LIVEACT_ENV}" ]]; then
  exec conda run --no-capture-output -p "${LIVEACT_ENV}" \
    env LIVEACT_WRAPPER_IN_ENV=1 bash "${BASH_SOURCE[0]}" "$@"
fi

cd "${ROOT_DIR}"
export LIVEACT_CKPT_DIR="${LIVEACT_CKPT_DIR:-${MODEL_ROOT}/LiveAct}"
export LIVEACT_WAV2VEC_DIR="${LIVEACT_WAV2VEC_DIR:-${MODEL_ROOT}/chinese-wav2vec2-base}"
export LIVEACT_AVATAR_CUDA_VISIBLE_DEVICES="${LIVEACT_AVATAR_CUDA_VISIBLE_DEVICES:-0,1}"
export LIVEACT_AVATAR_NPROC="${LIVEACT_AVATAR_NPROC:-2}"
export LIVEACT_AVATAR_PORT="${LIVEACT_AVATAR_PORT:-8092}"
export LIVEACT_AVATAR_SIZE="${LIVEACT_AVATAR_SIZE:-416*720}"

exec ./liveact_avatar/start_avatar_server.sh
