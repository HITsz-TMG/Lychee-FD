#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${LIVEACT_AVATAR_CUDA_VISIBLE_DEVICES:-0}"
export USE_CHANNELS_LAST_3D="${USE_CHANNELS_LAST_3D:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CKPT_DIR="${LIVEACT_CKPT_DIR:?LIVEACT_CKPT_DIR is required}"
WAV2VEC_DIR="${LIVEACT_WAV2VEC_DIR:?LIVEACT_WAV2VEC_DIR is required}"
HOST="${LIVEACT_AVATAR_HOST:-0.0.0.0}"
PORT="${LIVEACT_AVATAR_PORT:-8092}"
SIZE="${LIVEACT_AVATAR_SIZE:-416*720}"
NPROC="${LIVEACT_AVATAR_NPROC:-1}"
SESSION_TTL_SEC="${LIVEACT_AVATAR_SESSION_TTL_SEC:-45}"
STOP_JOIN_TIMEOUT_SEC="${LIVEACT_AVATAR_STOP_JOIN_TIMEOUT_SEC:-130}"

SERVER_MODULE="${LIVEACT_AVATAR_SERVER_MODULE:-liveact_avatar.avatar_http_server}"

COMMON_ARGS=(
  -m "${SERVER_MODULE}"
  --ckpt_dir "${CKPT_DIR}"
  --wav2vec_dir "${WAV2VEC_DIR}"
  --host "${HOST}"
  --port "${PORT}"
  --size "${SIZE}"
  --t5_cpu
  --session_ttl_sec "${SESSION_TTL_SEC}"
  --stop_join_timeout_sec "${STOP_JOIN_TIMEOUT_SEC}"
)

if [[ "${LIVEACT_AVATAR_FP8_KV_CACHE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--fp8_kv_cache)
fi
if [[ "${LIVEACT_AVATAR_BLOCK_OFFLOAD:-0}" == "1" ]]; then
  COMMON_ARGS+=(--block_offload)
fi
LEGACY_COMPILE="${LIVEACT_AVATAR_COMPILE:-0}"
if [[ "${LIVEACT_AVATAR_COMPILE_MODEL:-${LEGACY_COMPILE}}" == "1" ]]; then
  COMMON_ARGS+=(--compile_model)
fi
if [[ "${LIVEACT_AVATAR_COMPILE_VAE_DECODE:-${LEGACY_COMPILE}}" == "1" ]]; then
  COMMON_ARGS+=(--compile_vae_decode)
fi
if [[ "${LIVEACT_AVATAR_WARMUP:-0}" == "1" ]]; then
  COMMON_ARGS+=(--warmup)
fi
if [[ -n "${LIVEACT_AVATAR_PRELOAD_IMAGE_PATH:-}" ]]; then
  COMMON_ARGS+=(--preload_image_path "${LIVEACT_AVATAR_PRELOAD_IMAGE_PATH}")
  COMMON_ARGS+=(--preload_prompt "${LIVEACT_AVATAR_PRELOAD_PROMPT:-A person is having a natural conversation with the user.}")
fi
if [[ "${LIVEACT_AVATAR_NO_FP8_GEMM:-0}" == "1" ]]; then
  COMMON_ARGS+=(--no_fp8_gemm)
fi

echo "[avatar-launcher] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[avatar-launcher] nproc=${NPROC} size=${SIZE} port=${PORT}"
echo "[avatar-launcher] session_mode=single_active_audio_driven ttl=${SESSION_TTL_SEC}s stop_timeout=${STOP_JOIN_TIMEOUT_SEC}s"
echo "[avatar-launcher] ckpt=${CKPT_DIR}"
echo "[avatar-launcher] wav2vec=${WAV2VEC_DIR}"
echo "[avatar-launcher] fixed_image=${LIVEACT_AVATAR_PRELOAD_IMAGE_PATH:-<disabled>}"
echo "[avatar-launcher] fixed_prompt=${LIVEACT_AVATAR_PRELOAD_PROMPT:-<default>}"
echo "[avatar-launcher] server_module=${SERVER_MODULE}"

if [[ "${NPROC}" -gt 1 ]]; then
  exec torchrun --nproc_per_node="${NPROC}" --master_port="${LIVEACT_AVATAR_MASTER_PORT:-29622}" "${COMMON_ARGS[@]}"
fi

exec python "${COMMON_ARGS[@]}"
