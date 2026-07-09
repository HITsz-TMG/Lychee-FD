#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./scripts/start_backend.sh [stable|aggressive] [vllm|hf]
#
# stable:     推荐的低时延且更稳定的默认配置。
# aggressive: 更激进的时延调优，可能提升不稳定性或 OOM 风险。
# vllm:       使用 vLLM 作为模型后端（默认）。
# hf:         使用 HuggingFace 模型后端与 HF 风格增量会话。

MODE="${1:-stable}"
BACKEND_RAW="${2:-${LYCHEEFD_BACKEND:-vllm}}"
BACKEND_ARG_EXPLICIT="0"
if [[ $# -ge 2 ]]; then
  BACKEND_ARG_EXPLICIT="1"
fi
BACKEND="$(printf '%s' "${BACKEND_RAW}" | tr '[:upper:]' '[:lower:]')"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_SOURCE_DIR="${LYCHEEFD_VLLM_SOURCE_DIR:-${ROOT_DIR}/third_party/vllm}"
SYNC_FLASH_ATTN="${LYCHEEFD_VLLM_SYNC_FLASH_ATTN:-1}"
FORCE_SYNC_FLASH_ATTN="${LYCHEEFD_VLLM_FORCE_SYNC_FLASH_ATTN:-0}"

CONDA_ENV_PATH="${LYCHEEFD_CONDA_ENV_PATH:-}"
SERVER_NAME="${LYCHEEFD_SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${LYCHEEFD_SERVER_PORT:-7860}"
FORCE_DISABLE_PROXY="${LYCHEEFD_FORCE_DISABLE_PROXY:-1}"
if [[ -z "${LYCHEEFD_MODEL_PATH:-}" ]]; then
  echo "[backend][ERROR] LYCHEEFD_MODEL_PATH is required." >&2
  echo "[backend][ERROR] Example: LYCHEEFD_MODEL_PATH=/path/to/checkpoint ./scripts/start_backend.sh stable vllm" >&2
  exit 1
fi
MODEL_PATH="${LYCHEEFD_MODEL_PATH}"
if [[ -n "${CONDA_ENV_PATH}" && -x "${CONDA_ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

is_truthy() {
  local v
  v="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "${v}" == "1" || "${v}" == "true" || "${v}" == "yes" || "${v}" == "on" ]]
}

RUNTIME_LOG_ROOT="${LYCHEEFD_RUNTIME_LOG_ROOT:-${ROOT_DIR}/runtime_logs}"
RUNTIME_LOG_RUN_TAG="${LYCHEEFD_RUNTIME_LOG_RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUNTIME_LOG_DIR="${LYCHEEFD_RUNTIME_LOG_DIR:-${RUNTIME_LOG_ROOT}/${RUNTIME_LOG_RUN_TAG}}"
mkdir -p "${RUNTIME_LOG_DIR}"

if is_truthy "${LYCHEEFD_REALTIME_DEBUG:-0}"; then
  REALTIME_DEBUG_ENABLED="1"
else
  REALTIME_DEBUG_ENABLED="0"
fi

if [[ "${REALTIME_DEBUG_ENABLED}" == "1" ]]; then
  REALTIME_STAGE_TIMING_LOG_EFFECTIVE="1"
  VLLM_TOKEN_TRACE_EFFECTIVE="1"
else
  REALTIME_STAGE_TIMING_LOG_EFFECTIVE="${LYCHEEFD_REALTIME_STAGE_TIMING_LOG:-0}"
  VLLM_TOKEN_TRACE_EFFECTIVE="${LYCHEEFD_VLLM_TOKEN_TRACE:-0}"
fi

REALTIME_STAGE_TIMING_LOG_DIR_EFFECTIVE="${LYCHEEFD_REALTIME_STAGE_TIMING_LOG_DIR:-${RUNTIME_LOG_DIR}/realtime_stage_timing}"
REALTIME_CONTROL_PROB_TRACE_LOG_EFFECTIVE="${LYCHEEFD_REALTIME_CONTROL_PROB_TRACE_LOG:-0}"
REALTIME_CONTROL_PROB_TRACE_LOG_DIR_EFFECTIVE="${LYCHEEFD_REALTIME_CONTROL_PROB_TRACE_LOG_DIR:-${RUNTIME_LOG_DIR}/realtime_control_prob}"
VLLM_TOKEN_TRACE_PATH_EFFECTIVE="${LYCHEEFD_VLLM_TOKEN_TRACE_PATH:-${RUNTIME_LOG_DIR}/vllm_token_trace_${RUNTIME_LOG_RUN_TAG}.jsonl}"
STARTUP_CONFIG_LOG_PATH="${LYCHEEFD_STARTUP_CONFIG_LOG_PATH:-${RUNTIME_LOG_DIR}/startup_config.log}"

mkdir -p "${REALTIME_STAGE_TIMING_LOG_DIR_EFFECTIVE}" "${REALTIME_CONTROL_PROB_TRACE_LOG_DIR_EFFECTIVE}"
mkdir -p "$(dirname "${VLLM_TOKEN_TRACE_PATH_EFFECTIVE}")"

sync_vllm_native_artifacts() {
  if [[ "${SYNC_FLASH_ATTN}" != "1" ]]; then
    return
  fi
  if [[ ! -d "${VLLM_SOURCE_DIR}" ]]; then
    return
  fi

  local dst_pkg dst_dir dst_so src_pkg src_dir src_ver so_name
  dst_pkg="${VLLM_SOURCE_DIR}/vllm"
  dst_dir="${VLLM_SOURCE_DIR}/vllm/vllm_flash_attn"
  dst_so="${dst_dir}/vllm_flash_attn_c.abi3.so"

  if [[ ! -d "${dst_pkg}" ]]; then
    return
  fi

  src_pkg="$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata as im
from pathlib import Path
try:
    dist = im.distribution("vllm")
except Exception:
    print("")
    raise SystemExit(0)
pkg = Path(dist.locate_file("vllm"))
print(str(pkg) if pkg.exists() else "")
PY
)"
  if [[ -z "${src_pkg}" || ! -d "${src_pkg}" ]]; then
    echo "[backend][WARN] Cannot find installed vllm package from python=${PYTHON_BIN}; keep current local artifacts."
    return
  fi

  if [[ "${src_pkg}" == "${dst_pkg}" ]]; then
    echo "[backend] installed/local vllm packages are identical; skip native artifact sync."
    return
  fi

  for so_name in _C.abi3.so _moe_C.abi3.so; do
    if [[ -f "${src_pkg}/${so_name}" ]]; then
      if [[ "${FORCE_SYNC_FLASH_ATTN}" == "1" || ! -f "${dst_pkg}/${so_name}" ]]; then
        rm -f "${dst_pkg:?}/${so_name}"
        cp -f "${src_pkg}/${so_name}" "${dst_pkg}/${so_name}"
        echo "[backend] synced vllm native artifact: ${src_pkg}/${so_name} -> ${dst_pkg}/${so_name}"
      else
        echo "[backend] vllm native artifact already exists: ${dst_pkg}/${so_name}"
      fi
    fi
  done

  src_dir="${src_pkg}/vllm_flash_attn"
  if [[ -z "${src_dir}" || ! -d "${src_dir}" ]]; then
    echo "[backend][WARN] Cannot find installed vllm_flash_attn from python=${PYTHON_BIN}; keep current local artifacts."
    return
  fi
  if [[ "${src_dir}" == "${dst_dir}" ]]; then
    echo "[backend] vllm_flash_attn source/destination are identical; skip sync."
    return
  fi

  mkdir -p "${dst_dir}"
  cp -a "${src_dir}/." "${dst_dir}/"

  # Keep local source vllm version metadata aligned with the installed wheel.
  src_ver="$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata as im
from pathlib import Path
try:
    dist = im.distribution("vllm")
except Exception:
    print("")
    raise SystemExit(0)
ver = Path(dist.locate_file("vllm/_version.py"))
print(str(ver) if ver.exists() else "")
PY
)"
  if [[ -n "${src_ver}" && -f "${src_ver}" ]]; then
    cp -f "${src_ver}" "${VLLM_SOURCE_DIR}/vllm/_version.py" || true
  fi

  echo "[backend] synced vllm native/flash-attn artifacts from installed wheel: ${src_pkg} -> ${dst_pkg}"
}

case "${MODE}" in
  stable)
    SIDE_PARALLEL_BRANCH="${LYCHEEFD_VLLM_SIDE_PARALLEL_BRANCH:-0}"
    GPU_MEM_UTIL="${LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
    ;;
  aggressive)
    SIDE_PARALLEL_BRANCH="${LYCHEEFD_VLLM_SIDE_PARALLEL_BRANCH:-0}"
    GPU_MEM_UTIL="${LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 [stable|aggressive] [vllm|hf]" >&2
    exit 2
    ;;
esac

case "${BACKEND}" in
  vllm)
    DEFAULT_USE_VLLM="1"
    DEFAULT_REALTIME_INCREMENTAL_BACKEND="auto"
    ;;
  hf)
    DEFAULT_USE_VLLM="0"
    DEFAULT_REALTIME_INCREMENTAL_BACKEND="hf"
    ;;
  *)
    echo "Unknown backend: ${BACKEND_RAW}" >&2
    echo "Usage: $0 [stable|aggressive] [vllm|hf]" >&2
    exit 2
    ;;
esac

if [[ "${BACKEND_ARG_EXPLICIT}" == "1" && "${BACKEND}" == "hf" ]]; then
  # HF debug path is explicit and should not be silently overridden by shell env.
  EFFECTIVE_USE_VLLM="${DEFAULT_USE_VLLM}"
  REALTIME_INCREMENTAL_BACKEND="${DEFAULT_REALTIME_INCREMENTAL_BACKEND}"
else
  EFFECTIVE_USE_VLLM="${LYCHEEFD_USE_VLLM:-${DEFAULT_USE_VLLM}}"
  REALTIME_INCREMENTAL_BACKEND="${LYCHEEFD_REALTIME_INCREMENTAL_BACKEND:-${DEFAULT_REALTIME_INCREMENTAL_BACKEND}}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${EFFECTIVE_USE_VLLM}" == "1" ]]; then
  sync_vllm_native_artifacts
fi

if [[ "${EFFECTIVE_USE_VLLM}" == "1" && -d "${VLLM_SOURCE_DIR}" ]]; then
  export PYTHONPATH="${VLLM_SOURCE_DIR}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ "${EFFECTIVE_USE_VLLM}" == "1" ]]; then
    echo "[backend][WARN] vLLM source dir not found: ${VLLM_SOURCE_DIR}" >&2
  fi
fi

build_no_proxy_csv() {
  local raw combined host_ips ip
  raw="${NO_PROXY:-},${no_proxy:-},127.0.0.1,localhost,::1"
  if [[ "${SERVER_NAME}" != "0.0.0.0" && "${SERVER_NAME}" != "::" ]]; then
    raw="${raw},${SERVER_NAME}"
  fi
  host_ips="$(hostname -I 2>/dev/null || true)"
  for ip in ${host_ips}; do
    if [[ -n "${ip}" ]]; then
      raw="${raw},${ip}"
    fi
  done
  combined="$(printf '%s' "${raw}" \
    | tr ',' '\n' \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
    | awk 'NF && !seen[$0]++ { if (out == "") out=$0; else out=out "," $0 } END { print out }')"
  printf '%s\n' "${combined}"
}

NO_PROXY_MERGED="$(build_no_proxy_csv)"
export NO_PROXY="${NO_PROXY_MERGED}"
export no_proxy="${NO_PROXY_MERGED}"

if [[ "${FORCE_DISABLE_PROXY}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

echo "[backend] mode=${MODE}"
echo "[backend] backend=${BACKEND} (LYCHEEFD_USE_VLLM=${EFFECTIVE_USE_VLLM})"
echo "[backend] backend_arg_explicit=${BACKEND_ARG_EXPLICIT}"
echo "[backend] realtime_incremental_backend=${REALTIME_INCREMENTAL_BACKEND}"
echo "[backend] python=${PYTHON_BIN}"
echo "[backend] host=${SERVER_NAME} port=${SERVER_PORT}"
echo "[backend] model_path=${MODEL_PATH}"
echo "[backend] side_parallel=${SIDE_PARALLEL_BRANCH} gpu_mem_util=${GPU_MEM_UTIL}"
echo "[backend] startup_token2wav_first=${LYCHEEFD_STARTUP_TOKEN2WAV_FIRST:-1}"
echo "[backend] startup_warmup=${LYCHEEFD_STARTUP_WARMUP:-1}"
echo "[backend] startup_warmup_prompt_voice=${LYCHEEFD_STARTUP_WARMUP_PROMPT_VOICE:-male}"
echo "[backend] startup_warmup_token=${LYCHEEFD_STARTUP_WARMUP_TOKEN:-100}"
echo "[backend] vllm_clear_audio_each_step=0 (fixed)"
echo "[backend] realtime_debug=${REALTIME_DEBUG_ENABLED}"
echo "[backend] runtime_log_dir=${RUNTIME_LOG_DIR}"
echo "[backend] realtime_stage_timing_log=${REALTIME_STAGE_TIMING_LOG_EFFECTIVE}"
echo "[backend] realtime_stage_timing_log_dir=${REALTIME_STAGE_TIMING_LOG_DIR_EFFECTIVE}"
echo "[backend] realtime_control_prob_trace_log=${REALTIME_CONTROL_PROB_TRACE_LOG_EFFECTIVE}"
echo "[backend] realtime_control_prob_trace_log_dir=${REALTIME_CONTROL_PROB_TRACE_LOG_DIR_EFFECTIVE}"
echo "[backend] vllm_token_trace=${VLLM_TOKEN_TRACE_EFFECTIVE}"
echo "[backend] vllm_token_trace_path=${VLLM_TOKEN_TRACE_PATH_EFFECTIVE}"
echo "[backend] vllm_ignore_text_eos=${LYCHEEFD_VLLM_IGNORE_TEXT_EOS:-1}"
echo "[backend] force_disable_proxy=${FORCE_DISABLE_PROXY}"
echo "[backend] sync_flash_attn=${SYNC_FLASH_ATTN} force_sync_flash_attn=${FORCE_SYNC_FLASH_ATTN}"
echo "[backend] vllm_source_dir=${VLLM_SOURCE_DIR}"
echo "[backend] PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "[backend] NO_PROXY=${NO_PROXY}"
echo "[backend] http_proxy=${http_proxy:-<unset>} https_proxy=${https_proxy:-<unset>}"
echo "[backend] realtime_infer_window_ms=${LYCHEEFD_REALTIME_INFER_WINDOW_MS:-400} (min=${LYCHEEFD_REALTIME_INFER_WINDOW_MIN_MS:-160})"
echo "[backend] window_second=${LYCHEEFD_WINDOW_SECOND:-0.4}"
echo "[backend] stoken_delay_num=${LYCHEEFD_STOKEN_DELAY_NUM:-10}"

{
  echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "mode=${MODE}"
  echo "backend=${BACKEND}"
  echo "backend_arg_explicit=${BACKEND_ARG_EXPLICIT}"
  echo "effective_use_vllm=${EFFECTIVE_USE_VLLM}"
  echo "realtime_incremental_backend=${REALTIME_INCREMENTAL_BACKEND}"
  echo "model_path=${MODEL_PATH}"
  echo "server_name=${SERVER_NAME}"
  echo "server_port=${SERVER_PORT}"
  echo "python_bin=${PYTHON_BIN}"
  echo "runtime_log_dir=${RUNTIME_LOG_DIR}"
  echo "realtime_debug=${REALTIME_DEBUG_ENABLED}"
  echo "realtime_stage_timing_log=${REALTIME_STAGE_TIMING_LOG_EFFECTIVE}"
  echo "realtime_stage_timing_log_dir=${REALTIME_STAGE_TIMING_LOG_DIR_EFFECTIVE}"
  echo "realtime_control_prob_trace_log=${REALTIME_CONTROL_PROB_TRACE_LOG_EFFECTIVE}"
  echo "realtime_control_prob_trace_log_dir=${REALTIME_CONTROL_PROB_TRACE_LOG_DIR_EFFECTIVE}"
  echo "vllm_token_trace=${VLLM_TOKEN_TRACE_EFFECTIVE}"
  echo "vllm_token_trace_path=${VLLM_TOKEN_TRACE_PATH_EFFECTIVE}"
  echo "vllm_ignore_text_eos=${LYCHEEFD_VLLM_IGNORE_TEXT_EOS:-1}"
  echo "vllm_gpu_memory_utilization=${GPU_MEM_UTIL}"
  echo "vllm_side_parallel_branch=${SIDE_PARALLEL_BRANCH}"
  echo "force_disable_proxy=${FORCE_DISABLE_PROXY}"
} > "${STARTUP_CONFIG_LOG_PATH}"
echo "[backend] startup_config_log=${STARTUP_CONFIG_LOG_PATH}"

RESOLVED_VLLM="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
print(spec.origin if spec else "NOT_FOUND")
PY
)"
if [[ "${EFFECTIVE_USE_VLLM}" == "1" ]]; then
  echo "[backend] resolved_vllm=${RESOLVED_VLLM}"
  if [[ "${RESOLVED_VLLM}" == "NOT_FOUND" ]]; then
    echo "[backend][ERROR] python cannot find vllm module" >&2
    exit 1
  fi
  if [[ -d "${VLLM_SOURCE_DIR}" && "${RESOLVED_VLLM}" != "${VLLM_SOURCE_DIR}"* ]]; then
    echo "[backend][WARN] resolved vllm is not from LYCHEEFD_VLLM_SOURCE_DIR" >&2
  fi
fi

cd "${ROOT_DIR}"

exec env \
  LYCHEEFD_REALTIME_DEBUG="${REALTIME_DEBUG_ENABLED}" \
  LYCHEEFD_REALTIME_STAGE_TIMING_LOG="${REALTIME_STAGE_TIMING_LOG_EFFECTIVE}" \
  LYCHEEFD_REALTIME_STAGE_TIMING_LOG_DIR="${REALTIME_STAGE_TIMING_LOG_DIR_EFFECTIVE}" \
  LYCHEEFD_REALTIME_CONTROL_PROB_TRACE_LOG="${REALTIME_CONTROL_PROB_TRACE_LOG_EFFECTIVE}" \
  LYCHEEFD_REALTIME_CONTROL_PROB_TRACE_LOG_DIR="${REALTIME_CONTROL_PROB_TRACE_LOG_DIR_EFFECTIVE}" \
  LYCHEEFD_VLLM_TOKEN_TRACE="${VLLM_TOKEN_TRACE_EFFECTIVE}" \
  LYCHEEFD_VLLM_TOKEN_TRACE_PATH="${VLLM_TOKEN_TRACE_PATH_EFFECTIVE}" \
  LYCHEEFD_REALTIME_INFER_WINDOW_MS="${LYCHEEFD_REALTIME_INFER_WINDOW_MS:-400}" \
  LYCHEEFD_REALTIME_INFER_WINDOW_MIN_MS="${LYCHEEFD_REALTIME_INFER_WINDOW_MIN_MS:-160}" \
  LYCHEEFD_WINDOW_SECOND="${LYCHEEFD_WINDOW_SECOND:-0.4}" \
  LYCHEEFD_STOKEN_DELAY_NUM="${LYCHEEFD_STOKEN_DELAY_NUM:-20}" \
  LYCHEEFD_STOKEN_NO_REPEAT_NGRAM="${LYCHEEFD_STOKEN_NO_REPEAT_NGRAM:-4}" \
  LYCHEEFD_REALTIME_TTS_CHUNK_SIZE="${LYCHEEFD_REALTIME_TTS_CHUNK_SIZE:-1}" \
  LYCHEEFD_TTS_VOCODER_HOP_SIZE="${LYCHEEFD_TTS_VOCODER_HOP_SIZE:-25}" \
  LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN="${LYCHEEFD_T2W_STREAM_LOOKAHEAD_LEN:-3}" \
  LYCHEEFD_PROFILE_LATENCY="${LYCHEEFD_PROFILE_LATENCY:-0}" \
  LYCHEEFD_PROFILE_SIDE="${LYCHEEFD_PROFILE_SIDE:-0}" \
  LYCHEEFD_LOGITS_SANITIZE="${LYCHEEFD_LOGITS_SANITIZE:-0}" \
  LYCHEEFD_REALTIME_TRUE_INCREMENTAL_AUDIO="1" \
  LYCHEEFD_STARTUP_TOKEN2WAV_FIRST="${LYCHEEFD_STARTUP_TOKEN2WAV_FIRST:-1}" \
  LYCHEEFD_STARTUP_WARMUP="${LYCHEEFD_STARTUP_WARMUP:-1}" \
  LYCHEEFD_STARTUP_WARMUP_PROMPT_VOICE="${LYCHEEFD_STARTUP_WARMUP_PROMPT_VOICE:-male}" \
  LYCHEEFD_STARTUP_WARMUP_TOKEN="${LYCHEEFD_STARTUP_WARMUP_TOKEN:-100}" \
  LYCHEEFD_USE_VLLM="${EFFECTIVE_USE_VLLM}" \
  LYCHEEFD_REALTIME_INCREMENTAL_BACKEND="${REALTIME_INCREMENTAL_BACKEND}" \
  LYCHEEFD_VLLM_ENFORCE_EAGER="${LYCHEEFD_VLLM_ENFORCE_EAGER:-1}" \
  LYCHEEFD_VLLM_ENABLE_CHUNKED_PREFILL="${LYCHEEFD_VLLM_ENABLE_CHUNKED_PREFILL:-1}" \
  LYCHEEFD_VLLM_ENABLE_PREFIX_CACHING="${LYCHEEFD_VLLM_ENABLE_PREFIX_CACHING:-0}" \
  LYCHEEFD_VLLM_KEEP_ALIVE_SPEAKING="${LYCHEEFD_VLLM_KEEP_ALIVE_SPEAKING:-1}" \
  LYCHEEFD_VLLM_IGNORE_TEXT_EOS="${LYCHEEFD_VLLM_IGNORE_TEXT_EOS:-1}" \
  LYCHEEFD_VLLM_SIDE_SUBSET_PROJ="${LYCHEEFD_VLLM_SIDE_SUBSET_PROJ:-1}" \
  LYCHEEFD_VLLM_SIDE_SUBSET_REQUIRE_PROCESSOR="${LYCHEEFD_VLLM_SIDE_SUBSET_REQUIRE_PROCESSOR:-1}" \
  LYCHEEFD_VLLM_SIDE_PARALLEL_BRANCH="${SIDE_PARALLEL_BRANCH}" \
  LYCHEEFD_VLLM_TENSOR_PARALLEL_SIZE="${LYCHEEFD_VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  LYCHEEFD_VLLM_PIPELINE_PARALLEL_SIZE="${LYCHEEFD_VLLM_PIPELINE_PARALLEL_SIZE:-1}" \
  LYCHEEFD_VLLM_GPU_MEMORY_UTILIZATION="${GPU_MEM_UTIL}" \
  LYCHEEFD_VLLM_MAX_MODEL_LEN="${LYCHEEFD_VLLM_MAX_MODEL_LEN:-4096}" \
  LYCHEEFD_VLLM_MAX_NUM_SEQS="${LYCHEEFD_VLLM_MAX_NUM_SEQS:-1}" \
  LYCHEEFD_VLLM_MAX_NUM_BATCHED_TOKENS="${LYCHEEFD_VLLM_MAX_NUM_BATCHED_TOKENS:-1024}" \
  "${PYTHON_BIN}" -u -m lychee_fd.app \
    --model_path "${MODEL_PATH}" \
    --server_name "${SERVER_NAME}" \
    --server_port "${SERVER_PORT}" \
    --no-share
