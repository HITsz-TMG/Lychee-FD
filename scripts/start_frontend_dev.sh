#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./scripts/start_frontend_dev.sh [prod|dev] [public|private]
#
# DEV 版本: 使用 lychee_fd.controller 同时托管前端静态文件 + /admin/* 控制后端启停。
# 不会修改任何已有文件; 与 scripts/start_frontend.sh 互不影响。


MODE="${1:-prod}"
EXPOSE_MODE="${2:-${FRONTEND_EXPOSE_MODE:-public}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONT_DIR="${ROOT_DIR}/frontend"
PORT="${FRONTEND_PORT:-8084}"
FORCE_DISABLE_PROXY="${FRONTEND_FORCE_DISABLE_PROXY:-1}"
REALTIME_DEBUG="${LYCHEEFD_REALTIME_DEBUG:-0}"
export VUE_APP_REALTIME_DEBUG="${VUE_APP_REALTIME_DEBUG:-${REALTIME_DEBUG}}"

CONTROLLER_MODULE="${CONTROLLER_MODULE:-lychee_fd.controller}"
BACKEND_SCRIPT="${BACKEND_SCRIPT:-${ROOT_DIR}/scripts/start_backend.sh}"
PRESETS_FILE="${PRESETS_FILE:-${ROOT_DIR}/model_presets_dev.json}"
ALLOWED_MODEL_ROOT="${ALLOWED_MODEL_ROOT:-${ROOT_DIR}}"
ADMIN_TOKEN="${ADMIN_TOKEN:-}"
BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL:-http://127.0.0.1:${LYCHEEFD_SERVER_PORT:-7860}}"
AUTO_LOAD_DEFAULT="${AUTO_LOAD_DEFAULT:-0}"

CONDA_ENV_PATH="${LYCHEEFD_CONDA_ENV_PATH:-}"
if [[ -n "${CONDA_ENV_PATH}" && -x "${CONDA_ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

case "${EXPOSE_MODE}" in
  public)  HOST="${FRONTEND_HOST:-0.0.0.0}";;
  private) HOST="${FRONTEND_HOST:-127.0.0.1}";;
  *)
    echo "Unknown expose mode: ${EXPOSE_MODE}" >&2
    echo "Usage: $0 [prod|dev] [public|private]" >&2
    exit 2
    ;;
esac

if [[ ! -d "${FRONT_DIR}" ]]; then
  echo "Frontend dir not found: ${FRONT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${BACKEND_SCRIPT}" ]]; then
  echo "Backend script not found: ${BACKEND_SCRIPT}" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found in PATH" >&2
  exit 1
fi
if [[ ! -f "${FRONT_DIR}/vue.config_dev.js" ]]; then
  echo "Dev Vue config not found: ${FRONT_DIR}/vue.config_dev.js" >&2
  exit 1
fi

VUE_CLI_BIN="${FRONT_DIR}/node_modules/.bin/vue-cli-service"
if [[ ! -x "${VUE_CLI_BIN}" ]]; then
  VUE_CLI_BIN="npx vue-cli-service"
fi

build_no_proxy_csv() {
  local raw combined host_ips ip
  raw="${NO_PROXY:-},${no_proxy:-},127.0.0.1,localhost,::1"
  if [[ "${HOST}" != "0.0.0.0" && "${HOST}" != "::" ]]; then
    raw="${raw},${HOST}"
  fi
  host_ips="$(hostname -I 2>/dev/null || true)"
  for ip in ${host_ips}; do
    [[ -n "${ip}" ]] && raw="${raw},${ip}"
  done
  printf '%s\n' "${raw}" \
    | tr ',' '\n' \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
    | awk 'NF && !seen[$0]++ { if (out == "") out=$0; else out=out "," $0 } END { print out }'
}

NO_PROXY_MERGED="$(build_no_proxy_csv)"
export NO_PROXY="${NO_PROXY_MERGED}"
export no_proxy="${NO_PROXY_MERGED}"

if [[ "${FORCE_DISABLE_PROXY}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

echo "[frontend-dev] mode=${MODE} expose=${EXPOSE_MODE} host=${HOST} port=${PORT}"
echo "[frontend-dev] controller_module=${CONTROLLER_MODULE}"
echo "[frontend-dev] backend_script=${BACKEND_SCRIPT}"
echo "[frontend-dev] presets_file=${PRESETS_FILE}"
echo "[frontend-dev] allowed_root=${ALLOWED_MODEL_ROOT}"
echo "[frontend-dev] backend_health_url=${BACKEND_HEALTH_URL}"
echo "[frontend-dev] python=${PYTHON_BIN}"
echo "[frontend-dev] NO_PROXY=${NO_PROXY}"

cd "${FRONT_DIR}"

run_vue_cli_with_dev_config() {
  local backup
  backup="${FRONT_DIR}/vue.config.js.devbak.$$"
  cp "${FRONT_DIR}/vue.config.js" "${backup}"
  cp "${FRONT_DIR}/vue.config_dev.js" "${FRONT_DIR}/vue.config.js"
  "$@"
  local rc=$?
  mv "${backup}" "${FRONT_DIR}/vue.config.js"
  return "${rc}"
}

case "${MODE}" in
  prod)
    echo "[frontend-dev] building Vue with vue.config_dev.js ..."
    run_vue_cli_with_dev_config "${VUE_CLI_BIN}" build
    AUTO_LOAD_FLAG=""
    if [[ "${AUTO_LOAD_DEFAULT}" == "1" ]]; then
      AUTO_LOAD_FLAG="--auto-load-default"
    fi
    exec env \
      ADMIN_TOKEN="${ADMIN_TOKEN}" \
      BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL}" \
      ALLOWED_MODEL_ROOT="${ALLOWED_MODEL_ROOT}" \
      PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" -u -m "${CONTROLLER_MODULE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --static-dir "${FRONT_DIR}/dist" \
        --backend-script "${BACKEND_SCRIPT}" \
        --backend-root "${ROOT_DIR}" \
        --backend-health-url "${BACKEND_HEALTH_URL}" \
        --presets-file "${PRESETS_FILE}" \
        --allowed-root "${ALLOWED_MODEL_ROOT}" \
        ${AUTO_LOAD_FLAG}
    ;;
  dev)
    # dev 模式: 还是走 vue-cli 的 8084,但额外启一个 controller (默认 8085)
    CONTROLLER_PORT="${CONTROLLER_PORT:-8085}"
    echo "[frontend-dev] starting lychee_fd.controller on ${HOST}:${CONTROLLER_PORT} (separate)"
    AUTO_LOAD_FLAG=""
    if [[ "${AUTO_LOAD_DEFAULT}" == "1" ]]; then
      AUTO_LOAD_FLAG="--auto-load-default"
    fi
    env \
      ADMIN_TOKEN="${ADMIN_TOKEN}" \
      BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL}" \
      ALLOWED_MODEL_ROOT="${ALLOWED_MODEL_ROOT}" \
      PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" -u -m "${CONTROLLER_MODULE}" \
        --host "${HOST}" \
        --port "${CONTROLLER_PORT}" \
        --static-dir "${FRONT_DIR}/dist" \
        --backend-script "${BACKEND_SCRIPT}" \
        --backend-root "${ROOT_DIR}" \
        --backend-health-url "${BACKEND_HEALTH_URL}" \
        --presets-file "${PRESETS_FILE}" \
        --allowed-root "${ALLOWED_MODEL_ROOT}" \
        ${AUTO_LOAD_FLAG} &
    CONTROLLER_PID=$!
    trap "kill ${CONTROLLER_PID} 2>/dev/null || true" EXIT INT TERM
    echo "[frontend-dev] controller pid=${CONTROLLER_PID}"
    run_vue_cli_with_dev_config "${VUE_CLI_BIN}" serve -- --host "${HOST}" --port "${PORT}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 [prod|dev] [public|private]" >&2
    exit 2
    ;;
esac
