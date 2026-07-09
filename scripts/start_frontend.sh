#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   ./scripts/start_frontend.sh [prod|dev] [public|private]
#
# prod: 先构建，再用 python http.server 提供 dist 静态文件（推荐用于时延测试）。
# dev:  启动 vue-cli 开发服务器，便于前端迭代修改。
# public:  绑定到 0.0.0.0（若防火墙/安全组允许，可被外网访问）。
# private: 绑定到 127.0.0.1（仅本机可访问）。

MODE="${1:-prod}"
EXPOSE_MODE="${2:-${FRONTEND_EXPOSE_MODE:-public}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONT_DIR="${ROOT_DIR}/frontend"
PORT="${FRONTEND_PORT:-8084}"
FORCE_DISABLE_PROXY="${FRONTEND_FORCE_DISABLE_PROXY:-1}"
REALTIME_DEBUG="${LYCHEEFD_REALTIME_DEBUG:-0}"
export VUE_APP_REALTIME_DEBUG="${VUE_APP_REALTIME_DEBUG:-${REALTIME_DEBUG}}"

case "${EXPOSE_MODE}" in
  public)
    HOST="${FRONTEND_HOST:-0.0.0.0}"
    ;;
  private)
    HOST="${FRONTEND_HOST:-127.0.0.1}"
    ;;
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

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found in PATH" >&2
  exit 1
fi

pick_python() {
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo ""
}

build_no_proxy_csv() {
  local raw combined host_ips ip
  raw="${NO_PROXY:-},${no_proxy:-},127.0.0.1,localhost,::1"
  if [[ "${HOST}" != "0.0.0.0" && "${HOST}" != "::" ]]; then
    raw="${raw},${HOST}"
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

cd "${FRONT_DIR}"

case "${MODE}" in
  prod)
    echo "[frontend] mode=prod expose=${EXPOSE_MODE} host=${HOST} port=${PORT}"
    echo "[frontend] realtime_debug=${VUE_APP_REALTIME_DEBUG}"
    echo "[frontend] force_disable_proxy=${FORCE_DISABLE_PROXY}"
    echo "[frontend] NO_PROXY=${NO_PROXY}"
    echo "[frontend] http_proxy=${http_proxy:-<unset>} https_proxy=${https_proxy:-<unset>}"
    echo "[frontend] note: browser proxy also needs bypass for 127.0.0.1/localhost/your-server-ip."
    npm run build
    PYTHON_CMD="$(pick_python)"
    if [[ -z "${PYTHON_CMD}" ]]; then
      echo "Neither python nor python3 found for static file serving." >&2
      exit 1
    fi
    cd dist
    exec "${PYTHON_CMD}" -m http.server "${PORT}" --bind "${HOST}"
    ;;
  dev)
    echo "[frontend] mode=dev expose=${EXPOSE_MODE} host=${HOST} port=${PORT}"
    echo "[frontend] realtime_debug=${VUE_APP_REALTIME_DEBUG}"
    echo "[frontend] force_disable_proxy=${FORCE_DISABLE_PROXY}"
    echo "[frontend] NO_PROXY=${NO_PROXY}"
    echo "[frontend] http_proxy=${http_proxy:-<unset>} https_proxy=${https_proxy:-<unset>}"
    echo "[frontend] note: browser proxy also needs bypass for 127.0.0.1/localhost/your-server-ip."
    exec npm run serve -- --host "${HOST}" --port "${PORT}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 [prod|dev] [public|private]" >&2
    exit 2
    ;;
esac
