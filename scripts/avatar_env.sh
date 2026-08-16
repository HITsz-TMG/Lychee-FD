#!/usr/bin/env bash

# Shared configuration helpers for the integrated LiveAct avatar launchers.
# Callers must define ROOT_DIR before sourcing this file.

AVATAR_ENV_FILE="${AVATAR_ENV_FILE:-${ROOT_DIR}/.env.avatar}"
if [[ -f "${AVATAR_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${AVATAR_ENV_FILE}"
  set +a
fi

require_avatar_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[avatar-config][ERROR] ${name} is required." >&2
    echo "[avatar-config][ERROR] Copy .env.avatar.example to .env.avatar and edit it, or export ${name}." >&2
    exit 1
  fi
}
