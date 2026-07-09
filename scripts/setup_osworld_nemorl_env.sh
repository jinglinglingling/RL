#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
OSWORLD_SRC="${OSWORLD_SRC:-${PROJECT_ROOT}/third_party/OSWorld}"
NEMORL_ROOT="${NEMORL_ROOT:-${PROJECT_ROOT}/../Nemo-RL-main-1/RL-merge-2689}"

AUTO_CLONE_OSWORLD="${AUTO_CLONE_OSWORLD:-0}"
OSWORLD_REPO_URL="${OSWORLD_REPO_URL:-https://github.com/xlang-ai/OSWorld.git}"
INSTALL_WORKSPACE_DEPS="${INSTALL_WORKSPACE_DEPS:-1}"
INSTALL_OSWORLD_DEPS="${INSTALL_OSWORLD_DEPS:-1}"

echo "=== Setup OSWorld + NeMo-RL workspace ==="
echo "PROJECT_ROOT:           ${PROJECT_ROOT}"
echo "PYTHON_BIN:             ${PYTHON_BIN}"
echo "OSWORLD_SRC:            ${OSWORLD_SRC}"
echo "NEMORL_ROOT:            ${NEMORL_ROOT}"
echo "AUTO_CLONE_OSWORLD:     ${AUTO_CLONE_OSWORLD}"
echo "INSTALL_WORKSPACE_DEPS: ${INSTALL_WORKSPACE_DEPS}"
echo "INSTALL_OSWORLD_DEPS:   ${INSTALL_OSWORLD_DEPS}"
echo

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[FATAL] python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${NEMORL_ROOT}" ]]; then
  echo "[FATAL] NEMORL_ROOT not found: ${NEMORL_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${OSWORLD_SRC}" ]]; then
  if [[ "${AUTO_CLONE_OSWORLD}" == "1" ]]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "[FATAL] git is required for AUTO_CLONE_OSWORLD=1" >&2
      exit 1
    fi
    mkdir -p "$(dirname "${OSWORLD_SRC}")"
    git clone "${OSWORLD_REPO_URL}" "${OSWORLD_SRC}"
  else
    echo "[FATAL] OSWORLD_SRC not found: ${OSWORLD_SRC}" >&2
    echo "Set AUTO_CLONE_OSWORLD=1 or clone manually." >&2
    exit 1
  fi
fi

if [[ "${INSTALL_WORKSPACE_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

if [[ "${INSTALL_OSWORLD_DEPS}" == "1" ]]; then
  if [[ ! -f "${OSWORLD_SRC}/requirements.txt" ]]; then
    echo "[FATAL] Missing ${OSWORLD_SRC}/requirements.txt" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m pip install -r "${OSWORLD_SRC}/requirements.txt"
fi

echo
echo "✅ Setup finished."
