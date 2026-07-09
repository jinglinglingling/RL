#!/usr/bin/env bash
# Build an OSWorld runtime SIF from Docker without root privileges.
#
# This script bootstraps a user-space Apptainer binary from a .deb package,
# then runs `apptainer pull` to convert a Docker image into a .sif file.
#
# Example:
#   bash scripts/slurm/build_osworld_sif_with_local_apptainer.sh
#   OSWORLD_DOCKER_IMAGE=happysixd/osworld-docker:latest \
#   OUTPUT_DIR=/lustre/.../osworld-sif \
#   bash scripts/slurm/build_osworld_sif_with_local_apptainer.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APPTAINER_VERSION="${APPTAINER_VERSION:-1.4.1}"
APPTAINER_DEB_URL="${APPTAINER_DEB_URL:-https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_amd64.deb}"
APPTAINER_WORKDIR="${APPTAINER_WORKDIR:-${HOME}/local/apptainer-${APPTAINER_VERSION}}"
APPTAINER_DEB="${APPTAINER_WORKDIR}/apptainer_${APPTAINER_VERSION}_amd64.deb"
APPTAINER_EXTRACT="${APPTAINER_WORKDIR}/extract"
APPTAINER_BIN="${APPTAINER_EXTRACT}/usr/bin/apptainer"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${APPTAINER_WORKDIR}/cache}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${APPTAINER_WORKDIR}/tmp}"

# OSWorld Docker provider uses happysixd/osworld-docker in:
# third_party/OSWorld/desktop_env/providers/docker/provider.py
OSWORLD_DOCKER_IMAGE="${OSWORLD_DOCKER_IMAGE:-happysixd/osworld-docker:latest}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/sif-images}"
OSWORLD_SIF_NAME="${OSWORLD_SIF_NAME:-}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
VERIFY_RUN="${VERIFY_RUN:-0}"

mkdir -p "${APPTAINER_WORKDIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${OUTPUT_DIR}"

if [[ ! -x "${APPTAINER_BIN}" ]]; then
  echo "[setup] Bootstrapping Apptainer ${APPTAINER_VERSION} under ${APPTAINER_WORKDIR}"
  if [[ ! -f "${APPTAINER_DEB}" ]]; then
    wget -q "${APPTAINER_DEB_URL}" -O "${APPTAINER_DEB}"
  fi
  rm -rf "${APPTAINER_EXTRACT}"
  dpkg-deb -x "${APPTAINER_DEB}" "${APPTAINER_EXTRACT}"
fi

# Some packaged binaries expect config under usr/etc/apptainer.
mkdir -p "${APPTAINER_EXTRACT}/usr/etc"
ln -sfn "../../etc/apptainer" "${APPTAINER_EXTRACT}/usr/etc/apptainer"
# Some builds resolve localstatedir under ${prefix}/var.
mkdir -p "${APPTAINER_EXTRACT}/usr/var/lib/apptainer/mnt/session"
mkdir -p "${APPTAINER_EXTRACT}/usr/var/apptainer/mnt/session"

if [[ ! -x "${APPTAINER_BIN}" ]]; then
  echo "[FATAL] Apptainer binary not found after extraction: ${APPTAINER_BIN}" >&2
  exit 1
fi

if [[ "${OSWORLD_DOCKER_IMAGE}" == docker://* ]]; then
  DOCKER_REF="${OSWORLD_DOCKER_IMAGE}"
  IMAGE_FOR_NAME="${OSWORLD_DOCKER_IMAGE#docker://}"
else
  DOCKER_REF="docker://${OSWORLD_DOCKER_IMAGE}"
  IMAGE_FOR_NAME="${OSWORLD_DOCKER_IMAGE}"
fi

if [[ -z "${OSWORLD_SIF_NAME}" ]]; then
  SAFE_NAME="${IMAGE_FOR_NAME//\//_}"
  SAFE_NAME="${SAFE_NAME//:/_}"
  OSWORLD_SIF_NAME="${SAFE_NAME}.sif"
fi
OUT_SIF="${OUTPUT_DIR}/${OSWORLD_SIF_NAME}"

if [[ -f "${OUT_SIF}" && "${FORCE_REBUILD}" != "1" ]]; then
  echo "[skip] SIF already exists: ${OUT_SIF}"
  exit 0
fi
if [[ -f "${OUT_SIF}" ]]; then
  rm -f "${OUT_SIF}"
fi

export APPTAINER_CACHEDIR
export APPTAINER_TMPDIR

echo "=== Build OSWorld SIF ==="
echo "APPTAINER_BIN:      ${APPTAINER_BIN}"
echo "APPTAINER_CACHEDIR: ${APPTAINER_CACHEDIR}"
echo "APPTAINER_TMPDIR:   ${APPTAINER_TMPDIR}"
echo "DOCKER_REF:         ${DOCKER_REF}"
echo "OUT_SIF:            ${OUT_SIF}"
echo

"${APPTAINER_BIN}" --version
"${APPTAINER_BIN}" pull "${OUT_SIF}" "${DOCKER_REF}"

if [[ "${VERIFY_RUN}" == "1" ]]; then
  echo "[verify] Running a smoke command in the generated SIF..."
  "${APPTAINER_BIN}" exec "${OUT_SIF}" bash -lc 'echo "SIF ready: $(uname -a)"'
fi

echo
echo "Done."
echo "Generated SIF: ${OUT_SIF}"
