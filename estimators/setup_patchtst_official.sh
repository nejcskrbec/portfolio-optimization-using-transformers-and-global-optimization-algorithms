#!/usr/bin/env bash
# Install the official PatchTST repo at the pinned commit.
#
# Unlike MASTER there is no vendored copy and no local patch: patchtst_mu.py and
# benchmark/tasks/task_predictive.py import straight out of
# patchtst_official/PatchTST_supervised, so the clone IS the dependency.
#
# This script treats patchtst_official_commit.txt as the authoritative INPUT pin.
# It previously cloned --depth 1 (i.e. upstream HEAD) and then OVERWROTE the pin
# file with whatever it got, so the "pin" pinned nothing -- the same drift that
# left MASTER's pin disagreeing with its recorded gitlink. Re-running could
# silently move the thesis onto a different PatchTST than the results claim.
#
# Usage:
#   bash estimators/setup_patchtst_official.sh           # install / repair at the pin
#   REPIN=1 bash estimators/setup_patchtst_official.sh   # deliberately move pin to HEAD
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/patchtst_official"
PIN_FILE="${HERE}/patchtst_official_commit.txt"
REPO="https://github.com/yuqinie98/PatchTST.git"
BACKBONE="PatchTST_supervised/layers/PatchTST_backbone.py"
REPIN="${REPIN:-0}"

if [[ "${REPIN}" != "1" && ! -f "${PIN_FILE}" ]]; then
  echo "ERROR: missing pin file ${PIN_FILE}" >&2
  echo "       Refusing to guess by cloning upstream HEAD. If you really mean" >&2
  echo "       to adopt HEAD, re-run with REPIN=1." >&2
  exit 1
fi

# --- clone if absent (full clone: a shallow one cannot check out an old pin) ---
if [[ ! -d "${DEST}/.git" ]]; then
  echo "Cloning PatchTST into ${DEST} ..."
  rmdir "${DEST}" 2>/dev/null || true
  git clone --quiet "${REPO}" "${DEST}"
fi

if [[ "${REPIN}" == "1" ]]; then
  git -C "${DEST}" fetch --quiet origin
  git -C "${DEST}" checkout --quiet --detach origin/HEAD
  git -C "${DEST}" rev-parse HEAD > "${PIN_FILE}"
  echo "REPINNED to upstream HEAD: $(cat "${PIN_FILE}")"
  echo "Commit the updated pin file and re-run your benchmarks."
else
  PIN="$(tr -d '[:space:]' < "${PIN_FILE}")"

  if ! git -C "${DEST}" cat-file -e "${PIN}^{commit}" 2>/dev/null; then
    echo "Pinned commit ${PIN} not present locally; fetching ..."
    git -C "${DEST}" fetch --quiet origin || true
  fi

  if ! git -C "${DEST}" cat-file -e "${PIN}^{commit}" 2>/dev/null; then
    echo "ERROR: pinned commit ${PIN} does not exist in ${REPO}." >&2
    echo "       Do NOT silently adopt HEAD -- that changes which PatchTST the" >&2
    echo "       thesis results were produced with. Use REPIN=1 deliberately." >&2
    exit 1
  fi

  git -C "${DEST}" checkout --quiet --detach "${PIN}"
fi

# --- verify the import target the pipeline actually needs --------------------
if [[ ! -f "${DEST}/${BACKBONE}" ]]; then
  echo "ERROR: ${BACKBONE} not found at $(git -C "${DEST}" rev-parse HEAD)." >&2
  echo "       patchtst_mu.py imports from PatchTST_supervised and will fail." >&2
  exit 1
fi

echo "Official PatchTST installed."
echo "Pin:   $(git -C "${DEST}" rev-parse HEAD)"
echo "Path:  ${DEST}/PatchTST_supervised"
