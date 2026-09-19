#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/master_official"
LIB="${HERE}/master_lib"
REPO="https://github.com/SJTU-DMTai/MASTER.git"

if [[ -d "${DEST}/.git" ]]; then
  echo "MASTER official repo already present: ${DEST}"
else
  git clone --depth 1 "${REPO}" "${DEST}"
fi

git -C "${DEST}" rev-parse HEAD > "${HERE}/master_official_commit.txt"
mkdir -p "${LIB}"
cp "${DEST}/master.py" "${LIB}/master.py"
cp "${DEST}/base_model.py" "${LIB}/base_model.py"

if [[ ! -f "${LIB}/master.py" || ! -f "${LIB}/base_model.py" ]]; then
  echo "ERROR: official MASTER core files were not installed" >&2
  exit 1
fi

echo "Official MASTER core installed."
echo "Commit: $(cat "${HERE}/master_official_commit.txt")"
echo "Path:   ${LIB}"
