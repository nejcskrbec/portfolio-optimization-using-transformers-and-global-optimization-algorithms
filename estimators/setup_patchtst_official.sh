#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/patchtst_official"
REPO="https://github.com/yuqinie98/PatchTST.git"

if [[ -d "${DEST}/.git" ]]; then
  echo "PatchTST official repo already present: ${DEST}"
else
  git clone --depth 1 "${REPO}" "${DEST}"
fi

# Record the exact checkout actually used. Commit this text file together with
# thesis results if you want the run to be reproducible even if upstream moves.
git -C "${DEST}" rev-parse HEAD > "${HERE}/patchtst_official_commit.txt"

if [[ ! -f "${DEST}/PatchTST_supervised/layers/PatchTST_backbone.py" ]]; then
  echo "ERROR: official PatchTST supervised backbone not found" >&2
  exit 1
fi

echo "Official PatchTST installed."
echo "Commit: $(cat "${HERE}/patchtst_official_commit.txt")"
echo "Path:   ${DEST}/PatchTST_supervised"
