#!/usr/bin/env bash
# Install the official MASTER core into estimators/master_lib/.
#
# master_lib is NOT a plain copy of upstream: it is upstream-at-the-pin plus
# master_lib.patch. The patch carries two thesis-critical changes that must
# never be silently lost (see the patch header for the full rationale):
#
#   1. drop_extreme() small-universe guard for N < 40. Without it the whole
#      batch is masked -> nan loss -> zero gradient, and MASTER trains to
#      nothing WITHOUT raising. This is the dangerous one: it is silent.
#   2. fit() epoch selection / early stopping (patience, max_epochs,
#      returns best_epoch). master_us.py calls this signature, so losing it
#      is at least loud (TypeError).
#
# Therefore this script:
#   * treats master_official_commit.txt as the authoritative INPUT pin
#     (it used to clone latest and then overwrite the pin with HEAD, which
#     meant the "pin" pinned nothing and drifted);
#   * rebuilds master_lib from upstream + patch on every run, so re-running
#     is idempotent and byte-identical rather than clobbering;
#   * verifies both changes survived, and refuses to install if not.
#
# Usage:
#   bash estimators/setup_master_official.sh                # install / repair
#   bash estimators/setup_master_official.sh --regen-patch  # after editing master_lib
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${HERE}/master_official"
LIB="${HERE}/master_lib"
PATCH="${HERE}/master_lib.patch"
PIN_FILE="${HERE}/master_official_commit.txt"
REPO="https://github.com/SJTU-DMTai/MASTER.git"

MODE="${1:-install}"

if [[ ! -f "${PIN_FILE}" ]]; then
  echo "ERROR: missing pin file ${PIN_FILE}" >&2
  echo "       The pinned commit is required for reproducibility; refusing to" >&2
  echo "       guess by cloning upstream HEAD." >&2
  exit 1
fi
PIN="$(tr -d '[:space:]' < "${PIN_FILE}")"

# --- fetch upstream at the pinned commit -----------------------------------
if [[ ! -d "${DEST}/.git" ]]; then
  echo "Cloning MASTER into ${DEST} ..."
  rmdir "${DEST}" 2>/dev/null || true
  git clone --quiet "${REPO}" "${DEST}"
fi

if ! git -C "${DEST}" cat-file -e "${PIN}^{commit}" 2>/dev/null; then
  echo "Pinned commit ${PIN} not present locally; fetching ..."
  git -C "${DEST}" fetch --quiet origin || true
fi

if ! git -C "${DEST}" cat-file -e "${PIN}^{commit}" 2>/dev/null; then
  echo "ERROR: pinned commit ${PIN} does not exist in ${REPO}." >&2
  echo "       Do NOT 'fix' this by repinning to HEAD -- master_lib.patch is" >&2
  echo "       built against ${PIN} and may not apply to a different tree." >&2
  exit 1
fi

git -C "${DEST}" checkout --quiet --detach "${PIN}"
echo "Upstream MASTER checked out at pin ${PIN}"

# --- regen mode: refresh the patch from the current master_lib -------------
if [[ "${MODE}" == "--regen-patch" ]]; then
  diff -u --label a/base_model.py --label b/base_model.py \
    "${DEST}/base_model.py" "${LIB}/base_model.py" > "${PATCH}.new" || true
  if [[ ! -s "${PATCH}.new" ]]; then
    echo "ERROR: regenerated patch is empty -- master_lib matches upstream." >&2
    echo "       That would mean the local changes are already lost." >&2
    rm -f "${PATCH}.new"
    exit 1
  fi
  mv "${PATCH}.new" "${PATCH}"
  echo "Regenerated ${PATCH} ($(wc -l < "${PATCH}") lines) against pin ${PIN}."
  echo "Review it with: git diff -- ${PATCH}"
  exit 0
fi

# --- build master_lib = upstream + patch, in a staging dir -----------------
if [[ ! -f "${PATCH}" ]]; then
  echo "ERROR: missing ${PATCH}. Refusing to install an unpatched master_lib," >&2
  echo "       which would silently disable the N<40 drop_extreme guard." >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

cp "${DEST}/master.py" "${STAGE}/master.py"
cp "${DEST}/base_model.py" "${STAGE}/base_model.py"

if ! patch --silent -p1 "${STAGE}/base_model.py" < "${PATCH}"; then
  echo "ERROR: master_lib.patch did not apply to upstream base_model.py at ${PIN}." >&2
  echo "       Upstream probably moved. Resolve by hand and re-run with" >&2
  echo "       --regen-patch; do NOT install the unpatched file." >&2
  exit 1
fi

# --- verify both changes actually survived ---------------------------------
python3 - "${STAGE}/base_model.py" <<'PY'
import ast, sys

src = open(sys.argv[1]).read()
tree = ast.parse(src)
fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
errs = []

drop = fns.get("drop_extreme")
if drop is None:
    errs.append("drop_extreme() is missing entirely")
else:
    # The guard must be a real early return, not just a comment.
    has_guard = any(
        isinstance(node, ast.If) and any(isinstance(b, ast.Return) for b in node.body)
        for node in ast.walk(drop)
    )
    if not has_guard:
        errs.append(
            "drop_extreme() has no early-return guard -- the N<40 small-universe "
            "fix is NOT present. This regresses SILENTLY (nan loss, zero grad)."
        )

fit = fns.get("fit")
if fit is None:
    errs.append("fit() is missing entirely")
else:
    args = {a.arg for a in fit.args.args} | {a.arg for a in fit.args.kwonlyargs}
    for needed in ("patience", "max_epochs"):
        if needed not in args:
            errs.append(f"fit() does not accept '{needed}' -- master_us.py will TypeError")

if errs:
    print("VERIFICATION FAILED:", file=sys.stderr)
    for e in errs:
        print("  - " + e, file=sys.stderr)
    sys.exit(1)
print("Verified: drop_extreme guard + fit(patience, max_epochs) both present.")
PY

# --- install ----------------------------------------------------------------
mkdir -p "${LIB}"
touch "${LIB}/__init__.py"
mv "${STAGE}/master.py" "${LIB}/master.py"
mv "${STAGE}/base_model.py" "${LIB}/base_model.py"

echo "Official MASTER core installed."
echo "Pin:   ${PIN}"
echo "Path:  ${LIB} (upstream + master_lib.patch)"
