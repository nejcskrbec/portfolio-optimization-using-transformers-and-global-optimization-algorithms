#!/usr/bin/env bash
# Bring the pipeline from a fresh clone to runnable, or diagnose an existing tree.
#
#   bash setup.sh            # install everything that is missing, then verify
#   bash setup.sh --check    # verify only, change nothing (exit 1 if not ready)
#   bash setup.sh --force    # also recreate the conda env from environment.yml
#
# Steps: prerequisites -> conda env -> vendored transformers -> C++ optimizer
#        -> import + binary + dry-run verification.
#
# Safe to re-run: every step is idempotent and skips work already done.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="magistrska"
MODE="install"
case "${1:-}" in
  --check) MODE="check" ;;
  --force) MODE="force" ;;
  "")      MODE="install" ;;
  *) echo "usage: bash setup.sh [--check|--force]" >&2; exit 2 ;;
esac

FAIL=0
step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }

# --- 1. prerequisites --------------------------------------------------------
step "Prerequisites"
for tool in git conda make; do
  if command -v "$tool" >/dev/null 2>&1; then ok "$tool"; else bad "$tool not found in PATH"; fi
done
if command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1; then
  ok "C++ compiler"
else
  bad "no C++ compiler (need g++ or clang++; on macOS: xcode-select --install)"
fi
[ "$FAIL" -gt 0 ] && { echo; echo "Missing prerequisites; fix the above first."; exit 1; }

CONDA_BASE="$(conda info --base 2>/dev/null)"
# shellcheck disable=SC1091
[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ] && . "${CONDA_BASE}/etc/profile.d/conda.sh"
CONDA_RUN=(conda run --no-capture-output -n "${ENV_NAME}")

# --- 2. conda environment ----------------------------------------------------
step "Conda environment '${ENV_NAME}'"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [ "$MODE" = "force" ]; then
    echo "  updating from environment.yml ..."
    conda env update -n "${ENV_NAME}" -f "${ROOT}/environment.yml" --prune >/dev/null \
      && ok "environment updated" || bad "conda env update failed"
  else
    ok "exists (use --force to update from environment.yml)"
  fi
elif [ "$MODE" = "check" ]; then
  bad "environment '${ENV_NAME}' does not exist -- run: bash setup.sh"
else
  echo "  creating from environment.yml (this takes a few minutes) ..."
  conda env create -f "${ROOT}/environment.yml" >/dev/null \
    && ok "environment created" || bad "conda env create failed"
fi

# --- 3. vendored transformer sources ----------------------------------------
# MASTER: master_lib = upstream-at-pin + master_lib.patch (the patch carries a
# silent-failure fix; setup_master_official.sh verifies it survived).
# PatchTST: imported straight out of the clone, so the clone IS the dependency.
step "Vendored transformer sources"

if [ -f "${ROOT}/estimators/master_lib/master.py" ] && [ -f "${ROOT}/estimators/master_lib/base_model.py" ]; then
  ok "MASTER core present (estimators/master_lib)"
elif [ "$MODE" = "check" ]; then
  bad "estimators/master_lib is missing -- run: bash estimators/setup_master_official.sh"
else
  echo "  installing MASTER core ..."
  bash "${ROOT}/estimators/setup_master_official.sh" >/dev/null 2>&1 \
    && ok "MASTER core installed" \
    || bad "setup_master_official.sh failed -- run it directly to see why"
fi

if [ -f "${ROOT}/estimators/patchtst_official/PatchTST_supervised/layers/PatchTST_backbone.py" ]; then
  ok "PatchTST source present"
elif [ "$MODE" = "check" ]; then
  bad "PatchTST source missing -- run: bash estimators/setup_patchtst_official.sh"
else
  echo "  cloning PatchTST at its pinned commit ..."
  bash "${ROOT}/estimators/setup_patchtst_official.sh" >/dev/null 2>&1 \
    && ok "PatchTST installed" \
    || bad "setup_patchtst_official.sh failed -- run it directly to see why"
fi

# --- 4. C++ optimizer --------------------------------------------------------
step "C++ optimizer (PSO + SA)"
BIN="${ROOT}/portfolio_optimizers/portfolio_optimizer"
NEEDS_BUILD=0
if [ ! -x "$BIN" ]; then
  NEEDS_BUILD=1
else
  for src in main.cpp pso.cpp sa.cpp portfolio_common.h; do
    [ "${ROOT}/portfolio_optimizers/${src}" -nt "$BIN" ] && NEEDS_BUILD=1
  done
fi

if [ "$NEEDS_BUILD" = "0" ]; then
  ok "binary up to date"
elif [ "$MODE" = "check" ]; then
  [ -x "$BIN" ] && bad "binary is older than its sources -- run: make -C portfolio_optimizers" \
                || bad "binary not built -- run: make -C portfolio_optimizers"
else
  echo "  building ..."
  if make -C "${ROOT}/portfolio_optimizers" >/dev/null 2>&1; then ok "built"; else bad "make failed"; fi
fi

# --- 5. verification ---------------------------------------------------------
step "Verification"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if "${CONDA_RUN[@]}" python -c "
import sys; sys.path.insert(0, '${ROOT}')
from estimators.master_us   import train_master_us
from estimators.tft_mu      import train_tft
from estimators.patchtst_mu import train_patchtst
from portfolio_optimizers.bridge import run_optimizer, find_binary
" >/dev/null 2>&1; then
    ok "all three estimators + optimizer bridge import"
  else
    bad "import check failed -- rerun without redirection to see the traceback:"
    echo "        conda run -n ${ENV_NAME} python -c 'import estimators.tft_mu'"
  fi

  if "${CONDA_RUN[@]}" python "${ROOT}/benchmark/run.py" all --dry-run >/dev/null 2>&1; then
    ok "benchmark wiring (all --dry-run)"
  else
    bad "benchmark dry-run failed -- try: conda run -n ${ENV_NAME} python benchmark/run.py all --dry-run"
  fi
else
  warn "environment missing; skipped import and dry-run checks"
fi

[ -x "$BIN" ] && ok "optimizer binary executable" || bad "optimizer binary missing"

# --- summary -----------------------------------------------------------------
echo
if [ "$FAIL" -eq 0 ]; then
  printf '\033[32mPipeline ready.\033[0m\n\n'
  cat <<EOF
Next steps:
  conda activate ${ENV_NAME}
  python -u benchmark/run.py all --dry-run     # wiring only, seconds
  python -u benchmark/run.py leow --smoke      # short real run
  python -u benchmark/run.py all               # full thesis evaluation (hours, needs network)
EOF
  exit 0
else
  printf '\033[31m%d check(s) failed.\033[0m See the messages above.\n' "$FAIL"
  exit 1
fi
