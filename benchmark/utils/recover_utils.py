#!/usr/bin/env python3
"""Rebuild result CSVs from the stdout logs in test_results/_run_logs/.

The literature result directories under test_results/ were emptied on
2026-09-20 and were never tracked by git (`.gitignore:31`), so there is no
copy to restore from. What survived is the captured stdout of the runs that
produced the thesis numbers. The engine prints several of its result frames
verbatim (`DataFrame.to_string()`), so those frames can be read back.

Run from the repo root:
    python benchmark/utils/recover_utils.py            # write the CSVs
    python benchmark/utils/recover_utils.py --dry-run  # report only

What is and is not recoverable
------------------------------
Recovered, in full:
    forecast_summary.csv    the FORECAST SUMMARY block
    portfolio_summary.csv   the PORTFOLIO SUMMARY block
    risk_matched.csv        the RISK-MATCHED COMPARISON block
    wang_sharpe_table.csv   the "Our pipeline" block (Wang only)
    timing_train.csv        aggregated from the per-fit `[<model>] train <n>s`
                            lines; `n_assets` is dropped, it is never printed

NOT recoverable, and deliberately not faked:
    prediction_results.csv  per-asset mu, never printed
    portfolio_results.csv   per-decision rows, never printed -- so equity
                            curves cannot be redrawn without a real re-run
    aggregate_summary.csv, estimator_vs_historical.csv, risk_frontier.csv
    wang_sharpe_table's `sharpe_std` column (the printed table omits it; only
    its max/min survive, in thesis-paper/generated/wang_macros.tex)

Precision: the logs print what `to_string()` formatted, so recovered values
carry only the printed decimals -- 4 for the risk-matched block. Regenerating
the thesis tables from a recovered directory reproduces
`tab_leow_metrics_rows.tex` and `tab_wang_rows.tex` byte for byte, but moves
two 3-decimal macros in `leow_macros.tex` by one in the last place (Sharpe
1,416 -> 1,415). Treat the committed *.tex as authoritative over anything
regenerated from here.

Every directory written gets a RECOVERED.md recording exactly this.

Which log is canonical
----------------------
Several logs cover the same benchmark from different tuning attempts. The one
named per benchmark below is the run whose numbers match the committed
thesis-paper/generated/*.tex, verified value by value -- not simply the newest
file. For All-Weather that distinction matters: `leow_then_wang_chain.log` is
a *different* TFT run (risk-matched Sharpe 1.349 vs the thesis' 1.289) and
would silently install wrong numbers.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "test_results" / "_run_logs"
RESULTS = ROOT / "test_results" / "literature"

# benchmark output_dir -> (log stem, section index within that log)
# `section` picks the run when one log chains several benchmarks: 0 is the
# first `[benchmark] config:` block, 1 the second.
SOURCES = {
    "leow_allweather_direct": ("leow_val63_patience8", 0),
    "wang_sp500":             ("leow_then_wang_chain", 1),
    "aprea_djia":             ("aprea_djia_run", 0),
    "aprea_nasdaq100":        ("aprea_nasdaq_run", 0),
    "longterm_investor":      ("longterm_investor_run", 0),
}

# Blocks printed as DataFrame.to_string(). `text_col` names the one column
# whose values contain spaces, so a row can have more tokens than the header.
BLOCKS = {
    "forecast_summary.csv":  ("FORECAST SUMMARY", None),
    "portfolio_summary.csv": ("PORTFOLIO SUMMARY", None),
    "risk_matched.csv":      ("RISK-MATCHED COMPARISON", "rule"),
    "wang_sharpe_table.csv": ("=== Our pipeline", "method"),
}

_TRAIN = re.compile(r"^\s*\[([A-Za-z0-9-]+)\] train ([0-9.]+)s\s*$")
_CONFIG = re.compile(r"^\[benchmark\] config:")


def _sections(lines: list[str]) -> list[tuple[int, int]]:
    """(start, end) line spans, one per `[benchmark] config:` block in a log."""
    starts = [i for i, l in enumerate(lines) if _CONFIG.match(l)]
    if not starts:
        return [(0, len(lines))]
    # A section begins at its banner, which precedes the config line by one.
    starts = [max(0, i - 1) for i in starts]
    return [(s, starts[k + 1] if k + 1 < len(starts) else len(lines))
            for k, s in enumerate(starts)]


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _find_header(lines: list[str], marker: str) -> int | None:
    """Line index of the table header that follows `marker`.

    A header is recognised by its tokens all being snake_case identifiers,
    which is what separates it from the prose the risk-matched block prints
    first ("Each model is read at the P whose realized volatility ...").
    Matching on token count alone is not enough -- that prose is wider than
    the header it precedes.
    """
    for i, l in enumerate(lines):
        if marker not in l:
            continue
        for j in range(i + 1, min(i + 12, len(lines))):
            tok = lines[j].split()
            if len(tok) >= 3 and all(_IDENT.match(t) for t in tok):
                return j
        return None
    return None


def _parse_block(lines: list[str], header_idx: int, text_col: str | None):
    """Read a to_string() table back into a DataFrame.

    Rows stop at the first blank line or rule-off. A row with more tokens than
    the header has them absorbed into `text_col`, which is the only column
    whose values contain spaces (`best Sharpe at<=target`, `Thesis TFT`).
    """
    names = lines[header_idx].split()
    ti = names.index(text_col) if text_col and text_col in names else None
    rows = []
    for l in lines[header_idx + 1:]:
        s = l.rstrip()
        if not s.strip() or set(s.strip()) <= {"="} or s.startswith(("Saved:", "saved:", "[")):
            break
        tok = s.split()
        if len(tok) != len(names):
            if ti is None or len(tok) < len(names):
                raise ValueError(
                    f"cannot align row against header ({len(tok)} tokens vs "
                    f"{len(names)} columns) and no text column to absorb the "
                    f"difference:\n  {s[:120]}"
                )
            extra = len(tok) - len(names)
            tok = tok[:ti] + [" ".join(tok[ti:ti + 1 + extra])] + tok[ti + 1 + extra:]
        rows.append(tok)
    if not rows:
        raise ValueError(f"header at line {header_idx + 1} has no data rows")
    df = pd.DataFrame(rows, columns=names)
    for c in df.columns:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().all():        # leave mixed/text columns as strings
            df[c] = num
    return df


def _timing(lines: list[str], paper: str | None) -> pd.DataFrame | None:
    """Aggregate the per-fit `[<model>] train <n>s` lines like walkforward does.

    walkforward.py:786 groups the same numbers by model; `n_assets` is the one
    column it also stores that is never printed, so it is omitted here rather
    than invented.
    """
    fits: dict[str, list[float]] = {}
    for l in lines:
        m = _TRAIN.match(l)
        if m:
            fits.setdefault(m.group(1), []).append(float(m.group(2)))
    if not fits:
        return None
    df = pd.DataFrame(
        [{"model": k, "n_trainings": len(v),
          "train_sec_mean": sum(v) / len(v), "train_sec_total": sum(v)}
         for k, v in sorted(fits.items())]
    )
    if paper:
        df.insert(0, "paper", paper)
    return df.round(2)


NOTE = """# Recovered from stdout, not a real run

These CSVs were rebuilt by `benchmark/utils/recover_utils.py` from
`test_results/_run_logs/{log}.log` (section {section}). The original run
directory was emptied on 2026-09-20; `test_results/` is gitignored, so no
copy existed in any branch.

Recovered here: {written}

Missing, and NOT reconstructable from stdout -- it was never printed:

  * `prediction_results.csv` (per-asset mu)
  * `portfolio_results.csv` (per-decision rows). Note this is what
    `run.py equity-curves-all` reads, so the equity-curve figures cannot be
    redrawn from this directory.
  * `aggregate_summary.csv`, `estimator_vs_historical.csv`, `risk_frontier.csv`
  * `timing_train.csv`'s `n_assets` column
{extra}
Values carry only the decimals the log printed (4 in the risk-matched block),
so the committed `thesis-paper/generated/*.tex` stays authoritative over
anything regenerated from here. See the module docstring for how the source
log was chosen and how the numbers were verified.
"""


def recover(run: str, log_stem: str, section: int, dry: bool) -> list[str]:
    log = LOGS / f"{log_stem}.log"
    if not log.exists():
        print(f"  skip {run}: {log} missing")
        return []
    lines = log.read_text(errors="ignore").split("\n")
    spans = _sections(lines)
    if section >= len(spans):
        print(f"  skip {run}: {log.name} has {len(spans)} section(s), wanted {section}")
        return []
    lo, hi = spans[section]
    block = lines[lo:hi]

    out = RESULTS / run
    written = []
    for fname, (marker, text_col) in BLOCKS.items():
        idx = _find_header(block, marker)
        if idx is None:
            continue
        df = _parse_block(block, idx, text_col)
        if not dry:
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / fname, index=False)
        written.append(f"{fname} ({len(df)}x{len(df.columns)})")

    paper = next((l.split(":", 1)[1].strip() for l in block
                  if l.startswith("[benchmark] config:")), None)
    t = _timing(block, Path(paper).stem if paper else None)
    if t is not None:
        if not dry:
            out.mkdir(parents=True, exist_ok=True)
            t.to_csv(out / "timing_train.csv", index=False)
        written.append(f"timing_train.csv ({len(t)} models, "
                       f"{int(t['n_trainings'].sum())} fits)")

    if written and not dry:
        extra = ""
        if run == "wang_sp500":
            extra = ("  * `wang_sharpe_table.csv`'s `sharpe_std` column (the printed\n"
                     "    table omits it; only max/min survive, in wang_macros.tex)\n")
        (out / "RECOVERED.md").write_text(
            NOTE.format(log=log_stem, section=section,
                        written=", ".join(written), extra=extra),
            encoding="utf-8")
    print(f"  {run:26} <- {log.name} [{section}]: " +
          (", ".join(written) if written else "nothing found"))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and report, write nothing.")
    args = ap.parse_args(argv)

    print(f"Recovering result CSVs from {LOGS.relative_to(ROOT)}"
          + (" (dry run)" if args.dry_run else ""))
    total = 0
    for run, (stem, section) in SOURCES.items():
        total += len(recover(run, stem, section, args.dry_run))
    print(f"Done: {total} file(s).")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
