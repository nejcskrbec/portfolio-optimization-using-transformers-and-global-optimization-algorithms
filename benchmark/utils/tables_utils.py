#!/usr/bin/env python3
"""
tables_utils.py
===============
Reads benchmark result CSVs and writes LaTeX fragments to thesis-paper/generated/.

Run from the repo root:
    python benchmark/utils/tables_utils.py

Outputs
-------
thesis-paper/generated/tab_leow_metrics_rows.tex  -- our rows for tab:leow
thesis-paper/generated/leow_macros.tex            -- \newcommand macros, Leow prose
thesis-paper/generated/tab_wang_rows.tex          -- our rows for tab:wang
thesis-paper/generated/wang_macros.tex            -- \newcommand macros, Wang prose

Both tables carry one `Historical@<L>` row per distinct model conditioning
window (see estimators/pipeline.conditioning_window). There is no single "the
historical baseline" any more, so every transformer row is emitted next to the
baseline it is actually matched against.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test_results" / "literature"
OUT = ROOT / "thesis-paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# PSO is the representative metaheuristic used throughout the thesis; both
# solvers are stored per run, and pooling them would concatenate two return
# series into one whose std is neither solver's.
OPTIMIZER = "PSO"
MODEL_ORDER = ["MASTER", "PatchTST", "TFT"]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _num(v: float, dec: int = 3, signed: bool = False) -> str:
    """Slovene decimal-comma LaTeX number."""
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:+.{dec}f}" if signed else f"{v:.{dec}f}"
    return s.replace(".", "{,}")


def _pct(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    return f"${v * 100:+.1f}".replace(".", "{,}") + r"\,\%$"


def _plain_pct(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    return f"${v * 100:.1f}".replace(".", "{,}") + r"\,\%$"


def _dur(sec: float) -> str:
    """Training/solver wall clock, in the largest sensible unit."""
    if sec is None or not np.isfinite(sec) or sec <= 0:
        return "--"
    if sec < 60:
        return _num(sec, 1) + r"\,s"
    if sec < 3600:
        return _num(sec / 60, 1) + r"\,min"
    return _num(sec / 3600, 2) + r"\,h"


def _is_baseline(model: str) -> bool:
    return str(model).startswith("Historical@")


def _display(model: str, n_baselines: int) -> str:
    """Slovenian row label; the window suffix only appears when a run has >1."""
    if _is_baseline(model):
        if n_baselines <= 1:
            return "Zgodovinski"
        return "Zgodovinski\\,(" + str(model).split("@", 1)[1] + "\\,dni)"
    return str(model)


def _order(models) -> list[str]:
    """Baselines first (short window -> long), then transformers."""
    base = sorted((m for m in models if _is_baseline(m)),
                  key=lambda m: int(str(m).split("@", 1)[1]))
    return base + [m for m in MODEL_ORDER if m in set(models)]


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def _portfolio(path: Path, P: float, optimizer: str = OPTIMIZER) -> pd.DataFrame:
    """Per-model realized metrics at one risk point and one solver."""
    df = pd.read_csv(path / "portfolio_summary.csv")
    if "P" in df.columns and P is not None:
        df = df[df["P"] == P]
    if optimizer is not None and "optimizer" in df.columns:
        df = df[df["optimizer"].str.upper() == optimizer.upper()]
    return df.set_index("model")


def _risk_matched(path: Path, optimizer: str = OPTIMIZER) -> pd.DataFrame:
    """Per-model metrics at each model's RISK-MATCHED point (equal realized vol).

    Reads `risk_matched.csv` (written by task_risk_matched): one row per
    (model, optimizer) at the P whose realized volatility matches the target
    (the longest-window historical baseline's vol at the reference P). Used only
    for Leow, where the short single-crash window makes a fixed nominal P an
    unfair operating point and the experiment calls for an equal-risk read.
    """
    df = pd.read_csv(path / "risk_matched.csv")
    if optimizer is not None and "optimizer" in df.columns:
        df = df[df["optimizer"].str.upper() == optimizer.upper()]
    return df.set_index("model")


def _train_seconds(path: Path) -> dict:
    """model -> total training wall clock for the whole run (all seeds/refits).

    Historical rows never appear here: a sample mean has no fitted parameters,
    which is exactly why its lookback is directly comparable to a model's
    conditioning window. They render as `--`.
    """
    f = path / "timing_train.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f)
    return t.groupby("model")["train_sec_total"].sum().to_dict()


def _train_detail(path: Path) -> dict:
    """model -> (n_trainings, seconds per training)."""
    f = path / "timing_train.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f)
    out = {}
    for _, r in t.iterrows():
        n = int(r["n_trainings"])
        out[str(r["model"])] = (n, float(r["train_sec_total"]) / max(n, 1))
    return out


def _matched(path: Path) -> dict:
    """model -> its `Historical@<L>` row, read from the run's own output.

    Prefers the per-decision file, but `matched_baseline` is constant per model
    and is repeated in the summary, so the summary alone is enough.
    """
    f = path / "portfolio_results.csv"
    if not f.exists():
        f = path / "portfolio_summary.csv"
    df = pd.read_csv(f)
    if "matched_baseline" not in df.columns:
        return {}
    return (df[~df["model"].astype(str).str.startswith("Historical@")]
            .drop_duplicates("model")
            .set_index("model")["matched_baseline"].to_dict())


def _mac(name: str, body: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{body}}}"


def _write_rows(target: Path, rows: list[str]) -> None:
    r"""Write tabular rows for `\input` inside a tabular.

    The fragment deliberately does NOT end with `\\`: `\\` scans ahead for an
    optional `[`, that lookahead crosses the end of the inputted file, and the
    following `\hline` then lands outside the alignment (`Misplaced \noalign`).
    The caller supplies the closing `\\` in main.tex instead. The trailing `%`
    swallows the file's final newline so no stray space enters the cell.
    """
    body = "\n".join(rows)
    assert body.endswith(r" \\"), "rows must be emitted with a trailing \\\\"
    target.write_text(body[: -len(r" \\")] + "%\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Leow All-Weather
# ---------------------------------------------------------------------------

def generate_leow(P: float = 0.5) -> None:
    path = RESULTS / "leow_allweather_direct"
    if not path.exists():
        print(f"  skip Leow: {path} missing")
        return

    # Leow is reported RISK-MATCHED (each model at the P whose realized vol equals
    # the matched-baseline's) rather than at a fixed nominal P: the single-crash
    # window makes a fixed P an unfair operating point. Other benchmarks stay
    # fixed-P (Wang cannot be risk-matched -- too few independent observations).
    rm = _risk_matched(path)
    n_decisions = int(_portfolio(path, P)["n_decisions"].iloc[0])
    train = _train_seconds(path)
    matched = _matched(path)
    n_base = sum(1 for m in rm.index if _is_baseline(m))
    pf = rm  # all per-model metrics below are read at the risk-matched point

    rows = []
    for m in _order(pf.index):
        r = pf.loc[m]
        rows.append(
            "All-Weather & {label} & {ret} & {vol} & ${sh}$ & {tr} \\\\".format(
                label=_display(m, n_base),
                ret=_pct(float(r["annualized_arithmetic_return"])),
                vol=_plain_pct(float(r["annualized_vol"])),
                sh=_num(float(r["sharpe_annualized"]), 2, signed=True),
                tr=_dur(train.get(m, float("nan"))),
            )
        )
    _write_rows(OUT / "tab_leow_metrics_rows.tex", rows)
    print(f"  wrote {OUT / 'tab_leow_metrics_rows.tex'} ({len(rows)} rows)")

    macros = []
    for m in pf.index:
        if _is_baseline(m):
            continue
        macros.append(_mac(f"LeowAW{m}", _num(float(pf.loc[m, "sharpe_annualized"]), 3, True)))
        macros.append(_mac(f"LeowAW{m}Ret", _pct(float(pf.loc[m, "annualized_arithmetic_return"]))))
        base = matched.get(m)
        if base in pf.index:
            macros.append(_mac(f"LeowAWHistorical{m}",
                               _num(float(pf.loc[base, "sharpe_annualized"]), 3, True)))
            macros.append(_mac(f"LeowAWHistorical{m}Ret",
                               _pct(float(pf.loc[base, "annualized_arithmetic_return"]))))
            macros.append(_mac(f"LeowAWWindow{m}", str(base).split("@", 1)[1]))
            macros.append(_mac(f"LeowAWMargin{m}",
                               _num(float(pf.loc[m, "sharpe_annualized"])
                                    - float(pf.loc[base, "sharpe_annualized"]), 3, True)))

    total = sum(train.values())
    macros.append(_mac("LeowAWTrainTotal", _dur(total)))
    macros.append(_mac("LeowAWDecisions", str(n_decisions)))
    for m, (n, per) in _train_detail(path).items():
        macros.append(_mac(f"LeowAWTrainPer{m}", _dur(per)))
        macros.append(_mac(f"LeowAWTrainN{m}", str(n)))

    (OUT / "leow_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"  wrote {OUT / 'leow_macros.tex'} ({len(macros)} macros)")


# ---------------------------------------------------------------------------
# Wang S&P 500
# ---------------------------------------------------------------------------

def generate_wang(optimizer: str = OPTIMIZER) -> None:
    path = RESULTS / "wang_sp500"
    f = path / "wang_sharpe_table.csv"
    if not f.exists():
        print(f"  skip Wang: {f} missing")
        return

    # The Wang-style table is the one tab:wang quotes: realized annualized
    # Sharpe rebuilt from daily portfolio returns, comparable to their Table 3.
    t = pd.read_csv(f)
    t = t[t["optimizer"].str.upper() == optimizer.upper()]
    t = t.set_index("predictor")
    train = _train_seconds(path)
    matched = _matched(path)
    n_base = sum(1 for m in t.index if _is_baseline(m))

    rows = []
    for m in _order(t.index):
        r = t.loc[m]
        rows.append(
            "{label} & {ret} & {risk} & ${sh}$ & {tr} \\\\".format(
                label=_display(m, n_base),
                ret=_num(float(r["ann_return"])),
                risk=_num(float(r["ann_risk"])),
                sh=_num(float(r["sharpe"]), 3),
                tr=_dur(train.get(m, float("nan"))),
            )
        )
    _write_rows(OUT / "tab_wang_rows.tex", rows)
    print(f"  wrote {OUT / 'tab_wang_rows.tex'} ({len(rows)} rows)")

    # A LaTeX control sequence may contain letters only, so baseline macros are
    # named after the MODEL they are matched to (\WangHistoricalMASTER), never
    # after the window length (\WangHistorical67 would not compile).
    macros = []
    for m in t.index:
        if _is_baseline(m):
            continue
        macros.append(_mac(f"Wang{m}", _num(float(t.loc[m, "sharpe"]), 3)))
        macros.append(_mac(f"Wang{m}Ret", _num(float(t.loc[m, "ann_return"]))))
        macros.append(_mac(f"Wang{m}Risk", _num(float(t.loc[m, "ann_risk"]))))
    for m, base in matched.items():
        if m in t.index and base in t.index:
            macros.append(_mac(f"WangHistorical{m}", _num(float(t.loc[base, "sharpe"]), 3)))
            macros.append(_mac(f"WangHistorical{m}Ret", _num(float(t.loc[base, "ann_return"]))))
            macros.append(_mac(f"WangWindow{m}", str(base).split("@", 1)[1]))
            macros.append(_mac(f"WangMargin{m}",
                               _num(float(t.loc[m, "sharpe"])
                                    - float(t.loc[base, "sharpe"]), 3, True)))
    # sharpe_std >= sharpe for every row (83% holding-period overlap, 7
    # decisions ~ 2 independent observations) -- the caveat the prose must carry.
    macros.append(_mac("WangSharpeStdMax", _num(float(t["sharpe_std"].max()), 3)))
    macros.append(_mac("WangSharpeStdMin", _num(float(t["sharpe_std"].min()), 3)))
    macros.append(_mac("WangTrainTotal", _dur(sum(train.values()))))
    for m, (n, per) in _train_detail(path).items():
        macros.append(_mac(f"WangTrainPer{m}", _dur(per)))
        macros.append(_mac(f"WangTrainN{m}", str(n)))

    (OUT / "wang_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"  wrote {OUT / 'wang_macros.tex'} ({len(macros)} macros)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating LaTeX table fragments...")
    generate_leow()
    generate_wang()
    print("Done.")
