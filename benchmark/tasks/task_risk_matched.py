#!/usr/bin/env python3
"""Risk-matched model comparison.

Reading every model at the same nominal risk parameter P is not a fair
comparison: P only fixes the trade-off between mu'w and w'Sigma w, so two mu
sources with different cross-sectional dispersion land on different points of
the efficient frontier even at an identical P. Scaling mu by c is equivalent to
moving the risk parameter to P'/(1-P') = c * P/(1-P); the horizon-scaling fix in
`historical_mus` removes the *unit* part of that gap, but any genuine
difference in forecast dispersion remains.

This task therefore reports each model at the P whose *realized* annualized
volatility is closest to a common target, so returns and Sharpe are read at
equal risk. Default target = the longest-window historical baseline's realized
volatility at the reference P (`--target-P`, default 0.5), i.e. "what did each
mu source earn if you ran it at the baseline's risk?".

    python benchmark/run.py risk-matched --dir test_results/literature/leow_allweather_direct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLS = [
    "model", "optimizer", "P", "annualized_vol",
    "annualized_arithmetic_return", "cagr", "sharpe_annualized",
    "max_drawdown", "mean_turnover_l1",
]


def _match(summary: pd.DataFrame, target_vol: float, tol: float = 0.05) -> pd.DataFrame:
    """Per (model, optimizer), the best risk point at (or below) target_vol.

    NOT simply "nearest volatility". Realized volatility is not monotone in P --
    on the Leow All-Weather frontier TFT runs 0.187, 0.245, 0.284, 0.322, 0.313,
    ..., 0.275 as P goes 0.05 -> 0.95 -- so a nearest-vol rule can land on the
    low-P branch purely by a rounding-level volatility difference. That is
    exactly what happened once here: nearest-vol picked TFT at P=0.1 (Sharpe
    0.04) when P=0.9 sat 0.025 further from the target with Sharpe 2.01, and it
    made TFT look like the worst model when it was the second best.

    The defensible rule: among points whose volatility does not exceed the target
    (within `tol` relative slack), take the highest Sharpe. Sharpe is invariant to
    blending with cash, so any such portfolio can be levered to the target risk;
    picking the best one is the honest "what does this mu source earn at the
    baseline's risk?" answer. Falls back to nearest-vol if the whole frontier sits
    above the target.
    """
    rows = []
    cap = target_vol * (1.0 + tol)
    for (model, optimizer), g in summary.groupby(["model", "optimizer"], sort=True):
        g = g.dropna(subset=["annualized_vol"])
        if g.empty:
            continue
        eligible = g[g["annualized_vol"] <= cap]
        if not eligible.empty:
            pick = eligible.loc[eligible["sharpe_annualized"].idxmax()]
            rule = "best Sharpe at<=target"
        else:
            pick = g.iloc[(g["annualized_vol"] - target_vol).abs().to_numpy().argmin()]
            rule = "nearest vol (frontier above target)"
        row = {c: pick[c] for c in COLS if c in pick.index}
        row["vol_gap"] = float(pick["annualized_vol"] - target_vol)
        row["rule"] = rule
        # Unconstrained best on the frontier, for context: if this is far above
        # the matched value the model only earns by taking more risk.
        best = g.loc[g["sharpe_annualized"].idxmax()]
        row["frontier_best_sharpe"] = float(best["sharpe_annualized"])
        row["frontier_best_P"] = float(best["P"])
        row["frontier_best_vol"] = float(best["annualized_vol"])
        rows.append(row)
    return pd.DataFrame(rows)


def task_risk_matched(argv=None):
    ap = argparse.ArgumentParser(description="Risk-matched comparison of mu sources.")
    ap.add_argument("--dir", required=True,
                    help="Run output directory containing portfolio_summary.csv")
    ap.add_argument("--target-P", type=float, default=0.5,
                    help="Reference risk point defining the target volatility.")
    ap.add_argument("--target-vol", type=float, default=None,
                    help="Explicit target annualized volatility (overrides --target-P).")
    ap.add_argument("--reference-model", default="auto",
                    help="Model whose volatility at --target-P defines the target. "
                         "'auto' takes the LONGEST-window historical baseline "
                         "(`Historical@<max L>`) -- the target has to be a single "
                         "number shared by every model, and among the matched "
                         "baselines the longest lookback is the least noisy, hence "
                         "the strongest classical anchor.")
    args = ap.parse_args(argv)

    out = Path(args.dir)
    if not out.is_absolute():
        out = ROOT / out
    path = out / "portfolio_summary.csv"
    if not path.exists():
        raise SystemExit(f"No portfolio_summary.csv in {out}")

    summary = pd.read_csv(path)
    if "annualized_vol" not in summary.columns:
        raise SystemExit(
            "portfolio_summary.csv predates the annualized_vol column; re-run the "
            "benchmark before risk-matching."
        )

    # Average across seeds first so the matching is not driven by one draw.
    keys = ["P", "model", "optimizer"]
    agg = (summary.groupby(keys, as_index=False)
                  .agg({c: "mean" for c in COLS if c not in keys}))
    agg["n_seeds"] = summary.groupby(keys)["seed"].nunique().to_numpy()

    if args.target_vol is not None:
        target = float(args.target_vol)
        src = f"explicit target {target:.4f}"
    else:
        reference = args.reference_model
        if reference == "auto":
            base = [m for m in agg["model"].unique()
                    if str(m).startswith("Historical@")]
            if not base:
                raise SystemExit(
                    "no `Historical@<L>` baseline rows in this run; pass "
                    "--reference-model or --target-vol explicitly"
                )
            reference = max(base, key=lambda m: int(str(m).split("@", 1)[1]))
        ref = agg[(agg["model"] == reference) &
                  np.isclose(agg["P"], args.target_P)]
        if ref.empty:
            raise SystemExit(
                f"No {reference} rows at P={args.target_P}; "
                f"available P: {sorted(agg['P'].unique())}"
            )
        target = float(ref["annualized_vol"].mean())
        src = f"{reference} at P={args.target_P:g}"

    matched = _match(agg, target)

    print("=" * 100)
    print(f"RISK-MATCHED COMPARISON  |  {out.name}")
    print(f"target annualized volatility = {target:.4f}  (from {src})")
    print("Each model is read at the P whose realized volatility is closest to the target,")
    print("so returns/Sharpe are compared at equal risk rather than at an equal nominal P.")
    print("=" * 100)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(matched.round(4).to_string(index=False))

    dst = out / "risk_matched.csv"
    matched.to_csv(dst, index=False)
    print(f"\nSaved: {dst}")

    # The full frontier is what a reviewer will actually want to see.
    fdst = out / "risk_frontier.csv"
    agg.sort_values(["model", "optimizer", "P"]).to_csv(fdst, index=False)
    print(f"Saved: {fdst}")
    return matched


if __name__ == "__main__":
    task_risk_matched()
