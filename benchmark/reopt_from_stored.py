#!/usr/bin/env python3
"""Re-run portfolio optimization using stored mu predictions with a corrected
covariance window, without retraining any models.

Reads prediction_results.csv from the benchmark's output directory, recomputes
the covariance matrix using the current config (which may have a different
`moment_lookback_periods` than the original run), and re-runs the C++ optimizer.
Overwrites portfolio_results.csv / portfolio_summary.csv / aggregate_summary.csv /
estimator_vs_historical.csv; prediction_results.csv is unchanged.

Usage:
    python benchmark/reopt_from_stored.py benchmark/configs/config_wang_sp500.json
    python benchmark/reopt_from_stored.py benchmark/configs/config_leow_allweather_direct.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.utils.data_utils import load_prices
from benchmark.utils.benchmark_utils import _load_optimizer_config
from benchmark.utils.summaries_utils import (
    risk_points, _forecast_summary, _forecast_aggregate,
    _portfolio_summary, _aggregate_summary, _vs_historical,
)
from benchmark.walkforward import (
    resolve_windows, build_decisions, historical_covariance,
    actual_asset_return, row_windows as _row_windows_fn,
)
from estimators.pipeline import (
    matched_baseline_name, row_windows, normalize_for_optimizer,
)
from portfolio_optimizers.bridge import run_optimizer, find_binary, _mvo_utility


def reopt(config_path: str, *, smoke: bool = False):
    config = json.load(open(config_path))
    config["optimizer_config"] = _load_optimizer_config(config)
    prices, tickers = load_prices(config)
    windows = resolve_windows(prices, config)

    max_decisions = None
    if smoke:
        max_decisions = int(config.get("smoke", {}).get("max_decisions", 3))
    decisions = build_decisions(prices, windows, config, max_decisions=max_decisions)

    rpoints = risk_points(config)
    if smoke:
        smoke_risk = config.get("smoke", {}).get("risk_points")
        if smoke_risk:
            import copy
            tmp = copy.deepcopy(config)
            tmp["protocol"]["risk_grid"]["values"] = smoke_risk
            rpoints = risk_points(tmp)
        else:
            rpoints = rpoints[:min(3, len(rpoints))]

    binary = find_binary()

    out_dir = Path(config["output_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    preds_path = out_dir / "prediction_results.csv"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"No stored predictions at {preds_path} -- run a full benchmark first."
        )
    preds = pd.read_csv(preds_path, parse_dates=["decision_date"])

    # Refit cadence (mirrors run_experiment logic exactly).
    moment_refit = str(config["protocol"].get("moment_refit", "match_models"))
    if moment_refit == "match_models":
        moment_refit = str(config["protocol"].get("model_refit", "once"))

    row_win = row_windows(config)

    fixed_cov = None
    if moment_refit in ("once", "annual"):
        fixed_cov = historical_covariance(prices, windows["fit_end_pos"], config)
        lb = config["protocol"].get("moment_lookback_periods", "auto")
        print(f"[reopt] fixed cov computed at fit_end "
              f"(moment_lookback_periods={lb}, shape={fixed_cov.shape})")

    paper = config["literature"]["paper"]
    level = config["literature"].get("comparison_level", "adaptation")

    portfolio_rows = []
    drifted_prev: dict = {}
    seed = 0  # stored predictions are already ensemble-averaged

    # Annual moment state (mirrors run_experiment): Sigma starts at the fit_end
    # estimate and is re-estimated on the trailing window at the first decision
    # of each new OOS calendar year.
    cov_synced = fixed_cov
    moments_through_pos = int(windows["fit_end_pos"])
    moments_for_year = decisions[0].decision_date.year if decisions else None

    def _annual_cutoff(dec):
        """Position of the last trading day before dec's year, on year change."""
        nonlocal moments_for_year
        if dec.decision_date.year == moments_for_year:
            return None
        moments_for_year = dec.decision_date.year
        prior = prices.index[prices.index < pd.Timestamp(dec.decision_date.year, 1, 1)]
        if len(prior) == 0:
            return None
        return int(prices.index.get_loc(prior[-1]))

    for di, dec in enumerate(decisions):
        if moment_refit == "per_decision":
            cov = historical_covariance(prices, dec.decision_pos, config)
        elif moment_refit == "annual":
            m_cutoff = _annual_cutoff(dec)
            if m_cutoff is not None and m_cutoff > moments_through_pos:
                cov_synced = historical_covariance(prices, m_cutoff, config)
                moments_through_pos = m_cutoff
            cov = cov_synced
        else:  # "once"
            cov = fixed_cov

        # Build mu dict from stored predictions for this decision.
        slice_ = preds[preds["decision_date"] == dec.decision_date]
        if slice_.empty:
            raise RuntimeError(
                f"No stored predictions for decision_date={dec.decision_date.date()} "
                f"in {preds_path}. Available: {sorted(preds['decision_date'].dt.date.unique())}"
            )
        mus: dict[str, np.ndarray] = {}
        for model, grp in slice_.groupby("model"):
            grp_idx = grp.set_index("ticker")
            missing = [t for t in tickers if t not in grp_idx.index]
            if missing:
                raise RuntimeError(
                    f"Model {model!r} missing tickers at {dec.decision_date.date()}: "
                    f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
                )
            # Stored mu on whatever native/calibrated scale the run used. The
            # optimizer-boundary z-score below is invariant to any affine scale,
            # so the normalized inputs are identical regardless of how the stored
            # values were scaled -- no recalibration step is needed.
            mus[model] = np.array(
                [float(grp_idx.loc[t, "predicted_asset_return"]) for t in tickers]
            )

        actual_modeled, actual_simple = actual_asset_return(
            prices, dec, config["protocol"].get("return_type", "simple")
        )

        for ri, (risk_label, paper_par, P) in enumerate(rpoints):
            for model, mu in mus.items():
                for algo in ("pso", "sa"):
                    if not config.get("solvers", {}).get(algo, {}).get("enabled", True):
                        continue

                    opt_seed = int(
                        seed * 100000 + di * 1000 + ri * 10 +
                        (0 if algo == "pso" else 1)
                    )
                    mu_n, cov_n = normalize_for_optimizer(mu, cov)
                    res = run_optimizer(config, tickers, algo, mu_n, cov_n, P,
                                        opt_seed, binary)
                    w = np.asarray(res["weights"], dtype=float)

                    pred_ret = float(w @ mu)
                    actual_ret_modeled = float(w @ actual_modeled)
                    actual_ret_simple = float(w @ actual_simple)
                    variance = float(w @ cov @ w)

                    key = (risk_label, model, algo)
                    prev = drifted_prev.get(key)
                    turnover = (
                        float(np.abs(w - prev).sum()) if prev is not None else float("nan")
                    )

                    gross = 1.0 + actual_simple
                    drift = w * gross
                    if float(drift.sum()) > 0:
                        drift = drift / drift.sum()
                    drifted_prev[key] = drift

                    portfolio_rows.append({
                        "paper": paper,
                        "comparison_level": level,
                        "seed": seed,
                        "decision_date": dec.decision_date,
                        "realization_date": dec.realization_date,
                        "risk_label": risk_label,
                        "paper_risk_parameter": float(paper_par),
                        "P": float(P),
                        "model": model,
                        "conditioning_window": int(row_win.get(model, 0)),
                        "matched_baseline": matched_baseline_name(
                            row_win.get(model, 0)
                        ) if model in row_win else model,
                        "optimizer": algo.upper(),
                        "predicted_portfolio_return": pred_ret,
                        "actual_modeled_portfolio_return": actual_ret_modeled,
                        "actual_simple_portfolio_return": actual_ret_simple,
                        "predicted_variance": variance,
                        "predicted_utility": _mvo_utility(w, mu, cov, P),
                        "realized_utility": _mvo_utility(w, actual_modeled, cov, P),
                        "turnover_l1": turnover,
                        "active_count": int(res["active_count"]),
                        "min_active": float(res["min_active"]),
                        "max_active": float(res["max_active"]),
                        "solver_sec": float(res["solver_sec"]),
                        "weights_json": json.dumps(
                            {t: float(x) for t, x in zip(tickers, w) if x > 1e-8}
                        ),
                    })

        print(
            f"  decision {di+1:>3}/{len(decisions)} "
            f"{dec.decision_date.date()} -> {dec.realization_date.date()}"
        )

    portfolios = pd.DataFrame(portfolio_rows)
    portfolio_summary = _portfolio_summary(portfolios, config)
    aggregate_summary = _aggregate_summary(portfolio_summary)
    vs_historical = _vs_historical(portfolio_summary)

    portfolios.to_csv(out_dir / "portfolio_results.csv", index=False)
    portfolio_summary.to_csv(out_dir / "portfolio_summary.csv", index=False)
    aggregate_summary.to_csv(out_dir / "aggregate_summary.csv", index=False)
    vs_historical.to_csv(out_dir / "estimator_vs_historical.csv", index=False)

    print(f"\n[reopt] done — results saved to {out_dir}")
    cols = ["model", "optimizer", "annualized_arithmetic_return",
            "sharpe_annualized", "max_drawdown"]
    cols = [c for c in cols if c in portfolio_summary.columns]
    print(portfolio_summary[cols].to_string(index=False))
    return portfolio_summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to benchmark config JSON")
    p.add_argument("--smoke", action="store_true", help="Quick 3-decision smoke test")
    args = p.parse_args()
    reopt(args.config, smoke=args.smoke)
