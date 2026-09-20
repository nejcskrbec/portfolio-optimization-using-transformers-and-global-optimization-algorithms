#!/usr/bin/env python3
"""Wang et al. comparison: realized-Sharpe table, risk sweep, and article runner."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from benchmark.utils.benchmark_utils import (
    ROOT, BENCH, CFG, CONFIGS, _load_optimizer_config,
    _print_forecast, _print_portfolio, _dry, _run_named_literature,
)


def task_wang_sharpe(argv=None):
    #!/usr/bin/env python3
    """
    wang_sharpe.py
    ==============
    Wang et al. (ICLR 2023)-style realized-Sharpe table for our transformer +
    meta-heuristics pipeline, computed from an already-run Wang literature
    benchmark (``portfolio_results.csv``) plus the authors' daily price file.

    Their Table 3 / Fig. 5 report, per portfolio, the REALIZED annualized
    return, risk and Sharpe over a 120-trading-day holding window:

        r_p(t) = sum_i w_i * (P_i(t)/P_i(t-1) - 1)          # daily portfolio return
        ann_return = mean(r_p) * 252
        ann_risk   = std(r_p, ddof=1) * sqrt(252)
        Sharpe     = (ann_return - r_f) / ann_risk,  r_f = 3%

    averaged across the (monthly) starting dates in 2021.  We reproduce exactly
    this metric from our chosen weights so our rows are directly comparable to
    their published numbers (history-opt 0.673, LSTM+Gurobi 1.082, CardNN-GS 1.968).

    This is a SAME-DATA / SAME-CARDINALITY comparison: our optimiser maximises the
    mean-variance P-tradeoff (not Sharpe directly), so this is a method-substitution
    head-to-head, not an exact-objective replication.

    Usage
    -----
        python benchmark/wang_sharpe.py \
            --csv    test_results/literature/wang_sp500/portfolio_results.csv \
            --prices data/literature/wang/snp500.csv
    """

    import argparse
    import json
    import os

    import numpy as np
    import pandas as pd

    RF_ANNUAL = 0.03
    TRADING_DAYS = 252

    # Wang et al. published reference rows (Table 3 + Fig. 5 averages).
    WANG_REFERENCE = [
        # method, predictor, optimizer, ann_return, ann_risk, sharpe
        ("Wang history-opt",   "Historical", "Gurobi",    0.139, 0.162, 0.673),
        ("Wang predict-then",  "LSTM",       "Gurobi",    0.241, 0.195, 1.082),
        ("Wang predict-and",   "LSTM",       "CardNN-GS", 0.400, 0.188, 1.968),
        ("Wang S&P500 index",  "-",          "-",         0.235, np.nan, np.nan),
    ]


    def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.sort_index().pct_change(fill_method=None)


    def _window_stats(weights: dict, rets: pd.DataFrame,
                      d0: pd.Timestamp, d1: pd.Timestamp) -> tuple[float, float, float]:
        """Annualized (return, risk, Sharpe) of a fixed-weight portfolio held over
        (d0, d1].  Weights are renormalised over the tickers present in the price
        file (defensive; the Wang universe is fully covered)."""
        cols = [t for t in weights if t in rets.columns]
        if not cols:
            return (np.nan, np.nan, np.nan)
        w = np.array([weights[t] for t in cols], dtype=float)
        if w.sum() <= 0:
            return (np.nan, np.nan, np.nan)
        w = w / w.sum()
        win = rets.loc[(rets.index > d0) & (rets.index <= d1), cols]
        if len(win) < 2:
            return (np.nan, np.nan, np.nan)
        rp = win.to_numpy(float) @ w
        ann_ret = float(np.mean(rp) * TRADING_DAYS)
        ann_risk = float(np.std(rp, ddof=1) * np.sqrt(TRADING_DAYS))
        sharpe = (ann_ret - RF_ANNUAL) / ann_risk if ann_risk > 0 else np.nan
        return (ann_ret, ann_risk, sharpe)


    def build_table(csv: str, prices_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv)
        prices = pd.read_csv(prices_path)
        dcol = [c for c in prices.columns if str(c).lower().startswith("date")][0]
        prices.index = pd.to_datetime(prices[dcol])
        prices = prices.drop(columns=[dcol]).apply(pd.to_numeric, errors="coerce")
        rets = _daily_returns(prices)

        rows = []
        for (model, opt), g in df.groupby(["model", "optimizer"]):
            recs = []
            for _, r in g.iterrows():
                w = json.loads(r["weights_json"])
                d0 = pd.Timestamp(r["decision_date"])
                d1 = pd.Timestamp(r["realization_date"])
                recs.append(_window_stats(w, rets, d0, d1))
            arr = np.array(recs, dtype=float)
            arr = arr[~np.isnan(arr[:, 2])]
            if len(arr) == 0:
                continue
            rows.append({
                "method": f"Thesis {model}",
                "predictor": model,
                "optimizer": opt,
                "ann_return": float(np.mean(arr[:, 0])),
                "ann_risk": float(np.mean(arr[:, 1])),
                "sharpe": float(np.mean(arr[:, 2])),
                "sharpe_std": float(np.std(arr[:, 2], ddof=1)) if len(arr) > 1 else np.nan,
                "n_windows": int(len(arr)),
            })
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--csv", required=True,
                        help="Wang run portfolio_results.csv")
        ap.add_argument("--prices", default="data/literature/wang/snp500.csv")
        ap.add_argument("--out", default=None,
                        help="output CSV (default: alongside the input CSV)")
        args = ap.parse_args(argv)

        tbl = build_table(args.csv, args.prices)
        out = args.out or os.path.join(os.path.dirname(args.csv), "wang_sharpe_table.csv")
        tbl.to_csv(out, index=False, float_format="%.4f")

        ref = pd.DataFrame(WANG_REFERENCE,
                           columns=["method", "predictor", "optimizer",
                                    "ann_return", "ann_risk", "sharpe"])

        def _fmt(d):
            return d.to_string(index=False,
                               float_format=lambda x: f"{x:6.3f}" if pd.notna(x) else "   -  ")

        print("\n=== Wang et al. (ICLR 2023) — published reference ===")
        print(_fmt(ref))
        print("\n=== Our pipeline (same data, same K=20, realized annualized Sharpe) ===")
        print(_fmt(tbl[["method", "predictor", "optimizer",
                        "ann_return", "ann_risk", "sharpe", "n_windows"]]))
        print(f"\nsaved: {out}")

    main()


def task_wang_risk_sweep(argv=None):
    #!/usr/bin/env python3
    """
    wang_risk_sweep.py
    ==================
    Risk-matched Wang (ICLR 2023) comparison WITHOUT retraining.

    The models' predicted mu per asset/window are already stored in
    ``prediction_results.csv``; the covariance is a single fixed pre-test matrix
    (the Wang config uses covariance_mode=fixed_pretest). Risk-matching only
    touches the optimiser's risk parameter P (objective  P*mu'w - (1-P)*w'Sigma w),
    so we just re-run PSO/SA across a P grid on the frozen mu/Sigma and pick, per
    (model, optimiser), the P whose average realized annualized risk is nearest a
    target (default 19%, matching Wang's LSTM+Gurobi row). Realized risk/return/
    Sharpe are computed Wang-style (daily portfolio returns over the 120-day hold).

    Usage
    -----
        python benchmark/wang_risk_sweep.py \
            --run    test_results/literature/wang_sp500 \
            --prices data/literature/wang/snp500.csv \
            --target-risk 0.19
    """

    import argparse
    import json
    import os
    import sys

    import numpy as np
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from portfolio_optimizers.bridge import run_optimizer, find_binary
    from benchmark.utils.paths_utils import project_root

    WANG_REFERENCE = {}
    def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.sort_index().pct_change(fill_method=None)


    def _window_stats(weights: dict, rets: pd.DataFrame,
                      d0: pd.Timestamp, d1: pd.Timestamp) -> tuple[float, float, float]:
        """Annualized (return, risk, Sharpe) of a fixed-weight portfolio held over
        (d0, d1].  Weights are renormalised over the tickers present in the price
        file (defensive; the Wang universe is fully covered)."""
        cols = [t for t in weights if t in rets.columns]
        if not cols:
            return (np.nan, np.nan, np.nan)
        w = np.array([weights[t] for t in cols], dtype=float)
        if w.sum() <= 0:
            return (np.nan, np.nan, np.nan)
        w = w / w.sum()
        win = rets.loc[(rets.index > d0) & (rets.index <= d1), cols]
        if len(win) < 2:
            return (np.nan, np.nan, np.nan)
        rp = win.to_numpy(float) @ w
        ann_ret = float(np.mean(rp) * TRADING_DAYS)
        ann_risk = float(np.std(rp, ddof=1) * np.sqrt(TRADING_DAYS))
        sharpe = (ann_ret - RF_ANNUAL) / ann_risk if ann_risk > 0 else np.nan
        return (ann_ret, ann_risk, sharpe)




    RF_ANNUAL = 0.03
    TRADING_DAYS = 252
    FIT_START = "2018-01-02"
    FIT_END = "2020-12-30"
    ALGOS = ["pso", "sa"]


    def _base_config(K: int, w_min: float, w_max: float) -> dict:
        with open(project_root() / "portfolio_optimizers" / "config.json") as f:
            solver = json.load(f)  # flat: common, pso, sa
        solver["common"].update({"cardinality_K": K, "w_min": w_min, "w_max": w_max})
        return {"optimizer_config": solver, "run_settings": {}}


    def _fixed_cov(prices: pd.DataFrame, tickers: list[str]) -> np.ndarray:
        """Daily sample covariance over the pre-test fit window, in `tickers` order —
        replicates historical_moments(fixed_pretest) with moment_frequency=daily."""
        p = prices.loc[(prices.index >= pd.Timestamp(FIT_START)) &
                       (prices.index <= pd.Timestamp(FIT_END)), tickers]
        r = p.pct_change(fill_method=None).dropna()
        cov = np.cov(r.to_numpy(float), rowvar=False)
        cov = 0.5 * (cov + cov.T)
        e = float(np.linalg.eigvalsh(cov)[0])
        if e < 1e-10:
            cov = cov + np.eye(cov.shape[0]) * (1e-10 - e)
        return cov


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--run", default="test_results/literature/wang_sp500")
        ap.add_argument("--prices", default="data/literature/wang/snp500.csv")
        ap.add_argument("--target-risk", type=float, default=0.19)
        ap.add_argument("--pgrid", default="0.02,0.05,0.1,0.15,0.2,0.3,0.5")
        ap.add_argument("--K", type=int, default=20)
        ap.add_argument("--w-min", type=float, default=0.001)
        ap.add_argument("--w-max", type=float, default=1.0)
        ap.add_argument("--seed", type=int, default=42)
        args = ap.parse_args(argv)

        pgrid = [float(x) for x in args.pgrid.split(",")]
        preds = pd.read_csv(os.path.join(args.run, "prediction_results.csv"))

        prices = pd.read_csv(args.prices)
        dcol = [c for c in prices.columns if str(c).lower().startswith("date")][0]
        prices.index = pd.to_datetime(prices[dcol])
        prices = prices.drop(columns=[dcol]).apply(pd.to_numeric, errors="coerce")
        rets = _daily_returns(prices)

        # Canonical ticker order = predicted tickers present in the price file.
        tickers = sorted(set(preds["ticker"]) & set(prices.columns))
        cov = _fixed_cov(prices, tickers)
        base = _base_config(args.K, args.w_min, args.w_max)
        binary = find_binary()

        decisions = (preds[["decision_date", "realization_date"]]
                     .drop_duplicates().sort_values("decision_date")
                     .itertuples(index=False))
        decisions = list(decisions)
        models = sorted(preds["model"].unique())

        # mu lookup: (model, decision_date) -> mu vector in `tickers` order
        mu_map = {}
        for (m, dd), g in preds.groupby(["model", "decision_date"]):
            s = g.set_index("ticker")["predicted_asset_return"]
            mu_map[(m, dd)] = s.reindex(tickers).fillna(0.0).to_numpy(float)

        rows = []
        ncall = len(models) * len(ALGOS) * len(pgrid) * len(decisions)
        print(f"tickers={len(tickers)}  P grid={pgrid}  calls={ncall}")
        done = 0
        for m in models:
            for algo in ALGOS:
                for P in pgrid:
                    stats = []
                    for d in decisions:
                        mu = mu_map[(m, d.decision_date)]
                        res = run_optimizer(base, tickers, algo, mu, cov, P,
                                            args.seed, binary)
                        w = res["weights"]
                        wd = {t: float(wi) for t, wi in zip(tickers, w) if wi > 0}
                        ann_ret, ann_risk, sharpe = _window_stats(
                            wd, rets, pd.Timestamp(d.decision_date),
                            pd.Timestamp(d.realization_date))
                        stats.append((ann_ret, ann_risk, sharpe))
                        done += 1
                    a = np.array(stats, float)
                    a = a[~np.isnan(a[:, 2])]
                    rows.append({
                        "model": m, "optimizer": algo.upper(), "P": P,
                        "ann_return": float(np.nanmean(a[:, 0])),
                        "ann_risk": float(np.nanmean(a[:, 1])),
                        "sharpe": float(np.nanmean(a[:, 2])),
                        "n": int(len(a)),
                    })
                    print(f"  [{done:4d}/{ncall}] {m:11s} {algo.upper():3s} P={P:<5} "
                          f"risk={rows[-1]['ann_risk']:.3f} sharpe={rows[-1]['sharpe']:.3f}")

        sweep = pd.DataFrame(rows)
        sweep_path = os.path.join(args.run, "wang_risk_sweep_full.csv")
        sweep.to_csv(sweep_path, index=False, float_format="%.4f")

        # Per (model, optimizer): pick P whose mean realized risk is nearest target.
        picks = []
        for (m, opt), g in sweep.groupby(["model", "optimizer"]):
            i = (g["ann_risk"] - args.target_risk).abs().idxmin()
            picks.append(g.loc[i])
        matched = pd.DataFrame(picks).sort_values("sharpe", ascending=False)
        matched_path = os.path.join(args.run, "wang_risk_matched_table.csv")
        matched.to_csv(matched_path, index=False, float_format="%.4f")

        ref = pd.DataFrame(WANG_REFERENCE,
                           columns=["method", "predictor", "optimizer",
                                    "ann_return", "ann_risk", "sharpe"])

        def _fmt(d, cols):
            return d[cols].to_string(index=False,
                                     float_format=lambda x: f"{x:6.3f}" if pd.notna(x) else "   -  ")

        print("\n=== Wang et al. published reference ===")
        print(_fmt(ref, ["method", "predictor", "optimizer", "ann_return", "ann_risk", "sharpe"]))
        print(f"\n=== Risk-matched (target ~{args.target_risk:.0%} ann. risk) — our pipeline ===")
        print(_fmt(matched, ["model", "optimizer", "P", "ann_return", "ann_risk", "sharpe"]))
        print(f"\nsaved: {sweep_path}\n       {matched_path}")

    main()



def task_wang(smoke: bool = False, dry: bool = False):
    """Run the Wang S&P500 config, then the Wang-style realized-Sharpe table."""
    out = _run_named_literature("wang", smoke, dry)
    if dry or smoke:
        return out
    task_wang_sharpe(["--csv", str(ROOT / "test_results/literature/wang_sp500/portfolio_results.csv")])
    return out

