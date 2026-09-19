#!/usr/bin/env python3
"""P-parameter sweep for Leow benchmark (all three sub-datasets).

Uses stored per-asset μ predictions and realized returns from prediction_results.csv.
Runs the optimizer at each P in the grid with a fixed pre-test sample covariance
(2018-08-01 to 2019-12-31), applies the resulting weights to stored per-asset
realized returns, and reports the annualized Sharpe over the OOS period.

No model retraining — cheap re-run of the optimizer only.
"""
from __future__ import annotations

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from portfolio_optimizers.bridge import run_optimizer, find_binary

FIT_START = "2018-08-01"
FIT_END   = "2019-12-31"
SEED      = 42
ALGOS     = ["pso", "sa"]
P_GRID    = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIGDIR    = ROOT / "thesis-paper" / "fig"

DATASETS = {
    "allweather": {
        "rundir": ROOT / "test_results" / "literature" / "leow_allweather_direct",
        "K": 5, "w_min": 0.01, "w_max": 0.4,
        "label": "All-Weather (5)",
    },
}

MODEL_COLORS = {
    "TFT": "tab:blue", "PatchTST": "tab:orange", "MASTER": "tab:green",
}


def _color(model: str) -> str:
    # Baselines are `Historical@<L>`, one per model conditioning window.
    return MODEL_COLORS.get(model,
                            "tab:gray" if model.startswith("Historical@") else "black")


def _load_solver_cfg(K: int, w_min: float, w_max: float) -> dict:
    with open(ROOT / "portfolio_optimizers" / "config.json") as f:
        solver = json.load(f)
    solver["common"].update({"cardinality_K": K, "w_min": w_min, "w_max": w_max})
    return {"optimizer_config": solver, "run_settings": {}}


def _fetch_prices(tickers: list[str]) -> pd.DataFrame:
    print(f"  Downloading {len(tickers)} tickers ({FIT_START} -> {FIT_END}) ...")
    px = yf.download(tickers, start=FIT_START, end=FIT_END,
                     progress=False, auto_adjust=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px


def _sample_cov(prices: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    p = prices[tickers].dropna()
    r = p.pct_change(fill_method=None).dropna()
    cov = np.cov(r.to_numpy(float), rowvar=False)
    cov = 0.5 * (cov + cov.T)
    e = float(np.linalg.eigvalsh(cov)[0])
    if e < 1e-10:
        cov += np.eye(cov.shape[0]) * (1e-10 - e)
    return cov


def _period_sharpe(returns: np.ndarray, periods_per_year: int = 52) -> float:
    if len(returns) < 2:
        return np.nan
    mu  = np.mean(returns)
    sig = np.std(returns, ddof=1)
    if sig <= 0:
        return np.nan
    return float((mu / sig) * np.sqrt(periods_per_year))


def run_dataset(name: str, cfg: dict, binary) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Dataset: {cfg['label']}")
    print(f"{'='*60}")

    preds = pd.read_csv(cfg["rundir"] / "prediction_results.csv")
    preds["decision_date"] = pd.to_datetime(preds["decision_date"])

    tickers = sorted(preds["ticker"].unique())
    decs = (preds[["decision_date", "realization_date"]]
            .drop_duplicates().sort_values("decision_date").reset_index(drop=True))

    prices = _fetch_prices(tickers)
    cov = _sample_cov(prices, tickers)

    ret_map = {}
    for (m, dd, tk), row in preds.groupby(["model", "decision_date", "ticker"]).first().iterrows():
        ret_map[(m, pd.Timestamp(dd), tk)] = row["actual_simple_asset_return"]

    mu_map = {}
    for (m, dd), g in preds.groupby(["model", "decision_date"]):
        s = g.groupby("ticker")["predicted_asset_return"].mean()
        mu_map[(m, pd.Timestamp(dd)),] = s.reindex(tickers).fillna(0.0).to_numpy(float)

    base   = _load_solver_cfg(cfg["K"], cfg["w_min"], cfg["w_max"])
    models = sorted(preds["model"].unique())

    ncall = len(models) * len(ALGOS) * len(P_GRID) * len(decs)
    print(f"models={models}  total calls={ncall}")

    rows = []
    done = 0
    for m in models:
        for algo in ALGOS:
            for P in P_GRID:
                port_rets = []
                for _, d in decs.iterrows():
                    dd = d["decision_date"]
                    mu_key = (m, dd),
                    if mu_key not in mu_map:
                        port_rets.append(np.nan)
                        done += 1
                        continue
                    mu = mu_map[mu_key]
                    res = run_optimizer(base, tickers, algo, mu, cov, P, SEED, binary)
                    w  = res["weights"]
                    pr, wsum = 0.0, 0.0
                    for tk, wi in zip(tickers, w):
                        if wi > 1e-6:
                            r = ret_map.get((m, dd, tk), np.nan)
                            if np.isfinite(r):
                                pr += wi * r; wsum += wi
                    port_rets.append(pr / wsum if wsum > 1e-6 else np.nan)
                    done += 1

                arr   = np.array(port_rets, float)
                valid = arr[np.isfinite(arr)]
                sharpe = _period_sharpe(valid)
                rows.append({
                    "dataset": name, "model": m, "optimizer": algo.upper(),
                    "P": P, "sharpe_annualized": sharpe, "n": int(len(valid)),
                })
                print(f"  [{done:4d}/{ncall}] {m:11s} {algo.upper():3s} "
                      f"P={P:<4}  Sharpe={sharpe:.3f}")

    sweep = pd.DataFrame(rows)
    out_path = cfg["rundir"] / "leow_p_sweep.csv"
    sweep.to_csv(out_path, index=False, float_format="%.4f")
    print(f"Saved: {out_path}")

    best = sweep.loc[sweep.groupby(["model", "optimizer"])["sharpe_annualized"].idxmax()]
    print(f"\n=== Best P — {cfg['label']} ===")
    print(best[["model", "optimizer", "P", "sharpe_annualized"]].to_string(index=False))

    return sweep


def plot_all(all_sweeps: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, (name, sweep) in zip(axes, all_sweeps.items()):
        pso = sweep[sweep["optimizer"] == "PSO"]
        models = sorted(sweep["model"].unique())
        for m in models:
            sub = pso[pso["model"] == m]
            ax.plot(sub["P"], sub["sharpe_annualized"], "o-",
                    color=_color(m), lw=1.6, ms=5, label=m)
        ax.axvline(0.8, color="0.5", lw=0.8, ls="--")
        ax.set_xlabel("$P$")
        ax.set_title(DATASETS[name]["label"])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Letni Sharpov količnik")
    axes[0].legend(fontsize=8)
    fig.suptitle("Leow: mreža parametra $P$ (PSO)")
    fig.tight_layout()
    path = FIGDIR / "leow_p_sweep.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"\nFigure: {path}")


def main():
    binary = find_binary()
    all_sweeps = {}
    for name, cfg in DATASETS.items():
        all_sweeps[name] = run_dataset(name, cfg, binary)
    plot_all(all_sweeps)


if __name__ == "__main__":
    main()
