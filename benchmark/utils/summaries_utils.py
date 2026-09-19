#!/usr/bin/env python3
"""Forecast and portfolio summary/aggregate builders."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def risk_points(config: dict) -> list[tuple[str, float, float]]:
    """Return (label, paper_parameter, P_for_cpp)."""
    rcfg = config["protocol"].get("risk_grid", {"type": "P", "values": [0.8]})
    typ = str(rcfg.get("type", "P")).lower()

    if typ == "lambda_sweep":
        vals = rcfg.get("values", "0:1:0.02")
        if isinstance(vals, str):
            vals = [round(i * 0.02, 2) for i in range(51)]
        return [(f"lambda={float(v):.2f}", float(v), 1.0 - float(v)) for v in vals]

    if typ == "p":
        vals = [float(x) for x in rcfg.get("values", [0.8])]
        return [(f"P={v:.4g}", v, v) for v in vals]

    raise ValueError(f"Unsupported risk grid type: {typ}")


def _wealth_and_drawdown(simple_returns: np.ndarray):
    wealth = np.cumprod(1.0 + simple_returns)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])
    series = np.r_[1.0, wealth]
    dd = 1.0 - series / peak
    return wealth, float(np.max(dd))


def _forecast_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, model), g in pred.groupby(["seed", "model"], sort=True):
        y = g["actual_modeled_asset_return"].to_numpy(float)
        p = g["predicted_asset_return"].to_numpy(float)
        b = g["historical_asset_return"].to_numpy(float)

        sst = float(np.sum((y - y.mean()) ** 2))
        sse_b = float(np.sum((y - b) ** 2))
        sse = float(np.sum((y - p) ** 2))

        mask = np.abs(y) > 1e-6
        rows.append({
            "seed": int(seed),
            "model": model,
            "r2": (1.0 - sse / sst) if sst > 0 else np.nan,
            "r2_os_vs_historical": (1.0 - sse / sse_b) if sse_b > 0 else np.nan,
            "mae": float(np.mean(np.abs(p - y))),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mape": (
                float(np.mean(np.abs((p[mask] - y[mask]) / y[mask])))
                if mask.any() else np.nan
            ),
            "directional_accuracy": float(np.mean(np.sign(p) == np.sign(y))),
            "n_asset_predictions": int(len(g)),
        })
    return pd.DataFrame(rows)


def _portfolio_summary(port: pd.DataFrame, config: dict) -> pd.DataFrame:
    periods_per_year = float(config["protocol"].get("periods_per_year", 52.0))
    rows = []
    keys = ["seed", "risk_label", "paper_risk_parameter", "P", "model", "optimizer"]

    for vals, g in port.groupby(keys, sort=True):
        seed, risk_label, paper_par, P, model, optimizer = vals
        g = g.sort_values("decision_date")
        simple = g["actual_simple_portfolio_return"].to_numpy(float)
        modeled = g["actual_modeled_portfolio_return"].to_numpy(float)
        pred = g["predicted_portfolio_return"].to_numpy(float)

        wealth, mdd = _wealth_and_drawdown(simple)
        n = len(simple)
        years = n / periods_per_year if periods_per_year > 0 else np.nan
        final_wealth = float(wealth[-1]) if len(wealth) else 1.0
        cagr = (
            float(final_wealth ** (1.0 / years) - 1.0)
            if years and years > 0 and final_wealth > 0 else np.nan
        )

        mean = float(np.mean(modeled))
        std = float(np.std(modeled, ddof=1)) if n > 1 else np.nan
        sharpe_period = mean / std if std and std > 0 else np.nan
        sharpe_ann = (
            sharpe_period * math.sqrt(periods_per_year)
            if np.isfinite(sharpe_period) else np.nan
        )
        ann_arith = float(np.mean(simple) * periods_per_year)
        # Realized annualized volatility -- the axis the risk-matched comparison
        # is read on (models cannot be compared at a common nominal P because
        # their mu dispersion differs; they can be compared at a common risk).
        vol_ann = (
            float(std * math.sqrt(periods_per_year))
            if np.isfinite(std) else np.nan
        )

        var95 = (
            float(-np.quantile(simple, 0.05))
            if len(simple) >= 5 else np.nan
        )

        rows.append({
            "seed": int(seed),
            "risk_label": risk_label,
            "paper_risk_parameter": float(paper_par),
            "P": float(P),
            "model": model,
            # Constant within the group -- carried so `_vs_historical` can pair
            # each model with the baseline reading its own conditioning window.
            "matched_baseline": str(g["matched_baseline"].iloc[0]),
            "optimizer": optimizer,
            "mean_period_return": float(np.mean(simple)),
            "annualized_arithmetic_return": ann_arith,
            "cagr": cagr,
            "sharpe_period": sharpe_period,
            "sharpe_annualized": sharpe_ann,
            "annualized_vol": vol_ann,
            "max_drawdown": mdd,
            "final_wealth": final_wealth,
            "var95_loss": var95,
            "positive_rate": float(np.mean(simple > 0)),
            "negative_count": int(np.sum(simple < 0)),
            "portfolio_forecast_mae": float(np.mean(np.abs(pred - modeled))),
            "mean_realized_utility": float(g["realized_utility"].mean()),
            "mean_turnover_l1": float(g["turnover_l1"].dropna().mean())
                if g["turnover_l1"].notna().any() else np.nan,
            "mean_active_count": float(g["active_count"].mean()),
            "mean_solver_sec": float(g["solver_sec"].mean()),
            "n_decisions": int(n),
        })

    return pd.DataFrame(rows)


def _forecast_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    metrics = [
        "r2", "r2_os_vs_historical", "mae", "rmse", "mape",
        "directional_accuracy",
    ]
    rows = []
    for model, g in summary.groupby("model", sort=True):
        row = {"model": model, "n_seeds": int(g["seed"].nunique())}
        for m in metrics:
            row[m + "_mean"] = float(g[m].mean())
            row[m + "_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _vs_historical(summary: pd.DataFrame) -> pd.DataFrame:
    """Paired estimator deltas versus each model's MATCHED historical baseline.

    There is no single `Historical` row: every model is differenced against
    `Historical@<its own conditioning window>`, so the delta isolates method and
    not how much recent data the two sides were allowed to read.
    """
    if summary.empty:
        return summary
    keys = ["seed", "risk_label", "paper_risk_parameter", "P", "optimizer"]
    hist = summary[summary["model"].str.startswith("Historical@")].set_index(
        keys + ["model"]
    )
    rows = []
    metrics = [
        "cagr", "sharpe_period", "sharpe_annualized", "annualized_vol",
        "max_drawdown",
        "final_wealth", "positive_rate", "portfolio_forecast_mae",
        "mean_realized_utility", "mean_turnover_l1",
    ]
    for _, r in summary[~summary["model"].str.startswith("Historical@")].iterrows():
        key = tuple(r[k] for k in keys) + (r["matched_baseline"],)
        if key not in hist.index:
            continue
        h = hist.loc[key]
        row = {k: r[k] for k in keys}
        row["model"] = r["model"]
        row["matched_baseline"] = r["matched_baseline"]
        for m in metrics:
            row["delta_" + m] = float(r[m] - h[m]) if pd.notna(r[m]) and pd.notna(h[m]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    group = ["risk_label", "paper_risk_parameter", "P", "model", "optimizer"]
    metrics = [
        "mean_period_return", "annualized_arithmetic_return", "cagr",
        "sharpe_period", "sharpe_annualized", "annualized_vol",
        "max_drawdown", "final_wealth",
        "var95_loss", "positive_rate", "portfolio_forecast_mae",
        "mean_realized_utility", "mean_turnover_l1", "mean_solver_sec",
    ]
    rows = []
    for vals, g in summary.groupby(group, sort=True):
        row = dict(zip(group, vals))
        row["n_seeds"] = int(g["seed"].nunique())
        for m in metrics:
            row[m + "_mean"] = float(g[m].mean())
            row[m + "_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

