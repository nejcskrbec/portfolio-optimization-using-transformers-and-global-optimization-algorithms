#!/usr/bin/env python3
"""Shared benchmark helpers: config assembly, console printing,
and the generic literature runner used by every article task."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH = ROOT / "benchmark"
CFG = BENCH / "configs"

LITERATURE_BENCHMARK_VERSION = "2026.09.02-resolver-v2-patchtst-compact"
from benchmark.walkforward import run_experiment, save_outputs

CONFIGS = {
    # Leow: only the All-Weather basket is retained -- it is the sole universe the
    # paper reports numeric results for (their Tables 4 and 6). The sector-SPDR and
    # diversified-ETF variants were archived on 2026-09-10 (see
    # archive/leow_removed_20260910/).
    "leow-allweather": CFG / "leow_allweather.json",
    "wang": CFG / "wang.json",
    "practical": CFG / "practical.json",
    # Aprea & Sbaiz (2025) method-substitution benchmark: DJIA (n=28) and
    # NASDAQ-100 (n=54), monthly OOS 2016-2020, K=10 / w_max=0.2 (their u_i=0.2).
    "aprea-djia": CFG / "aprea_djia.json",
    "aprea-nasdaq": CFG / "aprea_nasdaq.json",
}


def _load_optimizer_config(config: dict) -> dict:
    with open(ROOT / "portfolio_optimizers" / "config.json", encoding="utf-8") as f:
        solver = json.load(f)
    pf = config["portfolio"]
    return {
        "common": {
            "population_size": solver["common"]["population_size"],
            "num_generations": solver["common"]["num_generations"],
            "seed": solver["common"]["seed"],
            "cardinality_K": int(pf["cardinality_K"]),
            "w_min": float(pf["w_min"]),
            "w_max": float(pf["w_max"]),
            "risk_parameter": 0.5,
        },
        "pso": solver["pso"],
        "sa": solver["sa"],
    }


def _print_forecast(df):
    print("\nFORECAST SUMMARY")
    print(df.sort_values(["seed", "model"]).to_string(index=False))


def _print_portfolio(df):
    print("\nPORTFOLIO SUMMARY")
    print(df.sort_values(["seed", "risk_label", "optimizer", "model"]).to_string(index=False))


def task_literature(config_path, smoke=False):
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    print(f"[benchmark] engine: {LITERATURE_BENCHMARK_VERSION}")
    print(f"[benchmark] config: {config_path}")
    config["optimizer_config"] = _load_optimizer_config(config)
    result = run_experiment(config, smoke=smoke)
    _print_forecast(result["forecast_summary"])
    _print_portfolio(result["portfolio_summary"])
    out = save_outputs(config, result)
    print(f"\nSaved to: {out}")
    return out


def _dry(label, dry):
    if dry:
        print(f"[dry-run] {label}")
        return True
    return False


def _run_named_literature(name, smoke=False, dry=False):
    if _dry(f"literature {name} ({CONFIGS[name].name})" + (" --smoke" if smoke else ""), dry):
        return
    # `literature`/`all` run several configs in one process; clear the per-run
    # MASTER data caches so they do not leak across benchmarks.
    from estimators.master_us import reset_cube_cache, reset_ohlcv_cache
    reset_ohlcv_cache()
    reset_cube_cache()
    return task_literature(CONFIGS[name], smoke=smoke)

