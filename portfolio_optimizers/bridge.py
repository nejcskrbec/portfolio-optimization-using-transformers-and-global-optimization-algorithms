#!/usr/bin/env python3
"""Python side of the C++ portfolio optimizer: writes the JSON bridge
files, invokes the compiled `portfolio_optimizer` binary, and parses its output."""
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


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_binary() -> str:
    root = project_root()
    candidates = [
        root / "portfolio_optimizer",
        root / "portfolio_optimization",
        root / "portfolio_optimizers" / "portfolio_optimizer",
        root / "portfolio_optimizers" / "portfolio_optimization",
    ]
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)

    for name in ("portfolio_optimizer", "portfolio_optimization"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError(
        "Compiled C++ optimizer binary not found. Expected e.g. "
        f"{root / 'portfolio_optimizers' / 'portfolio_optimizer'}"
    )


def _mvo_utility(w, mu, cov, P) -> float:
    w = np.asarray(w, dtype=float)
    return float(P * (np.asarray(mu) @ w) - (1.0 - P) * (w @ np.asarray(cov) @ w))


def _validate_solution(weights: np.ndarray, common: dict) -> dict:
    w = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(w)):
        raise RuntimeError("Optimizer returned non-finite weights")

    K = int(common["cardinality_K"])
    w_min = float(common["w_min"])
    w_max = float(common["w_max"])
    tol = 5e-4

    raw_sum = float(w.sum())
    if raw_sum <= 0 or abs(raw_sum - 1.0) > 5e-3:
        raise RuntimeError(f"Budget violation: sum(weights)={raw_sum:.8f}")

    # Only remove harmless text-output rounding after validating the raw sum.
    w = w / raw_sum
    active = w > 1e-8
    n_active = int(active.sum())

    if n_active != K:
        raise RuntimeError(f"Cardinality violation: active={n_active}, expected={K}")

    aw = w[active]
    if np.any(aw < w_min - tol):
        raise RuntimeError(
            f"w_min violation: min active={aw.min():.8f}, expected >= {w_min:.8f}"
        )
    if np.any(aw > w_max + tol):
        raise RuntimeError(
            f"w_max violation: max active={aw.max():.8f}, expected <= {w_max:.8f}"
        )
    if np.any(w < -tol):
        raise RuntimeError(f"Long-only violation: min={w.min():.8f}")

    return {
        "weights": w,
        "active_count": n_active,
        "min_active": float(aw.min()),
        "max_active": float(aw.max()),
    }


def run_optimizer(base_config: dict, tickers: list[str], algo: str,
                  mu: np.ndarray, cov: np.ndarray, P: float,
                  seed: int, binary: str) -> dict:
    algo = algo.lower()
    if algo not in ("pso", "sa"):
        raise ValueError(f"Unsupported optimizer: {algo}")

    cfg = copy.deepcopy(base_config)
    cfg.setdefault("run_settings", {})["optimizer_type"] = algo
    common = cfg["optimizer_config"]["common"]
    common["seed"] = int(seed)
    common["risk_parameter"] = float(P)

    n = len(tickers)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if mu.shape != (n,):
        raise ValueError(f"mu shape={mu.shape}, expected {(n,)}")
    if cov.shape != (n, n):
        raise ValueError(f"cov shape={cov.shape}, expected {(n, n)}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(cov)):
        raise ValueError("Non-finite mu/cov")

    bridge = {
        "n": n,
        "tickers": list(tickers),
        "mu": mu.tolist(),
        "cov": cov.ravel().tolist(),
    }

    binary = os.path.abspath(binary)
    with tempfile.TemporaryDirectory(prefix="literature_benchmark_") as td:
        cfg_path = os.path.join(td, "cfg.json")
        data_path = os.path.join(td, "bridge.json")
        cfg["run_settings"]["data_bridge_file"] = data_path

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(bridge, f)

        t0 = time.perf_counter()
        proc = subprocess.run(
            [binary, cfg_path],
            capture_output=True,
            text=True,
            cwd=td,
        )
        elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            f"{algo.upper()} failed ({proc.returncode}).\n"
            f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )

    weights = np.zeros(n, dtype=float)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    number_at_end = re.compile(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(%)?\s*$"
    )

    parsed = 0
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        ticker = parts[0].rstrip(":")
        if ticker not in ticker_to_idx:
            continue
        m = number_at_end.search(line)
        if not m:
            continue
        value = float(m.group(1))
        if m.group(2):
            value /= 100.0
        weights[ticker_to_idx[ticker]] = value
        parsed += 1

    if parsed == 0:
        raise RuntimeError(
            f"No portfolio weights parsed from {algo.upper()} output.\n{proc.stdout}"
        )

    audit = _validate_solution(weights, common)
    w = audit.pop("weights")
    utility = _mvo_utility(w, mu, cov, P)
    return {
        "weights": w,
        "utility": utility,
        "objective": -utility,
        "solver_sec": float(elapsed),
        **audit,
    }

