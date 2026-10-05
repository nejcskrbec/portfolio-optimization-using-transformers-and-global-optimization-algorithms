#!/usr/bin/env python3
"""OKN and the portfolios found by RD and SO on all five OR-Library sets.

Runs the C++ optimizers exactly as `benchmark/run.py orlib --orlib-full` does
(K = 10, eps = 0.01, delta = 1, 50 equally spaced weights, pop 100, 750 generations, seed 42),
so the points are the ones behind the thesis table.

Writes thesis-paper/fig/orlib_frontiers.pdf.
Run from the repo root:  python thesis-paper/fig/draw_orlib_frontiers.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from benchmark.tasks.task_orlib import ORLIB_DATASETS, load_frontier, load_orlib
from portfolio_optimizers.bridge import find_binary, run_optimizer

OUT = Path(__file__).resolve().parent / "orlib_frontiers.pdf"
K, EPS, DELTA, SEED, POP, GEN = 10, 0.01, 1.0, 42, 100, 750
STYLE = {"pso": ("tab:red", "RD"), "sa": ("tab:green", "SO")}


def base_config():
    with open(ROOT / "portfolio_optimizers" / "config.json") as f:
        solver = json.load(f)
    solver["common"].update({"population_size": POP, "num_generations": GEN, "cardinality_K": K,
                             "w_min": EPS, "w_max": DELTA, "seed": SEED})
    return {"optimizer_config": solver, "run_settings": {"tickers": [], "lookahead_days": 1}}


def trace(cfg, tickers, algo, mu, cov, binary):
    pts = []
    for lam in np.linspace(0.0, 1.0, 50):
        w = run_optimizer(cfg, tickers, algo, mu, cov, float(np.clip(1.0 - lam, 0, 1)), SEED, binary)["weights"]
        if w.sum() >= 1e-9:
            pts.append((float(w @ mu), float(w @ cov @ w)))
    return np.array(pts)


plt.rcParams.update({"font.family": "serif"})
INK, MUTED = "0.15", "0.45"
binary, cfg = find_binary(), base_config()
fig, axes = plt.subplots(1, 5, figsize=(5.2, 1.6))
for ax, (ds, meta) in zip(axes, ORLIB_DATASETS.items()):
    mu, cov, _ = load_orlib(ds)
    fr = load_frontier(ds)
    tick = [f"A{i}" for i in range(len(mu))]
    ax.plot(np.sqrt(fr[:, 1]), fr[:, 0], "-", color="0.2", lw=1.1, zorder=2)
    for algo, (col, lab) in STYLE.items():
        pts = trace(cfg, tick, algo, mu, cov, binary)
        ax.scatter(np.sqrt(pts[:, 1]), pts[:, 0], s=3.5, color=col, alpha=0.85, zorder=3, linewidths=0)
    ax.set_title(f"{meta['name']} ({meta['N']})", fontsize=5.4, color=INK)
    ax.tick_params(labelsize=4.4, colors=MUTED, length=1.5, pad=1.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.set_xlabel("Tveganje $\\sigma$", fontsize=5.0, color=INK, labelpad=1.2)
    ax.grid(color="0.92", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.6")
    print(ds, "done", flush=True)
axes[0].set_ylabel("Pričakovani donos $\\mu$", fontsize=5.0, color=INK, labelpad=1.2)
handles = [Line2D([], [], color="0.2", lw=1.3, label="OKN"),
           Line2D([], [], ls="none", marker="o", ms=4, color="tab:red", label="RD"),
           Line2D([], [], ls="none", marker="o", ms=4, color="tab:green", label="SO")]
fig.legend(handles=handles, ncol=3, fontsize=5.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.6)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT)
