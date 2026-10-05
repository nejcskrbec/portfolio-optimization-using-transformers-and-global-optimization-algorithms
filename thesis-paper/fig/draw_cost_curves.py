#!/usr/bin/env python3
"""Portfolio value of each model with and without transaction costs (long-term scenario, RD, P = 0.5).

Solid line: gross value. Dashed line: net value with costs of 10 bps per unit of L1 turnover,
subtracted from each month's return before chaining. Dotted: S&P 500 index.

Writes thesis-paper/fig/cost_curves.pdf.
Run from the repo root:  python thesis-paper/fig/draw_cost_curves.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "cost_curves.pdf"
COST_BPS = 10.0
COLOR = {"MASTER": "tab:green", "PatchTST": "tab:blue", "TFT": "tab:orange"}

df = pd.read_csv(ROOT / "test_results" / "literature" / "longterm_investor" / "portfolio_results.csv")
df_all = df
summ = pd.read_csv(ROOT / "test_results" / "literature" / "longterm_investor" / "portfolio_summary.csv")
summ = summ[np.isclose(summ["P"].astype(float), 0.5)]
df = df[(df["optimizer"].str.upper() == "PSO") & np.isclose(df["P"].astype(float), 0.5)]
decs = df[["decision_date", "realization_date"]].drop_duplicates().sort_values("decision_date").reset_index(drop=True)
dates = pd.to_datetime([decs["decision_date"].iloc[0]] + list(decs["realization_date"]))
raw = yf.download("^GSPC", start=(dates[0] - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                  end=(dates[-1] + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                  progress=False, auto_adjust=True)["Close"].dropna()
idx = raw.reindex(dates, method="ffill").to_numpy(float).ravel()
idx = idx / idx[0]


def wealth(model, cost_bps, opt="PSO"):
    g = df_all[(df_all["model"] == model) & (df_all["optimizer"].str.upper() == opt)
               & np.isclose(df_all["P"].astype(float), 0.5)]
    r = g.groupby("decision_date")["actual_simple_portfolio_return"].mean().reindex(decs["decision_date"]).to_numpy(float)
    t = g.groupby("decision_date")["turnover_l1"].mean().reindex(decs["decision_date"]).to_numpy(float)
    r = r - np.nan_to_num(t) * cost_bps / 1e4
    return np.r_[1.0, np.cumprod(1.0 + r)]


def num(v):
    return f"{v:.2f}".replace(".", ",")


plt.rcParams.update({"font.family": "serif"})
INK, MUTED = "0.15", "0.45"
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3), sharey=True)
OPT_LABEL = {"PSO": "RD", "SA": "SO"}
for ax, m in zip(axes, ("MASTER", "PatchTST", "TFT")):
    # the better solver for this model (higher Sharpe, then terminal wealth, then RD), as in equity_pairs
    key = lambda o: (round(float(summ[(summ["model"] == m) & (summ["optimizer"].str.upper() == o)]["sharpe_annualized"].iloc[0]), 2),
                     round(wealth(m, 0.0, o)[-1], 4))
    best = "SA" if key("SA") > key("PSO") else "PSO"
    g, n = wealth(m, 0.0, best), wealth(m, COST_BPS, best)
    ax.plot(dates, idx, ls=":", color="black", lw=1.0, zorder=2)
    ax.plot(dates, g, ls="-", color=COLOR[m], lw=1.6, zorder=4)
    ax.plot(dates, n, ls=(0, (3, 1.5)), color=COLOR[m], lw=1.4, alpha=0.75, zorder=3)
    ax.text(dates[-1], g[-1], " " + num(g[-1]), color=COLOR[m], fontsize=7, va="center", ha="left", clip_on=False)
    ax.text(dates[-1], n[-1] - 0.12 * (g[-1] - n[-1]) , " " + num(n[-1]), color=COLOR[m], fontsize=7, va="top", ha="left", clip_on=False, alpha=0.9)
    ax.set_title(f"{m} + {OPT_LABEL[best]}", fontsize=8, color=INK)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    ax.grid(color="0.92", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.6")
    ax.margins(x=0.02)
axes[0].set_ylabel("Vrednost portfelja", fontsize=7.5, color=INK)
handles = [Line2D([], [], color="0.2", lw=1.6, label="brez stroškov"),
           Line2D([], [], color="0.2", lw=1.4, ls=(0, (3, 1.5)), label=f"stroški {COST_BPS:g} b.t."),
           Line2D([], [], color="black", lw=1.0, ls=":", label="S&P 500")]
fig.legend(handles=handles, ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 0.97, 0.9))
fig.savefig(OUT)
print("wrote", OUT, {m: (round(wealth(m, 0)[-1], 2), round(wealth(m, COST_BPS)[-1], 2)) for m in COLOR})
