#!/usr/bin/env python3
"""Portfolio value of each model against its matched historical mean (same input window).

One panel per (model, historical mean) pair, one figure per experiment; the passive
index is drawn dotted for reference. Models are read at P = 0.5 (the crisis scenario at each
model's own risk-matched P, as in the tables). The long-term scenario is drawn twice: gross and net
(10 bps per unit of L1 turnover), one figure each. The half-year scenario is skipped: its holding periods
overlap and cannot be chained into one wealth curve.

For every panel the model is drawn with the BETTER of the two solvers (RD = PSO or SO = SA): higher Sharpe
ratio (2 decimals), then higher terminal wealth, then RD. Its matched historical mean is drawn with the SAME
solver, so the pair still isolates the forecast. The panel title names the combination.

Writes thesis-paper/fig/equity_pairs_<experiment>.pdf.
Run from the repo root:  python thesis-paper/fig/draw_equity_pairs.py
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
RES = ROOT / "test_results" / "literature"
FIG = Path(__file__).resolve().parent
OPTS = ("PSO", "SA")
OPT_LABEL = {"PSO": "RD", "SA": "SO"}   # every curve is a (model + optimizer) combination
COLOR = {"MASTER": "tab:green", "PatchTST": "tab:blue", "TFT": "tab:orange"}
plt.rcParams.update({"font.family": "serif"})
INK, MUTED = "0.15", "0.45"

# name, run folder, index ticker, index label, risk-matched?, rows [(label, cost in bps)], date format
EXPERIMENTS = [
    ("krizni", "leow_allweather_direct", "^GSPC", "S&P 500", True, [("", 0.0)], "%d.%m."),
    ("mesecno", "aprea_djia", "^DJI", "DJIA", False, [("", 0.0)], "%Y"),
    # the long-term scenario is drawn twice, gross and net (10 bps per unit of L1 turnover), one row each
    ("dolgorocno_bruto", "longterm_investor", "^GSPC", "S&P 500", False, [("Bruto", 0.0)], "%Y"),
    ("dolgorocno_neto", "longterm_investor", "^GSPC", "S&P 500", False, [("Neto", 10.0)], "%Y"),
]


def num(v):
    return f"{v:.2f}".replace(".", ",")


def build(name, folder, tick, tick_lab, risk_matched, rows, datefmt):
    path = RES / folder
    df = pd.read_csv(path / "portfolio_results.csv")
    summ = pd.read_csv(path / "portfolio_summary.csv")
    mt = (summ[~summ["model"].astype(str).str.startswith("Historical@")]
          .drop_duplicates("model").set_index("model")["matched_baseline"].to_dict())
    models = [m for m in ("MASTER", "PatchTST", "TFT") if m in mt]
    p_by = {}   # (optimizer, model) -> risk-matched P (each solver has its own equal-risk point)
    if risk_matched:
        rm = pd.read_csv(path / "risk_matched.csv")
        for o in OPTS:
            sub = rm[rm["optimizer"].str.upper() == o]
            p_by.update({(o, m): p for m, p in zip(sub["model"], sub["P"])})
    decs = (df[["decision_date", "realization_date"]].drop_duplicates()
            .sort_values("decision_date").reset_index(drop=True))
    dates = pd.to_datetime([decs["decision_date"].iloc[0]] + list(decs["realization_date"]))
    raw = yf.download(tick, start=(dates[0] - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                      end=(dates[-1] + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                      progress=False, auto_adjust=True)["Close"].dropna()
    idx = raw.reindex(dates, method="ffill").to_numpy(float).ravel()
    idx = idx / idx[0]

    def wealth(model, cost_bps, opt):
        p = p_by.get((opt, model), 0.5)
        g = df[(df["model"] == model) & (df["optimizer"].str.upper() == opt) & (np.isclose(df["P"].astype(float), p))]
        r = g.groupby("decision_date")["actual_simple_portfolio_return"].mean().reindex(decs["decision_date"]).to_numpy(float)
        if cost_bps:
            t = g.groupby("decision_date")["turnover_l1"].mean().reindex(decs["decision_date"]).to_numpy(float)
            r = r - np.nan_to_num(t) * cost_bps / 1e4
        return np.r_[1.0, np.cumprod(1.0 + r)]

    # Sharpe of every (model, solver) at the point the tables read it (risk-matched P for Leow, else 0.5)
    sh_src = pd.read_csv(path / "risk_matched.csv") if risk_matched else summ[np.isclose(summ["P"].astype(float), 0.5)]

    def sharpe(model, opt):
        return float(sh_src[(sh_src["model"] == model) & (sh_src["optimizer"].str.upper() == opt)]["sharpe_annualized"].iloc[0])

    def best_opt(model):
        key = lambda o: (round(sharpe(model, o), 2), round(wealth(model, 0.0, o)[-1], 4))
        return "SA" if key("SA") > key("PSO") else "PSO"

    sel = {m: best_opt(m) for m in models}
    sel.update({mt[m]: sel[m] for m in models})   # the matched historical mean uses the model's solver
    nr, nc = len(rows), len(models)
    fig, axes = plt.subplots(nr, nc, figsize=(2.15 * nc + 0.5, 2.15 * nr + 0.6), sharey="row", squeeze=False)
    for ri, (rlab, cost) in enumerate(rows):
        lo = min(min(wealth(m, cost, sel[m]).min(), wealth(mt[m], cost, sel[mt[m]]).min()) for m in models)
        hi = max(max(wealth(m, cost, sel[m]).max(), wealth(mt[m], cost, sel[mt[m]]).max()) for m in models)
        for ci, m in enumerate(models):
            ax = axes[ri][ci]
            base = mt[m]
            wm, wb = wealth(m, cost, sel[m]), wealth(base, cost, sel[base])
            ax.plot(dates, idx, ls=":", color="black", lw=1.0, zorder=2)
            ax.plot(dates, wb, ls=(0, (4, 2)), color="0.45", lw=1.3, zorder=3)
            ax.plot(dates, wm, ls="-", color=COLOR[m], lw=1.6, zorder=4)
            # direct labels of the final values; push apart when close
            ends = sorted([(wm[-1], COLOR[m]), (wb[-1], "0.35")], key=lambda t: t[0])
            gap = (hi - lo) * 0.09
            ys = [e[0] for e in ends]
            if ys[1] - ys[0] < gap:
                mid = (ys[0] + ys[1]) / 2
                ys = [mid - gap / 2, mid + gap / 2]
            for (v, c), y in zip(ends, ys):
                ax.text(dates[-1], y, " " + num(v), color=c, fontsize=7, va="center", ha="left", clip_on=False)
            ax.axhline(1.0, color="0.8", lw=0.6, zorder=1)
            title = f"{m} + {OPT_LABEL[sel[m]]} /\nhist. + {OPT_LABEL[sel[base]]} ({base.split('@')[1]} dni)"
            ax.set_title(title, fontsize=8, color=INK)
            ax.xaxis.set_major_locator(mdates.MonthLocator() if datefmt == "%d.%m." else mdates.AutoDateLocator(maxticks=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(datefmt))
            ax.tick_params(labelsize=7, colors=MUTED, length=2)
            ax.grid(color="0.92", lw=0.6, zorder=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color("0.6")
            ax.margins(x=0.02)
            if ci == 0:
                ax.set_ylabel((rlab + "\n" if rlab else "") + "Vrednost portfelja", fontsize=7.5, color=INK)
    handles = [Line2D([], [], color="0.2", lw=1.6, label="model + optimizator"),
               Line2D([], [], color="0.45", lw=1.3, ls=(0, (4, 2)), label="zgodovinsko povprečje + optimizator"),
               Line2D([], [], color="black", lw=1.0, ls=":", label=tick_lab)]
    fig.legend(handles=handles, ncol=3, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 0.97, 0.93))
    out = FIG / f"equity_pairs_{name}.pdf"
    fig.savefig(out)
    print("wrote", out.name, "idx", round(idx[-1], 2))
    for m in models:
        wm, wb = wealth(m, 0.0, sel[m]), wealth(mt[m], 0.0, sel[mt[m]])
        print(f"   {m}+{OPT_LABEL[sel[m]]}: {wm[-1]:.2f}   {mt[m]}+{OPT_LABEL[sel[mt[m]]]}: {wb[-1]:.2f}   "
              f"ratio {wm[-1] / wb[-1]:.2f}  months above hist {np.mean(wm[1:] >= wb[1:] - 1e-12) * 100:.0f}%")


for e in EXPERIMENTS:
    build(*e)
