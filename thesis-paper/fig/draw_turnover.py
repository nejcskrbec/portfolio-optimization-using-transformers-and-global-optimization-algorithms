"""Annual turnover per calendar year of each model vs its matched historical mean (long-term scenario, RD, P = 0.5).

Annual turnover of a calendar year = (mean per-rebalance L1 turnover in that year) * 12; the mean (not a raw sum) keeps
2024 comparable, since it has only 11 decisions. The first decision (initial allocation) has no turnover and is excluded.

Writes thesis-paper/fig/turnover_letno.pdf (not yet referenced from main.tex).
Run from the repo root:  python thesis-paper/fig/draw_turnover.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "test_results/literature/longterm_investor/portfolio_results.csv")
df = df[(df["optimizer"] == "PSO") & np.isclose(df["P"].astype(float), 0.5)]
t = df.pivot_table(index="decision_date", columns="model", values="turnover_l1", aggfunc="mean")
t.index = pd.to_datetime(t.index)
# pivot_table already drops the first decision (initial allocation, turnover undefined = NaN)
assert t.notna().all().all() and len(t) == 167
# Annual turnover per calendar year = (mean per-rebalance turnover) * 12; 2024 has 11 months, so mean*12 (not a raw sum)
ann = t.groupby(t.index.year).mean() * 12
cnt = t.groupby(t.index.year).size()
print(cnt.to_dict())
print(ann.round(1).to_string())
print("range:", {m: (round(ann[m].min(), 1), round(ann[m].max(), 1)) for m in ann})

PAIRS = [("MASTER", "Historical@67", "tab:green"),
         ("PatchTST", "Historical@336", "tab:blue"),
         ("TFT", "Historical@271", "tab:orange")]
plt.rcParams.update({"font.family": "serif"})
INK, MUTED = "0.15", "0.45"
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for m, h, c in PAIRS:
    ax.plot(ann.index, ann[h], ls=(0, (3, 1.5)), color=c, lw=1.4, alpha=0.75, zorder=3)
    ax.plot(ann.index, ann[m], ls="-", color=c, lw=1.6, zorder=4)
    # direct label at the right end of the solid (model) lines only; the dashed lines are explained in the legend
    ax.text(ann.index[-1] + 0.25, ann[m].iloc[-1], m, color=INK, fontsize=7.5, va="center", ha="left", clip_on=False)
ax.set_xticks([2011, 2014, 2017, 2020, 2024])
ax.set_ylim(0, 22)
ax.set_yticks([0, 5, 10, 15, 20])
ax.tick_params(labelsize=7.5, colors=MUTED, length=2)
ax.grid(color="0.92", lw=0.6, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color("0.6")
ax.margins(x=0.02)
ax.set_ylabel(r"Letni promet $\mathrm{TO}_{\mathrm{letno}}$", fontsize=8, color=INK)
handles = [Line2D([], [], color="0.2", lw=1.6, label="model"),
           Line2D([], [], color="0.2", lw=1.4, ls=(0, (3, 1.5)), label="zgodovinsko povprečje z istim oknom")]
fig.legend(handles=handles, ncol=2, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 0.88, 0.92))
fig.savefig(OUT / "turnover_letno.pdf")
fig.savefig(OUT / "turnover_letno.png", dpi=200)
