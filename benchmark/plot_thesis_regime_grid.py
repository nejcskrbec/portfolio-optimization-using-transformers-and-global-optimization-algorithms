"""
benchmark/plot_thesis_regime_grid.py
====================================
Kompaktna mreža (2×3) čez vseh šest tržnih režimov. V vsakem panelu je
transformerski cevovod narisan s POSAMEZNIMI metahevristikami (PSO, SA, GA),
kot referenca pa klasični zgodovinski cevovod (siva črtkana, PSO). Namen:
z enim pogledom pokazati robustnost transformerskega signala čez režime in
da se solverji (PSO/SA/GA) povsod tesno ujemajo.

Bere SAMO že-piklane rezultate (test_results/results_*.pkl). Zaženi iz repo roota:
    python benchmark/plot_thesis_regime_grid.py
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "test_results")

ALGO_COLOR = {"pso": "#2ca02c", "sa": "#d62728", "ga": "#9467bd"}
ALGO_LABEL = {"pso": "PSO", "sa": "SA", "ga": "GA"}

# (pkl, začetek obdobja, kratek naslov panela)
REGIMES = [
    ("results_200701_201012_gfc2008.pkl",     "2007-01-01", "GFC (kriza 2008)"),
    ("results_201301_201712.pkl",             "2013-01-01", "2013–2017 (umirjen trg)"),
    ("results_201906_202112_covid2020.pkl",   "2019-06-01", "COVID (kriza 2020)"),
    ("results_202201_202412.pkl",             "2022-01-01", "2022–2024 (dvig obr. mer)"),
    ("results_201901_202212_divuniverse.pkl", "2019-01-01", "divuniverse (brez velikanov)"),
    ("results_201901_202212.pkl",             "2019-01-01", "headline (z velikani)"),
]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7.5,
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _cum(rets_list):
    r = np.concatenate([np.asarray(w, dtype=float) for w in rets_list])
    return (np.cumprod(1.0 + r) - 1.0) * 100.0


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    axes = axes.ravel()

    for ax, (pkl, start, title) in zip(axes, REGIMES):
        path = os.path.join(RESULTS_DIR, pkl)
        if not os.path.isfile(path):
            ax.set_visible(False)
            continue
        d = pickle.load(open(path, "rb"))
        ar = d["all_results"]
        n = len(np.concatenate(
            [np.asarray(w, float) for w in ar["Transformer"]["pso"]]))
        dates = pd.bdate_range(start=start, periods=n)

        for algo in ["pso", "sa", "ga"]:
            rl = ar.get("Transformer", {}).get(algo)
            if not rl:
                continue
            cum = _cum(rl)
            ax.plot(dates[:len(cum)], cum, color=ALGO_COLOR[algo],
                    linewidth=1.4, alpha=0.95, label=ALGO_LABEL[algo])
        zg = ar.get("Zgodovinski", {}).get("pso")
        if zg:
            cum = _cum(zg)
            ax.plot(dates[:len(cum)], cum, color="#7f7f7f", linewidth=1.4,
                    linestyle="--", label="Zgodovinski")

        ax.set_title(title, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=0)
        ax.legend(loc="upper left", framealpha=0.9, ncol=2, columnspacing=0.8,
                  handlelength=1.2)

    for i in (0, 3):
        axes[i].set_ylabel("Kumulativen donos (%)")
    fig.suptitle("Transformerski cevovod (PSO / SA / GA) proti zgodovinskemu "
                 "čez šest tržnih režimov", fontweight="bold", y=1.005)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "regime_grid_algos.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  shranjeno: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
