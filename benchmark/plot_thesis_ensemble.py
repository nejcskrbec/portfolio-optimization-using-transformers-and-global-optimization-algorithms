"""
benchmark/plot_thesis_ensemble.py
=================================
Nariše tezno sliko, ki JASNO pokaže prednost ansambla v režimu, kjer
transformerski cevovod zaostane za klasičnim zgodovinskim. Izberemo režim
headline 2019--2022 (z velikani), kjer transformer (Sharpe 0,37) zaostane za
zgodovinskim (0,43), spletna kombinacija Hedge (Ansambel, 0,48) pa prekaša
OBA eksperta --- učbeniški primer, da kombinacija napovedi ne zaostane za
najboljšim posameznim virom (meja obžalovanja).

Trije paneli (Zgodovinski / Transformer / Ansambel) s SKUPNO osjo y, v vsakem
POSAMEZNE metahevristike (PSO, SA, GA) --- da se vidi, da se solverji znotraj
scenarija tesno ujemajo, panel Ansambel pa se konča najvišje.

Bere SAMO že-piklane rezultate (test_results/results_201901_202212.pkl);
datumsko os rekonstruira iz obdobja režima (delovni dnevi). Zaženi iz repo roota:
    python benchmark/plot_thesis_ensemble.py
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
PKL   = "results_201901_202212.pkl"       # headline 2019--2022 (z velikani)
START = "2019-01-01"

ALGO_COLOR = {"pso": "#2ca02c", "sa": "#d62728", "ga": "#9467bd"}
ALGO_LABEL = {"pso": "PSO", "sa": "SA", "ga": "GA"}

# (scenarij, naslov panela)
PANELS = [
    ("Zgodovinski", "Zgodovinski"),
    ("Transformer", "Transformer"),
    ("Ansambel",    "Ansambel (Hedge)"),
]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _cum(rets_list):
    r = np.concatenate([np.asarray(w, dtype=float) for w in rets_list])
    return (np.cumprod(1.0 + r) - 1.0) * 100.0


def _ann_sharpe(rets_list):
    r = np.concatenate([np.asarray(w, dtype=float) for w in rets_list])
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def main():
    d = pickle.load(open(os.path.join(RESULTS_DIR, PKL), "rb"))
    ar, br = d["all_results"], d.get("baseline_results", {})

    n = len(np.concatenate(
        [np.asarray(w, float) for w in ar["Transformer"]["pso"]]))
    dates = pd.bdate_range(start=START, periods=n)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for ax, (scen, title) in zip(axes, PANELS):
        for algo in ["pso", "sa", "ga"]:
            rl = ar.get(scen, {}).get(algo)
            if not rl:
                continue
            cum = _cum(rl)
            ax.plot(dates[:len(cum)], cum, color=ALGO_COLOR[algo],
                    linewidth=1.7, alpha=0.95, label=ALGO_LABEL[algo])
        if "1/N" in br:
            cum = _cum(br["1/N"])
            ax.plot(dates[:len(cum)], cum, color="black", linewidth=1.1,
                    linestyle=":", label="1/N")
        # statistiko v naslovu poročamo za reprezentativni solver PSO
        rl_pso = ar.get(scen, {}).get("pso")
        sh_pso = _ann_sharpe(rl_pso) if rl_pso else 0.0
        fin_pso = _cum(rl_pso)[-1] if rl_pso else 0.0
        sub = f"{title}\nSharpe {sh_pso:.2f} · konč. {fin_pso:.0f}%".replace(".", ",")
        ax.set_title(sub, fontweight="bold")
        ax.set_xlabel("Datum")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    axes[0].set_ylabel("Kumulativen donos (%)")
    fig.suptitle("headline 2019–2022: transformer zaostane za zgodovinskim, "
                 "ansambel prekaša oba", fontweight="bold", y=1.02)

    out = os.path.join(RESULTS_DIR, "ensemble_advantage_headline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  shranjeno: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
