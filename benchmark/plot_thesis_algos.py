"""
benchmark/plot_thesis_algos.py
==============================
Nariše tezno sliko, ki v dveh panelih (Transformer / Ansambel) prikaže vse TRI
metahevristike (PSO, SA, GA) — da se vidi, da vsi solverji dosežejo prednost
transformerskega cevovoda in se med seboj skladajo. Kot kontekst dodamo klasični
cevovod (Zgodovinski) in pasivno 1/N.

Bere SAMO že-piklane rezultate (test_results/results_*.pkl); datumsko os
rekonstruira iz obdobja režima (delovni dnevi). Zaženi iz repo roota:
    python benchmark/plot_thesis_algos.py
"""
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "test_results")

ALGO_COLOR = {
    "pso": "#2ca02c", "sa": "#d62728", "ga": "#9467bd",
}
ALGO_LABEL = {"pso": "PSO", "sa": "SA", "ga": "GA"}

# (pkl, začetek obdobja, naslov panela)
REGIMES = [
    ("results_201901_202212_divuniverse.pkl", "2019-01-01",
     "divuniverse 2019–2022 (umirjen trg, brez tehnoloških velikanov)"),
    ("results_200701_201012_gfc2008.pkl", "2007-01-01",
     "Globalna finančna kriza 2007–2010"),
]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _cum(rets_list):
    r = np.concatenate([np.asarray(w, dtype=float) for w in rets_list])
    return (np.cumprod(1.0 + r) - 1.0) * 100.0


def _dates(start, n):
    return pd.bdate_range(start=start, periods=n)


def main():
    for pkl, start, title in REGIMES:
        path = os.path.join(RESULTS_DIR, pkl)
        if not os.path.isfile(path):
            print(f"  manjka: {pkl}")
            continue
        d = pickle.load(open(path, "rb"))

        # dolžino niza vzamemo iz Transformer/pso in postavimo datumsko os
        n = len(np.concatenate(
            [np.asarray(w, float) for w in d["all_results"]["Transformer"]["pso"]]))
        dates = _dates(start, n)

        ar, br = d["all_results"], d.get("baseline_results", {})

        # Dva panela: levo vse metahevristike v scenariju Transformer, desno vse
        # metahevristike v scenariju Ansambel. Skupna os y za neposredno primerjavo.
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

        def _draw(ax, scen, panel_title):
            for algo in ["pso", "sa", "ga"]:
                rl = ar.get(scen, {}).get(algo)
                if not rl:
                    continue
                cum = _cum(rl)
                ax.plot(dates[:len(cum)], cum, color=ALGO_COLOR[algo],
                        linewidth=1.7, alpha=0.95, label=ALGO_LABEL[algo])
            # referenci v obeh panelih
            zg = ar.get("Zgodovinski", {}).get("pso")
            if zg:
                cum = _cum(zg)
                ax.plot(dates[:len(cum)], cum, color="#7f7f7f", linewidth=1.7,
                        linestyle="--", label="Zgodovinski")
            if "1/N" in br:
                cum = _cum(br["1/N"])
                ax.plot(dates[:len(cum)], cum, color="black", linewidth=1.4,
                        linestyle=":", label="1/N")
            ax.set_title(panel_title, fontweight="bold")
            ax.set_xlabel("Datum")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

        _draw(axes[0], "Transformer", "Transformer (MASTER)")
        # Ansambel/Hedge removed 2026-08-25; only Transformer shown
        axes[1].axis("off")
        axes[0].set_ylabel("Kumulativen donos (%)")
        fig.suptitle(title, fontweight="bold", y=1.02)

        tag = "divuniverse" if "divuniverse" in pkl else "gfc"
        out = os.path.join(RESULTS_DIR, f"walkforward_{tag}_panels.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  shranjeno: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
