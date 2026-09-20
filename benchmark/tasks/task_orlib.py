#!/usr/bin/env python3
"""OR-Library optimizer-quality benchmark + its data loaders."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from benchmark.utils.benchmark_utils import (
    ROOT, BENCH, CFG, CONFIGS, _load_optimizer_config,
    _print_forecast, _print_portfolio, _dry, _run_named_literature,
)


# Chang et al. (2000) OR-Library instances: port1-5 (returns/covariances) and
# portef1-5 (their published unconstrained efficient frontiers, used as the
# reference the solvers are scored against).
ORLIB_DATA_DIR = os.path.join(ROOT, "data", "orlib")
ORLIB_DATASETS = {
    "port1": {"name": "Hang Seng", "N": 31},
    "port2": {"name": "DAX 100",   "N": 85},
    "port3": {"name": "FTSE 100",  "N": 89},
    "port4": {"name": "S&P 100",   "N": 98},
    "port5": {"name": "Nikkei",    "N": 225},
}


def load_orlib(name: str, data_dir: str = ORLIB_DATA_DIR):
    path = os.path.join(data_dir, f"{name}.txt")
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it))
    mu = np.zeros(N)
    sigma = np.zeros(N)
    for i in range(N):
        mu[i] = float(next(it)); sigma[i] = float(next(it))
    corr = np.zeros((N, N))
    while True:
        try:
            i = int(next(it)) - 1; j = int(next(it)) - 1; r = float(next(it))
        except StopIteration:
            break
        corr[i, j] = r; corr[j, i] = r
    cov = corr * np.outer(sigma, sigma)
    return mu, cov, sigma


def load_frontier(name: str, data_dir: str = ORLIB_DATA_DIR) -> np.ndarray:
    path = os.path.join(data_dir, f"portef{name[-1]}.txt")
    pts = np.loadtxt(path)
    ret, var = pts[:, 0], pts[:, 1]
    order = np.argsort(ret)
    return np.column_stack([ret[order], var[order]])


def _interp_var_at_return(frontier: np.ndarray, r: float) -> float:
    ret, var = frontier[:, 0], frontier[:, 1]
    if r <= ret[0]: return float(var[0])
    if r >= ret[-1]: return float(var[-1])
    return float(np.interp(r, ret, var))


def _interp_return_at_var(frontier: np.ndarray, v: float) -> float:
    ret, var = frontier[:, 0], frontier[:, 1]
    o = np.argsort(var); var_s, ret_s = var[o], ret[o]
    if v <= var_s[0]: return float(ret_s[0])
    if v >= var_s[-1]: return float(ret_s[-1])
    return float(np.interp(v, var_s, ret_s))


def frontier_error(points: np.ndarray, frontier: np.ndarray) -> dict:
    if len(points) == 0:
        return {k: np.nan for k in ("mean", "median", "min", "max", "std", "n")}
    errs = []
    for r, v in points:
        v_star = _interp_var_at_return(frontier, r)
        r_star = _interp_return_at_var(frontier, v)
        var_err = 100.0*(v-v_star)/abs(v_star) if v_star != 0 else np.nan
        ret_err = 100.0*(r_star-r)/abs(r_star) if r_star != 0 else np.nan
        cand = [e for e in (var_err, ret_err) if not np.isnan(e)]
        if cand: errs.append(max(0.0, min(cand)))
    errs = np.asarray(errs)
    if len(errs) == 0:
        return {k: np.nan for k in ("mean", "median", "min", "max", "std", "n")}
    return {"mean": float(errs.mean()), "median": float(np.median(errs)),
            "min": float(errs.min()), "max": float(errs.max()),
            "std": float(errs.std(ddof=1)) if len(errs)>1 else 0.0, "n": int(len(errs))}



def task_orlib(argv=None):
    #!/usr/bin/env python3
    """
    benchmark/run.py orlib
    ======================
    Literaturni benchmark metahevristik na standardnih OR-Library instancah
    (Chang et al. 2000). Za vsak instance in vsak algoritem trasira KARDINALNO
    OMEJENO efficient frontier (sweep po λ) in izmeri standardno odstotno napako
    glede na priloženo NEomejeno fronto — natanko metrika, ki jo poročajo
    Chang et al. (2000), Crama & Schyns (2003), Cura (2009).

    Standardne nastavitve (Chang et al. 2000):
        K = 10,  εᵢ = 0.01,  δᵢ = 1.0.

    Uporaba:
        python benchmark/run.py orlib                 # privzeto: port1–3, 25 λ
        python benchmark/run.py orlib --datasets port1 port2 port3 port4 port5
        python benchmark/run.py orlib --full          # vse instance, 50 λ
        python benchmark/run.py orlib

    C++ optimizator mora biti zgrajen:  cd portfolio_optimizers && make
    """

    import argparse
    import json
    import os
    import sys
    import time

    import numpy as np

    # Skripta živi v benchmark/ → repo koren je nadrejeni imenik (na sys.path
    # zato, da deluje absolutni uvoz `from benchmark.… import …`).
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, ROOT)

    from portfolio_optimizers.bridge import find_binary, run_optimizer
    DATASETS = ORLIB_DATASETS
    DATA_DIR = ORLIB_DATA_DIR

    # The tree ships PSO + SA; bridge.run_optimizer accepts exactly these two.
    METAHEURISTICS = ["pso", "sa"]

    # Style map (previously imported from the deleted benchmark_report.ALGO_STYLE).
    ALGO_STYLE = {
        "pso": {"color": "tab:red",   "label": "PSO"},
        "sa":  {"color": "tab:green", "label": "SA"},
    }

    RESULTS_DIR = os.path.join(ROOT, "test_results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Chang et al. (2000) standardne nastavitve.
    K_STD, EPS_STD, DELTA_STD = 10, 0.01, 1.0


    def _base_config(pop: int, gen: int, K: int, w_min: float, w_max: float,
                     seed: int) -> dict:
        """Minimalni config za C++ optimizator. Solver hiperparametri (per-algo +
        populacija/generacije/seed) iz portfolio_optimizers/config.json; problemske
        omejitve (K, w_min, w_max) nastavi ta benchmark sam na Chang-ove standarde."""
        with open(os.path.join(ROOT, "portfolio_optimizers", "config.json")) as f:
            solver = json.load(f)   # flat: common, pso, sa
        solver["common"].update({"population_size": pop, "num_generations": gen,
                                 "cardinality_K": K, "w_min": w_min, "w_max": w_max,
                                 "seed": seed})
        return {"optimizer_config": solver,
                "run_settings": {"tickers": [], "lookahead_days": 1}}


    def trace_frontier(cfg, tickers, algo, mu, cov, lambdas, seed, binary):
        """Sweep po λ → seznam (donos, varianca) omejene fronte + skupni čas.

        Standardni ε=0.01, δ=1.0 se prenašata prek optimizer_config.common
        (w_min/w_max), ki ju nastavi _base_config; literature_benchmark.run_optimizer
        ne sprejema per-asset eps/delta, ampak globalni meji iz configa."""
        pts, t_tot = [], 0.0
        for lam in lambdas:
            # C++ maksimizira P·μw − (1−P)·wΣw.  Chang: min λ·wΣw − (1−λ)·μw
            # ⇒ P = risk_parameter = 1 − λ.
            P = float(np.clip(1.0 - lam, 0.0, 1.0))
            t0 = time.perf_counter()
            res = run_optimizer(cfg, tickers, algo, mu, cov, P, seed, binary)
            t_tot += time.perf_counter() - t0
            w = res["weights"]
            if w.sum() < 1e-9:
                continue
            pts.append((float(w @ mu), float(w @ cov @ w)))
        return np.array(pts), t_tot


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--datasets", nargs="+", default=["port1", "port2", "port3"],
                        choices=list(DATASETS.keys()))
        ap.add_argument("--algos", nargs="+", default=METAHEURISTICS)
        ap.add_argument("--lambdas", type=int, default=25,
                        help="Število λ točk za trasiranje fronte (Chang: ~50).")
        ap.add_argument("--pop", type=int, default=100)
        ap.add_argument("--gen", type=int, default=750)
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--full", action="store_true",
                        help="Vse instance (port1–5) in 50 λ točk.")
        ap.add_argument("--plot", action="store_true", help="Shrani grafe front.")
        args = ap.parse_args(argv)

        if args.full:
            args.datasets = list(DATASETS.keys())
            args.lambdas  = 50

        lambdas = np.linspace(0.0, 1.0, args.lambdas)
        binary  = find_binary()
        if not os.path.isfile(binary):
            print(f"Napaka: C++ optimizator ni zgrajen ({binary}).\n"
                  f"Zaženi:  cd portfolio_optimizers && make")
            sys.exit(1)

        print(f"{'='*78}")
        print(f"  OR-LIBRARY BENCHMARK (Chang et al. 2000)  "
              f"K={K_STD}, ε={EPS_STD}, δ={DELTA_STD}")
        print(f"  λ točk: {args.lambdas}   pop={args.pop}  gen={args.gen}  "
              f"algos: {', '.join(args.algos)}")
        print(f"  Optimizator: {binary}")
        print(f"{'='*78}")

        cfg = _base_config(args.pop, args.gen, K_STD, EPS_STD, DELTA_STD, args.seed)
        rows, all_pts = [], {}

        for ds in args.datasets:
            mu, cov, sig = load_orlib(ds, DATA_DIR)
            frontier     = load_frontier(ds, DATA_DIR)
            tickers      = [f"A{i}" for i in range(len(mu))]
            meta         = DATASETS[ds]
            all_pts[ds]  = {"_frontier": frontier}
            print(f"\n── {ds}  ({meta['name']}, N={meta['N']}) "
                  f"{'─'*max(0, 50-len(meta['name']))}")
            print(f"  {'algo':<12} {'mean%':>8} {'median%':>8} {'min%':>7} "
                  f"{'max%':>8} {'std%':>7} {'pts':>4} {'čas(s)':>8}")

            algo_list = list(args.algos)
            for algo in algo_list:
                pts, t_tot = trace_frontier(cfg, tickers, algo, mu, cov,
                                            lambdas, args.seed, binary)
                err = frontier_error(pts, frontier)
                all_pts[ds][algo] = pts
                rows.append({"dataset": ds, "index": meta["name"], "N": meta["N"],
                             "algorithm": algo, **{f"err_{k}": v for k, v in err.items()},
                             "time_sec": round(t_tot, 2)})
                print(f"  {algo:<12} {err['mean']:>8.4f} {err['median']:>8.4f} "
                      f"{err['min']:>7.4f} {err['max']:>8.4f} {err['std']:>7.4f} "
                      f"{err['n']:>4d} {t_tot:>8.2f}")

        # Izvoz CSV
        import pandas as pd
        df = pd.DataFrame(rows)
        csv_p = os.path.join(RESULTS_DIR, "orlib_benchmark.csv")
        df.to_csv(csv_p, index=False, float_format="%.6f")
        print(f"\n  Izvoz: {csv_p}")

        # Povzetek: povprečna mean-napaka po algoritmu čez vse instance.
        print(f"\n{'='*78}")
        print(f"  POVZETEK — povprečna mean% napaka čez {len(args.datasets)} instance")
        print(f"{'='*78}")
        summ = (df.groupby("algorithm")["err_mean"].mean()
                  .sort_values())
        for algo, val in summ.items():
            print(f"  {algo:<14} {val:>8.4f}%")

        if args.plot:
            _plot_frontiers(all_pts)


    def _plot_frontiers(all_pts: dict):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for ds, d in all_pts.items():
            fr = d["_frontier"]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(np.sqrt(fr[:, 1]), fr[:, 0], "-", color="0.25", lw=1.8,
                    label="USEF (neomejena fronta, ref.)")
            for algo, pts in d.items():
                if algo == "_frontier" or len(pts) == 0:
                    continue
                st = ALGO_STYLE.get(algo, {"color": "C0", "label": algo})
                ax.scatter(np.sqrt(pts[:, 1]), pts[:, 0], s=18,
                           color=st["color"], label=st["label"], alpha=0.8)
            ax.set_xlabel("Tveganje σ (std)")
            ax.set_ylabel("Pričakovani donos μ")
            ax.set_title(f"{ds} — {DATASETS[ds]['name']} (K={K_STD})")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(alpha=0.3)
            p = os.path.join(RESULTS_DIR, f"orlib_frontier_{ds}.png")
            fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
            print(f"  Graf: {p}")

    main()

