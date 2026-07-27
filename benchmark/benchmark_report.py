"""
benchmark_report.py
===================
Vse PREDSTAVITVENE funkcije walk-forward benchmarka na enem mestu — grafi
(matplotlib) IN izhod (konzolne tabele + CSV izvoz). Ločeno od logike
(benchmark_core.py). Nastane z združitvijo nekdanjih benchmark_plots.py +
benchmark_export.py (oba sta bila čist prikaz in sta se navzkrižno uvažala).

Razdelki:
  1. Skupne konstante + slogi
  2. Pomožne funkcije (zlepljanje oken, meje)
  3. Grafi        — plot_combined_walkforward / plot_all_algorithms_grid /
                    plot_weights_grid
  4. Metrike      — calc_metrics / window_turnover / apply_transaction_costs
  5. Konzolne tabele — print_summary_table / print_prediction_quality /
                    print_turnover_and_costs / print_scenario_significance
  6. CSV izvoz    — export_csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 1.  Skupne konstante + slogi

RESULTS_DIR = "test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_ALGOS = ["pso", "sa"]
# Naše metahevristike (naš prispevek).
METAHEURISTIC_ALGOS = ["pso", "sa"]

ALGO_STYLE = {
    "pso":        {"color": "#2ca02c", "label": "PSO"},
    "sa":         {"color": "#d62728", "label": "SA"},
}
# Kratke oznake za konzolne tabele (izpeljane iz ALGO_STYLE label).
ALGO_LABEL = {k: v["label"] for k, v in ALGO_STYLE.items()}

SCENARIO_STYLE = {
    "Transformer":   {"linestyle": "-",  "alpha": 1.0},
    "Transformer+Σ": {"linestyle": ":",  "alpha": 0.85},
    "Zgodovinski":   {"linestyle": "--", "alpha": 0.55},
}
# Neznani scenariji ne smejo sesuti risanja → varen default.
_SCENARIO_STYLE_DEFAULT = {"linestyle": "-", "alpha": 0.9}

# Barva PO SCENARIJU (za razčiščen combined graf, kjer prikažemo le en
# reprezentativni algoritem na scenarij → barva nosi scenarij, ne algoritem).
SCENARIO_COLOR = {
    "Zgodovinski": "#7f7f7f",   # siva  — klasična osnova
    "Transformer": "#1f77b4",   # modra — transformer pipeline
    "TACTiS":      "#2ca02c",   # zelena — TACTiS-2 μ+Σ pipeline
    "MASTER":      "#9467bd",   # vijolična — MASTER (Alpha158 + tržni gating)
    "TFT":         "#e377c2",   # roza — Temporal Fusion Transformer
}
# En reprezentativni METAHEVRISTIČNI solver za combined graf: naše metahevristike
# rešujejo isti CCMV problem skoraj identično (razlike vidne v all_algorithms_grid),
# zato combined graf strne dimenzijo algoritma na to eno. PSO (Cura 2009) = hitra in
# robustna reprezentativna metahevristika.
REP_ALGO_COMBINED = "pso"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


# 2.  Pomožne funkcije

def build_full_series(rets_list, df_test, test_days):
    """Zlepi per-okno arraye v en kontinuiran niz (donosi, datumi)."""
    all_rets, all_dates = [], []
    for w_idx, r in enumerate(rets_list):
        ts = w_idx * test_days
        te = ts + test_days
        if te > len(df_test): break
        d = df_test.index[ts:te]
        all_rets.append(r)
        all_dates.append(d[1:len(r)+1])
    if not all_rets:
        return np.array([]), pd.DatetimeIndex([])
    full_dates = all_dates[0].append(all_dates[1:]) if len(all_dates) > 1 else all_dates[0]
    return np.concatenate(all_rets), full_dates


def _window_boundaries(baseline_list, df_test, test_days):
    return [
        df_test.index[w * test_days]
        for w in range(1, len(baseline_list))
        if w * test_days < len(df_test)
    ]


# 3.  Grafi

# Kumulativni donosi (combined)

def plot_combined_walkforward(all_results, baseline_results,
                               df_test, test_days, period_tag="",
                               rep_algo=REP_ALGO_COMBINED):
    """Razčiščen combined graf: le TRIJE scenariji (Zgodovinski/Transformer/
    Ansambel) kot ena reprezentativna krivulja na scenarij (solver=rep_algo,
    barva po scenariju) + 1/N in Market kot pasivni kontekst. Podrobno primerjavo
    po algoritmih pokaže plot_all_algorithms_grid; tu je poudarek na scenarijih."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pasivni kontekst: samo naivni 1/N in tržni buy&hold (ostale baseline
    # portfelje — GMV/RiskParity/1N@K/Markowitz — kažejo druge tabele/grafi).
    bl_cfg = {
        "1/N":    ("black", ":", 2.0, "1/N (enakomerno)"),
        "Market": ("teal",  ":", 2.2, "Trg (buy&hold)"),
    }
    for key, (col, ls, lw, lbl) in bl_cfg.items():
        if key not in baseline_results:
            continue
        rets, dates = build_full_series(baseline_results[key], df_test, test_days)
        if len(rets):
            ax.plot(dates[:len(rets)], (np.cumprod(1+rets)-1)*100,
                    color=col, linestyle=ls, linewidth=lw, label=lbl)

    # Trije scenariji — po ena krivulja (rep_algo), barvana po scenariju.
    for scen_name in ["Zgodovinski", "TACTiS", "MASTER", "TFT", "Transformer"]:
        algos_dict = all_results.get(scen_name, {})
        rets_list = algos_dict.get(rep_algo) or next(
            (v for v in algos_dict.values() if v), [])   # varen fallback
        if not rets_list: continue
        rets, dates = build_full_series(rets_list, df_test, test_days)
        if not len(rets): continue
        ax.plot(dates[:len(rets)], (np.cumprod(1+rets)-1)*100,
                color=SCENARIO_COLOR.get(scen_name, "#333333"),
                linestyle="-", linewidth=2.2, label=scen_name)

    for bd in _window_boundaries(baseline_results["1/N"], df_test, test_days):
        ax.axvline(bd, color="black", linewidth=0.6, linestyle="--", alpha=0.2)

    ax.set_title(f"Primerjava scenarijev  [{period_tag.replace('_',' → ')}]"
                 f"  ·  solver: {ALGO_LABEL.get(rep_algo, rep_algo)}",
                 fontweight="bold")
    ax.set_ylabel("Kumulativen donos (%)")
    ax.set_xlabel("Datum")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.9)

    path = os.path.join(RESULTS_DIR, f"combined_walkforward_{period_tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Shranjeno: {path}")


# Grid po algoritmih

def plot_all_algorithms_grid(all_results, baseline_results,
                              df_test, test_days, period_tag="",
                              algos=METAHEURISTIC_ALGOS):
    plot_algos = list(algos)
    num_algos = len(plot_algos)
    fig, axes = plt.subplots(nrows=num_algos, ncols=1,
                              figsize=(12, 3*num_algos), sharex=True)
    if num_algos == 1: axes = [axes]

    bl_cfg = {
        "1/N":        ("black", ":",  1.5, "1/N (vse)"),
        "1/N@K_zgod": ("gray",  "-.", 1.5, "1/N@K (Zgod.)"),
    }
    bl_series = {k: build_full_series(v, df_test, test_days)
                 for k, v in baseline_results.items()}
    bl_cum    = {k: (np.cumprod(1+r) if len(r) else np.array([]), d)
                 for k, (r, d) in bl_series.items()}

    bounds = _window_boundaries(baseline_results["1/N"], df_test, test_days)

    for i, algo in enumerate(plot_algos):
        ax = axes[i]
        for key, (col, ls, lw, lbl) in bl_cfg.items():
            cum, dates = bl_cum[key]
            if len(cum): ax.plot(dates[:len(cum)], (cum-1)*100,
                                  color=col, linestyle=ls, linewidth=lw, label=lbl)

        for scen_name in ["Zgodovinski", "TACTiS", "MASTER", "TFT", "Transformer"]:
            if scen_name not in all_results: continue
            rets_list = all_results[scen_name].get(algo, [])
            if not rets_list: continue
            rets, dates = build_full_series(rets_list, df_test, test_days)
            if not len(rets): continue
            cum = np.cumprod(1 + rets)
            ax.plot(dates[:len(cum)], (cum-1)*100,
                    color=ALGO_STYLE[algo]["color"],
                    linestyle=SCENARIO_STYLE.get(scen_name, _SCENARIO_STYLE_DEFAULT)["linestyle"],
                    linewidth=2,
                    label=f"{ALGO_STYLE[algo]['label']} ({scen_name})")

        for bd in bounds:
            ax.axvline(bd, color="black", linewidth=0.5, linestyle="--", alpha=0.2)

        ax.set_title(f"Algoritem: {algo.upper()}", fontweight="bold", fontsize=11)
        ax.set_ylabel("Kum. donos (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                      fontsize=8, frameon=False)

    axes[-1].set_xlabel("Datum")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"all_algorithms_grid_{period_tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Shranjeno: {path}")


# Uteži portfelja

def plot_weights_grid(all_results_weights, df_test, test_days,
                      avail_tickers, period_tag="", algos=METAHEURISTIC_ALGOS):
    scenarios_present = [s for s in ["Zgodovinski", "TACTiS", "MASTER", "TFT", "Transformer"]
                         if s in all_results_weights]
    num_cols = len(scenarios_present)
    if not num_cols: return

    plot_algos = list(algos)
    num_algos = len(plot_algos)
    cmap = plt.cm.get_cmap("tab20", max(len(avail_tickers), 1))
    ticker_colors = {t: cmap(i) for i, t in enumerate(avail_tickers)}
    legend_patches = [plt.matplotlib.patches.Patch(
        facecolor=ticker_colors[t], label=t) for t in avail_tickers]

    fig, axes = plt.subplots(nrows=num_algos, ncols=num_cols,
                              figsize=(7*num_cols, 3.5*num_algos),
                              sharex=True, squeeze=False)

    for row_i, algo in enumerate(plot_algos):
        for col_j, scen_name in enumerate(scenarios_present):
            ax = axes[row_i][col_j]
            wlist = all_results_weights.get(scen_name, {}).get(algo, [])
            if not wlist:
                ax.text(0.5, 0.5, "Ni podatkov", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_ylim(0, 100); continue

            all_w_dates, all_w_vals = [], []
            for win_i, w in enumerate(wlist):
                ts = win_i * test_days
                te = min(ts + test_days, len(df_test))
                if ts >= len(df_test): break
                w_arr = np.array(w, dtype=float)
                if w_arr.sum() > 1e-10: w_arr /= w_arr.sum()
                win_dates = df_test.index[ts:te]
                all_w_dates.append(win_dates)
                all_w_vals.append(np.tile(w_arr, (len(win_dates), 1)))

            if not all_w_dates: continue
            plot_dates = (all_w_dates[0].append(all_w_dates[1:])
                          if len(all_w_dates) > 1 else all_w_dates[0])
            weight_matrix = np.vstack(all_w_vals)
            bottoms = np.zeros(len(plot_dates))

            for t_i, ticker in enumerate(avail_tickers):
                vals = weight_matrix[:, t_i]
                if vals.max() < 1e-6: continue
                ax.fill_between(plot_dates, bottoms, bottoms + vals*100,
                                color=ticker_colors[ticker], alpha=0.88,
                                step="post", linewidth=0)
                ax.step(plot_dates, bottoms + vals*100,
                        where="post", color="white", linewidth=0.4, alpha=0.6)
                bottoms += vals * 100

            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.set_title(f"{ALGO_STYLE[algo]['label']} — {scen_name}",
                         fontsize=10, fontweight="bold")
            ax.set_ylabel("Delež (%)" if col_j == 0 else "")
            ax.grid(True, alpha=0.2, axis="y")

    for col_j, scen_name in enumerate(scenarios_present):
        axes[0][col_j].annotate(scen_name, xy=(0.5, 1.0), xycoords="axes fraction",
                                 xytext=(0, 30), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    for col_j in range(num_cols):
        axes[-1][col_j].set_xlabel("Datum")

    fig.legend(handles=legend_patches, loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=True,
               title="Delnice", title_fontsize=9)
    plt.suptitle(f"Sestava portfelja  [{period_tag.replace('_',' → ')}]",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"weights_grid_{period_tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Shranjeno: {path}")


# 4.  Metrike

def calc_metrics(rets: np.ndarray, ann: int = 252) -> dict:
    if len(rets) < 2:
        return {k: np.nan for k in
                ["cum_ret", "ann_ret", "cagr", "sharpe", "sortino", "omega",
                 "calmar", "max_dd", "dd_mean", "dd_std", "ann_vol"]}
    cum    = np.prod(1 + rets)
    T_yr   = len(rets) / ann
    ann_r  = cum ** (1/T_yr) - 1 if T_yr > 0 else np.nan
    mu_d   = np.mean(rets)
    std_d  = np.std(rets, ddof=1) + 1e-15
    sharpe = (mu_d / std_d) * np.sqrt(ann)
    neg    = rets[rets < 0]
    ds     = np.sqrt(np.mean(neg**2)) * np.sqrt(ann) if len(neg) > 0 else 1e-15
    sort   = (mu_d * ann) / (ds + 1e-15)
    # Omega ratio (Keating & Shadwick 2002) at threshold 0: gains/losses ratio.
    pos    = float(rets[rets > 0].sum())
    negsum = float(-rets[rets < 0].sum())
    omega  = pos / (negsum + 1e-15) if negsum > 0 else np.inf
    # Drawdown series → max, mean and std (Kaucic et al. 2022 report mean/std DD).
    wealth = np.cumprod(1 + rets)
    peak   = np.maximum.accumulate(wealth)
    dd_ser = (peak - wealth) / (peak + 1e-15)
    dd     = dd_ser.max()
    cal    = ann_r / (dd + 1e-15) if dd > 0 else np.nan
    return {
        "cum_ret": round(float(cum-1)*100, 2),
        "ann_ret": round(float(ann_r)*100, 2),
        "cagr":    round(float(ann_r)*100, 2),   # alias (compound annual growth rate)
        "sharpe":  round(float(sharpe), 3),
        "sortino": round(float(sort),   3),
        "omega":   round(float(omega),  3),
        "calmar":  round(float(cal),    3),
        "max_dd":  round(float(dd)*100, 2),
        "dd_mean": round(float(dd_ser.mean())*100, 2),
        "dd_std":  round(float(dd_ser.std(ddof=1))*100, 2),
        "ann_vol": round(float(std_d*np.sqrt(ann))*100, 2),
    }


def window_turnover(weights_list) -> np.ndarray:
    """
    Enosmerni turnover po oknih: τ₀ = ½Σ|w₀| (začetni nakup, =0.5 če Σw=1),
    τₜ = ½Σ|wₜ − wₜ₋₁|. Vrne array dolžine #oken.
    """
    tau = []
    prev = None
    for w in weights_list:
        w = np.asarray(w, dtype=float)
        if prev is None:
            tau.append(0.5 * np.abs(w).sum())
        else:
            tau.append(0.5 * np.abs(w - prev).sum())
        prev = w
    return np.array(tau)


def apply_transaction_costs(rets_list, weights_list, df_test, test_days,
                            tc_bps: float):
    """
    Odšteje transakcijske stroške ob vsakem rebalansu (prvi dan okna).
    strošek_okna = (tc_bps/1e4) · τ_okna. Vrne (net_daily_series, dates).
    """
    tc  = tc_bps / 1e4
    tau = window_turnover(weights_list) if weights_list else None
    net = []
    for w_idx, r in enumerate(rets_list):
        ts = w_idx * test_days
        te = ts + test_days
        if te > len(df_test):
            break
        r = np.array(r, dtype=float).copy()
        if tau is not None and w_idx < len(tau) and len(r) > 0:
            r[0] -= tc * tau[w_idx]      # strošek na prvi dan okna
        net.append(r)
    if not net:
        return np.array([]), None
    _, dates = build_full_series(rets_list, df_test, test_days)
    return np.concatenate(net), dates


# 5.  Konzolne tabele

# Prikazni vrstni red + oznake scenarija za benchmark strategije.
BASELINE_ROWS = [
    ("1/N",             "naivna"),
    ("Market",          "pasivna"),
    ("GMV",             "samo Σ"),
    ("RiskParity",      "samo Σ"),
    ("HRP",             "samo Σ"),
    ("1/N@K_zgod",      "Zgodovinski"),
    ("Markowitz_zgod",  "Zgodovinski"),
]


def print_summary_table(all_results, baseline_results, df_test, test_days,
                        tc_bps: float = 0.0, weights=None, baseline_weights=None):
    """Izpiše agregirano tabelo metrik v konzolo (bruto + neto po stroških)."""
    W = 108
    print(f"\n{'='*W}")
    print(f"{'SKUPNA TABELA REZULTATOV (bruto | neto po '+str(tc_bps)+'bps stroških)':^{W}}")
    print(f"{'='*W}")
    hdr = (f"  {'Model':<18} {'Scenarij':<14} {'Kum%':>7} "
           f"{'Ann%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Vol%':>7} "
           f"{'Turn%':>7} {'NetAnn%':>8} {'NetShrp':>8}")
    print(hdr)
    print(f"  {'-'*(W-2)}")

    # Benchmark strategije — s turnover/net stolpci kjer imamo uteži.
    bw = baseline_weights or {}
    for key, lbl in BASELINE_ROWS:
        if key not in baseline_results:
            continue
        rets_list = baseline_results[key]
        rets, _   = build_full_series(rets_list, df_test, test_days)
        if not len(rets):
            continue
        m = calc_metrics(rets)
        wlist = bw.get(key)
        if wlist:
            turn = float(np.mean(window_turnover(wlist)) * 100)
            net_rets, _ = apply_transaction_costs(rets_list, wlist,
                                                  df_test, test_days, tc_bps)
            nm = calc_metrics(net_rets) if len(net_rets) else m
            turn_s, netann_s, netsh_s = (f"{turn:>7.2f}",
                                         f"{nm['ann_ret']:>8.2f}",
                                         f"{nm['sharpe']:>8.3f}")
        else:
            turn_s, netann_s, netsh_s = f"{'—':>7}", f"{'—':>8}", f"{'—':>8}"
        print(f"  {key:<18} {lbl:<14} "
              f"{m['cum_ret']:>7.2f} {m['ann_ret']:>7.2f} "
              f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>8.2f} {m['ann_vol']:>7.2f} "
              f"{turn_s} {netann_s} {netsh_s}")

    print(f"  {'-'*(W-2)}")
    for scen_name, algos_dict in all_results.items():
        for algo, rets_list in algos_dict.items():
            if not rets_list:
                continue
            rets, _ = build_full_series(rets_list, df_test, test_days)
            if not len(rets):
                continue
            m = calc_metrics(rets)
            wlist = (weights.get(scen_name, {}).get(algo)
                     if weights else None)
            if wlist:
                turn = float(np.mean(window_turnover(wlist)) * 100)
                net_rets, _ = apply_transaction_costs(
                    rets_list, wlist, df_test, test_days, tc_bps)
                nm = calc_metrics(net_rets) if len(net_rets) else m
                turn_s   = f"{turn:>7.2f}"
                netann_s = f"{nm['ann_ret']:>8.2f}"
                netsh_s  = f"{nm['sharpe']:>8.3f}"
            else:
                turn_s, netann_s, netsh_s = f"{'—':>7}", f"{'—':>8}", f"{'—':>8}"
            print(f"  {ALGO_LABEL.get(algo, algo):<18} {scen_name:<14} "
                  f"{m['cum_ret']:>7.2f} {m['ann_ret']:>7.2f} "
                  f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
                  f"{m['max_dd']:>8.2f} {m['ann_vol']:>7.2f} "
                  f"{turn_s} {netann_s} {netsh_s}")

    print(f"{'='*W}\n")


# Kvaliteta napovedi μ (IC, dir-acc)

def print_prediction_quality(pred_quality: dict):
    """Povprečni IC in directional accuracy μ napovedi po scenariju."""
    if not pred_quality:
        return
    W = 72
    print(f"\n{'='*W}")
    print(f"{'KVALITETA NAPOVEDI μ (out-of-sample)':^{W}}")
    print(f"{'='*W}")
    print(f"  {'Vir μ':<16} {'IC (Spearman)':>16} {'IC t-stat':>12} "
          f"{'Dir.acc.':>10} {'N oken':>8}")
    print(f"  {'-'*(W-2)}")
    for scen, rows in pred_quality.items():
        ics = np.array([r["ic"] for r in rows if not np.isnan(r["ic"])])
        das = np.array([r["dir_acc"] for r in rows if not np.isnan(r["dir_acc"])])
        if len(ics) == 0:
            continue
        ic_mean = ics.mean()
        ic_t    = (ic_mean / (ics.std(ddof=1) + 1e-15) * np.sqrt(len(ics))
                   if len(ics) > 1 else np.nan)
        print(f"  {scen:<16} {ic_mean:>16.4f} {ic_t:>12.2f} "
              f"{das.mean():>10.3f} {len(ics):>8d}")
    print("  IC≈0 → μ signal ni napovedljiv; noben optimizator ga ne reši.")
    print(f"{'='*W}\n")


# Turnover & transakcijski stroški

def print_turnover_and_costs(all_results_weights, all_results,
                             df_test, test_days, tc_bps: float):
    """Povprečni turnover, bruto/neto donos in strošek-drag po scenariju/algo."""
    W = 92
    print(f"\n{'='*W}")
    print(f"{('TURNOVER & STROŠKI (tc='+str(tc_bps)+'bps)'):^{W}}")
    print(f"{'='*W}")
    print(f"  {'Model':<16} {'Scenarij':<14} {'AvgTurn%':>9} "
          f"{'GrossAnn%':>10} {'NetAnn%':>9} {'Drag(pp)':>9}")
    print(f"  {'-'*(W-2)}")
    for scen_name, algos_dict in all_results_weights.items():
        for algo, wlist in algos_dict.items():
            if not wlist:
                continue
            rets_list = all_results.get(scen_name, {}).get(algo, [])
            gross, _ = build_full_series(rets_list, df_test, test_days)
            if not len(gross):
                continue
            net, _ = apply_transaction_costs(rets_list, wlist,
                                             df_test, test_days, tc_bps)
            gm = calc_metrics(gross)
            nm = calc_metrics(net) if len(net) else gm
            turn = float(np.mean(window_turnover(wlist)) * 100)
            drag = gm["ann_ret"] - nm["ann_ret"]
            print(f"  {ALGO_LABEL.get(algo, algo):<16} {scen_name:<14} "
                  f"{turn:>9.2f} {gm['ann_ret']:>10.2f} "
                  f"{nm['ann_ret']:>9.2f} {drag:>9.2f}")
    print(f"{'='*W}\n")


# Statistična signifikanca: Transformer vs Zgodovinski

def _paired_test(a: np.ndarray, b: np.ndarray, method: str):
    """Vrne (statistika-ime, p-value). Fallback brez scipy → paired t prek numpy."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3:
        return ("n<3", np.nan)
    try:
        from scipy import stats
        if method == "wilcoxon":
            diff = a - b
            if np.allclose(diff, 0):
                return ("wilcoxon", 1.0)
            return ("wilcoxon", float(stats.wilcoxon(a, b).pvalue))
        return ("t-test", float(stats.ttest_rel(a, b).pvalue))
    except Exception:
        d = a - b
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-15)
        # dvostranski p prek normalne aproksimacije
        from math import erfc, sqrt
        return ("t~norm", float(erfc(abs(t) / sqrt(2))))


def _win_cums(rets_list):
    return np.array([np.prod(1 + np.asarray(r)) - 1 for r in rets_list if len(r)])


def _print_pair_significance(all_results, name_a, name_b, method, W):
    """Paren test čez okna za en par scenarijev (A−B) po algoritmih."""
    if name_a not in all_results or name_b not in all_results:
        return
    title = f"SIGNIFIKANCA: {name_a} vs {name_b} (paren test čez okna)"
    print(f"\n{'='*W}")
    print(f"{title:^{W}}")
    print(f"{'='*W}")
    print(f"  {'Algoritem':<14} {f'ΔKum% (A−B)':>13} {'medn(Δ)%':>10} "
          f"{'test':>10} {'p-value':>9} {'sig':>5}")
    print(f"  {'-'*(W-2)}")
    for algo in all_results[name_a]:
        a_list = all_results[name_a].get(algo, [])
        b_list = all_results[name_b].get(algo, [])
        m = min(len(a_list), len(b_list))
        if m < 3:
            continue
        ac = _win_cums(a_list[:m]); bc = _win_cums(b_list[:m])
        m = min(len(ac), len(bc))
        ac, bc = ac[:m], bc[:m]
        # Preskoči algoritem, kjer je ena stran "prazna" (vsi ~0) → primerjava proti
        # ničelni seriji bi bila artefakt (ne resničen ±).
        if np.allclose(ac, 0.0) or np.allclose(bc, 0.0):
            continue
        diff = ac - bc
        tname, p = _paired_test(ac, bc, method)
        star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"  {ALGO_LABEL.get(algo, algo):<14} {diff.mean()*100:>13.2f} "
              f"{np.median(diff)*100:>10.2f} {tname:>10} {p:>9.4f} {star:>5}")
    print("  *** p<0.01  ** p<0.05  * p<0.1")
    print(f"{'='*W}")


def print_scenario_significance(all_results, df_test, test_days,
                                method: str = "wilcoxon"):
    """
    Parni testi čez okna (per-okno kumulativni donos). TACTiS-2 je zdaj glavni
    napovedni (transformerski) pipeline (μ IN Σ iz skupne porazdelitve); primerjamo
    ga z zgodovinskim baselineom in močnimi klasičnimi baseline-i:
      • TACTiS vs Zgodovinski     — transformerski pipeline vs klasičen (vzorčna Σ).

      • TACTiS vs SimpleML/BL/LSTM — vs preprost ML / klasičen BL / sekvenčni baseline.
    Pari, kjer scenarij manjka, se samodejno preskočijo (_print_pair_significance).
    """
    W = 78
    bases = ["Zgodovinski", "SimpleML", "BlackLitterman", "LSTM"]
    # Glavni transformerski pipeline(i), ki so v tem zagonu prisotni: TACTiS in/ali
    # MASTER. Za vsakega naredimo parne teste proti baseline-om (manjkajoči par se
    # samodejno preskoči). Tako deluje tudi zagon SAMO z MASTER (brez TACTiS).
    for head in ["TACTiS", "MASTER", "TFT"]:
        if head not in all_results:
            continue
        for b in bases:
            _print_pair_significance(all_results, head, b, method, W)
    print()


# 6.  CSV izvoz

def export_csv(all_results, baseline_results, all_results_weights,
               df_test, test_days, avail_tickers, period_tag="",
               prediction_quality=None, tc_bps: float = 0.0):
    """Izvozi summary, daily_returns in weights CSV."""
    print(f"\n  Izvoz CSV ({period_tag})...")

    # daily_returns
    daily = {}
    for key, bl_list in baseline_results.items():
        rets, dates = build_full_series(bl_list, df_test, test_days)
        if len(rets):
            daily[key] = pd.Series(rets, index=dates[:len(rets)])

    for scen_name, algos_dict in all_results.items():
        for algo, rets_list in algos_dict.items():
            rets, dates = build_full_series(rets_list, df_test, test_days)
            if len(rets):
                col = f"{algo}_{scen_name[:4]}"
                daily[col] = pd.Series(rets, index=dates[:len(rets)])

    if daily:
        df_d = pd.DataFrame(daily)
        df_d.index.name = "date"
        p = os.path.join(RESULTS_DIR, f"daily_returns_{period_tag}.csv")
        df_d.to_csv(p, float_format="%.8f")
        print(f"    daily_returns: {p}")

    # summary
    rows = []
    for key, bl_list in baseline_results.items():
        for w_idx, rets in enumerate(bl_list):
            if len(rets):
                ts = w_idx * test_days
                m  = calc_metrics(rets)
                rows.append({"scenario": "Baseline", "algorithm": key,
                             "window": w_idx+1,
                             "date_start": df_test.index[ts].date() if ts < len(df_test) else "",
                             **m})
    for scen_name, algos_dict in all_results.items():
        for algo, rets_list in algos_dict.items():
            for w_idx, rets in enumerate(rets_list):
                if len(rets):
                    ts = w_idx * test_days
                    m  = calc_metrics(rets)
                    rows.append({"scenario": scen_name, "algorithm": algo,
                                 "window": w_idx+1,
                                 "date_start": df_test.index[ts].date() if ts < len(df_test) else "",
                                 **m})
    if rows:
        p = os.path.join(RESULTS_DIR, f"summary_{period_tag}.csv")
        pd.DataFrame(rows).to_csv(p, index=False, float_format="%.6f")
        print(f"    summary:       {p}")

    # weights
    w_rows = []
    for scen_name, algos_dict in all_results_weights.items():
        for algo, wlist in algos_dict.items():
            for w_idx, weights in enumerate(wlist):
                ts = w_idx * test_days
                r  = {"scenario": scen_name, "algorithm": algo,
                      "window": w_idx+1,
                      "date_start": df_test.index[ts].date() if ts < len(df_test) else ""}
                for t, ww in zip(avail_tickers, weights):
                    r[t] = float(ww)
                w_rows.append(r)
    if w_rows:
        p = os.path.join(RESULTS_DIR, f"weights_{period_tag}.csv")
        pd.DataFrame(w_rows).to_csv(p, index=False, float_format="%.6f")
        print(f"    weights:       {p}")

    # prediction_quality (μ IC / dir-acc per okno)
    if prediction_quality:
        pq_rows = []
        for scen, wins in prediction_quality.items():
            for w_idx, r in enumerate(wins):
                pq_rows.append({"scenario": scen, "window": w_idx + 1,
                                "ic": r.get("ic"), "dir_acc": r.get("dir_acc")})
        if pq_rows:
            p = os.path.join(RESULTS_DIR, f"prediction_quality_{period_tag}.csv")
            pd.DataFrame(pq_rows).to_csv(p, index=False, float_format="%.6f")
            print(f"    pred_quality:  {p}")

    # turnover & net returns per scenario/algo
    tc_rows = []
    for scen_name, algos_dict in all_results_weights.items():
        for algo, wlist in algos_dict.items():
            if not wlist:
                continue
            rets_list = all_results.get(scen_name, {}).get(algo, [])
            gross, _ = build_full_series(rets_list, df_test, test_days)
            if not len(gross):
                continue
            net, _ = apply_transaction_costs(rets_list, wlist,
                                             df_test, test_days, tc_bps)
            gm = calc_metrics(gross)
            nm = calc_metrics(net) if len(net) else gm
            tc_rows.append({
                "scenario": scen_name, "algorithm": algo,
                "avg_turnover_pct": round(float(np.mean(window_turnover(wlist)) * 100), 3),
                "gross_ann_pct": gm["ann_ret"], "net_ann_pct": nm["ann_ret"],
                "gross_sharpe": gm["sharpe"], "net_sharpe": nm["sharpe"],
                "cost_drag_pp": round(gm["ann_ret"] - nm["ann_ret"], 3),
            })
    if tc_rows:
        p = os.path.join(RESULTS_DIR, f"turnover_costs_{period_tag}.csv")
        pd.DataFrame(tc_rows).to_csv(p, index=False, float_format="%.6f")
        print(f"    turnover_costs:{p}")


def print_efficiency_ranking(ar, timing, period_tag=""):
    """Rank model+optimizer combinations by Sharpe/second (price-to-performance).
    ar: all_results dict (keyed by (scenario, algo))
    timing: timing dict with train_* and optimizer_* entries
    """
    print(f"\n{'='*90}")
    print(f"  UČINKOVITOST: SHARPE / ČASE (training + walk-forward optimization)")
    print(f"{'='*90}")

    efficiency = []
    for (scen, algo), daily_rets_list in ar.items():
        # Compute PSO Sharpe for this scenario+algo
        d_rets = np.concatenate(daily_rets_list)
        if len(d_rets) < 5:
            continue
        ann_ret = np.mean(d_rets) * 252
        ann_vol = np.std(d_rets) * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-12)

        # Training time: look for the model in this scenario name
        train_time = 0.0
        if "MASTER" in scen:
            train_time = timing.get("train_master", 0)
        elif "TFT" in scen:
            train_time = timing.get("train_tft", 0)
        elif "TACTiS" in scen:
            train_time = timing.get("train_tactis", 0)
        elif "SimpleML" in scen:
            train_time = timing.get("train_simpleml", 0)
        elif "LSTM" in scen:
            train_time = timing.get("train_lstm", 0)

        # Optimizer time (walk-forward total is dominated by walk-forward + optimization)
        opt_key = f"{scen}_{algo}"
        opt_time = timing.get("optimizer_times", {}).get(opt_key, 0)
        total_time = train_time + opt_time

        if total_time > 0.1:
            efficiency_ratio = sharpe / total_time
            efficiency.append({
                "scenario": scen,
                "algo": algo,
                "sharpe": sharpe,
                "train_s": train_time,
                "opt_s": opt_time,
                "total_s": total_time,
                "sharpe_per_sec": efficiency_ratio,
            })

    # Sort by efficiency descending
    efficiency.sort(key=lambda x: x["sharpe_per_sec"], reverse=True)

    # Print top 15
    print(f"{'Rank':<5} {'Scenario':<15} {'Algo':<5} {'Sharpe':>8} {'Time(s)':>8} {'Sharpe/s':>10}")
    print(f"{'-'*60}")
    for rank, row in enumerate(efficiency[:15], 1):
        print(f"{rank:<5} {row['scenario']:<15} {row['algo']:<5} "
              f"{row['sharpe']:>8.3f} {row['total_s']:>8.1f} {row['sharpe_per_sec']:>10.4f}")
    print(f"{'='*90}\n")
