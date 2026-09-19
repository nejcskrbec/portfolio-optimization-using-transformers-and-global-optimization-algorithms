#!/usr/bin/env python3
"""All drawing tasks (Wang cumulative-return plot, literature equity curves)."""
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


def task_wang_plot(argv=None):
    #!/usr/bin/env python3
    """
    wang_plot.py
    ============
    Wang et al. (ICLR 2023) Fig. 5 analog: realized 120-day return by 2021 start
    date. Wang's predictive-portfolio experiment is a set of INDEPENDENT overlapping
    120-day bets from monthly start dates, so it has no valid chained equity curve
    (see benchmark/equity_curves.py); the faithful visualization is the realized
    outcome per start date, exactly as in their Fig. 5.

    Weights are recomputed at a chosen risk level P (default 0.2, matching the
    risk-matched Sharpe table) from the stored per-asset mu (prediction_results.csv)
    and the fixed pre-test covariance, then the realized 120-day portfolio return is
    measured from the authors' daily prices. Three cardinality-consistent lines:
    best transformer, classical historical, and the S&P 500 index (buy-and-hold over
    the same window).

    Usage:  python benchmark/wang_plot.py            # P=0.2, writes fig PDF+PNG
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, ROOT)


    from portfolio_optimizers.bridge import run_optimizer, find_binary

    def _base_config(K: int, w_min: float, w_max: float) -> dict:
        with open(os.path.join(ROOT, "portfolio_optimizers", "config.json")) as f:
            solver = json.load(f)  # flat: common, pso, sa, ga
        solver["common"].update({"cardinality_K": K, "w_min": w_min, "w_max": w_max})
        return {"optimizer_config": solver, "run_settings": {}}


    def _fixed_cov(prices: pd.DataFrame, tickers: list[str]) -> np.ndarray:
        """Daily sample covariance over the pre-test fit window, in `tickers` order —
        replicates historical_moments(fixed_pretest) with moment_frequency=daily."""
        p = prices.loc[(prices.index >= pd.Timestamp(FIT_START)) &
                       (prices.index <= pd.Timestamp(FIT_END)), tickers]
        r = p.pct_change(fill_method=None).dropna()
        cov = np.cov(r.to_numpy(float), rowvar=False)
        cov = 0.5 * (cov + cov.T)
        e = float(np.linalg.eigvalsh(cov)[0])
        if e < 1e-10:
            cov = cov + np.eye(cov.shape[0]) * (1e-10 - e)
        return cov


    def _daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.sort_index().pct_change(fill_method=None)



    RUN = os.path.join(ROOT, "test_results", "literature", "wang_sp500")
    PRICES = os.path.join(ROOT, "data", "literature", "wang", "snp500.csv")
    FIG = os.path.join(ROOT, "thesis-paper", "fig", "wang_by_startdate")
    P = 0.2
    SEED = 42
    FIT_START = "2018-01-02"
    FIT_END = "2020-12-30"


    def _window_return(w: dict, rets: pd.DataFrame, d0, d1) -> float:
        cols = [t for t in w if t in rets.columns]
        wv = np.array([w[t] for t in cols], float)
        wv /= wv.sum()
        win = rets.loc[(rets.index > d0) & (rets.index <= d1), cols]
        return float((1 + win.to_numpy(float) @ wv).prod() - 1) * 100.0


    def main():
        import yfinance as yf
        preds = pd.read_csv(f"{RUN}/prediction_results.csv")
        prices = pd.read_csv(PRICES)
        dcol = [c for c in prices.columns if c.lower().startswith("date")][0]
        prices.index = pd.to_datetime(prices[dcol])
        prices = prices.drop(columns=[dcol]).apply(pd.to_numeric, errors="coerce")
        rets = _daily_returns(prices)

        tickers = sorted(set(preds["ticker"]) & set(prices.columns))
        cov = _fixed_cov(prices, tickers)
        base = _base_config(20, 0.001, 1.0)
        binary = find_binary()
        dec = preds[["decision_date", "realization_date"]].drop_duplicates().sort_values("decision_date")
        starts = pd.to_datetime(dec["decision_date"])

        def realized(model: str) -> np.ndarray:
            out = []
            for _, r in dec.iterrows():
                g = (preds[(preds["model"] == model) & (preds["decision_date"] == r["decision_date"])]
                     .set_index("ticker")["predicted_asset_return"])
                mu = g.reindex(tickers).fillna(0.0).to_numpy(float)
                res = run_optimizer(base, tickers, "pso", mu, cov, P, SEED, binary)
                w = {t: float(wi) for t, wi in zip(tickers, res["weights"]) if wi > 0}
                out.append(_window_return(w, rets, pd.Timestamp(r["decision_date"]),
                                          pd.Timestamp(r["realization_date"])))
            return np.array(out)

        transformers = [m for m in ["TFT", "MASTER", "PatchTST"] if m in preds["model"].unique()]
        tvals = {m: realized(m) for m in transformers}
        best = max(tvals, key=lambda m: np.mean(tvals[m]))
        # The classical line is the baseline matched to `best`'s conditioning
        # window, not a generic one -- see `_matched_hist`.
        hist_name = _matched_hist(preds, best)
        hist = realized(hist_name)

        px = yf.download("^GSPC", start="2020-12-15", end="2021-12-31",
                         progress=False, auto_adjust=True)["Close"].dropna()
        idx = []
        for _, r in dec.iterrows():
            p0 = float(px.reindex([pd.Timestamp(r["decision_date"])], method="ffill").iloc[0])
            p1 = float(px.reindex([pd.Timestamp(r["realization_date"])], method="ffill").iloc[0])
            idx.append((p1 / p0 - 1) * 100.0)
        idx = np.array(idx)

        # Cumulative (running) average across start dates: the average realized
        # 120-day return over all bets up to date t. This aggregates the same
        # per-window outcomes; it is not a chained wealth curve.
        def runavg(a):
            return np.cumsum(a) / np.arange(1, len(a) + 1)

        series = [
            (runavg(tvals[best]), "tab:blue", "o", f"{best} (transformerski)"),
            (runavg(hist),        "tab:gray", "s", "Klasični (zgodovinski)"),
            (runavg(idx),         "black",    "^", "S&P 500 (pasivno)"),
        ]
        for ext, dpi in [(".pdf", None), (".png", 130)]:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            for y, c, mk, lab in series:
                ax.plot(starts, y, marker=mk, color=c, lw=1.5, ms=6, label=lab)
            ax.axhline(0, color="0.6", lw=0.8, ls="--")
            ax.set_xlabel("Začetni datum naložbe (2021)")
            ax.set_ylabel("Povprečni realiziran donos v 120 dneh (%)")
            ax.legend(fontsize=8.5)
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            target = (FIG + ext) if ext == ".pdf" else ("/tmp/wang_by_startdate" + ext)
            fig.savefig(target, dpi=dpi) if dpi else fig.savefig(target)
            plt.close(fig)
        print(f"P={P} best={best}  mean%: {best}={tvals[best].mean():.1f} "
              f"Hist={hist.mean():.1f} idx={idx.mean():.1f}")

    main()


def task_equity_curves(argv=None):
    #!/usr/bin/env python3
    """
    equity_curves.py
    ================
    Cumulative-wealth ("equity") curves for the literature benchmarks, in one
    consistent style. Each curve chains the realized per-window returns already
    stored in a run's ``portfolio_results.csv`` (no re-run), and overlays a passive
    buy-and-hold market index. Three cardinality-consistent lines:

      * best transformer portfolio (our method, K assets),
      * the classical historical portfolio (same K, same solver),
      * the market index (passive real-world alternative).

    The naive 1/N portfolio is deliberately NOT shown: it holds all N assets and
    thus violates the cardinality constraint that defines the problem (and a random
    cardinality-K equal-weight has, in expectation, the same return as 1/N anyway).

    Only benchmarks with a NON-overlapping rebalancing schedule can be chained into
    a valid equity curve; Wang et al. uses overlapping 120-day windows and is
    therefore excluded here (it keeps its realized-Sharpe table instead).

    Usage:  python benchmark/equity_curves.py        # regenerates all figures
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FIGDIR = os.path.join(ROOT, "thesis-paper", "fig")
    OPT = "PSO"

    # run dir, index ticker, index label, output basename, title
    BENCHMARKS = [
        ("leow_allweather_direct",               "^GSPC", "S&P 500 (indeks)",    "equity_curve_leow",            "Leow: All-Weather (COVID-19)"),
    ]


    def _wealth(returns: np.ndarray) -> np.ndarray:
        w = [1.0]
        for r in returns:
            w.append(w[-1] * (1.0 + r))
        return np.array(w)


    def build(run: str, index_ticker: str, index_label: str, out: str, title: str):
        import yfinance as yf
        path = os.path.join(ROOT, "test_results", "literature", run, "portfolio_results.csv")
        port = pd.read_csv(path)
        decs = (port[["decision_date", "realization_date"]]
                .drop_duplicates().sort_values("decision_date").reset_index(drop=True))
        dates = [pd.Timestamp(port["decision_date"].min())] + list(pd.to_datetime(decs["realization_date"]))

        def realized(model: str) -> np.ndarray:
            # Average any duplicate rows (seeds / risk points) per decision date,
            # then order by the unique decision schedule.
            s = (port[(port["model"] == model) & (port["optimizer"] == OPT)]
                 .groupby("decision_date")["actual_simple_portfolio_return"].mean())
            return s.reindex(decs["decision_date"]).to_numpy(float)

        # our line = transformer with highest final wealth; classical = Historical
        transformers = [m for m in ["TFT", "MASTER", "PatchTST"] if m in port["model"].unique()]
        tcurves = {m: _wealth(realized(m)) for m in transformers}
        best = max(tcurves, key=lambda m: tcurves[m][-1])
        hist_name = _matched_hist(port, best)
        hist = _wealth(realized(hist_name))

        d0, d1 = dates[0] - pd.Timedelta(days=10), dates[-1] + pd.Timedelta(days=10)
        idx = yf.download(index_ticker, start=d0.strftime("%Y-%m-%d"),
                          end=d1.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)["Close"].dropna()
        idx = idx.reindex(pd.to_datetime(dates), method="ffill").to_numpy(float).ravel()
        idx = idx / idx[0]

        lines = [
            (tcurves[best], "tab:blue", "-",  f"{best} (transformerski)"),
            (hist,          "tab:gray", "-",  "Klasični (zgodovinski)"),
            (idx,           "black",    ":",  index_label + ", pasivno"),
        ]
        for ext, dpi in [(".pdf", None), (".png", 130)]:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            for y, c, ls, lab in lines:
                ax.plot(dates, y, ls, color=c, lw=1.7, label=lab)
            ax.set_xlabel("Datum")
            ax.set_ylabel("Vrednost portfelja (začetek = 1)")
            ax.legend(fontsize=8.5, loc="best")
            ax.grid(alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            target = os.path.join(FIGDIR if ext == ".pdf" else "/tmp", out + ext)
            fig.savefig(target, dpi=dpi) if dpi else fig.savefig(target)
            plt.close(fig)
        print(f"{run}: best={best} final wealth "
              f"{best}={tcurves[best][-1]:.3f} Hist={hist[-1]:.3f} idx={idx[-1]:.3f}  -> {out}.pdf")


    def main():
        os.makedirs(FIGDIR, exist_ok=True)
        for args in BENCHMARKS:
            build(*args)

    main()


# ---------------------------------------------------------------------------
# Shared palette used by both new plot tasks below
# ---------------------------------------------------------------------------
_MODEL_STYLE = {
    "Historical": ("tab:gray",   "-",  "Zgodovinski"),
    "TFT":        ("tab:orange", "-",  "TFT"),
    "PatchTST":   ("tab:blue",   "-",  "PatchTST"),
    "MASTER":     ("tab:green",  "-",  "MASTER"),
}
_TRANSFORMERS = ["TFT", "PatchTST", "MASTER"]
# Baseline rows are `Historical@<L>` (one per conditioning window), so their
# names are not known until a run's config is read. Dashes distinguish the
# windows within the shared gray.
_HIST_DASHES = ["-", "--", ":", "-."]


def _model_style(model: str):
    """(color, linestyle, legend label) for any mu-source row name."""
    if model in _MODEL_STYLE:
        return _MODEL_STYLE[model]
    if model.startswith("Historical@"):
        return ("tab:gray", "-", "Zgodovinski@" + model.split("@", 1)[1])
    return ("tab:red", "-", model)


def _model_order(models) -> list:
    """Historical baselines first (short window -> long), then the transformers."""
    present = set(models)
    hist = sorted((m for m in present if m.startswith("Historical@")),
                  key=lambda m: int(m.split("@", 1)[1]))
    rest = [m for m in _TRANSFORMERS if m in present]
    extra = sorted(present - set(hist) - set(rest))
    return hist + rest + extra


def _matched_hist(df, model: str) -> str:
    """The `Historical@<L>` row `model` must be compared against."""
    if "matched_baseline" in df.columns:
        m = df.loc[df["model"] == model, "matched_baseline"].dropna()
        if len(m):
            return str(m.iloc[0])
    hist = [x for x in df["model"].unique() if str(x).startswith("Historical@")]
    if len(hist) == 1:
        return hist[0]
    raise KeyError(
        f"cannot resolve the matched historical baseline for {model!r}: "
        f"no `matched_baseline` column and {len(hist)} candidates {hist}"
    )


def _pick_p(port: "pd.DataFrame") -> float:
    """Choose a single P value: 0.5 if present, otherwise the only available value."""
    available = sorted(port["P"].unique())
    return 0.5 if 0.5 in available else available[0]


def task_equity_curves_all(argv=None):
    """Equity curves showing ALL models for every chainable regime benchmark.

    Unlike task_equity_curves (which plots only the best transformer), this
    variant shows every available model in a consistent color scheme so that
    cross-model performance is immediately visible. Wang is excluded because
    its 120-day windows overlap and cannot be chained into a valid wealth curve.

    Writes <out>_all.{pdf,png} into thesis-paper/fig/.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yfinance as yf

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FIGDIR = os.path.join(ROOT, "thesis-paper", "fig")
    OPT = "PSO"

    # (run_dir, periods_per_year, index_ticker, index_label, out_basename, title, risk_matched)
    # risk_matched=True reads each model's own risk-matched P from risk_matched.csv
    # (the same read used for the reported Sharpe/return numbers) instead of pinning
    # every model to one nominal P; otherwise a fixed P=0.5 can visually contradict
    # the risk-matched headline numbers (e.g. Leow: TFT trails its baseline at
    # P=0.5 but leads by +0.83 Sharpe at its own risk-matched P).
    # 8th field: cost_bps (None = gross; a number = net of that many bps round-trip
    # cost per unit L1 turnover, subtracted from each period's return before chaining).
    BENCHMARKS = [
        ("leow_allweather_direct", 52,  "^GSPC", "S&P 500", "equity_all_leow_allweather", "All-Weather ETF (COVID-19, Leow) -- vsak model pri lastnem tveganju-ujemajočem $P$", True, None),
        ("longterm_investor",      12,  "SPY",   "SPY",     "equity_all_longterm",         "Dolgoročni vlagatelj (2011–2024) -- bruto", False, None),
        ("longterm_investor",      12,  "SPY",   "SPY",     "equity_all_longterm_net",      "Dolgoročni vlagatelj (2011–2024) -- neto (10 b.t. stroškov)", False, 10.0),
    ]

    def _wealth(rets):
        return np.r_[1.0, np.cumprod(1.0 + np.asarray(rets, float))]

    def _risk_matched_p(run) -> dict:
        """model -> its risk-matched P (PSO), read from risk_matched.csv."""
        f = os.path.join(ROOT, "test_results", "literature", run, "risk_matched.csv")
        if not os.path.exists(f):
            return {}
        rm = pd.read_csv(f)
        rm = rm[rm["optimizer"].str.upper() == OPT]
        return dict(zip(rm["model"], rm["P"]))

    def build(run, ppy, idx_tick, idx_lab, out, title, risk_matched, cost_bps=None):
        path = os.path.join(ROOT, "test_results", "literature", run, "portfolio_results.csv")
        if not os.path.exists(path):
            print(f"  skip {run} (no results)")
            return
        port_full = pd.read_csv(path)
        models = _model_order(port_full["model"].unique())
        p_by_model = _risk_matched_p(run) if risk_matched else {}
        p_default = _pick_p(port_full)
        cost = (cost_bps or 0.0) / 1e4

        # dates are shared across models/P (same decision schedule), so any P slice works
        decs = (port_full[["decision_date", "realization_date"]]
                .drop_duplicates().sort_values("decision_date").reset_index(drop=True))
        dates = ([pd.Timestamp(decs["decision_date"].iloc[0])]
                 + list(pd.to_datetime(decs["realization_date"])))

        curves = {}
        p_used = {}
        for m in models:
            p_m = p_by_model.get(m, p_default)
            p_used[m] = p_m
            g = port_full[(port_full["model"] == m) & (port_full["optimizer"] == OPT)
                          & (port_full["P"] == p_m)]
            s = g.groupby("decision_date")["actual_simple_portfolio_return"].mean()
            rets = s.reindex(decs["decision_date"]).to_numpy(float)
            if cost > 0:
                turn = (g.groupby("decision_date")["turnover_l1"].mean()
                        .reindex(decs["decision_date"]).to_numpy(float)
                        if "turnover_l1" in g else np.zeros_like(rets))
                rets = rets - np.nan_to_num(turn, nan=0.0) * cost
            curves[m] = _wealth(rets)

        d0 = pd.Timestamp(dates[0]) - pd.Timedelta(days=10)
        d1 = pd.Timestamp(dates[-1]) + pd.Timedelta(days=10)
        raw = yf.download(idx_tick, start=d0.strftime("%Y-%m-%d"),
                          end=d1.strftime("%Y-%m-%d"), progress=False,
                          auto_adjust=True)["Close"].dropna()
        idx_vals = raw.reindex(pd.to_datetime(dates), method="ffill").to_numpy(float).ravel()
        idx_vals = idx_vals / idx_vals[0]

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for m in models:
            c, ls, lab = _model_style(m)
            if risk_matched:
                lab = f"{lab} ($P{{=}}{p_used[m]:g}$)"
            ax.plot(dates, curves[m], ls, color=c, lw=1.7, label=lab)
        ax.plot(dates, idx_vals, ":", color="black", lw=1.2, label=f"{idx_lab} (pasivno)")
        ax.axhline(1.0, color="0.7", lw=0.7, ls="--")
        ax.set_xlabel("Datum")
        ax.set_ylabel("Vrednost portfelja (začetek = 1)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext, dpi in [(".pdf", None), (".png", 130)]:
            dest = os.path.join(FIGDIR if ext == ".pdf" else "/tmp", out + ext)
            fig.savefig(dest, dpi=dpi) if dpi else fig.savefig(dest)
        plt.close(fig)
        summary = "  ".join(f"{m}(P={p_used[m]:g})={curves[m][-1]:.3f}" for m in models)
        print(f"{run}: {summary}  idx={idx_vals[-1]:.3f}  -> {out}.pdf")

    for args in BENCHMARKS:
        build(*args)


def task_metrics_bars(argv=None):
    """2×2 core-metrics bar chart for every chainable regime benchmark.

    Panels: Annualised Return / Annualised Volatility / Sharpe Ratio / Max Drawdown.
    Metrics are computed from the raw per-window returns in portfolio_results.csv
    (PSO, all available P values averaged) so they are consistent with the equity
    curves produced by task_equity_curves_all.

    Writes <out>_metrics.{pdf,png} into thesis-paper/fig/.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FIGDIR = os.path.join(ROOT, "thesis-paper", "fig")
    OPT = "PSO"
    RF = 0.0          # risk-free rate used for Sharpe (0 = excess return Sharpe)

    BENCHMARKS = [
        ("leow_allweather_direct", 52,  "metrics_leow_allweather", "All-Weather ETF (COVID-19, Leow)"),
        ("longterm_investor",      12,  "metrics_longterm",        "Dolgoročni vlagatelj (2011–2024)"),
    ]

    def _metrics(rets, ppy):
        n = len(rets)
        wealth = np.r_[1.0, np.cumprod(1.0 + rets)]
        ann_ret = wealth[-1] ** (ppy / n) - 1.0
        ann_vol = float(np.std(rets, ddof=1)) * np.sqrt(ppy)
        sharpe  = (ann_ret - RF) / ann_vol if ann_vol > 0 else 0.0
        peak    = np.maximum.accumulate(wealth)
        maxdd   = float(np.max(1.0 - wealth / peak))
        return ann_ret, ann_vol, sharpe, maxdd

    def build(run, ppy, out, title):
        path = os.path.join(ROOT, "test_results", "literature", run, "portfolio_results.csv")
        if not os.path.exists(path):
            print(f"  skip {run} (no results)")
            return
        port = pd.read_csv(path)
        p_val = _pick_p(port)
        port = port[port["P"] == p_val]
        decs = (port[["decision_date", "realization_date"]]
                .drop_duplicates().sort_values("decision_date").reset_index(drop=True))

        models = _model_order(port["model"].unique())
        met = {}
        for m in models:
            s = (port[(port["model"] == m) & (port["optimizer"] == OPT)]
                 .groupby("decision_date")["actual_simple_portfolio_return"].mean())
            rets = s.reindex(decs["decision_date"]).to_numpy(float)
            met[m] = _metrics(rets, ppy)

        labels  = [_model_style(m)[2] for m in models]
        colors  = [_model_style(m)[0] for m in models]
        metrics = [
            ("Letni donos (%)",        [met[m][0] * 100 for m in models], "%"),
            ("Letna volatilnost (%)",  [met[m][1] * 100 for m in models], "%"),
            ("Sharpov količnik",       [met[m][2]       for m in models], ""),
            ("Maks. padec (%)",        [met[m][3] * 100 for m in models], "%"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.5))
        fig.suptitle(title, fontsize=10, y=1.01)
        for ax, (metric_name, vals, unit) in zip(axes.ravel(), metrics):
            bars = ax.barh(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_title(metric_name, fontsize=8.5)
            ax.axvline(0, color="0.5", lw=0.7)
            ax.grid(axis="x", alpha=0.25)
            ax.tick_params(labelsize=8)
            for bar, v in zip(bars, vals):
                x = bar.get_width()
                ha = "left" if x >= 0 else "right"
                offset = 0.5 if x >= 0 else -0.5
                ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}{unit}", va="center", ha=ha, fontsize=7.5)
        fig.tight_layout()
        os.makedirs(FIGDIR, exist_ok=True)
        for ext, dpi in [(".pdf", None), (".png", 130)]:
            dest = os.path.join(FIGDIR if ext == ".pdf" else "/tmp", out + ext)
            fig.savefig(dest, dpi=dpi, bbox_inches="tight") if dpi else fig.savefig(dest, bbox_inches="tight")
        plt.close(fig)
        print(f"{run} (P={p_val}): metrics -> {out}.pdf")

    for args in BENCHMARKS:
        build(*args)


def task_wang_bars(argv=None):
    """Two-panel Wang figure.

    Top panel  — per-window realized 120-day return for each model (grouped bars
                 by decision date), analogous to Wang et al. Fig. 5.
    Bottom panel — aggregate realized Sharpe (Wang-style, from wang_sharpe_table.csv)
                   for our models vs. Wang published reference values.

    Wang uses overlapping 120-day windows, so a chained equity curve is not valid;
    this per-window + aggregate view is the methodologically appropriate alternative.

    Writes wang_bars.{pdf,png} into thesis-paper/fig/.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RUN  = os.path.join(ROOT, "test_results", "literature", "wang_sp500")
    FIGDIR = os.path.join(ROOT, "thesis-paper", "fig")
    OPT  = "PSO"

    port  = pd.read_csv(os.path.join(RUN, "portfolio_results.csv"))
    p_val = _pick_p(port)
    port  = port[(port["P"] == p_val) & (port["optimizer"] == OPT)]

    decs   = sorted(port["decision_date"].unique())
    models = _model_order(port["model"].unique())

    # ── top panel: per-window 120-day returns ──────────────────────────────
    rets = {}
    for m in models:
        s = port[port["model"] == m].set_index("decision_date")["actual_simple_portfolio_return"]
        rets[m] = [float(s.get(d, np.nan)) * 100 for d in decs]

    n_mod   = len(models)
    width   = 0.7 / n_mod
    x       = np.arange(len(decs))
    offsets = np.linspace(-(n_mod - 1) / 2, (n_mod - 1) / 2, n_mod) * width
    xlabels = [pd.Timestamp(d).strftime("%b %Y") for d in decs]

    # ── bottom panel: aggregate Sharpe ─────────────────────────────────────
    sharpe_path = os.path.join(RUN, "wang_sharpe_table.csv")
    our_sharpe = {}
    if os.path.exists(sharpe_path):
        st = pd.read_csv(sharpe_path)
        st = st[st["optimizer"] == OPT]
        for m in models:
            row = st[st["predictor"] == m]
            if not row.empty:
                our_sharpe[m] = float(row["sharpe"].iloc[0])

    wang_ref = {
        "Wang: zgod. (Gurobi)": 0.673,
        "Wang: LSTM+Gurobi":    1.082,
        "Wang: CardNN-GS":      1.968,
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.0),
                                   gridspec_kw={"height_ratios": [1.6, 1]})

    # top
    for i, m in enumerate(models):
        c, _, lab = _model_style(m)
        ax1.bar(x + offsets[i], rets[m], width, label=lab, color=c,
                edgecolor="white", linewidth=0.4)
    ax1.axhline(0, color="0.4", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=8)
    ax1.set_ylabel("Realiziran donos v 120 dneh (%)")
    ax1.set_title("Realizirani donos po oknu naložbe (Wang protokol, 2021)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.25)

    # bottom — horizontal bars: Wang reference + our models
    all_labels, all_vals, all_colors = [], [], []
    for label, val in wang_ref.items():
        all_labels.append(label)
        all_vals.append(val)
        all_colors.append("0.75")
    for m in models:
        if m in our_sharpe:
            _, _, lab = _model_style(m)
            all_labels.append(lab)
            all_vals.append(our_sharpe[m])
            all_colors.append(_model_style(m)[0])

    ypos = np.arange(len(all_labels))
    ax2.barh(ypos, all_vals, color=all_colors, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(all_labels, fontsize=8)
    ax2.set_xlabel("Realiziran Sharpov količnik (Wang-stil, letno)")
    ax2.set_title("Skupni Sharpov količnik — naše metode vs. Wang in sod.", fontsize=9)
    ax2.axvline(0, color="0.4", lw=0.8)
    ax2.grid(axis="x", alpha=0.25)
    for bar, v in zip(ax2.patches, all_vals):
        ax2.text(v + 0.03, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=7.5)

    fig.tight_layout(h_pad=1.5)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext, dpi in [(".pdf", None), (".png", 130)]:
        dest = os.path.join(FIGDIR if ext == ".pdf" else "/tmp", "wang_bars" + ext)
        fig.savefig(dest, dpi=dpi) if dpi else fig.savefig(dest)
    plt.close(fig)
    print(f"wang_sp500 (P={p_val}): -> wang_bars.pdf")


def task_protocol_timeline(argv=None):
    """Walk-forward protocol timeline: one standalone Gantt-style panel per
    (benchmark, model) combination, meant to be arranged as a subfigure grid
    per benchmark in the thesis. Each panel has one row per actual model
    refit: a grey bar for gradient-training data, a hatched darker tail for
    the validation/early-stopping holdout, and -- immediately after the
    cutoff -- a short model-coloured bar spanning to the next decision date,
    showing the prediction target window (the period whose returns the model
    is forecasting). The coloured segment sits strictly after the grey
    training bar, so training and prediction targets never overlap.
    Below all rows: the out-of-sample test window with retrain-date ticks.

    Reads fit_start/fit_end/test_start/test_end/validation_observations/
    model_refit from each benchmark's config, each model's matched
    conditioning window L, and the true decision dates -- all from the
    `matched_baseline`/`decision_date` columns of its portfolio_results.csv
    (the same source already used for the reported table numbers), so the
    diagram cannot drift from the numbers it illustrates.

    Writes protocol_timeline_<run>_<model>.{pdf,png} into thesis-paper/fig/,
    one file per (benchmark, model) combination.
    """
    import json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FIGDIR = os.path.join(ROOT, "thesis-paper", "fig")
    CFGDIR = os.path.join(ROOT, "benchmark", "configs")

    # (config_file, run_dir, title, out_basename)
    BENCHMARKS = [
        ("config_leow_allweather_direct.json", "leow_allweather_direct", "Leow (All-Weather ETF)",  "protocol_timeline_leow"),
        ("config_wang_sp500.json",              "wang_sp500",             "Wang (S&P 500)",          "protocol_timeline_wang"),
        ("config_aprea_djia.json",              "aprea_djia",             "Aprea (DJIA)",            "protocol_timeline_apreadjia"),
        ("config_aprea_nasdaq100.json",         "aprea_nasdaq100",        "Aprea (NASDAQ 100)",      "protocol_timeline_apreanasdaq"),
        ("config_longterm_investor.json",       "longterm_investor",      "Dolgoročni vlagatelj",    "protocol_timeline_longterm"),
    ]

    def _model_L(run) -> dict:
        """model -> its matched conditioning window L (trading days)."""
        f = os.path.join(ROOT, "test_results", "literature", run, "portfolio_results.csv")
        if not os.path.exists(f):
            return {}
        df = pd.read_csv(f, usecols=["model", "matched_baseline"]).drop_duplicates()
        out = {}
        for _, r in df.iterrows():
            m, mb = r["model"], r["matched_baseline"]
            if isinstance(m, str) and not m.startswith("Historical@") and str(mb).startswith("Historical@"):
                out[m] = int(str(mb).split("@", 1)[1])
        return out

    def _decision_dates(run) -> list:
        f = os.path.join(ROOT, "test_results", "literature", run, "portfolio_results.csv")
        if not os.path.exists(f):
            return []
        df = pd.read_csv(f, usecols=["decision_date"])
        return sorted(pd.to_datetime(df["decision_date"].unique()))

    def _retrain_dates(model_refit: str, decisions: list) -> list:
        """Approximate dates on which the model is actually retrained during
        the test window, mirroring `_refit_cutoff` in walkforward.py: annual
        retrains on the first decision of each new calendar year (the very
        first decision is covered by the pretest fit already, so it is not a
        retrain event); per_decision retrains at every decision; once never
        retrains again."""
        if not decisions or model_refit == "once":
            return []
        if model_refit == "per_decision":
            return list(decisions)
        if model_refit == "annual":
            out, seen_years = [], {decisions[0].year}
            for d in decisions[1:]:
                if d.year not in seen_years:
                    seen_years.add(d.year)
                    out.append(d)
            return out
        return []

    written = []
    for cfg_file, run, title, out in BENCHMARKS:
        with open(os.path.join(CFGDIR, cfg_file), encoding="utf-8") as fh:
            cfg = json.load(fh)
        prot = cfg["protocol"]
        fit_start = pd.Timestamp(prot["fit_start"])
        fit_end = pd.Timestamp(prot["fit_end"])
        test_start = pd.Timestamp(prot["test_start"])
        test_end = pd.Timestamp(prot["test_end"])
        val_obs = prot.get("validation_observations")
        model_refit = str(prot.get("model_refit", "once")).lower()
        # protocol.train_window_days is absent from every config here, so
        # walkforward.py's default ("auto") applies: a ROLLING window whose
        # length equals the initial pretest fit span, sliding forward to end
        # at each refit's cutoff date (`_train_start` in walkforward.py).
        fit_span_bdays = len(pd.bdate_range(fit_start, fit_end))

        # One row per actual refit: the initial pretest fit (cutoff=fit_end)
        # plus every later retrain event, in chronological order.
        decisions = _decision_dates(run)
        retrains = _retrain_dates(model_refit, decisions)
        # Drop retrain "events" whose cutoff coincides with the previous row's
        # (e.g. a per_decision schedule's first decision landing on fit_end):
        # the real engine skips these (`cutoff_pos > trained_through_pos`
        # guard in `_refit_cutoff`) since they add no new training data.
        cutoffs = [fit_end]
        for d in retrains:
            if d != cutoffs[-1]:
                cutoffs.append(d)
        n_rows = len(cutoffs)
        n_retr_actual = n_rows - 1

        row_h = 0.09
        gap = 0.045
        rows_top = 0.20 + n_rows * (row_h + gap)

        fig_h = 0.45 + n_rows * (row_h + gap)
        L_by_model = _model_L(run)
        os.makedirs(FIGDIR, exist_ok=True)

        for m in _model_order(list(L_by_model.keys())):
            L = L_by_model[m]
            c, _, lab = _model_style(m)
            fig, ax = plt.subplots(figsize=(7.0, min(fig_h, 6.5)))

            span = test_end - fit_start
            min_pred_width = span * 0.018   # minimum visible prediction block

            for i, cutoff in enumerate(cutoffs):
                y = rows_top - i * (row_h + gap)
                train_start = cutoff - pd.tseries.offsets.BDay(fit_span_bdays - 1)
                hist_start  = max(cutoff - pd.tseries.offsets.BDay(L), train_start)
                val_start   = (cutoff - pd.tseries.offsets.BDay(val_obs)) if val_obs else cutoff

                # The training window has up to four non-overlapping segments:
                #   A [train_start → min(hist_start,val_start)]: pure gradient training → grey
                #   B [hist_start → val_start] if L>val_obs: in L-window, still gradient training → light color
                #   B [val_start → hist_start] if L<val_obs: validation, not yet in L-window → hatched grey
                #   C [max(hist_start,val_start) → cutoff]: in L-window AND validation → hatched light color
                # This ensures validation and gradient-training never share a visual region.

                left_edge = min(hist_start, val_start)

                # Segment A: pure gradient training (grey)
                if left_edge > train_start:
                    ax.broken_barh([(train_start, left_edge - train_start)], (y, row_h),
                                    facecolors="0.80", edgecolors="white", linewidth=0.3)

                # Segment B: middle region between hist_start and val_start
                if hist_start < val_start:
                    # L extends before validation: gradient training + in L-window → light color
                    ax.broken_barh([(hist_start, val_start - hist_start)], (y, row_h),
                                    facecolors=c, alpha=0.38, edgecolors="white", linewidth=0.3)
                elif val_obs and val_start < hist_start:
                    # Validation extends before L-window: validation only → hatched grey
                    ax.broken_barh([(val_start, hist_start - val_start)], (y, row_h),
                                    facecolors="0.65", edgecolors="0.40", linewidth=0.5, hatch="///")

                # Segment C: in L-window AND validation → hatched light color
                right_start = max(hist_start, val_start)
                if right_start < cutoff:
                    ax.broken_barh([(right_start, cutoff - right_start)], (y, row_h),
                                    facecolors=c, alpha=0.38, edgecolors="0.40",
                                    linewidth=0.5, hatch="///")

                # 3. Prediction target: tall solid block after the cutoff —
                #    the period whose returns go into the optimizer.
                next_dec = next((d for d in decisions if d > cutoff), None)
                if next_dec is not None:
                    pred_width = max(min(next_dec, test_end) - cutoff, min_pred_width)
                    ax.broken_barh([(cutoff, pred_width)],
                                    (y, row_h),
                                    facecolors=c, alpha=0.90,
                                    edgecolors="white", linewidth=0.3)


            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="0.80", edgecolor="none", label="učenje"),
                Patch(facecolor=c, alpha=0.38, edgecolor="none", label=f"vhod (L={L}d)"),
                Patch(facecolor="0.80", edgecolor="0.40", hatch="///", label="validacija (zgodnja ustavitev)"),
                Patch(facecolor=c, alpha=0.90, edgecolor="none", label="držanje"),
            ]
            ax.legend(handles=legend_elements, fontsize=5.5, loc="upper left",
                      framealpha=0.85, handlelength=1.2, handleheight=0.9,
                      borderpad=0.5, labelspacing=0.25, frameon=True,
                      edgecolor="0.80")

            # Bottom timeline: model-coloured rail on the bottom spine.
            # Custom ticks extend only upward so they don't clash with date labels.
            # Matplotlib's own tick marks are hidden; only year labels remain.
            tl_y = 0.0
            tick_h = row_h * 0.55
            ax.spines["bottom"].set_position(("data", tl_y))
            ax.plot([test_start, test_end], [tl_y, tl_y],
                    color=c, lw=2.0, solid_capstyle="butt", zorder=3, alpha=0.8)
            for d in [test_start] + list(cutoffs[1:]) + [test_end]:
                ax.plot([d, d], [tl_y, tl_y + tick_h],
                        color="0.2", lw=1.0, zorder=4, clip_on=False)

            ax.set_title(f"{title} -- {lab}", fontsize=8.5, loc="left", pad=3)
            y_bottom = -0.02
            ax.set_ylim(y_bottom, rows_top + row_h + 0.08)
            ax.set_yticks([])
            ax.grid(axis="x", alpha=0.2)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis="x", labelsize=6.5, pad=2, length=0)
            fig.tight_layout(pad=0.3)

            model_slug = m.lower().replace(" ", "")
            fname = f"{out}_{model_slug}"
            for ext, dpi in [(".pdf", None), (".png", 130)]:
                dest = os.path.join(FIGDIR if ext == ".pdf" else "/tmp", fname + ext)
                fig.savefig(dest, dpi=dpi, bbox_inches="tight") if dpi else fig.savefig(dest, bbox_inches="tight")
            plt.close(fig)
            written.append(fname)
            print(f"{run}/{m}: model_refit={model_refit}, {n_retr_actual} retrain(s), "
                  f"{n_rows} rows -> {fname}.pdf")

    print("written:", ", ".join(written))



def task_pipeline_schema(argv=None):
    """Vector schematic of the end-to-end pipeline used in the thesis body.

    Replaces the former raster ``fig/pipeline.png``. Six numbered stages, drawn
    as a single walk-forward cycle: the mu branch (neural model) and the Sigma
    branch (sample covariance) run in parallel off the same history, meet in the
    CCMV solver, and the realised return of the held portfolio feeds the
    out-of-sample metrics before the window advances.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    FIGDIR = os.path.join(ROOT, "thesis-paper", "fig")
    os.makedirs(FIGDIR, exist_ok=True)

    C_DATA, C_MU, C_COV, C_OPT, C_HOLD, C_EVAL = (
        "0.45", "tab:blue", "tab:purple", "tab:red", "tab:orange", "tab:green")

    fig, ax = plt.subplots(figsize=(7.0, 2.55))

    # (x0, y0, x1, y1, roman, label, colour)
    boxes = [
        (1, 20, 18, 34, "I", "Zgodovinski\ntržni podatki\ndo $t$", C_DATA),
        (26, 37, 48, 51, "II", "Napovedni model\n(PatchTST / TFT / MASTER)", C_MU),
        (26, 3, 48, 17, "III", "Ocena kovariančne\nmatrike", C_COV),
        (56, 20, 72, 34, "IV", "Optimizator KONP\n(RD / SO)", C_OPT),
        (80, 20, 99, 34, "V", "Držanje portfelja\n$[t{+}1,\\,t{+}H]$", C_HOLD),
        (80, 3, 99, 17, "VI", "Zunajvzorčne mere\nuspešnosti", C_EVAL),
    ]
    for x0, y0, x1, y1, roman, label, col in boxes:
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=1.0, edgecolor=col, facecolor=col, alpha=0.13, zorder=2))
        ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=1.0, edgecolor=col, facecolor="none", zorder=3))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 1.0, label, ha="center",
                va="center", fontsize=7.6, linespacing=1.45, zorder=4)
        ax.text(x0 + 1.4, y1 - 1.4, roman, ha="left", va="top", fontsize=7.0,
                color=col, fontweight="bold", zorder=4)

    def arrow(xy, xytext, label=None, lx=0, ly=0, rad=0.0, ha="center"):
        ax.add_patch(FancyArrowPatch(
            xytext, xy, arrowstyle="-|>", mutation_scale=9, lw=0.9,
            color="0.35", shrinkA=0, shrinkB=0, zorder=1,
            connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text(lx, ly, label, ha=ha, va="center", fontsize=7.0,
                    color="0.2", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    arrow((26, 44), (18, 30), rad=0.12)
    arrow((26, 10), (18, 24), rad=-0.12)
    arrow((56, 30), (48, 44), r"$\hat{\vec{\mu}}_t$", 52.0, 40.0, rad=0.12)
    arrow((56, 24), (48, 10), r"$\hat{\vec{\Sigma}}_t$", 52.0, 14.0, rad=-0.12)
    arrow((80, 27), (72, 27), r"$\vec{w}_t$", 76.0, 30.5)
    arrow((89.5, 17), (89.5, 20), "realizirani donos", 87.5, 18.5, ha="right")

    # Walk-forward loop: the window advances and the cycle repeats.
    Y_LOOP = 58.0
    ax.plot([89.5, 89.5, 9.5], [34, Y_LOOP, Y_LOOP], color="0.45", lw=0.9,
            linestyle=(0, (4, 2)), zorder=1, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        (9.5, Y_LOOP), (9.5, 34), arrowstyle="-|>", mutation_scale=9, lw=0.9,
        color="0.45", linestyle=(0, (4, 2)), shrinkA=0, shrinkB=0, zorder=1))
    ax.text(49.5, Y_LOOP, r"pomik okna: $t \leftarrow t+H$", ha="center",
            va="center", fontsize=7.0, color="0.35", style="italic",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"), zorder=5)

    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 62)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIGDIR, "pipeline.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("written: pipeline.pdf")


def task_protocol_schema(argv=None):
    """One generic, data-free schematic of the walk-forward protocol.

    Replaces the per-(benchmark, model) Gantt panels in the thesis body: those
    differ only in dates and cadence (a table conveys that better), while this
    figure answers the questions the real panels could NOT show -- which window
    the matched historical mean is taken over, and where Sigma comes from.

    Everything is drawn relative to a single decision date at x=0. All estimation
    windows terminate at the cutoff; the future is shaded so the absence of
    look-ahead is visually obvious. Lane labels double as the legend, so no
    separate legend box is drawn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGDIR = os.path.join(ROOT, "thesis-paper", "fig")
    os.makedirs(FIGDIR, exist_ok=True)

    C_MODEL, C_MU, C_COV, C_HOLD = "tab:blue", "tab:green", "tab:purple", "tab:orange"

    # Generic geometry (arbitrary units; the axis carries symbolic labels).
    T_TRAIN, T_VAL, L, L_MAX, H = 10.0, 2.5, 4.0, 6.0, 3.0

    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    row_h = 0.44

    # Vertical extent: five lanes at y=0..4 plus a little headroom for the two
    # callouts above the top lane. Kept as names so the guide band below can be
    # derived from them instead of hard-coded axes fractions.
    Y_LO, Y_HI = -0.45, 4.85

    # Guide band highlighting that the model input and the matched historical
    # mean are computed over the *same* L trading days. It must cover exactly
    # the "vhod v model" (y=3) and "zgodovinsko povprečje" (y=2) lanes.
    def _yfrac(y):
        return (y - Y_LO) / (Y_HI - Y_LO)

    ax.axvspan(-L, 0, ymin=_yfrac(2 - row_h / 2), ymax=_yfrac(3 + row_h / 2),
               color=C_MODEL, alpha=0.09, zorder=1)

    lanes = [
        (4, "Učenje modela",
         [(-T_TRAIN, T_TRAIN - T_VAL, "0.80", 1.0, None),
          (-T_VAL, T_VAL, "0.80", 1.0, "///")]),
        (3, "Vhod v model ($L$ dni)",
         [(-L, L, C_MODEL, 0.85, None)]),
        (2, r"Zgodovinsko povprečje $\hat{\mu}$ ($L$ dni)",
         [(-L, L, C_MU, 0.85, None)]),
        (1, r"Vzorčna kovarianca $\hat{\Sigma}$ ($L_{\max}$ dni)",
         [(-L_MAX, L_MAX, C_COV, 0.85, None)]),
        (0, "Držanje portfelja ($H$ dni)",
         [(0, H, C_HOLD, 0.95, None)]),
    ]

    for y, label, segs in lanes:
        for x0, w, fc, al, hz in segs:
            ax.broken_barh([(x0, w)], (y - row_h / 2, row_h),
                           facecolors=fc, alpha=al, hatch=hz,
                           edgecolors="0.35" if hz else "white",
                           linewidth=0.6 if hz else 0.4, zorder=3)
        ax.text(-T_TRAIN - 0.5, y, label, ha="right", va="center", fontsize=8)

    # Annotate the hatched tail in place of a legend entry.
    ax.annotate("validacija\n(zgodnja ustavitev)",
                xy=(-T_VAL / 2, 4 + row_h / 2), xytext=(-T_VAL / 2 - 4.8, 4.52),
                fontsize=6.6, ha="center", va="bottom", color="0.3",
                linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))

    # "same window" marker between the model-input and historical-mean lanes.
    ax.annotate("", xy=(-L, 2.5), xytext=(0, 2.5),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))
    ax.text(-L / 2, 2.56, "isto okno", fontsize=6.6, ha="center", va="bottom",
            color="0.25", style="italic",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))

    # Future shading + the cutoff line.
    ax.axvspan(0, H + 1.6, color="0.93", zorder=0)
    ax.axvline(0, color="0.15", lw=1.4, ls="--", zorder=5)
    ax.text(0.2, 4.55, "odločitveni trenutek", fontsize=7.5,
            ha="left", va="center", style="italic")
    ax.text(H + 1.45, 2.2, "prihodnost:\nob odločitvi\nni na voljo", fontsize=6.6,
            ha="right", va="center", color="0.4", linespacing=1.35)

    ax.set_xlim(-T_TRAIN - 8.0, H + 1.7)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks([])
    ax.set_xticks([-T_TRAIN, -L_MAX, -L, 0, H])
    ax.set_xticklabels(["$-T$", r"$-L_{\max}$", "$-L$", "0", "$+H$"], fontsize=8)
    ax.set_xlabel("trgovalni dnevi glede na odločitveni trenutek", fontsize=8)
    ax.tick_params(axis="x", labelsize=8, length=2)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=0.3)
    for ext, dpi in [(".pdf", None), (".png", 150)]:
        dest = os.path.join(FIGDIR if ext == ".pdf" else "/tmp",
                            "protocol_schema" + ext)
        fig.savefig(dest, dpi=dpi, bbox_inches="tight") if dpi \
            else fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print("written: protocol_schema.pdf")
