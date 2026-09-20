#!/usr/bin/env python3
"""Long-term practical investor benefit post-processing."""
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


def task_investor(argv=None):
    #!/usr/bin/env python3
    """
    investor_benefit.py
    ===================
    Answers the practical question of the long-term-investor scenario: "If somebody
    had used our pipeline to invest over 2011-2024, would they have benefited?"

    From the stored monthly, non-overlapping walk-forward (practical.json),
    it chains realized per-window returns into a long-term wealth path for the best
    transformer pipeline and the classical historical pipeline, NET of transaction
    costs (turnover x cost), and overlays a passive SPY buy-and-hold. It reports the
    tangible investor outcomes (terminal wealth of a $10k stake, CAGR, annualized
    Sharpe, maximum drawdown, average annual turnover, gross vs net CAGR) and saves
    an equity-curve figure.

    Two comparisons are kept distinct:
      * transformer vs historical  -> does the transformer add value (survivorship-
        neutral, same universe);
      * vs SPY                      -> tangible benefit vs doing nothing (the index
        gap is a lower bound because the universe is survivorship-filtered).

    Usage:  python benchmark/investor_benefit.py --cost-bps 10
    """

    import argparse
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RUN = os.path.join(ROOT, "test_results", "literature", "longterm_investor")
    FIGDIR = os.path.join(ROOT, "thesis-paper", "fig")
    OPT = "PSO"
    PPY = 12  # monthly


    def _metrics(net_r, gross_r, turn, ppy=PPY):
        wealth = np.cumprod(1 + net_r)
        years = len(net_r) / ppy
        cagr = wealth[-1] ** (1 / years) - 1
        cagr_gross = np.cumprod(1 + gross_r)[-1] ** (1 / years) - 1
        sharpe = (net_r.mean() / net_r.std(ddof=1)) * np.sqrt(ppy) if net_r.std() > 0 else np.nan
        peak = np.maximum.accumulate(np.r_[1.0, wealth])
        mdd = float(np.max(1 - np.r_[1.0, wealth] / peak))
        ann_turn = float(np.nanmean(turn) * ppy) if turn is not None else np.nan
        return {"terminal_10k": 10000 * wealth[-1], "cagr": cagr, "cagr_gross": cagr_gross,
                "sharpe": sharpe, "max_drawdown": mdd, "ann_turnover": ann_turn}, wealth


    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--cost-bps", type=float, default=10.0,
                        help="round-trip cost in basis points per unit L1 turnover")
        ap.add_argument("--index", default="^GSPC")
        args = ap.parse_args(argv)
        cost = args.cost_bps / 1e4

        port = pd.read_csv(os.path.join(RUN, "portfolio_results.csv"))
        decs = (port[["decision_date", "realization_date"]].drop_duplicates()
                .sort_values("decision_date").reset_index(drop=True))
        dates = [pd.Timestamp(port["decision_date"].min())] + list(pd.to_datetime(decs["realization_date"]))

        def series(model):
            g = port[(port["model"] == model) & (port["optimizer"] == OPT)]
            gross = g.groupby("decision_date")["actual_simple_portfolio_return"].mean().reindex(decs["decision_date"]).to_numpy(float)
            turn = (g.groupby("decision_date")["turnover_l1"].mean().reindex(decs["decision_date"]).to_numpy(float)
                    if "turnover_l1" in g else np.zeros_like(gross))
            turn = np.nan_to_num(turn, nan=0.0)
            net = gross - turn * cost
            return gross, net, turn

        models = [m for m in ["TFT", "MASTER", "PatchTST"] if m in port["model"].unique()]
        # One historical baseline per distinct model conditioning window, so each
        # model's matched classical counterpart reads the same recent data it does.
        baselines = sorted((m for m in port["model"].unique()
                            if str(m).startswith("Historical@")),
                           key=lambda m: int(str(m).split("@", 1)[1]))
        curves, table = {}, []
        best, best_w = None, -1
        for m in models + baselines:
            gross, net, turn = series(m)
            met, wealth = _metrics(net, gross, turn)
            curves[m] = wealth
            table.append({"strategy": m, **met})
            if m in models and wealth[-1] > best_w:
                best, best_w = m, wealth[-1]

        # passive index buy-and-hold
        import yfinance as yf
        px = yf.download(args.index, start=dates[0].strftime("%Y-%m-%d"),
                         end=(dates[-1] + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)["Close"].dropna()
        idx = px.reindex(pd.to_datetime(dates), method="ffill").to_numpy(float).ravel()
        idx_w = idx / idx[0]
        idx_r = np.diff(idx_w) / idx_w[:-1]
        years = (len(dates) - 1) / PPY
        peak = np.maximum.accumulate(idx_w)
        table.append({"strategy": "SPY (indeks)", "terminal_10k": 10000 * idx_w[-1],
                      "cagr": idx_w[-1] ** (1 / years) - 1, "cagr_gross": idx_w[-1] ** (1 / years) - 1,
                      "sharpe": (idx_r.mean() / idx_r.std(ddof=1)) * np.sqrt(PPY),
                      "max_drawdown": float(np.max(1 - idx_w / peak)), "ann_turnover": 0.0})

        tbl = pd.DataFrame(table)
        tbl.to_csv(os.path.join(RUN, "investor_benefit_table.csv"), index=False, float_format="%.4f")

        # equity curve (net)
        S = 10000.0  # a $10k stake, shown in real dollars (a tangible wealth-growth plot)
        STYLE = {"TFT": ("tab:blue", "-"), "MASTER": ("tab:green", "-"),
                 "PatchTST": ("tab:purple", "-")}
        lines = []
        for m in [x for x in ["TFT", "MASTER", "PatchTST"] if x in curves]:
            c, ls = STYLE[m]
            lines.append((S * np.r_[1.0, curves[m]], c, ls, f"{m} (transformerski)"))
        for m, ls in zip(baselines, ["--", ":", "-."]):
            lab = ("Klasični (zgodovinski)" if len(baselines) == 1
                   else f"Klasični, {str(m).split('@', 1)[1]} dni")
            lines.append((S * np.r_[1.0, curves[m]], "tab:gray", ls, lab))
        lines.append((S * idx_w, "black", ":", "SPY (indeks, pasivno)"))
        from matplotlib.ticker import FuncFormatter
        for out, dpi in [(os.path.join(FIGDIR, "equity_curve_longterm.pdf"), None),
                         ("/tmp/equity_curve_longterm.png", 130)]:
            fig, ax = plt.subplots(figsize=(6.6, 4.1))
            for y, c, ls, lab in lines:
                ax.plot(dates, y, ls, color=c, lw=1.7, label=lab)
                ax.annotate(f"${y[-1]/1000:.0f}k", (dates[-1], y[-1]), fontsize=8,
                            color=c, xytext=(4, 0), textcoords="offset points", va="center")
            ax.set_xlabel("Datum")
            ax.set_ylabel("Vrednost naložbe 10.000 USD")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
            ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3)
            fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(out, dpi=dpi) if dpi else fig.savefig(out); plt.close(fig)

        # cost sweep: CAGR vs transaction cost (0–50 bps)
        sweep_bps = np.linspace(0, 50, 26)
        sweep_cagr = {m: [] for m in models + baselines}
        for bps in sweep_bps:
            c_ = bps / 1e4
            for m in models + baselines:
                g = port[(port["model"] == m) & (port["optimizer"] == OPT)]
                gr = g.groupby("decision_date")["actual_simple_portfolio_return"].mean().reindex(decs["decision_date"]).to_numpy(float)
                tu = (g.groupby("decision_date")["turnover_l1"].mean().reindex(decs["decision_date"]).to_numpy(float)
                      if "turnover_l1" in g else np.zeros_like(gr))
                tu = np.nan_to_num(tu, nan=0.0)
                nr = gr - tu * c_
                w = np.cumprod(1 + nr)
                sweep_cagr[m].append(w[-1] ** (12 / len(nr)) - 1)

        SWEEP_STYLE = {"TFT": ("tab:blue", "-"), "MASTER": ("tab:green", "-"),
                       "PatchTST": ("tab:purple", "-")}
        SWEEP_LABEL = {"TFT": "TFT", "MASTER": "MASTER", "PatchTST": "PatchTST"}
        for m, ls in zip(baselines, ["--", ":", "-."]):
            SWEEP_STYLE[m] = ("tab:gray", ls)
            SWEEP_LABEL[m] = ("Klasični" if len(baselines) == 1
                              else f"Klasični {str(m).split('@', 1)[1]}d")
        for out, dpi in [(os.path.join(FIGDIR, "longterm_cost_sweep.pdf"), None),
                         ("/tmp/longterm_cost_sweep.png", 130)]:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            for m in models + baselines:
                c_, ls_ = SWEEP_STYLE[m]
                ax.plot(sweep_bps, [v * 100 for v in sweep_cagr[m]],
                        ls_, color=c_, lw=1.7, label=SWEEP_LABEL[m])
            ax.axvline(args.cost_bps, color="0.5", lw=0.9, ls=":")
            ax.set_xlabel("Transakcijski stroški (b.t. na enoto prometa)")
            ax.set_ylabel("CAGR neto (%)")
            ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out, dpi=dpi) if dpi else fig.savefig(out); plt.close(fig)

        pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
        print(f"cost = {args.cost_bps:.0f} bps per unit turnover;  best transformer = {best}\n")
        print(tbl.to_string(index=False))
        print(f"\nsaved: {RUN}/investor_benefit_table.csv\n       {FIGDIR}/equity_curve_longterm.pdf\n       {FIGDIR}/longterm_cost_sweep.pdf")

    main()

