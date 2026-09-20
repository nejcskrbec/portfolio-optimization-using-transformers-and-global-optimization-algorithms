#!/usr/bin/env python3
"""Drawing tasks for the figures the thesis actually includes: all-model equity
curves, the pipeline schematic and the protocol schematic."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.utils.benchmark_utils import ROOT


# ---------------------------------------------------------------------------
# Shared palette for the equity-curve task
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


def _pick_p(port: "pd.DataFrame") -> float:
    """Choose a single P value: 0.5 if present, otherwise the only available value."""
    available = sorted(port["P"].unique())
    return 0.5 if 0.5 in available else available[0]


def task_equity_curves_all(argv=None):
    """Equity curves showing ALL models for every chainable regime benchmark.

    Every available model is drawn in a consistent color scheme so that
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


def task_pipeline_schema(argv=None):
    """Vector schematic of the end-to-end pipeline used in the thesis body.

    Replaces the former raster ``fig/pipeline.png``. Six numbered stages, drawn
    as a single walk-forward cycle: the mu branch (neural model) and the Sigma
    branch (sample covariance) run in parallel off the same history, meet in the
    CCMV solver, and the realised return of the held portfolio feeds the
    out-of-sample metrics before the window advances.

    Two language variants share this one drawing routine:

      --lang sl  -> thesis-paper/fig/pipeline.pdf  (vector, for the thesis body)
      --lang en  -> docs/pipeline.png              (raster, embedded in README)

    The README needs a raster because GitHub does not render PDF inline.
    """
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    ap = argparse.ArgumentParser(prog="pipeline-schema")
    ap.add_argument("--lang", choices=["sl", "en"], default="sl")
    ap.add_argument("--out", default=None, help="Override the output path.")
    args = ap.parse_args(argv or [])

    # Only the text differs between variants; geometry is shared.
    TEXT = {
        "sl": {
            "data":  "Zgodovinski\ntržni podatki\ndo $t$",
            "mu":    "Napovedni model\n(PatchTST / TFT / MASTER)",
            "cov":   "Ocena kovariančne\nmatrike",
            "opt":   "Optimizator KONP\n(RD / SO)",
            "hold":  "Držanje portfelja\n$[t{+}1,\\,t{+}H]$",
            "eval":  "Zunajvzorčne mere\nuspešnosti",
            "ret":   "realizirani donos",
            "loop":  r"pomik okna: $t \leftarrow t+H$",
        },
        "en": {
            "data":  "Historical\nmarket data\nup to $t$",
            "mu":    "Forecasting model\n(PatchTST / TFT / MASTER)",
            "cov":   "Covariance matrix\nestimate",
            "opt":   "CCMV optimizer\n(PSO / SA)",
            "hold":  "Hold portfolio\n$[t{+}1,\\,t{+}H]$",
            "eval":  "Out-of-sample\nperformance metrics",
            "ret":   "realised return",
            "loop":  r"window advances: $t \leftarrow t+H$",
        },
    }[args.lang]

    if args.out:
        outpath = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    elif args.lang == "en":
        outpath = os.path.join(ROOT, "docs", "pipeline.png")
    else:
        outpath = os.path.join(ROOT, "thesis-paper", "fig", "pipeline.pdf")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    C_DATA, C_MU, C_COV, C_OPT, C_HOLD, C_EVAL = (
        "0.45", "tab:blue", "tab:purple", "tab:red", "tab:orange", "tab:green")

    fig, ax = plt.subplots(figsize=(7.0, 2.55))

    # (x0, y0, x1, y1, roman, label, colour)
    boxes = [
        (1, 20, 18, 34, "I", TEXT["data"], C_DATA),
        (26, 37, 48, 51, "II", TEXT["mu"], C_MU),
        (26, 3, 48, 17, "III", TEXT["cov"], C_COV),
        (56, 20, 72, 34, "IV", TEXT["opt"], C_OPT),
        (80, 20, 99, 34, "V", TEXT["hold"], C_HOLD),
        (80, 3, 99, 17, "VI", TEXT["eval"], C_EVAL),
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
    arrow((89.5, 17), (89.5, 20), TEXT["ret"], 87.5, 18.5, ha="right")

    # Walk-forward loop: the window advances and the cycle repeats.
    Y_LOOP = 58.0
    ax.plot([89.5, 89.5, 9.5], [34, Y_LOOP, Y_LOOP], color="0.45", lw=0.9,
            linestyle=(0, (4, 2)), zorder=1, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        (9.5, Y_LOOP), (9.5, 34), arrowstyle="-|>", mutation_scale=9, lw=0.9,
        color="0.45", linestyle=(0, (4, 2)), shrinkA=0, shrinkB=0, zorder=1))
    ax.text(49.5, Y_LOOP, TEXT["loop"], ha="center",
            va="center", fontsize=7.0, color="0.35", style="italic",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"), zorder=5)

    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 62)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    save_kw = {"bbox_inches": "tight"}
    if outpath.endswith(".png"):
        save_kw["dpi"] = 200
    fig.savefig(outpath, **save_kw)
    plt.close(fig)
    print(f"written: {os.path.relpath(outpath, ROOT)}")


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
