#!/usr/bin/env python3
"""
tables_utils.py
===============
Reads benchmark result CSVs and writes LaTeX fragments to thesis-paper/generated/.

Run from the repo root:
    python benchmark/utils/tables_utils.py

Outputs
-------
thesis-paper/generated/tab_leow_metrics_rows.tex  -- our rows for tab:leow
thesis-paper/generated/leow_macros.tex            -- \newcommand macros, Leow prose
thesis-paper/generated/tab_wang_rows.tex          -- our rows for tab:wang
thesis-paper/generated/wang_macros.tex            -- \newcommand macros, Wang prose

Both tables carry one `Historical@<L>` row per distinct model conditioning
window (see estimators/pipeline.conditioning_window). There is no single "the
historical baseline" any more, so every transformer row is emitted next to the
baseline it is actually matched against.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test_results" / "literature"
OUT = ROOT / "thesis-paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# PSO is the representative metaheuristic used throughout the thesis; both
# solvers are stored per run, and pooling them would concatenate two return
# series into one whose std is neither solver's.
OPTIMIZER = "PSO"
OTHER_OPTIMIZER = "SA"   # reported next to PSO (RD | SO column pairs) in every benchmark table
MODEL_ORDER = ["MASTER", "PatchTST", "TFT"]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _num(v: float, dec: int = 3, signed: bool = False) -> str:
    """Slovene decimal-comma LaTeX number."""
    if v is None or not np.isfinite(v):
        return "--"
    s = f"{v:+.{dec}f}" if signed else f"{v:.{dec}f}"
    return s.replace(".", "{,}")


def _pct(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    return f"${v * 100:+.1f}".replace(".", "{,}") + r"\,\%$"


def _plain_pct(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    return f"${v * 100:.1f}".replace(".", "{,}") + r"\,\%$"


def _dur(sec: float) -> str:
    """Training/solver wall clock, in the largest sensible unit."""
    if sec is None or not np.isfinite(sec) or sec <= 0:
        return "--"
    # numbers in math mode (same font as every other number in the tables), unit in text
    if sec < 1:
        return "$" + _num(sec * 1000, 0) + r"$\,ms"
    if sec < 60:
        return "$" + _num(sec, 1) + r"$\,s"
    if sec < 3600:
        return "$" + _num(sec / 60, 1) + r"$\,min"
    return "$" + _num(sec / 3600, 2) + r"$\,h"


def _bold(cell: str) -> str:
    """Bold a table cell; math cells keep their (serif) digits via \\mathbf."""
    if cell.startswith("$") and cell.endswith("$"):
        return "$\\mathbf{" + cell[1:-1] + "}$"
    return "\\textbf{" + cell + "}"


OPT_LABEL = {"PSO": "RD", "SA": "SO"}


def best_optimizer(sharpe: dict, tiebreak: dict) -> str:
    """Solver a model is REPORTED with in the prose: higher Sharpe (2 decimals), then the tie-break
    (higher return / terminal wealth), then RD. The matched historical mean is read with the SAME solver,
    so the head-to-head still isolates the forecast, not the optimizer."""
    key = lambda o: (round(sharpe[o], 2), round(tiebreak[o], 4))
    return "SA" if key("SA") > key("PSO") else "PSO"


def _winners(vals: dict, higher: bool, dec: int, scale: float = 1.0) -> set:
    """Keys of the best value(s) of one metric over all of OUR rows (models + historical means, RD and SO).

    Compared after rounding to the shown precision, so ties are all winners.
    """
    rounded = {k: round(v * scale, dec) for k, v in vals.items() if v is not None and np.isfinite(v)}
    best = max(rounded.values()) if higher else min(rounded.values())
    return {k for k, v in rounded.items() if v == best}


def _cell(v: float, fmt, win: bool) -> str:
    """Table cell. Body cells are NOT bold (only the header rows with symbols and RD | SO are emphasised);
    `win` is kept so that highlighting winners can be switched back on by returning _bold(fmt(v))."""
    return fmt(v)


def _is_baseline(model: str) -> bool:
    return str(model).startswith("Historical@")


def _display(model: str, n_baselines: int) -> str:
    """Slovenian row label; the window suffix only appears when a run has >1."""
    if _is_baseline(model):
        if n_baselines <= 1:
            return "Zgodovinsko povprečje"
        return "Zgodovinsko povprečje\\,(" + str(model).split("@", 1)[1] + "\\,dni)"
    return str(model)


def _order(models) -> list[str]:
    """Baselines first (short window -> long), then transformers."""
    base = sorted((m for m in models if _is_baseline(m)),
                  key=lambda m: int(str(m).split("@", 1)[1]))
    return base + [m for m in MODEL_ORDER if m in set(models)]


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def _portfolio(path: Path, P: float, optimizer: str = OPTIMIZER) -> pd.DataFrame:
    """Per-model realized metrics at one risk point and one solver."""
    df = pd.read_csv(path / "portfolio_summary.csv")
    if "P" in df.columns and P is not None:
        df = df[df["P"] == P]
    if optimizer is not None and "optimizer" in df.columns:
        df = df[df["optimizer"].str.upper() == optimizer.upper()]
    return df.set_index("model")


def _risk_matched(path: Path, optimizer: str = OPTIMIZER) -> pd.DataFrame:
    """Per-model metrics at each model's RISK-MATCHED point (equal realized vol).

    Reads `risk_matched.csv` (written by task_risk_matched): one row per
    (model, optimizer) at the P whose realized volatility matches the target
    (the longest-window historical baseline's vol at the reference P). Used only
    for Leow, where the short single-crash window makes a fixed nominal P an
    unfair operating point and the experiment calls for an equal-risk read.
    """
    df = pd.read_csv(path / "risk_matched.csv")
    if optimizer is not None and "optimizer" in df.columns:
        df = df[df["optimizer"].str.upper() == optimizer.upper()]
    return df.set_index("model")


def _train_seconds(path: Path) -> dict:
    """model -> total training wall clock for the whole run (all seeds/refits).

    Historical rows never appear here: a sample mean has no fitted parameters,
    which is exactly why its lookback is directly comparable to a model's
    conditioning window. They render as `--`.
    """
    f = path / "timing_train.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f)
    return t.groupby("model")["train_sec_total"].sum().to_dict()


def _train_detail(path: Path) -> dict:
    """model -> (n_trainings, seconds per training)."""
    f = path / "timing_train.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f)
    out = {}
    for _, r in t.iterrows():
        n = int(r["n_trainings"])
        out[str(r["model"])] = (n, float(r["train_sec_total"]) / max(n, 1))
    return out


def _solver_seconds(path: Path, P, optimizer: str = OPTIMIZER) -> dict:
    """model -> mean optimizer wall clock per decision (seconds).

    `P` is either one risk point or a model -> P mapping (Leow reads every model
    at its own risk-matched P). The cost is set by N, K and the solver budget,
    not by the mu source, so it is near-identical across rows of one experiment.
    """
    df = pd.read_csv(path / "portfolio_summary.csv")
    df = df[df["optimizer"].str.upper() == optimizer.upper()]
    out = {}
    for _, r in df.iterrows():
        want = P.get(r["model"]) if isinstance(P, dict) else P
        if want is not None and abs(float(r["P"]) - float(want)) < 1e-9:
            out[str(r["model"])] = float(r["mean_solver_sec"])
    return out


def _final_wealth(path: Path, model: str, P: float, optimizer: str = OPTIMIZER) -> float:
    """Terminal portfolio value W_T of one model: its realized per-decision returns
    chained over the whole test window (enačba cumulative), as in the equity-curve
    figures. Only valid when the decision periods do not overlap."""
    df = pd.read_csv(path / "portfolio_results.csv")
    g = df[(df["model"] == model) & (df["optimizer"].str.upper() == optimizer.upper())
           & (np.isclose(df["P"].astype(float), float(P)))]
    r = g.groupby("decision_date")["actual_simple_portfolio_return"].mean().sort_index()
    return float(np.prod(1.0 + r.to_numpy(float)))


def _ntrain(detail: dict, model: str) -> str:
    """Number of fits behind a model's row; baselines have none."""
    return "$" + str(detail[model][0]) + "$" if model in detail else "--"


def _per_train(detail: dict, model: str) -> str:
    """Wall clock of ONE fit (mean over the run's refits and seeds)."""
    return _dur(detail[model][1]) if model in detail else "--"


def _matched(path: Path) -> dict:
    """model -> its `Historical@<L>` row, read from the run's own output.

    Prefers the per-decision file, but `matched_baseline` is constant per model
    and is repeated in the summary, so the summary alone is enough.
    """
    f = path / "portfolio_results.csv"
    if not f.exists():
        f = path / "portfolio_summary.csv"
    df = pd.read_csv(f)
    if "matched_baseline" not in df.columns:
        return {}
    return (df[~df["model"].astype(str).str.startswith("Historical@")]
            .drop_duplicates("model")
            .set_index("model")["matched_baseline"].to_dict())


def _mac(name: str, body: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{body}}}"


def _write_rows(target: Path, rows: list[str]) -> None:
    r"""Write tabular rows for `\input` inside a tabular.

    The fragment deliberately does NOT end with `\\`: `\\` scans ahead for an
    optional `[`, that lookahead crosses the end of the inputted file, and the
    following `\hline` then lands outside the alignment (`Misplaced \noalign`).
    The caller supplies the closing `\\` in main.tex instead. The trailing `%`
    swallows the file's final newline so no stray space enters the cell.
    """
    body = "\n".join(rows)
    assert body.endswith(r" \\"), "rows must be emitted with a trailing \\\\"
    target.write_text(body[: -len(r" \\")] + "%\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Leow All-Weather
# ---------------------------------------------------------------------------

def generate_leow(P: float = 0.5) -> None:
    path = RESULTS / "leow_allweather_direct"
    if not path.exists():
        print(f"  skip Leow: {path} missing")
        return

    # Leow is reported RISK-MATCHED (each model at the P whose realized vol equals
    # the matched-baseline's) rather than at a fixed nominal P: the single-crash
    # window makes a fixed P an unfair operating point. Other benchmarks stay
    # fixed-P (Wang cannot be risk-matched -- too few independent observations).
    rm = _risk_matched(path)
    n_decisions = int(_portfolio(path, P)["n_decisions"].iloc[0])
    train = _train_seconds(path)
    detail = _train_detail(path)
    solve = _solver_seconds(path, rm["P"].to_dict())
    rm_sa = _risk_matched(path, OTHER_OPTIMIZER)
    solve_sa = _solver_seconds(path, rm_sa["P"].to_dict(), OTHER_OPTIMIZER)
    matched = _matched(path)
    n_base = sum(1 for m in rm.index if _is_baseline(m))
    pf = rm  # all per-model metrics below are read at the risk-matched point

    # Every measure is reported for BOTH solvers (RD | SO), each at its own
    # risk-matched P (the equal-risk point differs slightly between solvers).
    # only the WINNER of each metric (over all our rows, RD and SO) is bold
    shsh = lambda v: "$" + _num(v, 2) + "$"
    tbl = {"PSO": pf, "SA": rm_sa}
    models_ = _order(pf.index)
    val = {
        "ret": {(m, o): float(tbl[o].loc[m, "annualized_arithmetic_return"]) for m in models_ for o in tbl},
        "vol": {(m, o): float(tbl[o].loc[m, "annualized_vol"]) for m in models_ for o in tbl},
        "sh": {(m, o): float(tbl[o].loc[m, "sharpe_annualized"]) for m in models_ for o in tbl},
        "wt": {(m, o): _final_wealth(path, m, float(tbl[o].loc[m, "P"]), o) for m in models_ for o in tbl},
    }
    win = {"ret": _winners(val["ret"], True, 1, 100), "vol": _winners(val["vol"], False, 1, 100),
           "sh": _winners(val["sh"], True, 2), "wt": _winners(val["wt"], True, 2)}
    fm = {"ret": _plain_pct, "vol": _plain_pct, "sh": shsh, "wt": shsh}
    rows = []
    for m in models_:
        cells = []
        for k in ("ret", "vol", "sh", "wt"):
            cells += [_cell(val[k][(m, o)], fm[k], (m, o) in win[k]) for o in ("PSO", "SA")]
        rows.append(_display(m, n_base) + " & " + " & ".join(cells) + " \\\\")
    _write_rows(OUT / "tab_leow_metrics_rows.tex", rows)
    print(f"  wrote {OUT / 'tab_leow_metrics_rows.tex'} ({len(rows)} rows)")

    macros = []
    for m in pf.index:
        if _is_baseline(m):
            continue
        base = matched.get(m)
        if base not in pf.index:
            continue
        # solver this model is reported with: higher Sharpe at its own risk-matched P, then terminal wealth
        o = best_optimizer({k: float(tbl[k].loc[m, "sharpe_annualized"]) for k in tbl},
                           {k: _final_wealth(path, m, float(tbl[k].loc[m, "P"]), k) for k in tbl})
        a = tbl[o]
        macros.append(_mac(f"LeowAW{m}Opt", OPT_LABEL[o]))
        macros.append(_mac(f"LeowAW{m}", _num(float(a.loc[m, "sharpe_annualized"]), 2)))
        macros.append(_mac(f"LeowAW{m}Ret", _plain_pct(float(a.loc[m, "annualized_arithmetic_return"]))))
        macros.append(_mac(f"LeowAWHistorical{m}", _num(float(a.loc[base, "sharpe_annualized"]), 2)))
        macros.append(_mac(f"LeowAWHistorical{m}Ret", _plain_pct(float(a.loc[base, "annualized_arithmetic_return"]))))
        macros.append(_mac(f"LeowAWWindow{m}", str(base).split("@", 1)[1]))
        macros.append(_mac(f"LeowAWMargin{m}",
                           _num(float(a.loc[m, "sharpe_annualized"]) - float(a.loc[base, "sharpe_annualized"]), 2, True)))

    total = sum(train.values())
    macros.append(_mac("LeowAWTrainTotal", _dur(total)))
    macros.append(_mac("LeowAWDecisions", str(n_decisions)))
    macros.append(_mac("LeowAWSolvePSO", _dur(float(np.mean(list(solve.values()))))))
    macros.append(_mac("LeowAWSolveSA", _dur(float(np.mean(list(solve_sa.values()))))))
    for m, (n, per) in _train_detail(path).items():
        macros.append(_mac(f"LeowAWTrainPer{m}", _dur(per)))
        macros.append(_mac(f"LeowAWTrainN{m}", str(n)))

    (OUT / "leow_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"  wrote {OUT / 'leow_macros.tex'} ({len(macros)} macros)")


# ---------------------------------------------------------------------------
# Wang S&P 500
# ---------------------------------------------------------------------------

def generate_wang(optimizer: str = OPTIMIZER) -> None:
    path = RESULTS / "wang_sp500"
    f = path / "wang_sharpe_table.csv"
    if not f.exists():
        print(f"  skip Wang: {f} missing")
        return

    # The Wang-style table is the one tab:wang quotes: realized annualized
    # Sharpe rebuilt from daily portfolio returns, comparable to their Table 3.
    t_all = pd.read_csv(f)
    t = t_all[t_all["optimizer"].str.upper() == optimizer.upper()].set_index("predictor")
    t_so = t_all[t_all["optimizer"].str.upper() == OTHER_OPTIMIZER.upper()].set_index("predictor")
    train = _train_seconds(path)
    detail = _train_detail(path)
    solve = _solver_seconds(path, 0.5, optimizer)
    solve_sa = _solver_seconds(path, 0.5, OTHER_OPTIMIZER)
    matched = _matched(path)
    n_base = sum(1 for m in t.index if _is_baseline(m))

    shsh = lambda v: "$" + _num(v, 2) + "$"
    tb = {"PSO": t, "SA": t_so}
    models_ = _order(t.index)
    val = {"ret": {(m, o): float(tb[o].loc[m, "ann_return"]) for m in models_ for o in tb},
           "risk": {(m, o): float(tb[o].loc[m, "ann_risk"]) for m in models_ for o in tb},
           "sh": {(m, o): float(tb[o].loc[m, "sharpe"]) for m in models_ for o in tb}}
    win = {"ret": _winners(val["ret"], True, 1, 100), "risk": _winners(val["risk"], False, 1, 100),
           "sh": _winners(val["sh"], True, 2)}
    fm = {"ret": _plain_pct, "risk": _plain_pct, "sh": shsh}
    rows = []
    for m in models_:
        cells = []
        for k in ("ret", "risk", "sh"):
            cells += [_cell(val[k][(m, o)], fm[k], (m, o) in win[k]) for o in ("PSO", "SA")]
        rows.append(_display(m, n_base) + " & " + " & ".join(cells) + " \\\\")
    _write_rows(OUT / "tab_wang_rows.tex", rows)
    print(f"  wrote {OUT / 'tab_wang_rows.tex'} ({len(rows)} rows)")

    # A LaTeX control sequence may contain letters only, so baseline macros are
    # named after the MODEL they are matched to (\WangHistoricalMASTER), never
    # after the window length (\WangHistorical67 would not compile).
    macros = []
    for m, base in matched.items():
        if m not in t.index or base not in t.index:
            continue
        o = best_optimizer({k: float(tb[k].loc[m, "sharpe"]) for k in tb}, {k: float(tb[k].loc[m, "ann_return"]) for k in tb})
        a, bl = tb[o], tb[o]
        macros.append(_mac(f"Wang{m}Opt", OPT_LABEL[o]))
        macros.append(_mac(f"Wang{m}", _num(float(a.loc[m, "sharpe"]), 2)))
        macros.append(_mac(f"Wang{m}Ret", _plain_pct(float(a.loc[m, "ann_return"]))))
        macros.append(_mac(f"Wang{m}Risk", _plain_pct(float(a.loc[m, "ann_risk"]))))
        macros.append(_mac(f"WangHistorical{m}", _num(float(bl.loc[base, "sharpe"]), 2)))
        macros.append(_mac(f"WangHistorical{m}Ret", _plain_pct(float(bl.loc[base, "ann_return"]))))
        macros.append(_mac(f"WangWindow{m}", str(base).split("@", 1)[1]))
        macros.append(_mac(f"WangMargin{m}", _num(float(a.loc[m, "sharpe"]) - float(bl.loc[base, "sharpe"]), 2, True)))
    # sharpe_std >= sharpe for every row (83% holding-period overlap, 7
    # decisions ~ 2 independent observations) -- the caveat the prose must carry.
    macros.append(_mac("WangSharpeStdMax", _num(float(t["sharpe_std"].max()), 2)))
    macros.append(_mac("WangSharpeStdMin", _num(float(t["sharpe_std"].min()), 2)))
    macros.append(_mac("WangTrainTotal", _dur(sum(train.values()))))
    macros.append(_mac("WangSolvePSO", _dur(float(np.mean(list(solve.values()))))))
    macros.append(_mac("WangSolveSA", _dur(float(np.mean(list(solve_sa.values()))))))
    for m, (n, per) in _train_detail(path).items():
        macros.append(_mac(f"WangTrainPer{m}", _dur(per)))
        macros.append(_mac(f"WangTrainN{m}", str(n)))

    (OUT / "wang_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"  wrote {OUT / 'wang_macros.tex'} ({len(macros)} macros)")


# ---------------------------------------------------------------------------
# Aprea & Sbaiz (DJIA), long-term investor, turnover -- RD | SO column pairs
# ---------------------------------------------------------------------------

def _pair_rows(rows_by_model: dict, order, n_base, label_prefix: str = "") -> list[str]:
    return [label_prefix + _display(m, n_base) + " & " + rows_by_model[m] + " \\\\" for m in order]


def generate_aprea(P: float = 0.5) -> None:
    path = RESULTS / "aprea_djia"
    if not path.exists():
        print(f"  skip Aprea: {path} missing")
        return
    a, b = _portfolio(path, P, OPTIMIZER), _portfolio(path, P, OTHER_OPTIMIZER)
    n_base = sum(1 for m in a.index if _is_baseline(m))

    pct = _plain_pct
    two = lambda v: "$" + _num(v, 2) + "$"
    tb = {"PSO": a, "SA": b}
    models_ = _order(a.index)
    # common metrics first (same order as every other experiment), then the article's own CAGR
    spec = [("annualized_arithmetic_return", pct, True, 1, 100), ("annualized_vol", pct, False, 1, 100),
            ("sharpe_annualized", two, True, 2, 1), ("final_wealth", two, True, 2, 1),
            ("cagr", pct, True, 1, 100)]
    cells = {}
    wins = {col: _winners({(m, o): float(tb[o].loc[m, col]) for m in models_ for o in tb}, hi, dec, sc)
            for col, _, hi, dec, sc in spec}
    for m in models_:
        out = []
        for col, fmt, hi, dec, sc in spec:
            out += [_cell(float(tb[o].loc[m, col]), fmt, (m, o) in wins[col]) for o in ("PSO", "SA")]
        cells[m] = " & ".join(out)
    rows = _pair_rows(cells, _order(a.index), n_base)
    _write_rows(OUT / "tab_aprea_rows.tex", rows)
    print(f"  wrote {OUT / 'tab_aprea_rows.tex'} ({len(rows)} rows)")


def _longterm_metrics(df: pd.DataFrame, optimizer: str, model: str, cost: float, P: float = 0.5) -> dict:
    """Gross and net (cost per unit of L1 turnover) metrics of one strategy -- the quantities of tab:longterm."""
    g = df[(df["optimizer"].str.upper() == optimizer.upper()) & np.isclose(df["P"].astype(float), P)
           & (df["model"] == model)].sort_values("decision_date")
    gross = g["actual_simple_portfolio_return"].to_numpy(float)
    modeled = g["actual_modeled_portfolio_return"].to_numpy(float)
    turn = np.nan_to_num(g["turnover_l1"].to_numpy(float), nan=0.0)
    net = gross - turn * cost
    ppy = 12.0
    # Sharpe is built from the modelled (log) returns for gross and from the net simple
    # returns for net, exactly as in benchmark/tasks/task_investor.py
    return {
        "g_ret": gross.mean() * ppy, "g_sh": modeled.mean() / modeled.std(ddof=1) * np.sqrt(ppy),
        "g_w": float(np.prod(1 + gross)),
        "n_ret": net.mean() * ppy, "n_sh": net.mean() / net.std(ddof=1) * np.sqrt(ppy),
        "n_w": float(np.prod(1 + net)),
        "to": float(np.nanmean(g["turnover_l1"].to_numpy(float))),
    }


def generate_longterm(cost_bps: float = 10.0) -> None:
    path = RESULTS / "longterm_investor"
    f = path / "portfolio_results.csv"
    if not f.exists():
        print(f"  skip long-term: {f} missing")
        return
    df = pd.read_csv(f)
    models = _order(df["model"].unique())
    n_base = sum(1 for m in models if _is_baseline(m))
    cost = cost_bps / 1e4
    two = lambda v: "$" + _num(v, 2) + "$"
    met = {(m, o): _longterm_metrics(df, o, m, cost) for m in models for o in (OPTIMIZER, OTHER_OPTIMIZER)}
    spec = [("g_ret", _plain_pct, 1, 100), ("g_sh", two, 2, 1), ("g_w", two, 2, 1),
            ("n_ret", _plain_pct, 1, 100), ("n_sh", two, 2, 1), ("n_w", two, 2, 1)]
    wins = {k: _winners({key: v[k] for key, v in met.items()}, True, dec, sc) for k, _, dec, sc in spec}
    rows = []
    for m in models:
        cells = []
        for k, fmt, dec, sc in spec:
            cells += [_cell(met[(m, o)][k], fmt, (m, o) in wins[k]) for o in (OPTIMIZER, OTHER_OPTIMIZER)]
        cells += [two(met[(m, o)]["to"]) for o in (OPTIMIZER, OTHER_OPTIMIZER)]   # turnover: never bold
        rows.append(_display(m, n_base) + " & " + " & ".join(cells) + " \\\\")
    _write_rows(OUT / "tab_longterm_rows.tex", rows)
    print(f"  wrote {OUT / 'tab_longterm_rows.tex'} ({len(rows)} rows)")


def generate_turnover(P: float = 0.5) -> None:
    """Mean per-decision L1 turnover of each transformer in every experiment, RD | SO (tab:whytft)."""
    spec = [("Polletno vlaganje", "wang_sp500"), ("Krizno vlaganje", "leow_allweather_direct"),
            ("Mesečno vlaganje", "aprea_djia"), ("Dolgoročno vlaganje", "longterm_investor")]
    rows = []
    for label, d in spec:
        path = RESULTS / d
        if not path.exists():
            continue
        cells = []
        for m in ("TFT", "MASTER", "PatchTST"):
            vals = []
            for opt in (OPTIMIZER, OTHER_OPTIMIZER):
                tbl = _portfolio(path, P, opt)
                vals.append("$" + _num(float(tbl.loc[m, "mean_turnover_l1"]), 2) + "$" if m in tbl.index else "--")
            cells += vals
        rows.append(label + " & " + " & ".join(cells) + " \\\\")
    _write_rows(OUT / "tab_turnover_rows.tex", rows)
    print(f"  wrote {OUT / 'tab_turnover_rows.tex'} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Total compute time per experiment (training and optimisation)
# ---------------------------------------------------------------------------

TIME_EXPERIMENTS = [("Polletno vlaganje", "wang_sp500"), ("Krizno vlaganje", "leow_allweather_direct"),
                    ("Mesečno vlaganje", "aprea_djia"), ("Dolgoročno vlaganje", "longterm_investor")]


def generate_time_totals() -> None:
    """Total training time per model and total solver time per optimiser, per experiment.

    Training: sum of every fit of a model (`timing_train.csv`, column train_sec_total).
    Optimisation: sum of every solver run of the experiment -- all models and matched
    historical means, all P points and decisions (`portfolio_results.csv`, solver_sec).
    Last column: training of all models plus all solver runs, i.e. the whole experiment.
    """
    rows = []
    for label, folder in TIME_EXPERIMENTS:
        path = RESULTS / folder
        if not path.exists():
            print(f"  skip total time {label}: {path} missing")
            return
        train = pd.read_csv(path / "timing_train.csv").set_index("model")["train_sec_total"]
        res = pd.read_csv(path / "portfolio_results.csv")
        solve = res.groupby(res["optimizer"].str.upper())["solver_sec"].sum()
        cells = [_dur(float(train[m])) if m in train.index else "--" for m in ("MASTER", "PatchTST", "TFT")]
        cells += [_dur(float(solve[o])) for o in ("PSO", "SA")]
        cells.append(_dur(float(train.sum() + solve.sum())))
        rows.append(label + " & " + " & ".join(cells) + " \\\\")
    _write_rows(OUT / "tab_time_totals_rows.tex", rows)
    print(f"  wrote {OUT / 'tab_time_totals_rows.tex'} ({len(rows)} rows)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating LaTeX table fragments...")
    generate_leow()
    generate_wang()
    generate_aprea()
    generate_longterm()
    generate_turnover()
    generate_time_totals()
    print("Done.")
