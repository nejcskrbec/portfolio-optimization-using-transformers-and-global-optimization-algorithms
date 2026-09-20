#!/usr/bin/env python3
"""Single entry point for all thesis benchmarks (central orchestrator).

Engine lives in benchmark/walkforward.py; each benchmark is a task in
benchmark/tasks/; shared helpers and drawing live in benchmark/utils/; the C++
optimizer bridge in portfolio_optimizers/bridge.py; training with the estimators
(estimators/pipeline.py). This file only wires arguments to tasks.

    python -u benchmark/run.py predictive | orlib | leow | wang
                               | practical | literature | all
                               | wang-risk-sweep | equity-curves-all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.utils.benchmark_utils import CONFIGS, _dry, _run_named_literature
from benchmark.utils.plotting_utils import (
    task_equity_curves_all,
    task_pipeline_schema,
    task_protocol_schema,
)
from benchmark.tasks.task_predictive import task_predictive
from benchmark.tasks.task_orlib import task_orlib
from benchmark.tasks.task_investor import task_investor
from benchmark.tasks.task_leow import task_leow
from benchmark.tasks.task_wang import task_wang, task_wang_sharpe, task_wang_risk_sweep
from benchmark.tasks.task_aprea import task_aprea_djia, task_aprea_nasdaq
from benchmark.tasks.task_risk_matched import task_risk_matched


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Benchmarks for Historical + MASTER + TFT + PatchTST."
    )
    ap.add_argument("benchmark", choices=[
        "predictive", "orlib",
        "leow", "leow-allweather",
        "wang", "wang-risk-sweep", "risk-matched",
        "aprea-djia", "aprea-nasdaq",
        "practical", "literature", "all",
        "equity-curves-all", "protocol-schema", "pipeline-schema",
    ])
    ap.add_argument("--smoke", action="store_true",
                    help="Short literature run: first seed/few decisions/reduced risk grid.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--orlib-full", action="store_true",
                    help="OR-Library port1-port5 with 50 lambda points.")
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="Transaction cost for practical post-processing.")
    ap.add_argument("--dir", default=None,
                    help="Run output directory (risk-matched).")
    ap.add_argument("--target-P", type=float, default=0.5,
                    help="Reference risk point for the risk-matched comparison.")
    ap.add_argument("--lang", choices=["sl", "en"], default="sl",
                    help="Label language for pipeline-schema (sl -> "
                         "thesis-paper/fig/pipeline.pdf, en -> docs/pipeline.png).")
    args = ap.parse_args(argv)

    b = args.benchmark

    if b == "predictive":
        if not _dry("predictive IC/RankIC", args.dry_run):
            task_predictive([])
        return

    if b == "orlib":
        av = ["--full"] if args.orlib_full else []
        if not _dry("OR-Library" + (" --full" if args.orlib_full else ""), args.dry_run):
            task_orlib(av)
        return

    if b == "leow":
        task_leow(args.smoke, args.dry_run)
        return

    if b == "wang":
        task_wang(args.smoke, args.dry_run)
        return

    if b == "aprea-djia":
        task_aprea_djia(args.smoke, args.dry_run)
        return

    if b == "aprea-nasdaq":
        task_aprea_nasdaq(args.smoke, args.dry_run)
        return

    if b in CONFIGS:
        _run_named_literature(b, args.smoke, args.dry_run)
        if args.dry_run or args.smoke:
            return
        if b == "practical":
            task_investor(["--cost-bps", str(args.cost_bps)])
        return

    if b == "risk-matched":
        if not args.dir:
            raise SystemExit("risk-matched requires --dir <run output directory>")
        if not _dry("risk-matched comparison", args.dry_run):
            task_risk_matched(["--dir", args.dir, "--target-P", str(args.target_P)])
        return

    if b == "wang-risk-sweep":
        if not _dry("Wang risk sweep", args.dry_run):
            task_wang_risk_sweep([])
        return

    if b == "equity-curves-all":
        if not _dry("all-model equity curves", args.dry_run):
            task_equity_curves_all([])
        return

    if b == "pipeline-schema":
        if not _dry(f"pipeline schema ({args.lang})", args.dry_run):
            task_pipeline_schema(["--lang", args.lang])
        return

    if b == "protocol-schema":
        if not _dry("protocol schema", args.dry_run):
            task_protocol_schema([])
        return

    if b == "literature":
        for x in ("leow-allweather", "wang"):
            _run_named_literature(x, args.smoke, args.dry_run)
        if not args.dry_run and not args.smoke:
            task_wang_sharpe(["--csv", str(ROOT / "test_results/literature/wang_sp500/portfolio_results.csv")])
        return

    if b == "all":
        if not args.dry_run:
            task_predictive([])
            task_orlib(["--full"] if args.orlib_full else [])
        else:
            _dry("predictive IC/RankIC", True); _dry("OR-Library", True)
        for x in ("leow-allweather", "wang", "practical"):
            _run_named_literature(x, args.smoke, args.dry_run)
        if not args.dry_run and not args.smoke:
            task_wang_sharpe(["--csv", str(ROOT / "test_results/literature/wang_sp500/portfolio_results.csv")])
            task_investor(["--cost-bps", str(args.cost_bps)])
        return


if __name__ == "__main__":
    main()
