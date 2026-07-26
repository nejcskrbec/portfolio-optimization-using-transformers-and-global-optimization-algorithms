# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis: portfolio optimization using a transformer-based return estimator and C++ metaheuristic optimizers. The pipeline downloads historical price data, trains a cross-sectional multi-stock transformer to predict expected returns (μ) and a covariance matrix (Σ) from a shared representation, then runs global optimization algorithms to find optimal portfolio weights.

**Thesis framing**: the central claim is "the transformer *pipeline* (model μ + model Σ) *improves on* the classical historical pipeline (sample-mean μ + sample Σ) via combination," not "replaces" it. The `Transformer`-vs-`Zgodovinski` head-to-head is the headline comparison — an end-to-end pipeline comparison (each side uses its native Σ), not a μ-only ablation. An online forecast-combination ensemble (`Ansambel`, `estimators/mu_ensemble.py`) is a secondary robustness layer that blends history + transformer μ (over the model Σ) and provably (Hedge regret bound) cannot trail the best expert in any regime.

## Commands

### Build the C++ optimizer
```bash
cd portfolio_optimizers && make
```
This produces the `portfolio_optimizer` binary. The binary must exist before running any benchmark.

### Run the full benchmark
```bash
python benchmark/run_benchmark.py benchmark/configs/config_thesis_multistock.json       # 2019–2022 (COVID/momentum regime)
python benchmark/run_benchmark.py benchmark/configs/config_thesis_2013_multistock.json  # 2013–2017 (calm/mean-reversion regime)
```
Run from the repo root (the scripts add the repo root to `sys.path` themselves).
Entry scripts live in `benchmark/`; run-configs live in `benchmark/configs/`.
Outputs plots and CSV to `test_results/`. Requires the `magistrska` conda env
(`source /opt/homebrew/anaconda3/etc/profile.d/conda.sh; conda activate magistrska`) — base
lacks `yfinance`/`torch`/`arch`. A full run takes ~40 min; raw results are pickled
to `test_results/results_<period>.pkl` **before** plotting so a plot/export error
never discards the walk-forward computation.

### Rebuild and run in one step
```bash
cd portfolio_optimizers && make && cd .. && python benchmark/run_benchmark.py benchmark/configs/config_thesis_multistock.json
```

### Run the OR-Library literature benchmark
```bash
python benchmark/run_orlib_benchmark.py                    # port1–3, 25 λ (quick)
python benchmark/run_orlib_benchmark.py --full --plot      # all port1–5, 50 λ, frontier plots
python benchmark/run_orlib_benchmark.py --near-exact       # add near-exact constrained frontier
```
Validates the C++ metaheuristics against the standard Chang et al. (2000)
OR-Library instances (`benchmark/orlib_data/portN.txt`) by tracing each
algorithm's cardinality-constrained efficient frontier (sweep over λ) and
reporting the standard percentage error vs. the published *unconstrained*
frontier. Outputs `test_results/orlib_benchmark.csv` and per-instance frontier
plots. Requires the compiled `portfolio_optimizer` binary. This benchmark is
independent of `yfinance`/transformer training — it is a pure solver-quality test.
It reuses only the solver hyperparameters from
`portfolio_optimizers/config.json` (no data download; K/ε/δ are set to the
Chang standards internally).

## Architecture

### End-to-end flow (`run_benchmark.py`)
1. Load config (e.g. `config_thesis_multistock.json`)
2. Download train/test price data via `yfinance`
3. Fit baseline covariance estimator once on full training returns
4. Train the return model once on training data
5. Run walk-forward loop: for each time window, compute μ and Σ on the growing history, call the C++ optimizer for each algorithm × scenarios. Three scenarios, each an end-to-end pipeline pairing a μ source with its native Σ source: **`Zgodovinski`** (historical rolling sample-mean μ + sample Σ), **`Transformer`** (transformer μ + shared-rep covariance-head Σ), and **`Ansambel`** (Hedge forecast-combination of history + transformer μ + shared-rep covariance-head Σ). Sample Σ is used **only** for `Zgodovinski`; the transformer and ensemble pipelines use the model's covariance head (falling back to sample Σ only if the model has no cov head).
6. Pickle raw results, then plot and export

### Estimators (`estimators/`)
The `estimators/` package holds the **moment estimators** — the model-based sources of the mean-variance inputs μ (first moment) and Σ (second moment): `multistock_master.py` (the MASTER-lite transformer, emits both μ and Σ) and `mu_ensemble.py` (the Hedge forecast-combination μ-layer). The two remaining sources — historical rolling-mean μ and sample Σ — are one-liners computed inline in `benchmark_core.py`, so they don't get their own module.

### Return estimator (`estimators/multistock_master.py`)
A single cross-sectional model — the **MASTER-lite** transformer (Li et al. 2024) — predicts the forward log-return (rank) over `lookahead` days for **all** assets simultaneously. There is no factory/dispatch layer; `run_benchmark.py` and `benchmark_core.py` call the module's functions directly:

- `train_multistock(config, df_train, avail_tickers, lookahead)` → registry dict (3-seed ensemble). Called once on training data.
- `get_mu_multistock(registry, df_hist, avail_tickers, lookahead)` → `np.ndarray` of shape `(N,)`. Called each walk-forward window.
- `get_cov_multistock(registry, df_hist, avail_tickers, lookahead)` → `(N,N)` from the shared-representation covariance head.

The model emits both the μ-rank and a shared-representation covariance `Σ = ββᵀ + diag(ψ)` (SPD by construction, `n_factors` factors) from the same per-stock embedding. Trained multi-task (rank-IC/ListNet return loss + Gaussian NLL cov loss); early-stopping tracks the **return** loss so the cov head is a shared-rep regularizer that never changes μ selection. Loss knobs are config-driven: `return_estimator.{topw_alpha, cov_lambda, n_factors}`.

`multistock` is the only supported `return_model`; any other value raises `ValueError` (the timesfm backend, the return-model factory, and the univariate/multivariate per-ticker architectures were all removed to slim the codebase).

**`topw_alpha` (winner-tilt)**: `topw_loss` blends `(1-α)·rank_ic_loss + α·listnet_loss`; `α=0` short-circuits to pure rank-IC (neutral model). Both thesis configs ship `topw_alpha=0.0` — the winner-tilt was moved from a bake-in loss term to an **inference-time** signed-power dial inside the ensemble (see below). α is kept only as an ablation lever.

### μ-ensemble (`estimators/mu_ensemble.py`)
`HedgeMuEnsemble` — online forecast-combination (multiplicative-weights / Hedge, Freund-Schapire 1997) driving the `Ansambel` scenario. Experts: historical rolling-mean μ + the neutral transformer μ under a fixed signed-power γ-grid `sign(z)·|z|^γ`, γ∈{0.5,1,2} (γ>1 sharpens winners — the job `topw_alpha` used to do, now an inference dial). Per window it blends the standardized experts, rescales the result to the transformer μ's mean/std (so P and the Σ trade-off behave identically to the `Transformer` scenario — only cross-sectional *shape* changes), and after each window's realized returns updates the weights `w *= exp(-η·loss)` with `η=√(8 ln K / T)` from theory (not tuned on regimes). Causal (weights use only past windows) and adds **zero** training cost — one neutral model, pure numpy inference, one extra optimizer pass per window. The Hedge regret bound guarantees the blend cannot trail the best expert (including the historical baseline) in any regime.

### Covariance (Σ)
There is no `cov_estimators/` package — it was over-engineering for what is now a single method. Two Σ sources, routed **by scenario** so each pipeline uses its native Σ:
- **Sample Σ** (`benchmark_core._sample_cov(returns, lookahead)` — numpy sample covariance scaled by lookahead, PD-repaired, computed each window): used **only** by the `Zgodovinski` scenario — the fully classical baseline (historical μ + historical Σ).
- **Model Σ** — the return model's **shared-representation covariance head** (`get_cov_multistock`, `Σ = ββᵀ + diag(ψ)`): used by the `Transformer` and `Ansambel` scenarios when the multistock registry reports `_has_cov` (falling back to sample Σ otherwise). The transformer and ensemble pipelines are thus fully model-based (model μ + model Σ). (A *standalone* transformer cov estimator existed but was removed: with the shared-rep head active its output was always overwritten, so it only wasted training time.)

### C++ portfolio optimizers (`portfolio_optimizers/`)
Single compilation unit: `main.cpp` `#include`s all algorithm files (`pso.cpp`, `sa.cpp`, `ga.cpp`) and the shared header `portfolio_common.h`. Three metaheuristics, each a single paper-faithful implementation: PSO (Cura 2009), `run_sa` (Crama & Schyns 2003 — purely random moves, no μ/σ in the search step), `run_ga` (Chang et al. 2000 — steady-state, roulette-wheel selection, uniform crossover on the membership vector). (A fourth, PBILDE, was removed to keep the thesis lean — it never uniquely won any regime.) Faithful moves keep the historical-μ-vs-transformer-μ comparison clean: μ enters only through the objective, never the search operator.

**Python↔C++ bridge**: Python writes two JSON files to a `tempfile.TemporaryDirectory()`:
- `cfg.json` — full config with `run_settings.data_bridge_file` set to bridge path
- `bridge.json` — `{n, tickers, mu, cov, eps?, delta?}`

The binary reads both, runs the selected algorithm, and prints results to stdout. Python parses ticker weights from stdout lines.

**Objective function**: `-(P * μᵀw - (1-P) * wᵀΣw)` — mean-variance with cardinality constraint K (at most K assets with weight in `[w_min, w_max]`).

**Dynamic risk parameter**: `benchmark_core.py` adjusts `P` each walk-forward window based on 21-day realized volatility (Ang & Bekaert 2004 regime logic): bull market → higher P, bear market → lower P.

### Benchmark orchestration (`benchmark/`)
- `benchmark_core.py`: `run_walkforward()` — the main walk-forward loop. Wires in the `Ansambel` scenario via `HedgeMuEnsemble` (from `estimators/mu_ensemble.py`, enabled by `evaluation.mu_ensemble.enabled`, default on). Also contains `find_binary()` which searches for the compiled `portfolio_optimizer`.
- `benchmark_report.py`: all presentation logic (merger of the former `benchmark_plots.py` + `benchmark_export.py`, which were both pure output and cross-imported). Matplotlib plots saved to `test_results/` (`plot_combined_walkforward`, `plot_all_algorithms_grid`, `plot_weights_grid`), the console summary table, and per-window CSV export (`export_csv`). `print_scenario_significance` runs paired significance tests (Wilcoxon by default) for three pairs: `Transformer`-vs-`Zgodovinski`, `Ansambel`-vs-`Zgodovinski`, `Ansambel`-vs-`Transformer`.
- `near_exact.py`: `solve_near_exact()` — near-optimal CCMV reference (support enumeration when `C(N,K)` is small, else warm-started multi-start 1-swap local search) for measuring the metaheuristics' optimality gap.
- `orlib.py`: loads the OR-Library `portN.txt` instances + published unconstrained frontiers (`portefN.txt`) and computes the Chang et al. (2000) standard percentage-error metric. Driven by `run_orlib_benchmark.py`.
- `benchmark_strategies.py`: classical benchmark portfolios for the historical walk-forward — `gmv_weights` (Σ-only), `max_sharpe_weights` (Markowitz tangency, no cardinality), `risk_parity_weights` (ERC). Chosen to isolate the hybrid's design choices: GMV/risk-parity ignore μ (do they beat transformer-μ?), Markowitz uses the same μ/Σ without cardinality (cost of the constraint + metaheuristic). Computed per-window inside `run_walkforward` and reported as extra `baseline_results` rows. A passive **market** buy-&-hold (`evaluation.market_ticker`, default `SPY`) is downloaded in `run_benchmark.py` and passed as `market_prices`.

## Configuration

Config is **split by concern across two files**, merged at runtime:

**Benchmark config** (`benchmark/configs/config_thesis_multistock.json`, `benchmark/configs/config_thesis_2013_multistock.json`) — the *experiment* (owned per-run):
- `tickers` — list of stock symbols
- `test_config` — `start_date`, `end_date`, `windows` (number of walk-forward windows)
- `return_estimator` — `return_model` (must be `multistock`), `{topw_alpha, cov_lambda, n_factors}` + hyperparameters, `start_date`/`end_date` for training
- `evaluation.mu_ensemble` — `enabled`, `gamma_grid`, `eta` for the `Ansambel` Hedge layer
- `portfolio` — the **problem definition**: `cardinality_K`, `w_min`, `w_max`, `risk_parameter` (base P). These define *which* CCMV problem is solved and may differ per regime.
- Covariance needs no config: `Zgodovinski` always uses sample Σ; `Transformer` and `Ansambel` auto-use the multistock shared-rep cov head when present.

**Solver config** (`portfolio_optimizers/config.json`) — the *metaheuristic knobs* (owned once, shared by all runs): `common` (`population_size`, `num_generations`, `seed`) + per-algorithm sections (`pso`/`sa`/`ga`). Also the standalone default the C++ binary reads when invoked with no config-path arg.

`run_benchmark.py._load_optimizer_config()` merges the two into the `optimizer_config` (`common` + per-algo) that `benchmark_core` and the C++ bridge expect: solver `common` (pop/gens/seed) ∪ benchmark `portfolio` (K/w_min/w_max/risk) ∪ per-algo sections. `run_orlib_benchmark.py` reads the solver config directly (and sets K/ε/δ to Chang standards itself).

The `run_settings` section (and `optimizer_config`) are injected at runtime by `run_benchmark.py` — do not add them to the benchmark config manually.

## Important Design Decisions

- **Rolling historical baseline + pipeline comparison**: The thesis contribution is "the transformer *pipeline* (model μ + model Σ) beats the classical pipeline (sample-mean μ + sample Σ), both fed to the same metaheuristic." Each scenario is an **end-to-end pipeline** using its native Σ (`Zgodovinski` = sample Σ, `Transformer`/`Ansambel` = shared-rep cov head), so the head-to-head compares whole approaches, not the μ source in isolation. The historical μ/Σ roll every window over the same recent window the transformer sees at inference (`roll_window = max(sequence_length, 21)`) — the real literature baseline (rolling sample mean/covariance). NB: because `Transformer` and `Zgodovinski` now differ in *both* μ and Σ, a difference cannot be attributed to μ alone; if you need to isolate the μ source, temporarily route `Transformer` through sample Σ (`trans_cov = "baseline"` in `benchmark_core.py`) as an ablation.
- **Rolling window for historical μ**: uses a window equal to `sequence_length` so both scenarios see the same amount of recent history.
- **No saved model persistence**: the model is re-trained from scratch on every `run_benchmark.py` call (no `saved_models/`).
- **Ensemble is a robustness layer, not the headline**: the primary thesis result is the pure `Transformer`-vs-`Zgodovinski` μ-source comparison. `Ansambel` is reported as a secondary regime-robust result (nothing tuned on the test regimes); its per-window weights are evidence — history's weight drops toward 0 when the transformer wins. Report improvements as economic/directional (with ~46 monthly windows nothing reaches p<0.05); do not overstate statistical significance.
