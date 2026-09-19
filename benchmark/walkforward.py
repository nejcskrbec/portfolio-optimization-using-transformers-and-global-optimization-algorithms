#!/usr/bin/env python3
"""Walk-forward engine: decision/window scheduling, moment estimation,
and run_experiment/save_outputs that drive the whole out-of-sample pipeline."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.utils.paths_utils import project_root
from benchmark.utils.data_utils import load_prices
from benchmark.utils.summaries_utils import (
    risk_points, _wealth_and_drawdown, _forecast_summary, _portfolio_summary,
    _forecast_aggregate, _vs_historical, _aggregate_summary,
)
from portfolio_optimizers.bridge import run_optimizer, find_binary, _mvo_utility
from estimators.pipeline import (
    train_models, predict_mus, required_windows, row_windows,
    matched_baseline_name, verify_conditioning_windows, normalize_for_optimizer,
)


@dataclass
class Decision:
    decision_pos: int
    realization_pos: int
    decision_date: pd.Timestamp
    realization_date: pd.Timestamp


def _position_on_or_before(index: pd.DatetimeIndex, date) -> int:
    pos = int(index.searchsorted(pd.Timestamp(date), side="right") - 1)
    if pos < 0:
        raise RuntimeError(f"No observation on/before {date}")
    return pos


def _position_on_or_after(index: pd.DatetimeIndex, date) -> int:
    pos = int(index.searchsorted(pd.Timestamp(date), side="left"))
    if pos >= len(index):
        raise RuntimeError(f"No observation on/after {date}")
    return pos


def resolve_windows(prices: pd.DataFrame, config: dict) -> dict:
    pcfg = config["protocol"]
    mode = pcfg.get("split_mode", "explicit")

    if mode == "fractions":
        tr = float(pcfg.get("train_fraction", 0.70))
        va = float(pcfg.get("validation_fraction", 0.15))
        n = len(prices)
        i_train = int(math.floor(n * tr))
        i_test = int(math.floor(n * (tr + va)))
        if not (20 < i_train < i_test < n - 1):
            raise RuntimeError("Invalid fraction split")
        return {
            "fit_start_pos": 0,
            "train_end_pos": i_train - 1,
            "fit_end_pos": i_test - 1,
            "test_start_pos": i_test,
            "test_end_pos": n - 1,
            "validation_observations": i_test - i_train,
        }

    fit_start = pcfg.get("fit_start", config["data"]["start_date"])
    fit_end = pcfg["fit_end"]
    test_start = pcfg["test_start"]
    test_end = pcfg.get("test_end", config["data"]["end_date"])

    fit_start_pos = _position_on_or_after(prices.index, fit_start)
    fit_end_pos = _position_on_or_before(prices.index, fit_end)
    test_start_pos = _position_on_or_after(prices.index, test_start)
    test_end_pos = _position_on_or_before(prices.index, test_end)

    if fit_end_pos >= test_start_pos:
        raise RuntimeError("fit_end must precede test_start")
    if test_end_pos <= test_start_pos:
        raise RuntimeError("test_end must follow test_start")

    # Early-stopping holdout, single global value fed to every model. Preferred
    # form is `validation_fraction` (a share of the training-window length), so
    # the holdout scales with the window (~10% is standard) instead of being a
    # fixed count that is 18% on a short protocol but 3% on a long one. A smaller
    # holdout also tightens symmetry with the historical baseline (which uses data
    # up to the decision date), since the model's weights are then fit on data
    # ending closer to the decision. Falls back to an explicit
    # `validation_observations` count.
    fit_span = fit_end_pos - fit_start_pos + 1
    val_frac = pcfg.get("validation_fraction")
    if val_frac is not None:
        validation_observations = max(int(round(float(val_frac) * fit_span)), 5)
    else:
        validation_observations = int(pcfg.get("validation_observations", 126))

    return {
        "fit_start_pos": fit_start_pos,
        "train_end_pos": None,
        "fit_end_pos": fit_end_pos,
        "test_start_pos": test_start_pos,
        "test_end_pos": test_end_pos,
        "validation_observations": validation_observations,
    }


def build_decisions(prices: pd.DataFrame, windows: dict, config: dict,
                    max_decisions=None) -> list[Decision]:
    pcfg = config["protocol"]
    schedule = str(pcfg.get("decision_schedule", "nonoverlap_days"))
    start = windows["test_start_pos"]
    end = windows["test_end_pos"]
    out: list[Decision] = []

    if schedule == "nonoverlap_days":
        horizon = int(pcfg["forecast_horizon_days"])
        d = max(start - 1, 0)
        # First decision must forecast a realization inside test.
        if d + horizon < start:
            d = start
        while d + horizon <= end:
            out.append(Decision(d, d + horizon, prices.index[d], prices.index[d + horizon]))
            d += int(pcfg.get("decision_step_days", horizon))
            if max_decisions is not None and len(out) >= int(max_decisions):
                break

    elif schedule == "daily":
        horizon = int(pcfg["forecast_horizon_days"])
        d0 = max(start - 1, 0)
        for d in range(d0, end - horizon + 1):
            if d + horizon < start:
                continue
            out.append(Decision(d, d + horizon, prices.index[d], prices.index[d + horizon]))
            if max_decisions is not None and len(out) >= int(max_decisions):
                break

    elif schedule == "month_end":
        # Decision at last trading day before each OOS month; realization at
        # the next month-end. Model training uses a fixed approximate horizon
        # (typically 21 trading days), while realized return is exact
        # month-end to month-end.
        oos_idx = prices.index[start:end + 1]
        month_ends = (
            pd.Series(oos_idx, index=oos_idx)
            .groupby(oos_idx.to_period("M"))
            .max()
            .tolist()
        )
        # Pre-test month end acts as the first decision date.
        pre = prices.index[max(start - 1, 0)]
        dates = [pre] + month_ends
        for a, b in zip(dates[:-1], dates[1:]):
            d = int(prices.index.get_loc(a))
            r = int(prices.index.get_loc(b))
            if r <= d:
                continue
            out.append(Decision(d, r, a, b))
            if max_decisions is not None and len(out) >= int(max_decisions):
                break

    else:
        raise ValueError(f"Unsupported decision_schedule: {schedule}")

    if not out:
        raise RuntimeError("No out-of-sample decisions created")
    return out


def _period_returns_from_positions(prices: pd.DataFrame, positions: list[tuple[int, int]],
                                   return_type: str) -> pd.DataFrame:
    rows, dates = [], []
    x = prices.values.astype(float)
    for a, b in positions:
        if a < 0 or b >= len(prices) or b <= a:
            continue
        if return_type == "log":
            r = np.log(x[b] / x[a])
        else:
            r = x[b] / x[a] - 1.0
        rows.append(r)
        dates.append(prices.index[b])
    if not rows:
        raise RuntimeError("No return samples available")
    return pd.DataFrame(rows, index=dates, columns=prices.columns)


def return_samples(prices: pd.DataFrame, end_pos: int, config: dict) -> pd.DataFrame:
    pcfg = config["protocol"]
    freq = str(pcfg.get("moment_frequency", "horizon_blocks"))
    rtype = str(pcfg.get("return_type", "simple"))

    if freq == "daily":
        if pcfg.get("moment_start_date") is not None:
            start_pos = _position_on_or_after(prices.index, pcfg["moment_start_date"])
        else:
            start_pos = int(pcfg.get("moment_start_pos", 0))
        p = prices.iloc[start_pos:end_pos + 1]
        if rtype == "log":
            ret = np.log(p / p.shift(1))
        else:
            ret = p.pct_change()
        return ret.dropna()

    if freq == "monthly":
        if pcfg.get("moment_start_date") is not None:
            start_pos = _position_on_or_after(prices.index, pcfg["moment_start_date"])
        else:
            start_pos = int(pcfg.get("moment_start_pos", 0))
        p = prices.iloc[start_pos:end_pos + 1]
        monthly = p.groupby(p.index.to_period("M")).tail(1)
        if rtype == "log":
            ret = np.log(monthly / monthly.shift(1))
        else:
            ret = monthly.pct_change()
        return ret.dropna()

    if freq == "horizon_blocks":
        h = int(pcfg["forecast_horizon_days"])
        if pcfg.get("moment_start_date") is not None:
            start_pos = _position_on_or_after(prices.index, pcfg["moment_start_date"])
        else:
            start_pos = int(pcfg.get("moment_start_pos", 0))
        positions = []
        b = start_pos + h
        while b <= end_pos:
            positions.append((b - h, b))
            b += h
        return _period_returns_from_positions(prices, positions, rtype)

    raise ValueError(f"Unsupported moment_frequency: {freq}")


# Horizon scaling (ALWAYS ON -- not configurable). mu and Sigma must live on
# the same horizon as the models' forecast. The transformers emit a raw
# H-period return, but "daily"/"monthly" sampling yields 1-period moments, so
# unscaled the model mu is inflated ~H-fold against the risk term and the
# nominal P silently means a different risk aversion for every model
# (scaling mu by c moves the trade-off to P'/(1-P') = c*P/(1-P)). Scaling BOTH
# moments by H is a no-op for the historical baseline -- H factors straight out
# of P*mu'w - (1-P)*w'Sigma w -- so this can only ever repair units, never
# move the baseline, which is why there is no switch for it. Under an i.i.d.
# random walk H*mu_1 and H*Sigma_1 are the exact H-period moments.
# "horizon_blocks" sampling already returns H-period returns, so it needs no
# scaling.
def _horizon_factor(config: dict) -> int:
    freq = str(config["protocol"].get("moment_frequency", "horizon_blocks"))
    if freq == "horizon_blocks":
        return 1
    h = int(config["protocol"].get(
        "model_forecast_horizon_days",
        config["protocol"]["forecast_horizon_days"],
    ))
    if freq == "monthly":
        h = max(1, int(round(h / 21.0)))
    return h


def _days_to_periods(config: dict, days: int) -> int:
    """Conditioning window in trading days -> number of moment samples.

    Conditioning windows are declared in trading days (that is the unit a model's
    `seq_len` is in), but the historical mean is taken over `return_samples`
    rows, whose spacing depends on `moment_frequency`.
    """
    freq = str(config["protocol"].get("moment_frequency", "horizon_blocks"))
    if freq == "daily":
        n = int(days)
    elif freq == "monthly":
        n = int(round(days / 21.0))
    else:
        n = int(round(days / int(config["protocol"]["forecast_horizon_days"])))
    if n < 2:
        raise ValueError(
            f"a {days}-day conditioning window is only {n} '{freq}' sample(s); "
            f"the matched historical mean would be meaningless -- use "
            f"moment_frequency='daily' for this protocol"
        )
    return n


def historical_covariance(prices: pd.DataFrame, end_pos: int, config: dict):
    """Sample Sigma over `moment_lookback_periods`, horizon-scaled and PSD-repaired.

    Sigma is deliberately NOT matched per model: `run_experiment` hands the same
    Sigma to every mu source, so its lookback and refit cadence degrade all of
    them identically and cannot tilt the head-to-head. Only mu is matched.

    When `moment_lookback_periods` is absent from the config the window defaults
    to the longest conditioning window across all enabled models (i.e. the most
    data any model reads).  This keeps Sigma and mu on a consistent timescale:
    no model's mu is paired with a Sigma estimated from a much longer history.
    """
    samples = return_samples(prices, end_pos, config)
    lookback = config["protocol"].get("moment_lookback_periods")
    if lookback is None:
        wins = required_windows(config)
        if wins:
            lookback = _days_to_periods(config, max(wins))
    if lookback is not None:
        samples = samples.tail(int(lookback))
    if len(samples) < 5:
        raise RuntimeError(f"Only {len(samples)} moment observations")

    cov = np.cov(samples.to_numpy(float), rowvar=False)
    cov = 0.5 * (cov + cov.T)
    cov = cov * _horizon_factor(config)

    # Tiny PSD repair only for numerical noise.
    eig_min = float(np.linalg.eigvalsh(cov)[0])
    if eig_min < 1e-10:
        cov = cov + np.eye(cov.shape[0]) * (1e-10 - eig_min)
    return cov


def historical_mus(prices: pd.DataFrame, end_pos: int, config: dict,
                   cond_windows) -> dict:
    """One historical sample mean per conditioning window, keyed by window length.

    Unlike Sigma these are recomputed at EVERY decision date regardless of
    `moment_refit`. That is not a cadence choice: the whole point of a matched
    baseline is that it reads the same recent bars the model reads, and a model's
    conditioning window always rolls forward with the decision date even when its
    weights are frozen. Freezing the mean would reintroduce exactly the staleness
    the matching exists to remove -- and it costs nothing (a numpy mean).
    """
    samples = return_samples(prices, end_pos, config)
    h = _horizon_factor(config)
    out = {}
    for w in cond_windows:
        n = _days_to_periods(config, int(w))
        block = samples.tail(n)
        if len(block) < 2:
            raise RuntimeError(
                f"only {len(block)} samples available for a {w}-day matched "
                f"baseline at position {end_pos}"
            )
        out[int(w)] = block.mean().to_numpy(float) * h
    return out


def actual_asset_return(prices: pd.DataFrame, d: Decision, return_type: str):
    p0 = prices.iloc[d.decision_pos].to_numpy(float)
    p1 = prices.iloc[d.realization_pos].to_numpy(float)
    simple = p1 / p0 - 1.0
    if return_type == "log":
        modeled = np.log(p1 / p0)
    else:
        modeled = simple
    return modeled, simple



def run_experiment(config: dict, *, smoke: bool = False):
    prices, tickers = load_prices(config)
    windows = resolve_windows(prices, config)

    max_decisions = config.get("run_settings", {}).get("max_decisions")
    if smoke:
        max_decisions = int(config.get("smoke", {}).get("max_decisions", 3))

    decisions = build_decisions(prices, windows, config, max_decisions=max_decisions)
    rpoints = risk_points(config)
    if smoke:
        smoke_risk = config.get("smoke", {}).get("risk_points")
        if smoke_risk:
            # Values are interpreted according to the configured grid type.
            tmp = copy.deepcopy(config)
            tmp["protocol"]["risk_grid"]["values"] = smoke_risk
            rpoints = risk_points(tmp)
        else:
            rpoints = rpoints[: min(3, len(rpoints))]

    seeds = [int(s) for s in config.get("run_settings", {}).get("seeds", [42])]
    if smoke:
        seeds = [seeds[0]]

    # Seed ensembling: instead of running each seed as an independent experiment
    # (and reporting a mean +/- std over highly dispersed single-seed Sharpes),
    # train one model per seed and average their mu forecasts before a single
    # optimizer call. This is the standard variance-reduction step for neural
    # forecasters (Gu-Kelly-Xiu 2020 train ensembles of identical nets) and is
    # applied uniformly to every model, so it cannot favour one of them.
    seed_ensemble = bool(config.get("run_settings", {}).get("seed_ensemble", False))
    if smoke:
        seed_ensemble = False
    # run_groups: (label written to the `seed` column, seeds trained for that run)
    if seed_ensemble:
        run_groups = [(0, list(seeds))]
    else:
        run_groups = [(s, [s]) for s in seeds]

    # Rolling vs expanding training window.
    #   default / "auto"     -- ROLLING: every (re)fit trains on the trailing
    #                           `fit_span` trading days (the initial fit length),
    #                           slid forward. Bounded per-refit cost: a perpetual
    #                           investor never trains on an ever-growing history,
    #                           and the window is self-calibrating (whatever each
    #                           config deemed enough for its first fit).
    #   <int>                -- ROLLING with that explicit trailing window.
    #   "expanding" / null   -- EXPANDING: train on everything through the cutoff
    #                           (anchored at data start). Paper-faithful ablation.
    # Only the historical mu/Sigma cadence (`moment_refit`) is separate; the
    # matched-baseline mu still rolls every decision regardless.
    _fit_span = int(windows["fit_end_pos"] - windows["fit_start_pos"] + 1)
    _tw = config["protocol"].get("train_window_days", "auto")
    if _tw in (None, "expanding"):
        train_window = None
    elif _tw == "auto":
        train_window = _fit_span
    else:
        train_window = int(_tw)
        # Floor guard for EXPLICIT overrides only: an auto window equals the
        # initial fit span, which trained successfully by construction, so it is
        # never checked. A hand-set window must still hold enough history for the
        # widest conditioning window + the validation holdout + a sample margin.
        floor = (max(required_windows(config))
                 + int(windows["validation_observations"]) + 60)
        if train_window < floor:
            raise ValueError(
                f"protocol.train_window_days={train_window} is below the "
                f"trainable floor {floor} (max conditioning window "
                f"{max(required_windows(config))} + validation "
                f"{int(windows['validation_observations'])} + 60-day margin)"
            )

    def _train_start(cutoff_pos: int) -> int:
        """First row of the training slice ending at `cutoff_pos` (inclusive)."""
        if train_window is None:
            return 0
        return max(0, cutoff_pos + 1 - train_window)

    print(f"training window: "
          + (f"rolling {train_window} trading days" if train_window is not None
             else "expanding (anchored at data start)"))

    # The initial fit ends at fit_end; under a rolling window equal to the fit
    # span this is exactly [fit_start, fit_end], so the default leaves the first
    # fit unchanged and only later refits drop stale history.
    fit_prices = prices.iloc[
        _train_start(windows["fit_end_pos"]): windows["fit_end_pos"] + 1
    ].copy()

    # `moment_refit` is the sibling of `model_refit` below: same three values,
    # one governs the historical mu/Sigma, the other the transformers.
    #   "once"         -- estimated once at fit_end, frozen for the whole OOS run.
    #   "annual"       -- re-estimated at the start of each OOS calendar year.
    #   "per_decision" -- re-estimated at every decision date.
    #   "match_models" -- (default) follow whatever `model_refit` does.
    # The default matters: when the two sides refit on different schedules the
    # comparison is not identified. On the Leow All-Weather protocol, letting the
    # baseline re-estimate weekly while the models stayed frozen was worth +4.4
    # Sharpe to the baseline, while the transformer moved 0.02 -- i.e. the entire
    # historical-vs-transformer result was a cadence artifact.
    moment_refit = str(config["protocol"].get("moment_refit", "match_models"))
    _REFITS = ("once", "annual", "per_decision")
    if moment_refit not in _REFITS + ("match_models",):
        raise ValueError(f"Unsupported moment_refit: {moment_refit}")
    if moment_refit == "match_models":
        moment_refit = str(config["protocol"].get("model_refit", "once"))
    # Conditioning windows of the enabled models. Each becomes a `Historical@<L>`
    # row that reads exactly the recent data its model reads.
    cond_windows = required_windows(config)
    row_win = row_windows(config)
    print(f"matched historical baselines: "
          + ", ".join(f"{n}->{matched_baseline_name(w)}"
                      for n, w in row_windows(config).items()
                      if not n.startswith("Historical")))

    fixed_cov = None
    if moment_refit in ("once", "annual"):
        fixed_cov = historical_covariance(prices, windows["fit_end_pos"], config)

    binary = find_binary()
    paper = config["literature"]["paper"]
    level = config["literature"].get("comparison_level", "adaptation")

    print("=" * 100)
    print(f"{paper}")
    print(f"comparison level: {level}")
    print(
        f"N={len(tickers)} | fit={fit_prices.index[0].date()}..{fit_prices.index[-1].date()} "
        f"| test={prices.index[windows['test_start_pos']].date()}.."
        f"{prices.index[windows['test_end_pos']].date()} "
        f"| decisions={len(decisions)} | risk points={len(rpoints)}"
    )
    for note in config["literature"].get("notes", []):
        print(f"[note] {note}")

    portfolio_rows = []
    prediction_rows = []

    # Walk-forward retraining cadence. "once" (default) = legacy single fit on
    # the pretest window; "annual" = refit on the expanding window at the start
    # of each new OOS calendar year, matching the historical baseline's per-window
    # re-estimation (Gu-Kelly-Xiu style). Removes the frozen-2015-weights penalty.
    model_refit = str(config["protocol"].get("model_refit", "once")).lower()
    if model_refit not in _REFITS:
        raise ValueError(f"Unsupported model_refit: {model_refit}")

    # Reject the pre-rename keys loudly rather than silently ignoring them and
    # running an unintended (and possibly unfair) protocol.
    for old, new in (("retrain_frequency", "model_refit"),
                     ("covariance_mode", "moment_refit"),
                     ("moment_horizon_scaling", None),
                     ("moment_mu_lookback_periods", None)):
        if old in config["protocol"]:
            hint = (f"renamed to '{new}'" if new else
                    "removed -- horizon scaling is now always on"
                    if old == "moment_horizon_scaling" else
                    "removed -- the historical mu lookback is no longer global; "
                    "each model is matched against a baseline reading its own "
                    "conditioning window")
            raise ValueError(f"protocol.{old} is obsolete ({hint})")

    training_times = []

    def _train_through(cutoff_pos: int, seed: int):
        train_px = prices.iloc[_train_start(cutoff_pos): cutoff_pos + 1].copy()
        return train_models(
            config, train_px, tickers, seed,
            validation_obs=windows["validation_observations"],
            timing_sink=training_times,
        )

    def _predict_avg(reg_list, df_hist, hist_mus):
        """Average each model's mu across the seed-ensemble members."""
        per_seed = [predict_mus(r, df_hist, tickers, hist_mus, config) for r in reg_list]
        if len(per_seed) == 1:
            return per_seed[0]
        out = {}
        for model in per_seed[0]:
            stack = np.vstack([np.asarray(p[model], dtype=float) for p in per_seed])
            out[model] = stack.mean(axis=0)
        return out

    for seed, seed_group in run_groups:
        if len(seed_group) > 1:
            print(f"\n{'='*72}\nSEED ENSEMBLE {seed_group} (mu averaged)\n{'='*72}")
        else:
            print(f"\n{'='*72}\nSEED {seed}\n{'='*72}")
        # Initial fit on the pretest window (through fit_end).
        regs = [
            train_models(
                config, fit_prices, tickers, s,
                validation_obs=windows["validation_observations"],
                timing_sink=training_times,
            )
            for s in seed_group
        ]
        # The pretest window is the SMALLEST fit window of the run, so a seq_len
        # that survives it survives every later refit too.
        for r in regs:
            verify_conditioning_windows(config, r)
        trained_for_year = decisions[0].decision_date.year if decisions else None
        # Position we've already trained through (initial fit ends at fit_end).
        trained_through_pos = int(windows["fit_end_pos"])
        # Baseline Sigma on the models' current information set.
        cov_synced = fixed_cov
        moments_through_pos = int(windows["fit_end_pos"])
        moments_for_year = trained_for_year

        # Turnover state is paired by risk/model/optimizer.
        drifted_prev = {}

        def _refit_cutoff(schedule: str, di: int, dec: Decision):
            """Data cutoff this decision refits on under `schedule`, else None.

            "once"         -- never refits after the initial pretest fit.
            "annual"       -- first decision of a new calendar year, on everything
                              through the end of the previous year.
            "per_decision" -- every decision, on the data available at that date.
            """
            if schedule == "annual":
                if dec.decision_date.year == _year_state[0]:
                    return None
                _year_state[0] = dec.decision_date.year
                prior = prices.index[prices.index < pd.Timestamp(dec.decision_date.year, 1, 1)]
                if len(prior) == 0:
                    return None
                return int(prices.index.get_loc(prior[-1]))
            if schedule == "per_decision":
                return int(dec.decision_pos)
            return None

        for di, dec in enumerate(decisions):
            _year_state = [trained_for_year]
            cutoff_pos = _refit_cutoff(model_refit, di, dec)
            trained_for_year = _year_state[0]

            # Skip if that cutoff adds no new data over what we already trained on
            # (avoids a redundant refit on the exact pretest window).
            if cutoff_pos is not None and cutoff_pos > trained_through_pos:
                label = (f"year={dec.decision_date.year}" if model_refit == "annual"
                         else f"decision {di+1}/{len(decisions)}")
                print(
                    f"\n[refit:models] seed={seed} {label} "
                    f"expanding fit through {prices.index[cutoff_pos].date()} ..."
                )
                # Explicitly release the old registry (models +
                # TimeSeriesDataSet objects) before allocating the next
                # refit, so peak RAM stays bounded across refits.
                import gc, torch
                del regs
                gc.collect()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                regs = [_train_through(cutoff_pos, s) for s in seed_group]
                trained_through_pos = cutoff_pos

            # Sigma follows `moment_refit`. When it was "match_models" it now
            # literally equals model_refit, so Sigma refreshes on exactly the
            # cutoffs the models were refit on.
            if moment_refit == "once":
                cov = fixed_cov
            elif moment_refit == "per_decision":
                cov = historical_covariance(prices, dec.decision_pos, config)
            else:  # "annual"
                _year_state = [moments_for_year]
                m_cutoff = _refit_cutoff("annual", di, dec)
                moments_for_year = _year_state[0]
                if m_cutoff is not None and m_cutoff > moments_through_pos:
                    cov_synced = historical_covariance(prices, m_cutoff, config)
                    moments_through_pos = m_cutoff
                cov = cov_synced

            # The matched historical means always roll with the decision date --
            # see `historical_mus` for why this is not a cadence knob.
            hist_mus = historical_mus(prices, dec.decision_pos, config, cond_windows)

            df_hist = prices.iloc[:dec.decision_pos + 1].copy()
            mus = _predict_avg(regs, df_hist, hist_mus)
            actual_modeled, actual_simple = actual_asset_return(
                prices, dec, config["protocol"].get("return_type", "simple")
            )

            # Asset-level predictions are independent of solver/risk point.
            for model, mu in mus.items():
                # `historical_asset_return` is this row's OWN matched baseline
                # (for a `Historical@L` row, itself), so any per-row comparison
                # against it is window-matched by construction.
                anchor = hist_mus[row_win[model]]
                for j, ticker in enumerate(tickers):
                    prediction_rows.append({
                        "seed": seed,
                        "decision_date": dec.decision_date,
                        "realization_date": dec.realization_date,
                        "ticker": ticker,
                        "model": model,
                        "conditioning_window": int(row_win[model]),
                        "matched_baseline": matched_baseline_name(row_win[model]),
                        "predicted_asset_return": float(mu[j]),
                        "historical_asset_return": float(anchor[j]),
                        "actual_modeled_asset_return": float(actual_modeled[j]),
                        "actual_simple_asset_return": float(actual_simple[j]),
                    })

            for ri, (risk_label, paper_par, P) in enumerate(rpoints):
                for model, mu in mus.items():
                    for algo in ("pso", "sa"):
                        if not config.get("solvers", {}).get(algo, {}).get("enabled", True):
                            continue

                        opt_seed = int(
                            seed * 100000 + di * 1000 + ri * 10 +
                            (0 if algo == "pso" else 1)
                        )
                        # Mandatory common-scale convention: z-score mu + mean-
                        # diagonal normalized Sigma, so every mu-source is on one
                        # dimensionless scale and P is a universal knob. Only the
                        # optimizer sees the normalized inputs; all reported
                        # diagnostics below stay in raw return units.
                        mu_n, cov_n = normalize_for_optimizer(mu, cov)
                        res = run_optimizer(
                            config, tickers, algo, mu_n, cov_n, P,
                            opt_seed, binary,
                        )
                        w = np.asarray(res["weights"], dtype=float)

                        pred_ret = float(w @ mu)
                        actual_ret_modeled = float(w @ actual_modeled)
                        actual_ret_simple = float(w @ actual_simple)
                        variance = float(w @ cov @ w)

                        key = (risk_label, model, algo)
                        prev = drifted_prev.get(key)
                        turnover = (
                            float(np.abs(w - prev).sum())
                            if prev is not None else np.nan
                        )

                        # Drift target weights through the just-realized period.
                        gross = 1.0 + actual_simple
                        drift = w * gross
                        if float(drift.sum()) > 0:
                            drift = drift / drift.sum()
                        drifted_prev[key] = drift

                        portfolio_rows.append({
                            "paper": paper,
                            "comparison_level": level,
                            "seed": seed,
                            "decision_date": dec.decision_date,
                            "realization_date": dec.realization_date,
                            "risk_label": risk_label,
                            "paper_risk_parameter": float(paper_par),
                            "P": float(P),
                            "model": model,
                            "conditioning_window": int(row_win[model]),
                            "matched_baseline": matched_baseline_name(row_win[model]),
                            "optimizer": algo.upper(),
                            "predicted_portfolio_return": pred_ret,
                            "actual_modeled_portfolio_return": actual_ret_modeled,
                            "actual_simple_portfolio_return": actual_ret_simple,
                            "predicted_variance": variance,
                            "predicted_utility": _mvo_utility(w, mu, cov, P),
                            "realized_utility": _mvo_utility(
                                w, actual_modeled, cov, P
                            ),
                            "turnover_l1": turnover,
                            "active_count": int(res["active_count"]),
                            "min_active": float(res["min_active"]),
                            "max_active": float(res["max_active"]),
                            "solver_sec": float(res["solver_sec"]),
                            "weights_json": json.dumps(
                                {t: float(x) for t, x in zip(tickers, w) if x > 1e-8}
                            ),
                        })

            if di == 0 or (di + 1) % 10 == 0 or di + 1 == len(decisions):
                print(
                    f"  decision {di+1:>4}/{len(decisions)} "
                    f"{dec.decision_date.date()} -> {dec.realization_date.date()}"
                )

    portfolios = pd.DataFrame(portfolio_rows)
    predictions = pd.DataFrame(prediction_rows)
    forecast_summary = _forecast_summary(predictions)
    forecast_aggregate = _forecast_aggregate(forecast_summary)
    portfolio_summary = _portfolio_summary(portfolios, config)
    aggregate_summary = _aggregate_summary(portfolio_summary)
    vs_historical = _vs_historical(portfolio_summary)

    return {
        "prices": prices,
        "tickers": tickers,
        "windows": windows,
        "decisions": decisions,
        "portfolios": portfolios,
        "predictions": predictions,
        "forecast_summary": forecast_summary,
        "forecast_aggregate": forecast_aggregate,
        "portfolio_summary": portfolio_summary,
        "aggregate_summary": aggregate_summary,
        "vs_historical": vs_historical,
        "training_times": training_times,
    }


def save_outputs(config: dict, result: dict):
    out = Path(config.get("output_dir", "test_results/literature_benchmark"))
    if not out.is_absolute():
        out = project_root() / out
    out.mkdir(parents=True, exist_ok=True)

    result["predictions"].to_csv(out / "prediction_results.csv", index=False)
    result["portfolios"].to_csv(out / "portfolio_results.csv", index=False)
    result["forecast_summary"].to_csv(out / "forecast_summary.csv", index=False)
    result["forecast_aggregate"].to_csv(out / "forecast_aggregate.csv", index=False)
    result["portfolio_summary"].to_csv(out / "portfolio_summary.csv", index=False)
    result["aggregate_summary"].to_csv(out / "aggregate_summary.csv", index=False)
    result["vs_historical"].to_csv(out / "estimator_vs_historical.csv", index=False)

    # Training-time record: wall-clock per-model fit times collected during the run
    # (folds in what the standalone timing benchmark used to measure).
    tt = pd.DataFrame(result.get("training_times", []))
    if not tt.empty:
        retrain = str(config["protocol"].get("model_refit", "once")).lower()
        timing = (tt.groupby("model")
                    .agg(n_assets=("n_assets", "max"),
                         n_trainings=("train_sec", "size"),
                         train_sec_mean=("train_sec", "mean"),
                         train_sec_total=("train_sec", "sum"))
                    .reset_index())
        timing.insert(0, "retrain", retrain)
        timing.insert(0, "paper", config["literature"]["paper"])
        timing.round(2).to_csv(out / "timing_train.csv", index=False)

    metadata = {
        "paper": config["literature"]["paper"],
        "comparison_level": config["literature"].get("comparison_level"),
        "notes": config["literature"].get("notes", []),
        "reference_values": config["literature"].get("reference_values", {}),
        "ticker_resolution_report": config.get("_runtime", {}).get(
            "ticker_resolution_report", []
        ),
        "tickers": result["tickers"],
        "n_assets": len(result["tickers"]),
        "n_decisions": len(result["decisions"]),
        # Which `Historical@<L>` row each model is matched against, and the
        # conditioning window (trading days) behind it. Recorded so a reader of
        # the CSVs can see the pairing without re-deriving it from the config.
        "conditioning_windows": {k: int(v) for k, v in row_windows(config).items()},
        "matched_baselines": {
            n: matched_baseline_name(w) for n, w in row_windows(config).items()
            if not n.startswith("Historical")
        },
        "config": config,
    }
    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return out

