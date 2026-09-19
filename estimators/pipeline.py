"""Model training/prediction orchestration (moved out of benchmark/core.py so
all training logic lives with the estimators it drives)."""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd


def _model_cfg(config: dict, name: str) -> dict:
    return copy.deepcopy(config.get("models", {}).get(name, {}))


# Conditioning window = the slice of RECENT data a mu-source consumes at forecast
# time to produce mu for one decision date. It is distinct from the *training*
# window (the data a model's weights were fit on): a transformer's weights stay
# frozen while its conditioning window rolls forward with every decision.
#
# The historical baseline has no training window at all -- a sample mean has no
# fitted parameters -- so its lookback IS its conditioning window. That makes it
# the one quantity matchable between the two approaches, and matching it is what
# identifies the head-to-head: if the two sides consume different recent data, a
# difference in mu cannot be attributed to method.
#
# A single global `moment_mu_lookback_periods` can only ever match ONE model (on
# the Wang protocol the models want 252 / 336 / 67). Hence per-model windows:
# every model is scored against a historical baseline reading exactly its own
# window.
def _declared_seq_len(config: dict, key: str) -> int:
    seq_len = config.get("models", {}).get(key, {}).get("seq_len")
    if seq_len is None:
        raise ValueError(
            f"model '{key}' declares no seq_len, so its conditioning window is "
            f"unknown and no matched historical baseline can be built"
        )
    return int(seq_len)


def conditioning_window(config: dict, name: str) -> int:
    """Length in trading days of `name`'s conditioning window.

    Every model reduces to `rows + feature_depth - 1`, where `rows` is how many
    timesteps the network reads and `feature_depth` is how many days of raw data
    each of those timesteps is built from. `seq_len` is only the FIRST term, so
    it equals the conditioning window solely for a model that eats raw bars:

      MASTER    T_LOOKBACK rows x Alpha158 stats over up to max(WINDOWS) days
      TFT       seq_len rows    x mom20/vol20 over FEATURE_LOOKBACK days
      PatchTST  seq_len rows    x raw log returns (depth 1) -> = seq_len

    Each depth is imported from the estimator module that defines the features,
    so a change to the features moves the matched baseline with it instead of
    silently unmatching it.
    """
    if name == "MASTER":
        from estimators.master_us import T_LOOKBACK, WINDOWS
        return int(T_LOOKBACK + max(WINDOWS) - 1)

    if name == "TFT":
        from estimators.tft_mu import FEATURE_LOOKBACK
        return int(_declared_seq_len(config, "tft") + FEATURE_LOOKBACK - 1)

    if name == "PatchTST":
        return int(_declared_seq_len(config, "patchtst"))

    raise ValueError(f"unknown mu source '{name}': no conditioning window defined")


def verify_conditioning_windows(config: dict, regs: dict) -> None:
    """Fail loudly if a trained model did not keep its declared `seq_len`.

    `patchtst_mu` silently CLAMPS seq_len when the fit window is too short and
    records the clamped value in its registry. Under a matched protocol that is
    not a graceful degradation -- it would pair the model with a baseline reading
    a window it never actually used, and (since fit windows grow across refits)
    could even change the window mid-run. Call this right after the first fit:
    the initial window is the smallest one, so passing here means no later refit
    can clamp either.
    """
    for name, reg in regs.items():
        if name == "MASTER" or not isinstance(reg, dict):
            continue
        actual = reg.get("seq_len")
        if actual is None:
            continue
        declared = _declared_seq_len(config, MODEL_NAMES[name])
        if int(actual) != declared:
            raise RuntimeError(
                f"{name} trained with seq_len={int(actual)} but the config "
                f"declares {declared} (the fit window is too short, so the "
                f"estimator clamped it). Its matched baseline would read the "
                f"wrong window -- set models.{MODEL_NAMES[name]}.seq_len to "
                f"{int(actual)} or lengthen the fit window."
            )


def matched_baseline_name(window: int) -> str:
    """Row label for the historical baseline reading a `window`-day lookback."""
    return f"Historical@{int(window)}"


MODEL_NAMES = {"MASTER": "master", "TFT": "tft", "PatchTST": "patchtst"}


def enabled_models(config: dict) -> list[str]:
    """Model names `train_models` will actually fit, in a stable order."""
    return [name for name, key in MODEL_NAMES.items()
            if _model_cfg(config, key).get("enabled", True)]


def required_windows(config: dict) -> list[int]:
    """Distinct conditioning windows the enabled models need, ascending.

    One historical mu is computed per entry; models sharing a window share a
    baseline row, so this is usually 2-3 numpy means per decision.
    """
    return sorted({conditioning_window(config, n) for n in enabled_models(config)})


def row_windows(config: dict) -> dict:
    """Every mu row -> the conditioning window it reads.

    Covers both the model rows and the `Historical@<L>` rows, so a caller can
    look up "which historical mu is this row matched against" without caring
    which kind of row it holds.
    """
    out = {matched_baseline_name(w): w for w in required_windows(config)}
    for name in enabled_models(config):
        out[name] = conditioning_window(config, name)
    return out


def normalize_for_optimizer(mu: np.ndarray, cov: np.ndarray):
    """Put mu and cov on a common, dimensionless scale for the CCMV objective.

    This is the single, mandatory scale convention for every mu-source (Historical,
    MASTER, TFT, PatchTST) and every benchmark -- there is no per-model or per-config
    switch. Two independent normalizations, each derived only from the inputs:

      * mu  -> cross-sectional z-score (mean 0, std 1). Every source is placed on
        one common scale, so a fixed risk parameter P means the same risk-aversion
        for all of them and a portfolio difference reflects the mu *ranking*, not
        the accidental magnitude of a model's raw output. (Rank-preserving, so it
        never touches forecast content: IC/RankIC are invariant to it.) The mean is
        irrelevant to selection anyway -- under the budget constraint 1^T w = 1,
        adding a constant to mu shifts the objective by a constant for every
        feasible portfolio -- so only the std normalization actually acts.

      * cov -> divided by its mean diagonal (average asset variance), giving a
        dimensionless risk term with mean variance 1. This is a single scalar, so
        every relative variance and correlation is preserved (unlike a correlation
        matrix, which would discard the asset volatilities).

    With both terms dimensionless the return/risk ratio no longer drifts with the
    horizon, universe or volatility regime, so P is a universal knob rather than a
    per-benchmark one. Portfolio *weights* are all that leave the optimizer, and the
    reported return/vol/Sharpe come from realized returns, so this convention only
    governs *which* portfolio is selected -- never how it is scored.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu has non-finite values before normalization")
    s = float(np.nanstd(mu))
    mu_z = (mu - float(np.nanmean(mu))) / s if s > 1e-12 else np.zeros_like(mu)
    d = float(np.mean(np.diag(cov)))
    cov_n = cov / d if d > 1e-16 else cov
    return mu_z, cov_n


# Training is two-stage per decision: (1) select the epoch count on a held-out
# validation slice via real validation-loss/IC early stopping, (2) retrain a
# FRESH model on the FULL window for exactly that many epochs. Stage 2's
# weights are current at the decision date -- symmetric with the historical
# baseline -- while stage 1 supplies a well-founded stopping signal (unlike a
# training-loss plateau heuristic, which a low/constant-LR model may never
# trigger early). See tft_mu.py, patchtst_mu.py, master_us.py.


def _record_train_time(sink, model, tickers, seed, dt):
    """Print and (if a sink is given) record one model's wall-clock training time."""
    print(f"  [{model}] train {dt:.1f}s")
    if sink is not None:
        sink.append({"model": model, "n_assets": len(tickers),
                     "seed": int(seed), "train_sec": round(float(dt), 2)})


def train_models(config: dict, prices_fit: pd.DataFrame,
                 tickers: list[str], seed: int, validation_obs: int,
                 timing_sink: list | None = None):
    regs = {}
    horizon = int(config["protocol"].get(
        "model_forecast_horizon_days",
        config["protocol"]["forecast_horizon_days"]
    ))

    # Estimator modules expect these project-level config sections.
    base = copy.deepcopy(config)
    base.setdefault("return_estimator", {})
    base["return_estimator"]["start_date"] = prices_fit.index[0].strftime("%Y-%m-%d")
    base["return_estimator"]["end_date"] = (
        prices_fit.index[-1] + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    # MASTER downloads its own OHLCV up to test_config.end_date (default 2022-01-01).
    # Feed it the real OOS end so inference features are never clamped to a stale
    # cut-off when a protocol runs past that default. Extra future rows are only used
    # at inference and are causally safe (normalization stats come from train_end).
    test_end = config["protocol"].get("test_end")
    if test_end is not None:
        base.setdefault("test_config", {})["end_date"] = (
            pd.Timestamp(test_end) + pd.Timedelta(days=5)
        ).strftime("%Y-%m-%d")

    if _model_cfg(config, "master").get("enabled", True):
        from estimators.master_us import train_master_us
        cfg = copy.deepcopy(base)
        cfg.setdefault("evaluation", {})["master"] = _model_cfg(config, "master")
        cfg["evaluation"]["master"]["validation_days"] = int(validation_obs)
        cfg["evaluation"]["master"]["seed"] = int(seed)
        t0 = time.time()
        regs["MASTER"] = train_master_us(cfg, prices_fit, tickers, horizon)
        _record_train_time(timing_sink, "MASTER", tickers, seed, time.time() - t0)

    if _model_cfg(config, "tft").get("enabled", True):
        from estimators.tft_mu import train_tft
        cfg = copy.deepcopy(base)
        cfg.setdefault("evaluation", {})["tft"] = _model_cfg(config, "tft")
        cfg["evaluation"]["tft"]["validation_days"] = int(validation_obs)
        cfg["evaluation"]["tft"]["seed"] = int(seed)
        t0 = time.time()
        regs["TFT"] = train_tft(cfg, prices_fit, tickers, horizon)
        _record_train_time(timing_sink, "TFT", tickers, seed, time.time() - t0)

    if _model_cfg(config, "patchtst").get("enabled", True):
        from estimators.patchtst_mu import train_patchtst
        cfg = copy.deepcopy(base)
        cfg.setdefault("evaluation", {})["patchtst"] = _model_cfg(config, "patchtst")
        cfg["evaluation"]["patchtst"]["validation_days"] = int(validation_obs)
        cfg["evaluation"]["patchtst"]["seed"] = int(seed)
        t0 = time.time()
        regs["PatchTST"] = train_patchtst(cfg, prices_fit, tickers, horizon)
        _record_train_time(timing_sink, "PatchTST", tickers, seed, time.time() - t0)

    return regs


def predict_mus(regs: dict, df_hist: pd.DataFrame, tickers: list[str],
                hist_mus: dict, config: dict):
    """Raw mu for every source at one decision date.

    Each source is returned on its OWN native scale -- no calibration or
    standardization here. The common-scale convention that makes the sources
    comparable to each other and commensurate with Sigma is applied once, at the
    optimizer boundary, by `normalize_for_optimizer` (z-score mu + mean-diagonal
    normalized Sigma). Because that step z-scores mu, it is invariant to any
    affine scale a source is emitted on, so the native scale here is irrelevant:
    MASTER's rank z-score, TFT/PatchTST's return forecast and the historical
    sample mean all collapse to the same normalized scale downstream.

    `hist_mus` maps a conditioning-window length -> the historical sample mean
    over exactly that many recent returns, so the returned dict carries one
    `Historical@<L>` row per distinct window (each model's matched baseline reads
    the same recent data its model reads -- see `conditioning_window`).
    """
    horizon = int(config["protocol"].get(
        "model_forecast_horizon_days",
        config["protocol"]["forecast_horizon_days"]
    ))

    mus = {matched_baseline_name(w): np.asarray(h, dtype=float)
           for w, h in hist_mus.items()}

    if "MASTER" in regs:
        from estimators.master_us import get_mu_master_us
        mus["MASTER"] = np.asarray(
            get_mu_master_us(regs["MASTER"], df_hist, tickers, horizon), dtype=float)

    if "TFT" in regs:
        from estimators.tft_mu import get_mu_tft
        mus["TFT"] = np.asarray(
            get_mu_tft(regs["TFT"], df_hist, tickers, horizon, standardize=False),
            dtype=float)

    if "PatchTST" in regs:
        from estimators.patchtst_mu import get_mu_patchtst
        mus["PatchTST"] = np.asarray(
            get_mu_patchtst(regs["PatchTST"], df_hist, tickers, horizon,
                            standardize=False), dtype=float)

    return mus


def matched_pairs(config: dict, regs: dict) -> dict:
    """model name -> the `Historical@<L>` row it must be compared against."""
    return {name: matched_baseline_name(conditioning_window(config, name))
            for name in regs}

