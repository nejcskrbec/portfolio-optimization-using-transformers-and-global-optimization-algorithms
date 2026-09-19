"""
tft_mu.py
=========
Temporal Fusion Transformer (Lim et al. 2021) kot μ-napovednik prek
pytorch-forecasting.

Ključne lastnosti:
  * ticker je static_categorical → TFT dobi učljiv embedding na delnico;
  * časovni split brez leakage-a (val_loss + early stopping + best ckpt);
  * fiksna dolžina encoderja in decoderja;
  * inference brez Lightning Trainer-ja (_direct_predict: tft(x) + to_prediction);
  * dnevni log-donosi → H-dnevni simple return z exp(sum)-1;
  * rezultat je cross-sectional z-score.

API:
  train_tft(config, df_train, avail_tickers, lookahead) -> registry
  get_mu_tft(registry, df_hist, avail_tickers, lookahead, standardize=True) -> np.ndarray (N,)
"""
from __future__ import annotations

import tempfile
import numpy as np
import pandas as pd

_EPS = 1e-12

# Depth, in return days and inclusive of the row itself, of the deepest feature
# built in `_long_frame`: mom20 and vol20 are both 20-day statistics.
#
# This is why seq_len is NOT the TFT's conditioning window. seq_len counts
# encoder ROWS, but each row is a summary of the FEATURE_LOOKBACK days ending at
# it, so the oldest row reaches FEATURE_LOOKBACK-1 days further back still: the
# true window is seq_len + FEATURE_LOOKBACK - 1. (PatchTST, by contrast, eats raw
# returns -- depth 1 -- so for it the window really is seq_len.)
#
# `estimators.pipeline.conditioning_window` reads this constant to size the
# matched historical baseline. Change the features and this constant together,
# never the features alone, or the baseline silently stops being matched.
FEATURE_LOOKBACK = 20


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def _long_frame(df_close: pd.DataFrame, tickers: list[str]):
    """Long panel [time_idx, ticker, ret, mom20, vol20] iz close cen.

    time_idx je dodeljen iz globalnega date_to_idx pred dropna, zato ostane
    zvezen tudi po odstranitvi prve NaN vrstice na ticker.
    """
    px = df_close[tickers].astype(float).copy()
    rets = np.log(px / px.shift(1))
    mom20 = px / px.shift(FEATURE_LOOKBACK) - 1.0
    vol20 = rets.rolling(FEATURE_LOOKBACK).std()

    dates = px.index
    date_to_idx = {d: i for i, d in enumerate(dates)}

    frames = []
    for tk in tickers:
        sub = pd.DataFrame({
            "time_idx": [date_to_idx[d] for d in dates],
            "ticker":   str(tk),
            "ret":      rets[tk].values,
            "mom20":    mom20[tk].values,
            "vol20":    vol20[tk].values,
        })
        sub = sub.dropna(subset=["ret"])
        sub["mom20"] = sub["mom20"].fillna(0.0)
        sub["vol20"] = sub["vol20"].fillna(0.0)
        if len(sub) > 40:
            frames.append(sub)

    if not frames:
        return pd.DataFrame(
            columns=["time_idx", "ticker", "ret", "mom20", "vol20"]
        ), dates

    long = pd.concat(frames, ignore_index=True)
    long["ticker"] = long["ticker"].astype(str).astype("category")
    return long, dates


def _make_dataset(long: pd.DataFrame, seq_len: int, horizon: int):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    return TimeSeriesDataSet(
        long,
        time_idx="time_idx",
        target="ret",
        group_ids=["ticker"],
        max_encoder_length=seq_len,
        min_encoder_length=seq_len,
        max_prediction_length=horizon,
        min_prediction_length=horizon,
        static_categoricals=["ticker"],
        time_varying_unknown_reals=["ret", "mom20", "vol20"],
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        randomize_length=False,
    )


def _split_train_validation(
    long: pd.DataFrame, seq_len: int, horizon: int, validation_days: int
):
    max_idx = int(long["time_idx"].max())
    validation_days = max(int(validation_days), horizon)
    validation_start = max_idx - validation_days + 1

    if validation_start <= seq_len + horizon:
        raise ValueError(
            f"TFT: premalo podatkov za validation split "
            f"(max_idx={max_idx}, validation_days={validation_days}, "
            f"seq_len={seq_len}, horizon={horizon})"
        )

    train_long = long[long["time_idx"] < validation_start].copy()
    training = _make_dataset(train_long, seq_len, horizon)

    context_start = max(0, validation_start - seq_len)
    val_long = long[long["time_idx"] >= context_start].copy()

    from pytorch_forecasting import TimeSeriesDataSet
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_long,
        min_prediction_idx=validation_start,
        stop_randomization=True,
        predict=False,
    )
    return training, validation, validation_start


# ---------------------------------------------------------------------------
# Prediction frame (adds H placeholder decoder rows)
# ---------------------------------------------------------------------------

def _prediction_frame(long: pd.DataFrame, tickers: list[str], horizon: int) -> pd.DataFrame:
    """Dodaj H prihodnjih placeholder vrstic za decoder okno."""
    last_idx = int(long["time_idx"].max())
    rows = []
    for tk in tickers:
        for h in range(1, horizon + 1):
            rows.append({
                "time_idx": last_idx + h,
                "ticker":   str(tk),
                "ret":      0.0,
                "mom20":    0.0,
                "vol20":    0.0,
            })

    future = pd.DataFrame(rows)
    combined = pd.concat([long, future], ignore_index=True)
    cats = list(map(str, tickers))
    combined["ticker"] = pd.Categorical(
        combined["ticker"].astype(str), categories=cats
    )
    return combined


# ---------------------------------------------------------------------------
# Direct inference (no Lightning Trainer per call)
# ---------------------------------------------------------------------------

def _resolve_inference_device(model, registry):
    import torch

    cached = registry.get("_inference_device")
    if cached is not None:
        return torch.device(cached)

    requested = registry.get("inference_device", "auto")
    if requested != "auto":
        device = torch.device(requested)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    registry["_inference_device"] = str(device)
    return device


def _move_x_to_device(x: dict, device):
    import torch

    out = {}
    for key, value in x.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=False)
        elif isinstance(value, (list, tuple)):
            moved = [
                v.to(device, non_blocking=False) if torch.is_tensor(v) else v
                for v in value
            ]
            out[key] = type(value)(moved)
        else:
            out[key] = value
    return out


def _direct_predict(tft, pred_ds, pred_dl, device):
    """Napovej brez Lightning Trainer-ja: tft(x) + to_prediction()."""
    import torch

    pred_batches = []
    index_batches = []

    tft.eval()
    with torch.inference_mode():
        for batch in pred_dl:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            idx = pred_ds.x_to_index(x)
            index_batches.append(idx)

            x_dev = _move_x_to_device(x, device)
            raw_out = tft(x_dev)
            pred = tft.to_prediction(raw_out)

            if isinstance(pred, (list, tuple)):
                if len(pred) != 1:
                    raise RuntimeError(
                        f"TFT: pričakovan en target, dobil {len(pred)} outputov"
                    )
                pred = pred[0]

            pred_batches.append(pred.detach().cpu())

    if not pred_batches:
        raise RuntimeError("TFT: prediction dataloader je prazen")

    preds = torch.cat(pred_batches, dim=0).numpy()
    index = pd.concat(index_batches, ignore_index=True)
    return preds, index


# ---------------------------------------------------------------------------
# IC-based early stopping (mirrors MASTER's val-IC logic)
# ---------------------------------------------------------------------------

class _ValICCallback:
    """Lightning Callback that early-stops on cross-sectional val IC.

    After each validation epoch we run a cheap forward pass through the
    val_dataset, compute the daily cross-sectional IC (Pearson correlation
    of predicted vs realised H-day log-return across tickers), and restore
    the best-IC weights at training end.  This exactly mirrors the MASTER
    early-stopping logic and is justified by Lim et al. (2021) §4.2 which
    note that validation loss and downstream task quality can diverge.

    Constructor args
    ----------------
    val_dataset : TimeSeriesDataSet  (predict=False, validation period)
    long        : full training long panel (pd.DataFrame with time_idx/ticker/ret)
    tickers     : list[str]
    horizon     : int  (H, same as max_prediction_length)
    patience    : int
    min_delta   : float  (min improvement in IC to reset the bad counter)
    """

    def __init__(self, val_dataset, long, tickers, horizon, patience, min_delta=0.0):
        import lightning.pytorch as pl

        self.val_dataset = val_dataset
        self.tickers = [str(t) for t in tickers]
        self.horizon = int(horizon)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_ic = -np.inf
        self.bad = 0
        self.best_epoch = 0
        self.best_state: dict | None = None
        self._stopped = False

        # Cached across epochs: the val dataloader is rebuilt identically every
        # epoch otherwise (train=False, so it is deterministic and stateless).
        self._val_dl = None

        # Precompute realized H-day forward log-returns from the full long panel.
        # realized[(ticker_str, last_encoder_time_idx)] = sum(ret[t+1 .. t+H])
        #
        # Vectorised per ticker. The previous nested Python loop did one dict
        # lookup per (ticker, t, h) -- n_tickers * n_days * H, i.e. ~44M lookups
        # on the Wang panel (494 tickers, ~750 days, H=120) -- to build what is
        # just a forward-looking rolling sum. Identical values, paid in numpy.
        self._realized: dict[tuple, float] = {}
        keep = set(self.tickers)
        panel = long[["ticker", "time_idx", "ret"]].copy()
        panel["ticker"] = panel["ticker"].astype(str)
        for tk, sub in panel.groupby("ticker", sort=False):
            if tk not in keep:
                continue
            s = sub.sort_values("time_idx").set_index("time_idx")["ret"].astype(float)
            observed = s.index
            # Reindex onto a contiguous axis so a gap in the panel propagates NaN
            # and is dropped, exactly as the per-h `r.get(t+h, nan)` lookup did.
            s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
            # sum(ret[t+1 .. t+H]): trailing H-sum ending at t+H, shifted back H.
            fwd = s.rolling(self.horizon).sum().shift(-self.horizon)
            # Only anchor on days the ticker actually has a row for: the old loop
            # iterated the observed time_idx values, so a gap day must not become
            # a key even when its forward window is fully populated.
            fwd = fwd.reindex(observed)
            fwd = fwd[np.isfinite(fwd.to_numpy())]
            self._realized.update(
                {(tk, int(t)): float(v) for t, v in fwd.items()}
            )

        # Attach Lightning hooks by making this a pl.Callback subclass at runtime.
        # We inherit dynamically to avoid importing pl at module load time.
        _ValICCallback._patch_as_callback(self, pl)

    @staticmethod
    def _patch_as_callback(obj, pl):
        """Make obj behave as a pl.Callback by inheriting from it at runtime."""
        obj.__class__ = type(
            "_ValICCallbackLC",
            (pl.Callback,),
            {
                "on_validation_epoch_end": _ValICCallback._on_val_epoch_end,
                "on_train_end": _ValICCallback._on_train_end,
            },
        )

    @staticmethod
    def _on_val_epoch_end(self, trainer, pl_module):
        if self._stopped:
            return

        ic = _ValICCallback._compute_ic(self, pl_module)
        print(
            f"  [TFT-IC] epoch {trainer.current_epoch}: "
            f"val_IC={ic:.4f}  best={self.best_ic:.4f}  bad={self.bad}/{self.patience}"
        )

        if ic > self.best_ic + self.min_delta:
            self.best_ic = ic
            self.bad = 0
            self.best_epoch = int(trainer.current_epoch)
            self.best_state = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
        else:
            self.bad += 1
            if self.bad >= self.patience:
                print(
                    f"  [TFT-IC] early stop at epoch {trainer.current_epoch}: "
                    f"IC stagnant for {self.patience} epochs (best={self.best_ic:.4f})"
                )
                trainer.should_stop = True
                self._stopped = True

    @staticmethod
    def _on_train_end(self, trainer, pl_module):
        if self.best_state is not None:
            pl_module.load_state_dict(self.best_state)
            print(f"  [TFT-IC] restored best val_IC={self.best_ic:.4f} weights")

    @staticmethod
    def _compute_ic(self, pl_module) -> float:
        """One forward pass through val_dataset → average cross-sectional IC."""
        import torch

        device = next(pl_module.parameters()).device
        pl_module.eval()

        if self._val_dl is None:
            self._val_dl = self.val_dataset.to_dataloader(
                train=False, batch_size=256, num_workers=0
            )
        val_dl = self._val_dl

        # (last_encoder_time_idx, ticker_str, mu_predicted)
        records: list[tuple[int, str, float]] = []

        with torch.inference_mode():
            for batch in val_dl:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch

                # decoder_time_idx[:, 0] = first decoder step = last_encoder + 1
                last_enc = (x["decoder_time_idx"][:, 0] - 1).cpu().numpy()

                idx = self.val_dataset.x_to_index(x)
                ticker_names = [str(t) for t in idx["ticker"].values]

                x_dev = _move_x_to_device(x, device)
                raw = pl_module(x_dev)
                pred = pl_module.to_prediction(raw)
                if isinstance(pred, (list, tuple)):
                    pred = pred[0]

                preds_np = pred.detach().cpu().numpy()
                if preds_np.ndim == 1:
                    preds_np = preds_np[:, None]
                log_h = np.clip(preds_np.sum(axis=1), -20.0, 20.0)

                for t, tk, lh in zip(last_enc, ticker_names, log_h):
                    records.append((int(t), tk, float(lh)))

        # Group by prediction date, compute cross-sectional IC
        from collections import defaultdict
        by_time: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for t, tk, mu_pred in records:
            key = (tk, t)
            if key in self._realized:
                by_time[t].append((mu_pred, self._realized[key]))

        ics = []
        for pairs in by_time.values():
            if len(pairs) >= 2:
                ps, rs = zip(*pairs)
                ic_t = float(np.corrcoef(ps, rs)[0, 1])
                if np.isfinite(ic_t):
                    ics.append(ic_t)

        return float(np.mean(ics)) if ics else 0.0


class _EpochSelector:
    """Stage-1-only epoch selector: tracks the best validation epoch (lower is
    better) for a Lightning-logged scalar metric (e.g. ``val_loss``) and stops
    after `patience` epochs without improvement. Only `.best_epoch` is read by
    the caller -- the model these weights belong to is discarded, so unlike
    `_ValICCallback` this does not bother snapshotting/restoring state.
    """

    def __init__(self, metric, patience, min_delta=0.0):
        import lightning.pytorch as pl

        self.metric = str(metric)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_epoch = 0
        self.bad = 0
        self.__class__ = type(
            "_EpochSelectorLC", (pl.Callback,),
            {"on_validation_epoch_end": _EpochSelector.on_validation_epoch_end},
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics.get(self.metric)
        if m is None:
            return
        v = float(m)
        print(f"  [TFT] epoch {trainer.current_epoch}: {self.metric}={v:.6f} "
              f"best={self.best:.6f} bad={self.bad}/{self.patience}")
        if v < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad = v, int(trainer.current_epoch), 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                print(f"  [TFT] zgodnja ustavitev pri epohi {trainer.current_epoch} "
                      f"(najboljša epoha {self.best_epoch}, {self.metric}={self.best:.6f})")
                trainer.should_stop = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tft(config, df_train, avail_tickers, lookahead):
    """Nauči TFT dvostopenjsko in vrne registry z modelom.

    1. stopnja (izbira epoh): trening/validation split, prava
       validation-loss/IC zgodnja ustavitev -> najboljša epoha N*.
    2. stopnja (končni fit): SVEŽ model na CELOTNEM oknu (do datuma odločitve),
       fiksno N* epoh, brez validacije -- uteži so tako aktualne na datum
       odločitve, simetrično z zgodovinsko osnovo.
    """
    import gc
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import RMSE, QuantileLoss

    tcfg = config.get("evaluation", {}).get("tft", {}) or {}

    seq_len              = int(tcfg.get("seq_len", 32))
    horizon              = int(lookahead)
    epochs               = int(tcfg.get("epochs", 30))
    seed                 = int(tcfg.get("seed", 0))
    batch_size           = int(tcfg.get("batch_size", 128))
    validation_days      = int(tcfg.get("validation_days", max(126, 2 * horizon)))
    patience             = int(tcfg.get("patience", 5))
    es_metric            = str(tcfg.get("early_stopping_metric", "val_loss"))

    if horizon < 1:
        raise ValueError("TFT: lookahead mora biti >= 1")
    if seq_len < 2:
        raise ValueError("TFT: seq_len mora biti >= 2")
    if es_metric not in ("val_loss", "ic"):
        raise ValueError(f"TFT: early_stopping_metric mora biti 'val_loss' ali 'ic', dobil '{es_metric}'")

    pl.seed_everything(seed, workers=True)

    tickers = list(map(str, avail_tickers))
    long, _ = _long_frame(df_train, tickers)
    if long.empty:
        raise ValueError("TFT: prazen training panel")
    # Record only tickers that survived _long_frame filtering (had enough data).
    actual_tickers = sorted(long["ticker"].astype(str).unique())

    # WHICH FUNCTIONAL DOES mu ESTIMATE?
    # QuantileLoss is the TFT paper's own loss, but `to_prediction()` then returns
    # the q=0.5 head -- a conditional *median*. The mean-variance objective
    # P*mu'w - (1-P)*w'Sigma*w wants a conditional *mean*, and every other mu
    # source in this study supplies one: the historical baseline is a sample mean,
    # PatchTST and MASTER both train on squared error. On skewed return
    # distributions median != mean, so TFT was estimating a different quantity
    # from everything it is benchmarked against.
    #   loss="quantile" -> paper-faithful TFT, mu = conditional median
    #   loss="rmse"     -> squared error, output_size=1, mu = conditional mean
    loss_name = str(tcfg.get("loss", "quantile")).lower()
    if loss_name not in ("quantile", "rmse", "mse"):
        raise ValueError(
            f"TFT: loss mora biti 'quantile' ali 'rmse'/'mse', dobil '{loss_name}'"
        )

    def _make_loss():
        return QuantileLoss() if loss_name == "quantile" else RMSE()

    def _make_tft(ds):
        return TemporalFusionTransformer.from_dataset(
            ds,
            learning_rate=float(tcfg.get("lr", 1e-3)),
            hidden_size=int(tcfg.get("hidden_size", 16)),
            attention_head_size=int(tcfg.get("heads", 2)),
            dropout=float(tcfg.get("dropout", 0.1)),
            hidden_continuous_size=int(tcfg.get("hidden_continuous_size", 8)),
            loss=_make_loss(),
            log_interval=-1,
            optimizer="adam",
            reduce_on_plateau_patience=int(tcfg.get("lr_patience", 3)),
        )

    # Paper-faithful TFT training: a CONSTANT learning rate. The original TFT
    # (Lim et al. 2021) regularizes with a fixed LR + gradient clipping (0.01) +
    # dropout, with no LR schedule. Override pytorch-forecasting's default
    # ReduceLROnPlateau with a plain constant-LR Adam in both stages.
    import types as _types
    _tft_lr = float(tcfg.get("lr", 1e-3))

    def _constant_lr(model):
        model.configure_optimizers = _types.MethodType(
            lambda self: torch.optim.Adam(self.parameters(), lr=_tft_lr), model
        )
        return model

    def _free(*objs):
        for o in objs:
            del o
        gc.collect()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # --- Stage 1: epoch selection on a held-out validation slice ---------
    training1, validation1, validation_start = _split_train_validation(
        long, seq_len, horizon, validation_days
    )
    train_dl1 = training1.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl1 = validation1.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    model1 = _constant_lr(_make_tft(training1))

    if es_metric == "ic":
        selector = _ValICCallback(validation1, long, tickers, horizon, patience)
    else:
        selector = _EpochSelector("val_loss", patience)

    print(
        f"  [TFT] 1. stopnja: izbira epoh (val={validation_days} dni, "
        f"patience={patience}, metrika={es_metric}, do {epochs} epoh) ..."
    )
    trainer1 = pl.Trainer(
        max_epochs=epochs,
        accelerator=tcfg.get("accelerator", "cpu"),
        devices=int(tcfg.get("devices", 1)),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=float(tcfg.get("gradient_clip_val", 0.1)),
        num_sanity_val_steps=0,
        deterministic=True,
        callbacks=[selector],
    )
    trainer1.fit(model1, train_dataloaders=train_dl1, val_dataloaders=val_dl1)
    n_epoch2 = max(1, int(selector.best_epoch) + 1)
    _free(trainer1, train_dl1, val_dl1, model1)

    # --- Stage 2: final fit, fresh model, FULL window, fixed N* epochs ----
    training2 = _make_dataset(long, seq_len, horizon)
    train_dl2 = training2.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    model2 = _constant_lr(_make_tft(training2))
    print(
        f"  [TFT] 2. stopnja: {n_epoch2} epoh na celotnem oknu, konst. LR={_tft_lr:.1e}, "
        f"seq={seq_len}, horizon={horizon}, "
        f"params={sum(p.numel() for p in model2.parameters())}) ..."
    )
    trainer2 = pl.Trainer(
        max_epochs=n_epoch2,
        accelerator=tcfg.get("accelerator", "cpu"),
        devices=int(tcfg.get("devices", 1)),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        gradient_clip_val=float(tcfg.get("gradient_clip_val", 0.1)),
        num_sanity_val_steps=0,
        deterministic=True,
    )
    trainer2.fit(model2, train_dataloaders=train_dl2)
    _free(trainer2, train_dl2)
    best_model = model2.eval()
    best_epoch = n_epoch2 - 1
    best_val = float(getattr(selector, "best", getattr(selector, "best_ic", float("nan"))))
    print(f"  [TFT] končano po {n_epoch2} epohah (N* iz 1. stopnje)")

    return {
        "_type":                    "tft",
        "model":                    best_model,
        "training":                 training2,
        "best_epoch":               int(best_epoch),
        "loss":                     loss_name,
        "mu_functional":            "median" if loss_name == "quantile" else "mean",
        "seq_len":                  seq_len,
        "horizon":                  horizon,
        "tickers":                  actual_tickers,
        "lookahead":                int(lookahead),
        "validation_start_time_idx": int(validation_start),
        "best_val_metric":          best_val,
        "val_metric_name":          es_metric,
        "seed":                     seed,
        "inference_device":         str(tcfg.get("inference_device", "auto")),
        "inference_batch_size":     int(tcfg.get("inference_batch_size", 256)),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_mu_tft(registry, df_hist, avail_tickers, lookahead, standardize=True):
    """TFT μ za prihodnjih H dni (cross-sectional z-score ali surovi return)."""
    from pytorch_forecasting import TimeSeriesDataSet

    tft      = registry["model"]
    training = registry["training"]
    horizon  = int(registry["horizon"])
    tickers  = list(registry["tickers"])

    if int(lookahead) != horizon:
        raise ValueError(
            f"TFT: inference lookahead={lookahead} se ne ujema s training horizon={horizon}"
        )
    # tickers = actual tickers that survived _long_frame during training.
    # Restrict inference to those; return 0 for anything unseen.
    avail_set     = set(map(str, avail_tickers))
    infer_tickers = [t for t in tickers if t in avail_set]

    long, _ = _long_frame(df_hist, infer_tickers)
    if long.empty:
        return np.zeros(len(avail_tickers), dtype=float)

    pred_frame = _prediction_frame(long, infer_tickers, horizon)
    pred_ds = TimeSeriesDataSet.from_dataset(
        training,
        pred_frame,
        predict=True,
        stop_randomization=True,
    )
    pred_dl = pred_ds.to_dataloader(
        train=False,
        batch_size=int(registry.get("inference_batch_size", 256)),
        num_workers=0,
    )

    device = _resolve_inference_device(tft, registry)
    preds, index = _direct_predict(tft, pred_ds, pred_dl, device)
    preds = np.asarray(preds)

    preds = np.squeeze(preds)
    if preds.ndim == 1:
        preds = preds[:, None]
    if preds.shape[1] != horizon:
        raise RuntimeError(
            f"TFT: pričakovan horizon={horizon}, dobil shape={preds.shape}"
        )

    log_h = np.clip(preds.sum(axis=1), -20.0, 20.0)
    mu_h  = np.expm1(log_h)

    idx_tickers = list(index["ticker"].astype(str).values)
    s = pd.Series(mu_h, index=idx_tickers).groupby(level=0).last()

    arr = s.reindex(list(map(str, avail_tickers))).values.astype(float)
    # Tickers absent from training universe (e.g. post-training spinoffs) get μ=0
    arr = np.where(np.isfinite(arr), arr, 0.0)

    if not standardize:
        return arr

    if np.isfinite(arr).sum() < 2:
        return arr

    std = float(np.nanstd(arr))
    if std < _EPS:
        return np.where(np.isfinite(arr), 0.0, arr)
    return (arr - float(np.nanmean(arr))) / (std + _EPS)
