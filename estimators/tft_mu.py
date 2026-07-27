"""
tft_mu.py
=========
Temporal Fusion Transformer (Lim et al. 2021, Google) kot μ-napovednik — prek
OFF-THE-SHELF paketa `pytorch-forecasting` (TemporalFusionTransformer). Splošni
interpretabilni napovedni transformer (variable-selection + gating + večglava
pozornost + kvantilni izhod); tu tretji transformer poleg TACTiS (skupna
porazdelitev) in MASTER (Alpha158 + tržni gating).

Kako dobimo μ: TFT napove NASLEDNJIH `lookahead` DNEVNIH donosov za vsako delnico
(multi-step); μ = vsota napovedanih dnevnih donosov (kumulativni H-dnevni donos) —
ista definicija kot pri TACTiS. Kavzalno: enkoder uporabi le podatke do datuma t.
μ-only: presečni z-score → v cevovodu reskaliran na zgodovinsko μ skalo (kot
SimpleML/LSTM/MASTER), parjen z vzorčno Σ.

API (vzporeden simple_ml/lstm_mu/master_us):
  train_tft(config, df_train, avail_tickers, lookahead) -> registry
  get_mu_tft(registry, df_hist, avail_tickers, lookahead) -> np.ndarray (N,)  (z-score)
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
_EPS = 1e-12


def _long_frame(df_close: pd.DataFrame, tickers: list):
    """Dolg (long) panel: [time_idx, ticker, ret, mom20, vol20] iz cen zaprtja.
    Vrne (long_df, date_index). Donosi so dnevni log-donosi."""
    px = df_close[tickers].astype(float)
    rets = np.log(px / px.shift(1))
    mom20 = px / px.shift(20) - 1.0
    vol20 = rets.rolling(20).std()
    dates = px.index
    date_to_idx = {d: i for i, d in enumerate(dates)}
    frames = []
    for tk in tickers:
        sub = pd.DataFrame({
            "time_idx": [date_to_idx[d] for d in dates],
            "ticker": tk,
            "ret": rets[tk].values,
            "mom20": mom20[tk].values,
            "vol20": vol20[tk].values,
        })
        sub = sub.dropna(subset=["ret"])
        if len(sub) > 40:
            frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    long["mom20"] = long["mom20"].fillna(0.0)
    long["vol20"] = long["vol20"].fillna(0.0)
    long["ticker"] = long["ticker"].astype(str).astype("category")
    return long, dates


def _make_dataset(long, seq_len, horizon):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    return TimeSeriesDataSet(
        long,
        time_idx="time_idx",
        target="ret",
        group_ids=["ticker"],
        max_encoder_length=seq_len,
        min_encoder_length=seq_len // 2,
        max_prediction_length=horizon,
        min_prediction_length=1,
        static_categoricals=["ticker"],
        time_varying_unknown_reals=["ret", "mom20", "vol20"],
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def train_tft(config, df_train, avail_tickers, lookahead):
    """Nauči off-the-shelf TFT (pytorch-forecasting). Vrne registry dict."""
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    tcfg = config.get("evaluation", {}).get("tft", {})
    seq_len = int(tcfg.get("seq_len", 32))
    horizon = int(lookahead)
    epochs = int(tcfg.get("epochs", 12))
    seed = int(tcfg.get("seed", 0))
    pl.seed_everything(seed, workers=True)

    long, _ = _long_frame(df_train, list(avail_tickers))
    training = _make_dataset(long, seq_len, horizon)
    train_dl = training.to_dataloader(train=True, batch_size=int(tcfg.get("batch_size", 128)),
                                      num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(tcfg.get("lr", 1e-3)),
        hidden_size=int(tcfg.get("hidden_size", 16)),
        attention_head_size=int(tcfg.get("heads", 2)),
        dropout=float(tcfg.get("dropout", 0.1)),
        hidden_continuous_size=int(tcfg.get("hidden_continuous_size", 8)),
        loss=QuantileLoss(),
        log_interval=-1,
        optimizer="adam",
    )
    print(f"  [TFT] učenje ({epochs} epoh, seq={seq_len}, horizon={horizon}, "
          f"params={sum(p.numel() for p in tft.parameters())}) ...")
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1,
        enable_progress_bar=False, enable_model_summary=False,
        logger=False, enable_checkpointing=False,
        gradient_clip_val=0.1,
    )
    trainer.fit(tft, train_dataloaders=train_dl)

    return {
        "_type": "tft",
        "model": tft, "training": training,
        "seq_len": seq_len, "horizon": horizon,
        "tickers": list(avail_tickers), "lookahead": lookahead,
    }


def get_mu_tft(registry, df_hist, avail_tickers, lookahead):
    """Presečni μ (z-score): TFT napove naslednjih `horizon` dnevnih donosov,
    μ_delnica = vsota. Kavzalno (enkoder do df_hist.index[-1])."""
    import torch
    from pytorch_forecasting import TimeSeriesDataSet

    tft = registry["model"]
    training = registry["training"]
    seq_len = registry["seq_len"]
    horizon = registry["horizon"]
    tickers = registry["tickers"]

    long, _ = _long_frame(df_hist, tickers)
    if long.empty:
        return np.zeros(len(avail_tickers))
    # predict=True → za vsako skupino vzame zadnje znano enkodersko okno in
    # napove `horizon` korakov naprej (kavzalno).
    pred_ds = TimeSeriesDataSet.from_dataset(training, long, predict=True,
                                             stop_randomization=True)
    pred_dl = pred_ds.to_dataloader(train=False, batch_size=256, num_workers=0)
    out = tft.predict(pred_dl, mode="prediction", return_index=True)
    preds = out.output if hasattr(out, "output") else out[0]
    index = out.index if hasattr(out, "index") else out[1]
    preds = np.asarray(preds)                      # [n_groups, horizon]
    mu_h = preds.sum(axis=1)                        # kumulativni H-dnevni donos
    idx_tickers = list(index["ticker"].astype(str).values)

    s = pd.Series(mu_h, index=idx_tickers)
    z = (s - s.mean()) / (s.std() + _EPS)
    return z.reindex(avail_tickers).fillna(0.0).values.astype(float)
