"""
Official-code PatchTST adapter for portfolio expected-return forecasting.

Architecture provenance
-----------------------
The neural backbone is NOT reimplemented here. This adapter imports
`PatchTST_backbone` from the authors' official ICLR-2023 repository:
    https://github.com/yuqinie98/PatchTST
expected at:
    estimators/patchtst_official/PatchTST_supervised/

Run `estimators/setup_patchtst_official.sh` once before training.

What this file adapts to the thesis pipeline:
- financial target: daily log returns;
- chronological train/validation split;
- training-only StandardScaler-style normalization, as in the official data loader;
- masked MSE only for missing financial observations;
- conversion of H predicted daily log returns to an H-day simple return;
- optional cross-sectional standardization for IC/rank-only evaluations.

The official PatchTST backbone retains the paper/repository mechanisms:
patching, channel independence, RevIN, end padding, residual attention,
BatchNorm-style encoder normalization, learned positional encoding and the
shared flatten forecasting head.
"""
from __future__ import annotations

import copy
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_EPS = 1e-12


def _log_returns(df_close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    px = df_close.reindex(columns=tickers).astype(float).copy()
    px = px.replace([np.inf, -np.inf], np.nan)
    r = np.log(px / px.shift(1)).replace([np.inf, -np.inf], np.nan)
    return r.iloc[1:].copy()


@dataclass
class _Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[None, :]) / self.std[None, :]


def _fit_scaler(rets: pd.DataFrame, train_end: int) -> _Scaler:
    """Match the official supervised data loader: fit scaling on train only."""
    arr = rets.iloc[:train_end].to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > _EPS), std, 1.0)
    return _Scaler(mean.astype(np.float32), std.astype(np.float32))


class _WindowDataset:
    def __init__(self, values, observed, starts, seq_len, horizon):
        import torch
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.observed = torch.as_tensor(observed, dtype=torch.bool)
        self.starts = list(map(int, starts))
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        b = s + self.seq_len
        c = b + self.horizon
        # x/y: [time, channels]
        return self.values[s:b], self.values[b:c], self.observed[b:c]


def _make_datasets(rets, seq_len, horizon, validation_days):
    n = len(rets)
    validation_days = max(int(validation_days), int(horizon))
    val_start = n - validation_days
    if val_start < seq_len + horizon:
        raise ValueError(
            f"PatchTST: premalo podatkov za časovno validacijo "
            f"(n={n}, seq={seq_len}, H={horizon}, val_days={validation_days})"
        )

    scaler = _fit_scaler(rets, val_start)
    raw = rets.to_numpy(dtype=float)
    observed = np.isfinite(raw)
    z = scaler.transform(raw)
    # Official benchmark datasets are complete. Financial panels may have gaps;
    # zero after train-fitted standardization is the neutral imputation.
    z = np.where(np.isfinite(z), z, 0.0).astype(np.float32)

    train_starts, val_starts = [], []
    max_start = n - seq_len - horizon
    for s in range(max_start + 1):
        ys = s + seq_len
        ye = ys + horizon
        if ye <= val_start:
            train_starts.append(s)
        elif ys >= val_start:
            val_starts.append(s)

    if not train_starts or not val_starts:
        raise ValueError(
            f"PatchTST: prazen train/validation split "
            f"(train={len(train_starts)}, val={len(val_starts)})"
        )

    return (
        _WindowDataset(z, observed, train_starts, seq_len, horizon),
        _WindowDataset(z, observed, val_starts, seq_len, horizon),
        scaler,
        val_start,
    )


def _official_supervised_root(pcfg: dict) -> Path:
    configured = pcfg.get("official_repo_path")
    if configured:
        root = Path(configured).expanduser().resolve()
        if root.name != "PatchTST_supervised":
            root = root / "PatchTST_supervised"
    else:
        root = Path(__file__).resolve().parent / "patchtst_official" / "PatchTST_supervised"

    required = root / "layers" / "PatchTST_backbone.py"
    if not required.is_file():
        raise FileNotFoundError(
            "Uradna koda PatchTST ni nameščena. Zaženi:\n"
            "  bash estimators/setup_patchtst_official.sh\n"
            f"Pričakovana datoteka: {required}"
        )
    return root


def _build_official_model(pcfg: dict, n_channels: int, seq_len: int, horizon: int):
    root = _official_supervised_root(pcfg)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Import the authors' exact backbone rather than a local rewrite.
    from layers.PatchTST_backbone import PatchTST_backbone

    patch_len = int(pcfg.get("patch_len", 16))
    stride = int(pcfg.get("stride", 8))
    if patch_len > seq_len:
        raise ValueError("PatchTST: patch_len ne sme presegati seq_len")

    model = PatchTST_backbone(
        c_in=n_channels,
        context_window=seq_len,
        target_window=horizon,
        patch_len=patch_len,
        stride=stride,
        n_layers=int(pcfg.get("e_layers", 3)),
        d_model=int(pcfg.get("d_model", 128)),
        n_heads=int(pcfg.get("n_heads", 16)),
        d_ff=int(pcfg.get("d_ff", 256)),
        norm=str(pcfg.get("norm", "BatchNorm")),
        attn_dropout=float(pcfg.get("attn_dropout", 0.0)),
        dropout=float(pcfg.get("dropout", 0.2)),
        act=str(pcfg.get("activation", "gelu")),
        res_attention=bool(pcfg.get("res_attention", True)),
        pre_norm=bool(pcfg.get("pre_norm", False)),
        pe=str(pcfg.get("pe", "zeros")),
        learn_pe=bool(pcfg.get("learn_pe", True)),
        fc_dropout=float(pcfg.get("fc_dropout", 0.2)),
        head_dropout=float(pcfg.get("head_dropout", 0.0)),
        padding_patch=pcfg.get("padding_patch", "end"),
        head_type="flatten",
        individual=bool(pcfg.get("individual", False)),
        revin=bool(pcfg.get("revin", True)),
        affine=bool(pcfg.get("affine", False)),
        subtract_last=bool(pcfg.get("subtract_last", False)),
    )
    return model, root


def _seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested="auto"):
    import torch
    requested = str(requested or "auto").lower()
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _masked_mse(pred, target, mask):
    """Financial robustness adaptation; equals ordinary MSE on complete panels."""
    m = mask.to(dtype=pred.dtype)
    denom = m.sum()
    if float(denom.detach().cpu()) <= 0:
        return None
    return (((pred - target) ** 2) * m).sum() / denom


def _forward(model, x):
    # Thesis dataset: x [B,L,N]. Official backbone: [B,N,L] -> [B,N,H].
    return model(x.transpose(1, 2)).transpose(1, 2)  # [B,H,N]


def _eval(model, dl, device):
    import torch
    model.eval()
    total, count = 0.0, 0.0
    with torch.inference_mode():
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            loss = _masked_mse(_forward(model, x), y, m)
            if loss is None:
                continue
            k = float(m.sum().detach().cpu())
            total += float(loss.detach().cpu()) * k
            count += k
    return total / count if count else math.inf


def _eval_ic(model, dl, device):
    """Average cross-sectional IC on the validation set (normalized space).

    For each window, sum predicted daily log-returns across the horizon to get
    the H-day signal per ticker, then compute Pearson IC across tickers.
    Normalisation is per-ticker (from the scaler), so the rank ordering across
    tickers is preserved and IC is valid in normalized space.
    """
    import torch
    model.eval()
    ics = []
    with torch.inference_mode():
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = _forward(model, x)   # (B, H, N)
            # H-day signal per ticker: sum across horizon dimension
            pred_h = pred.sum(dim=1)    # (B, N)
            real_h = y.sum(dim=1)       # (B, N)
            valid_h = m.all(dim=1)      # (B, N) – True if all horizon steps observed

            for b in range(pred_h.shape[0]):
                mask_b = valid_h[b].cpu().numpy()
                if mask_b.sum() < 2:
                    continue
                p = pred_h[b].cpu().numpy()[mask_b]
                r = real_h[b].cpu().numpy()[mask_b]
                ic_b = float(np.corrcoef(p, r)[0, 1])
                if np.isfinite(ic_b):
                    ics.append(ic_b)
    return float(np.mean(ics)) if ics else 0.0


def train_patchtst(config, df_train, avail_tickers, lookahead):
    """Train the authors' official supervised PatchTST backbone on financial returns.

    Two stages: (1) epoch selection on a held-out validation slice via real
    validation-loss/IC early stopping (constant LR -- the schedule doesn't
    matter, this model is discarded); (2) a FRESH model trained on the FULL
    window for exactly that many epochs, with the OneCycle schedule's
    `total_steps` recomputed for the full-window step count so it actually
    completes its anneal by the time training stops (unlike the previous
    train-loss-plateau heuristic, which cut the schedule off mid-anneal at
    an unpredictable epoch).
    """
    import torch
    from torch.utils.data import DataLoader

    pcfg = config.get("evaluation", {}).get("patchtst", {}) or {}
    seq_len = int(pcfg.get("seq_len", 336))
    horizon = int(lookahead)
    epochs = int(pcfg.get("epochs", 100))
    batch_size = int(pcfg.get("batch_size", 32))
    validation_days = int(pcfg.get("validation_days", max(126, 2 * horizon)))
    patience = int(pcfg.get("patience", 10))
    min_delta = float(pcfg.get("min_delta", 1e-5))
    es_metric = str(pcfg.get("early_stopping_metric", "val_loss"))
    if es_metric not in ("val_loss", "ic"):
        raise ValueError(f"PatchTST: early_stopping_metric mora biti 'val_loss' ali 'ic', dobil '{es_metric}'")
    seed = int(pcfg.get("seed", 2021))
    lr = float(pcfg.get("lr", 1e-4))
    pct_start = float(pcfg.get("pct_start", 0.2))
    use_onecycle = str(pcfg.get("lr_schedule", "TST")).lower() == "tst"

    if horizon < 1 or seq_len < 2:
        raise ValueError("PatchTST: neveljaven seq_len/lookahead")

    _seed_everything(seed)
    tickers = list(map(str, avail_tickers))
    rets = _log_returns(df_train, tickers)
    if rets.empty:
        raise ValueError("PatchTST: prazen učni panel")

    # Clamp seq_len and validation_days to available data so short regimes work.
    n = len(rets)
    max_seq = n - validation_days - horizon
    if max_seq < seq_len:
        # First try reducing validation_days proportionally, then seq_len.
        validation_days = max(horizon, n - seq_len - horizon)
        max_seq = n - validation_days - horizon
        if max_seq < 2:
            seq_len = max(2, n - validation_days - horizon)
            print(
                f"  [PatchTST] OPOZORILO: premalo podatkov (n={n}); "
                f"seq_len zmanjšan na {seq_len}, val_days={validation_days}"
            )
        elif max_seq < seq_len:
            seq_len = max_seq
            print(
                f"  [PatchTST] OPOZORILO: premalo podatkov (n={n}); "
                f"seq_len zmanjšan na {seq_len}, val_days={validation_days}"
            )

    device = _device(pcfg.get("accelerator", "auto"))
    official_root = _official_supervised_root(pcfg)

    def _one_epoch(model, dl, optimizer, scheduler):
        model.train()
        train_total, train_count = 0.0, 0.0
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = _forward(model, x)
            loss = _masked_mse(pred, y, m)
            if loss is None:
                continue
            loss.backward()
            # Official supervised training does not clip gradients by default.
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            k = float(m.sum().detach().cpu())
            train_total += float(loss.detach().cpu()) * k
            train_count += k
        return train_total / train_count if train_count else math.inf

    # --- Stage 1: epoch selection on a held-out validation slice -----------
    # Constant LR: the schedule doesn't matter here, this model is discarded --
    # only the epoch count at which validation stops improving is kept.
    train_ds1, val_ds1, scaler1, val_start = _make_datasets(rets, seq_len, horizon, validation_days)
    train_dl1 = DataLoader(
        train_ds1, batch_size=min(batch_size, len(train_ds1)), shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
    val_dl1 = DataLoader(val_ds1, batch_size=min(batch_size, len(val_ds1)), shuffle=False, num_workers=0)

    model1, _ = _build_official_model(pcfg, len(tickers), seq_len, horizon)
    model1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    print(
        f"  [PatchTST-official] 1. faza: izbira epohe (do {epochs} epoh, "
        f"train={len(train_ds1)}, val={len(val_ds1)} oken, patience={patience})"
    )
    best_score = -math.inf if es_metric == "ic" else math.inf
    best_epoch1 = 0
    bad = 0
    for epoch in range(epochs):
        train_loss = _one_epoch(model1, train_dl1, optimizer1, None)
        if es_metric == "ic":
            score = _eval_ic(model1, val_dl1, device)
            improved = score > best_score + min_delta
        else:
            score = _eval(model1, val_dl1, device)
            improved = score < best_score - min_delta
        print(
            f"  [PatchTST-official] 1.faza epoch={epoch+1:03d} "
            f"train={train_loss:.6f} val_{es_metric}={score:.6f} best={best_score:.6f} bad={bad}/{patience}"
        )
        if improved:
            best_score, best_epoch1, bad = score, epoch, 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [PatchTST-official] zgodnja ustavitev pri epohi {epoch+1} "
                      f"(najboljša epoha {best_epoch1+1}, val_{es_metric}={best_score:.6f})")
                break
    best_val = float(best_score)
    n_epoch2 = max(1, int(best_epoch1) + 1)
    del model1, optimizer1, train_dl1, val_dl1, train_ds1, val_ds1

    # --- Stage 2: final fit, fresh model, FULL window, fixed N* epochs -----
    # Weights are fit on data up to the decision date -- symmetric with the
    # historical baseline. The OneCycle schedule's `total_steps` is recomputed
    # from n_epoch2 (not the stage-1 `epochs` ceiling) so it actually completes
    # its anneal by the time training stops.
    scaler = _fit_scaler(rets, len(rets))
    raw = rets.to_numpy(dtype=float)
    observed = np.isfinite(raw)
    zt = scaler.transform(raw)
    z = np.where(np.isfinite(zt), zt, 0.0).astype(np.float32)
    full_starts = list(range(n - seq_len - horizon + 1))
    full_ds = _WindowDataset(z, observed, full_starts, seq_len, horizon)
    train_dl = DataLoader(
        full_ds, batch_size=min(batch_size, len(full_ds)), shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
    model, _ = _build_official_model(pcfg, len(tickers), seq_len, horizon)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    total_steps = max(1, len(train_dl)) * n_epoch2
    # OneCycleLR's two phases end at float(pct_start*total_steps)-1 and
    # total_steps-1; if either phase's (end_step - start_step) rounds to 0
    # it divides by zero on the very first scheduler.step(). Only enable it
    # when both phases have at least one real step.
    warmup_steps = pct_start * total_steps
    if use_onecycle and warmup_steps >= 2 and (total_steps - warmup_steps) >= 1:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, steps_per_epoch=max(1, len(train_dl)),
            pct_start=pct_start, epochs=n_epoch2, max_lr=lr,
        )
    print(
        f"  [PatchTST-official] 2. faza: {n_epoch2} epoh na celotnem oknu, "
        f"{len(full_ds)} oken, seq={seq_len}, horizon={horizon}, naprava={device})"
    )
    print(f"  [PatchTST-official] koda: {official_root}")
    for epoch in range(n_epoch2):
        train_loss = _one_epoch(model, train_dl, optimizer, scheduler)
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"  [PatchTST-official] 2.faza epoch={epoch+1:03d} "
            f"train={train_loss:.6f} lr={current_lr:.3e}"
        )
    best_epoch = n_epoch2 - 1

    model.to("cpu").eval()

    return {
        "_type": "patchtst_official",
        "model": model,
        "tickers": tickers,
        "seq_len": seq_len,
        "horizon": horizon,
        "lookahead": horizon,
        "scaler_mean": scaler.mean.copy(),
        "scaler_std": scaler.std.copy(),
        "best_val_metric": float(best_val),
        "val_metric_name": es_metric,
        "best_epoch": int(best_epoch),
        "validation_start_row": int(val_start),
        "seed": seed,
        "inference_device": str(pcfg.get("inference_device", "auto")),
        "official_supervised_root": str(official_root),
    }


def _inference_device(model, registry):
    import torch
    cached = registry.get("_inference_device")
    if cached is not None:
        return torch.device(cached)
    dev = _device(registry.get("inference_device", "auto"))
    model.to(dev).eval()
    registry["_inference_device"] = str(dev)
    return dev


def get_mu_patchtst(registry, df_hist, avail_tickers, lookahead, standardize=True):
    """PatchTST expected H-day simple returns for the current information set."""
    import torch

    model = registry["model"]
    seq_len = int(registry["seq_len"])
    horizon = int(registry["horizon"])
    train_tickers = list(map(str, registry["tickers"]))
    req = list(map(str, avail_tickers))

    if int(lookahead) != horizon:
        raise ValueError(
            f"PatchTST: inference lookahead={lookahead} != training horizon={horizon}"
        )
    if set(req) != set(train_tickers):
        raise ValueError("PatchTST: inference universe se razlikuje od učnega univerzuma")

    rets = _log_returns(df_hist, train_tickers)
    if len(rets) < seq_len:
        raise ValueError(
            f"PatchTST: premalo zgodovine za inferenco ({len(rets)} < {seq_len})"
        )

    mean = np.asarray(registry["scaler_mean"], dtype=np.float32)
    std = np.asarray(registry["scaler_std"], dtype=np.float32)
    x = rets.iloc[-seq_len:].to_numpy(dtype=float)
    x = (x - mean[None, :]) / std[None, :]
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)

    dev = _inference_device(model, registry)
    xt = torch.from_numpy(x).unsqueeze(0).to(dev)  # [1,L,N]
    with torch.inference_mode():
        pred_z = _forward(model, xt)[0].detach().cpu().numpy()  # [H,N]

    # Undo the outer StandardScaler. RevIN has already denormalized inside model.
    pred_log = pred_z * std[None, :] + mean[None, :]
    mu = np.expm1(np.clip(pred_log.sum(axis=0), -20.0, 20.0))
    arr = pd.Series(mu, index=train_tickers).reindex(req).to_numpy(dtype=float)

    if not standardize:
        return arr
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return arr
    s = float(np.nanstd(arr))
    if s < _EPS:
        return np.where(finite, 0.0, arr)
    return (arr - float(np.nanmean(arr))) / (s + _EPS)
