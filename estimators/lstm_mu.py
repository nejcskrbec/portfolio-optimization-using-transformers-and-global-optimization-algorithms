"""
estimators/lstm_mu.py
=====================
LSTM μ-baseline — POŠTENA sekvenčno-modelska primerjava za transformer μ.
Odgovarja NEPOSREDNO na predlagano primerjavo iz teme magistrske naloge
(proposal eksplicitno navaja "LSTM ali eno ML metodo" kot primerjalni model).

Zasnova (namenoma KLASIČEN sekvenčni model — recurrent, NE attention):
  • Vhod = standardizirana dnevna log-donosna sekvenca (SEQ_LEN dni) na delnico,
    ista dolžina konteksta kot transformer (sequence_length iz configa). Per-delnica
    normalizacija z učno mean/std donosov (kot transformer).
  • Tarča = ISTA kot transformer/simple_ml: naprej-gledani H-dnevni log-donos,
    cross-sectional demean po datumu → cross-sectional ranker (Li et al. 2024 MASTER).
  • Model = enoslojni LSTM (hidden iz configa) → linearna glava na zadnjem skritem
    stanju → cross-sectional score. Uči se na BATCHIH cross-sectionov (vse delnice
    istega datuma skupaj), izguba = negativna cross-sectional Pearsonova korelacija
    (rank-IC surogat), IDENTIČNA transformerjevi rank_ic_loss → razlika je SAMO
    arhitektura (LSTM vs MASTER attention).
  • 3-seed ansambel (kot transformer) → poštena primerjava (ne single-seed sreča).

To je "ali transformer prekaša LSTM" premisa iz predloga: enak cilj, enaka izguba,
enak kontekst, enak ansambel — spremeni se le hrbtenica modela.

Reference (za pisanje teze):
  Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory."
    Neural Computation 9(8):1735–1780.
  Fischer, T. & Krauss, C. (2018). "Deep learning with LSTM networks for financial
    market predictions." European Journal of Operational Research 270(2):654–669.
    (LSTM kot standardni deep-learning baseline za napovedovanje donosov.)

API (vzporeden z multistock_master / simple_ml):
  train_lstm(config, df_train, avail_tickers, lookahead)  → registry dict (3-seed)
  get_mu_lstm(registry, df_hist, avail_tickers, lookahead) → np.ndarray (N,), z-score
  (rescaling na donosno skalo naredi klicatelj, kot pri ansamblu/simple_ml).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# hiperparametri (usklajeni s transformerjem, kjer se prekrivajo)
SEQ_LEN_DEFAULT = 45          # ista dolžina konteksta kot MASTER-lite
TRAIN_STEP      = 2           # vzorči učne datume vsak 2. dan (kot transformer)
HIDDEN_DEFAULT  = 32          # skrito stanje LSTM (≈ transformer d_model)
EPOCHS, BATCH, LR, PATIENCE = 60, 16, 1e-3, 12
SEEDS           = [42, 1, 7]  # 3-seed ansambel (kot transformer)


class LSTMRanker(nn.Module):
    """Enoslojni LSTM → linearna glava. Vhod (B,N,L,1), izhod (B,N) score."""
    def __init__(self, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        B, N, L, F = x.shape
        h = x.reshape(B * N, L, F)
        out, _ = self.lstm(h)
        z = out[:, -1, :]                       # zadnje skrito stanje (B*N, hid)
        s = self.head(z).squeeze(-1)            # (B*N,)
        return s.reshape(B, N)                  # (B,N) cross-sectional score


def rank_ic_loss(pred, tgt):
    """Negativna cross-sectional Pearsonova korelacija po datumu (IDENTIČNA
    transformerjevi → izolira arhitekturo)."""
    pm = pred - pred.mean(dim=1, keepdim=True)
    tm = tgt - tgt.mean(dim=1, keepdim=True)
    pn = pm / (pm.std(dim=1, keepdim=True) + 1e-6)
    tn = tm / (tm.std(dim=1, keepdim=True) + 1e-6)
    return -(pn * tn).mean()


def _per_date_z(Y):
    mu = np.nanmean(Y, axis=1, keepdims=True)
    sg = np.nanstd(Y, axis=1, keepdims=True) + 1e-7
    return ((Y - mu) / sg).astype(np.float32)


def train_lstm(config: dict, df_train: pd.DataFrame,
               avail_tickers: list, lookahead: int) -> dict:
    """Natrenira 3-seed LSTM cross-sectional ranker na učnem oknu.
    Enak forward-H-dnevni cross-sectional cilj + rank-IC izguba kot transformer
    → poštena arhitekturna primerjava (LSTM vs MASTER attention)."""
    mc      = config.get("return_estimator", {})
    seq_len = int(mc.get("sequence_length", SEQ_LEN_DEFAULT))
    hidden  = int(mc.get("lstm_hidden", HIDDEN_DEFAULT))
    H       = int(lookahead)

    px = df_train[avail_tickers].ffill()
    C  = px.values.astype(np.float64)                       # (T,N)
    T, N = C.shape
    rr = np.zeros_like(C); rr[1:] = np.log(C[1:] / C[:-1] + 1e-12)   # dnevni log-donos

    # Per-delnica normalizacija donosov z UČNO statistiko (kot transformer).
    r_mu = np.nanmean(rr, axis=0, keepdims=True)
    r_sg = np.nanstd(rr, axis=0, keepdims=True) + 1e-7
    rn   = (rr - r_mu) / r_sg                               # (T,N)

    # fwd H-dnevni log-donos (cross-sectional demean → tarča kot MASTER).
    fwd = np.full((T, N), np.nan)
    fwd[:T - H] = np.log(C[H:] / C[:T - H] + 1e-12)

    # Učni datumi: [seq_len .. T-H), stride.
    P = np.array([t for t in range(seq_len, T - H, TRAIN_STEP)], dtype=int)

    def gather_X(idx):
        kidx = idx[:, None] - seq_len + 1 + np.arange(seq_len)     # (nP,L)
        X = rn[kidx]                                                # (nP,L,N)
        return np.transpose(X, (0, 2, 1))[..., None].astype(np.float32)  # (nP,N,L,1)

    Xtr = gather_X(P)                                      # (nP,N,L,1)
    Ytr = _per_date_z(fwd[P])                              # (nP,N)
    ok  = (np.isfinite(Xtr).all(axis=(1, 2, 3)) & np.isfinite(Ytr).all(axis=1))
    Xtr, Ytr = Xtr[ok], Ytr[ok]
    if len(Xtr) < 2:
        raise ValueError("lstm_mu: premalo učnih cross-sectionov (preveri df_train).")

    n = len(Xtr); nval = max(1, int(0.15 * n)); ntr = n - nval
    Xt = torch.tensor(Xtr[:ntr]); Yt = torch.tensor(Ytr[:ntr])
    Xv = torch.tensor(Xtr[ntr:]); Yv = torch.tensor(Ytr[ntr:])
    print(f"  [LSTM] učni cross-sectioni: {ntr} (val {nval}), N={N}, "
          f"seq_len={seq_len}, hidden={hidden}")

    states = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = LSTMRanker(hidden)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
        idx = np.arange(ntr); best, best_state, wait = float("inf"), None, 0
        for ep in range(EPOCHS):
            model.train(); np.random.shuffle(idx)
            for s in range(0, ntr, BATCH):
                b = idx[s:s + BATCH]
                opt.zero_grad()
                loss = rank_ic_loss(model(Xt[b]), Yt[b])
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                vl = rank_ic_loss(model(Xv), Yv).item()
            sch.step(vl)
            if vl < best - 1e-5:
                best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        states.append(best_state if best_state else model.state_dict())
        print(f"  [LSTM] seed {seed:2d}: val_rank_ic_loss={best:+.4f}")

    # per-delnica normalizacijska statistika za inferenco
    r_mu_d = {tk: float(r_mu[0, i]) for i, tk in enumerate(avail_tickers)}
    r_sg_d = {tk: float(r_sg[0, i]) for i, tk in enumerate(avail_tickers)}
    return {"_type": "lstm", "_tickers": avail_tickers, "_seq_len": seq_len,
            "hidden": hidden, "states": states, "r_mu": r_mu_d, "r_sg": r_sg_d}


def _load_models(registry: dict):
    models = []
    for state in registry["states"]:
        m = LSTMRanker(registry["hidden"])
        m.load_state_dict(state); m.eval()
        models.append(m)
    return models


def get_mu_lstm(registry: dict, df_hist: pd.DataFrame,
                avail_tickers: list, lookahead: int) -> np.ndarray:
    """μ napoved (cross-sectional z-score, mean0/std1 čez delnice) ob ZADNJEM
    datumu df_hist. 3-seed ansambel. Rescaling na donosno skalo naredi klicatelj."""
    seq_len = registry.get("_seq_len", SEQ_LEN_DEFAULT)
    r_mu = registry["r_mu"]; r_sg = registry["r_sg"]
    close = df_hist[avail_tickers].ffill()

    N = len(avail_tickers)
    X = np.zeros((N, seq_len, 1), dtype=np.float32)
    valid = np.zeros(N, dtype=bool)
    for i, tk in enumerate(avail_tickers):
        p = close[tk].values.astype(np.float64)
        if np.isfinite(p).sum() < seq_len + 2:
            continue
        r = np.diff(np.log(p + 1e-12))
        if len(r) < seq_len:
            continue
        rn = (r[-seq_len:] - r_mu.get(tk, 0.0)) / r_sg.get(tk, 1.0)
        X[i, :, 0] = rn
        valid[i] = True

    Xt = torch.tensor(np.nan_to_num(X)).unsqueeze(0)       # (1,N,L,1)
    preds = []
    for model in _load_models(registry):
        with torch.no_grad():
            preds.append(model(Xt).squeeze(0).cpu().numpy())
    score = np.mean(preds, axis=0)                         # (N,)

    sc = score.copy(); sc[~valid] = np.nan
    z = (sc - np.nanmean(sc)) / (np.nanstd(sc) + 1e-12)
    z[~valid] = 0.0                                        # nevtralno za nepokrite
    return np.nan_to_num(z, nan=0.0).astype(np.float64)
