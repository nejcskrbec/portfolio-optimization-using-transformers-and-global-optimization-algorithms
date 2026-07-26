"""
estimators/simple_ml.py
=======================
Preprost ML μ-baseline (gradient-boosted trees, LightGBM) — POŠTENA
"močnejša klasična" primerjava za transformer μ. Odgovarja na recenzentski
pomislek: "ali je transformer μ boljši od PREPROSTEGA ML modela, ne le od
zgodovinskega povprečja?".

Zasnova (namenoma standardna in interpretabilna — cross-sectional faktorski model):
  • Značilke = klasični momentum/reversal/vol faktorji (Jegadeesh & Titman 1993
    momentum; 12-1 momentum; kratkoročni reversal; realizirana vol), cross-sectional
    z-standardizirane PO DATUMU (kot transformer → cross-sectional ranker).
  • Tarča = ISTA kot transformer: naprej-gledani H-dnevni log-donos, cross-sectional
    demean po datumu (Li et al. 2024 MASTER; ujema estimators/multistock_master.py).
  • Model = LightGBM regresor (Ke et al. 2017), fiksni seed → determinističen.

Reference (za pisanje teze):
  Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers."
    Journal of Finance 48(1):65–91.  (momentum faktorji)
  Gu, S., Kelly, B. & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning."
    Review of Financial Studies 33(5):2223–2273.  (GBRT kot standardni ML baseline za donose)
  Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS.

API (vzporeden z multistock_master, brez faktorja/dispatch sloja):
  train_simple_ml(config, df_train, avail_tickers, lookahead)  → registry dict
  get_mu_simple_ml(registry, df_hist, avail_tickers, lookahead) → np.ndarray (N,), z-score
"""

import numpy as np
import pandas as pd

# Horizonti značilk (trgovalni dnevi). Standardni faktorski nabor.
_MOM_HORIZONS = [21, 63, 126, 252]   # 1m,3m,6m,12m momentum
_REV_HORIZON  = 5                    # 1w kratkoročni reversal
_VOL_HORIZON  = 21                   # 1m realizirana vol
_MOM_12_1     = (252, 21)            # 12-1 momentum (izpusti zadnji mesec)
_MIN_HIST     = 252 + 21             # dovolj zgodovine za najdaljšo značilko
_TRAIN_STRIDE = 5                    # vzorči učne datume vsak 5. dan (kot transformer TRAIN_STEP)
_SEED         = 42


def _features_at(logC: np.ndarray) -> np.ndarray:
    """Značilke za VSE delnice ob ZADNJEM stolpcu logC (T,N) → (N,F). Kavzalno."""
    T = logC.shape[0]
    feats = []
    for h in _MOM_HORIZONS:
        feats.append(logC[-1] - logC[-1 - h] if T > h else np.full(logC.shape[1], np.nan))
    feats.append(logC[-1] - logC[-1 - _REV_HORIZON] if T > _REV_HORIZON
                 else np.full(logC.shape[1], np.nan))
    # realizirana vol zadnjih _VOL_HORIZON dni
    if T > _VOL_HORIZON + 1:
        dr = np.diff(logC[-_VOL_HORIZON - 1:], axis=0)
        feats.append(dr.std(axis=0))
    else:
        feats.append(np.full(logC.shape[1], np.nan))
    # 12-1 momentum
    a, b = _MOM_12_1
    feats.append(logC[-1 - b] - logC[-1 - a] if T > a else np.full(logC.shape[1], np.nan))
    return np.stack(feats, axis=1)   # (N,F)


def _xsec_z(a: np.ndarray) -> np.ndarray:
    """Cross-sectional z-standardizacija po vrstici (čez delnice)."""
    mu = np.nanmean(a, axis=-1, keepdims=True)
    sd = np.nanstd(a, axis=-1, keepdims=True) + 1e-8
    return (a - mu) / sd


def train_simple_ml(config: dict, df_train: pd.DataFrame,
                    avail_tickers: list, lookahead: int) -> dict:
    """Natrenira LightGBM cross-sectional faktorski μ-model na učnem oknu.
    Enak forward-H-dnevni cilj kot transformer → poštena μ-primerjava."""
    # macOS libomp konflikt: torch je že naložil svoj OpenMP runtime; LightGBM
    # naloži svojega → dvojni libomp = SIGSEGV. Dovoli sobivanje (varno za GBRT)
    # in izklopi LGBM večnitnost (n_jobs=1) preden uvozimo lightgbm.
    import os as _os
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import lightgbm as lgb

    H = int(lookahead)
    px = df_train[avail_tickers].ffill().dropna(how="all")
    logC_full = np.log(px.values.astype(np.float64) + 1e-12)   # (T,N)
    T, N = logC_full.shape

    # Zgradi učne vzorce: za vsak datum t (stride) → značilke ob t, tarča = fwd H-log-donos.
    X_rows, y_rows = [], []
    for t in range(_MIN_HIST, T - H, _TRAIN_STRIDE):
        feat = _features_at(logC_full[:t + 1])                  # (N,F)
        feat = _xsec_z(feat)                                     # cross-sectional z
        tgt  = logC_full[t + H] - logC_full[t]                  # (N,) fwd H-log-donos
        tgt  = tgt - np.nanmean(tgt)                            # cross-sectional demean (kot MASTER)
        ok   = np.isfinite(feat).all(axis=1) & np.isfinite(tgt)
        if ok.sum() < 3:
            continue
        X_rows.append(feat[ok]); y_rows.append(tgt[ok])
    if not X_rows:
        raise ValueError("simple_ml: premalo učnih vzorcev (preveri dolžino df_train).")
    X = np.vstack(X_rows); y = np.concatenate(y_rows)

    model = lgb.LGBMRegressor(
        n_estimators=400, num_leaves=31, learning_rate=0.03,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=50, reg_lambda=1.0, random_state=_SEED, n_jobs=1,
        verbose=-1)
    model.fit(X, y)
    print(f"  [SimpleML] LightGBM faktorski μ: {len(y)} učnih cross-sectional vzorcev, "
          f"{X.shape[1]} značilk, H={H}")
    return {"_type": "simple_ml", "model": model, "H": H, "_seed": _SEED}


def get_mu_simple_ml(registry: dict, df_hist: pd.DataFrame,
                     avail_tickers: list, lookahead: int) -> np.ndarray:
    """μ napoved (cross-sectional z-score, mean0/std1 čez delnice) ob ZADNJEM
    datumu df_hist. Rescaling na donosno skalo naredi klicatelj (kot pri ansamblu)."""
    model = registry["model"]
    px = df_hist[avail_tickers].ffill()
    logC = np.log(px.values.astype(np.float64) + 1e-12)         # (T,N)
    feat = _xsec_z(_features_at(logC))                          # (N,F)
    feat_filled = np.where(np.isfinite(feat), feat, 0.0)        # manjkajoče → 0 (nevtralno)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # LGBM feature-name opozorilo (numpy vhod)
        pred = model.predict(feat_filled)                      # (N,)
    pred = np.asarray(pred, dtype=float)
    # cross-sectional z-score (monotono → IC nespremenjen; skalo poravna klicatelj)
    return (pred - pred.mean()) / (pred.std() + 1e-12)
