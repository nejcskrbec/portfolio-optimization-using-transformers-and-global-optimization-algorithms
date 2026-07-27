"""
master_us.py
============
MASTER (Market-Guided Stock Transformer, Li et al. AAAI 2024) na ameriških
delnicah — z ORIGINALNIM modelom iz repozitorija SJTU-DMTai/MASTER (vendored v
`estimators/master_lib/master.py` + `base_model.py`, nespremenjen), a s podatki,
ki jih zgradimo sami iz yfinance OHLCV — tako se izognemo Qlib/CSI podatkovni
verigi in torch-1.11 downgradu (model je čisti torch in teče na torch 2.x).

Kaj gradimo, da nahranimo model:
  * 158 Alpha158 faktorjev (9 KBAR + 4 cenovni + 29 kotalnih statistik × 5 oken
    [5,10,20,30,60] = 145) — presečno na delnico, iz OHLCV (Li et al./Qlib).
  * 63 tržnih značilk (market-guided gating): 3 ameriški indeksi (SPY≈CSI300,
    IWM≈CSI500, VTI≈CSI800/906) × 21 (donos + za d∈[5,10,20,30,60]:
    povprečje/std donosa in povprečje/std zneska/amount). Deljeno vsem delnicam.
  * oznaka (label) = Ref(close,-h)/Ref(close,-1)-1, h = lookahead.

Predobdelava kot v MASTER: RobustZScoreNorm (median/MAD iz UČNEGA obdobja, clip
±3, Fillna 0). Vsak vzorec je okno T=8 dni × 222 (158+63+1). Model odda presečni
rang-signal μ; v cevovodu ga (kot SimpleML/LSTM) afino reskaliramo na zgodovinsko
μ skalo in parimo z vzorčno/Ledoit-Wolf Σ (μ-only scenarij, brez modelske Σ).

API (vzporeden simple_ml/lstm_mu):
  train_master_us(config, df_train, avail_tickers, lookahead) -> registry
  get_mu_master_us(registry, df_hist, avail_tickers, lookahead) -> np.ndarray (N,)  (z-score)
"""
import os
import sys
import numpy as np
import pandas as pd

# vendored MASTER (original repo) na sys.path, da `from master import MASTERModel`
# in njegov `from base_model import SequenceModel` delujeta kot top-level uvoza.
_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

WINDOWS = [5, 10, 20, 30, 60]
T_LOOKBACK = 8
MARKET_TICKERS = ["SPY", "IWM", "VTI"]     # US ekvivalenti CSI300/CSI500/CSI800
_EPS = 1e-12


# ----------------------------------------------------------------------------
# Alpha158 v pandas (točno 158 stolpcev)
# ----------------------------------------------------------------------------
def _ols_rolling(close: pd.Series, d: int):
    """Kotalni OLS close~t (x=0..d-1). Vrne (slope, rsqr, resid_last)."""
    x = np.arange(d, dtype=float)
    x_mean = x.mean()
    Sxx = float(((x - x_mean) ** 2).sum())

    def _slope(a):
        y = a
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / (Sxx + _EPS))

    def _rsqr(a):
        y = a
        y_mean = y.mean()
        b = ((x - x_mean) * (y - y_mean)).sum() / (Sxx + _EPS)
        ss_tot = float(((y - y_mean) ** 2).sum())
        ss_res = float(((y - (y_mean + b * (x - x_mean))) ** 2).sum())
        if ss_tot < _EPS:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def _resid(a):
        y = a
        y_mean = y.mean()
        b = ((x - x_mean) * (y - y_mean)).sum() / (Sxx + _EPS)
        pred_last = y_mean + b * (x[-1] - x_mean)
        return float(y[-1] - pred_last)

    slope = close.rolling(d).apply(_slope, raw=True)
    rsqr = close.rolling(d).apply(_rsqr, raw=True)
    resid = close.rolling(d).apply(_resid, raw=True)
    return slope, rsqr, resid


def _alpha158_one(o, h, l, c, v):
    """158 Alpha158 značilk za ENO delnico (vse serije so pd.Series istega indeksa)."""
    vwap = (h + l + c) / 3.0
    feats = {}

    # --- KBAR (9) ---
    hl = (h - l) + _EPS
    feats["KMID"] = (c - o) / (o + _EPS)
    feats["KLEN"] = (h - l) / (o + _EPS)
    feats["KMID2"] = (c - o) / hl
    feats["KUP"] = (h - np.maximum(o, c)) / (o + _EPS)
    feats["KUP2"] = (h - np.maximum(o, c)) / hl
    feats["KLOW"] = (np.minimum(o, c) - l) / (o + _EPS)
    feats["KLOW2"] = (np.minimum(o, c) - l) / hl
    feats["KSFT"] = (2 * c - h - l) / (o + _EPS)
    feats["KSFT2"] = (2 * c - h - l) / hl

    # --- cenovne (4): OPEN/HIGH/LOW/VWAP glede na close ---
    feats["OPEN0"] = o / (c + _EPS)
    feats["HIGH0"] = h / (c + _EPS)
    feats["LOW0"] = l / (c + _EPS)
    feats["VWAP0"] = vwap / (c + _EPS)

    c1 = c.shift(1)
    v1 = v.shift(1)
    dclose = c - c1
    dvol = v - v1
    ret1 = c / (c1 + _EPS) - 1.0
    logv = np.log(v + 1.0)
    dlogv = np.log(v / (v1 + _EPS) + 1.0)

    # --- 29 kotalnih statistik × 5 oken (145) ---
    for d in WINDOWS:
        s = str(d)
        feats["ROC" + s] = c.shift(d) / (c + _EPS)
        feats["MA" + s] = c.rolling(d).mean() / (c + _EPS)
        feats["STD" + s] = c.rolling(d).std() / (c + _EPS)
        slope, rsqr, resid = _ols_rolling(c, d)
        feats["BETA" + s] = slope / (c + _EPS)
        feats["RSQR" + s] = rsqr
        feats["RESI" + s] = resid / (c + _EPS)
        feats["MAX" + s] = h.rolling(d).max() / (c + _EPS)
        feats["MIN" + s] = l.rolling(d).min() / (c + _EPS)
        feats["QTLU" + s] = c.rolling(d).quantile(0.8) / (c + _EPS)
        feats["QTLD" + s] = c.rolling(d).quantile(0.2) / (c + _EPS)
        feats["RANK" + s] = c.rolling(d).apply(
            lambda a: (a <= a[-1]).sum() / len(a), raw=True)
        mn = l.rolling(d).min()
        mx = h.rolling(d).max()
        feats["RSV" + s] = (c - mn) / (mx - mn + _EPS)
        feats["IMAX" + s] = h.rolling(d).apply(lambda a: (len(a) - 1 - np.argmax(a)) / d, raw=True)
        feats["IMIN" + s] = l.rolling(d).apply(lambda a: (len(a) - 1 - np.argmin(a)) / d, raw=True)
        feats["IMXD" + s] = h.rolling(d).apply(lambda a: np.argmax(a), raw=True) / d \
            - l.rolling(d).apply(lambda a: np.argmin(a), raw=True) / d
        feats["CORR" + s] = c.rolling(d).corr(logv)
        feats["CORD" + s] = ret1.rolling(d).corr(dlogv)
        up = (dclose > 0).astype(float)
        dn = (dclose < 0).astype(float)
        feats["CNTP" + s] = up.rolling(d).mean()
        feats["CNTN" + s] = dn.rolling(d).mean()
        feats["CNTD" + s] = up.rolling(d).mean() - dn.rolling(d).mean()
        pos = np.maximum(dclose, 0.0)
        neg = np.maximum(-dclose, 0.0)
        absd = dclose.abs()
        feats["SUMP" + s] = pos.rolling(d).sum() / (absd.rolling(d).sum() + _EPS)
        feats["SUMN" + s] = neg.rolling(d).sum() / (absd.rolling(d).sum() + _EPS)
        feats["SUMD" + s] = (pos.rolling(d).sum() - neg.rolling(d).sum()) \
            / (absd.rolling(d).sum() + _EPS)
        feats["VMA" + s] = v.rolling(d).mean() / (v + _EPS)
        feats["VSTD" + s] = v.rolling(d).std() / (v + _EPS)
        wv = (ret1.abs() * v)
        feats["WVMA" + s] = wv.rolling(d).std() / (wv.rolling(d).mean() + _EPS)
        vpos = np.maximum(dvol, 0.0)
        vneg = np.maximum(-dvol, 0.0)
        vabs = dvol.abs()
        feats["VSUMP" + s] = vpos.rolling(d).sum() / (vabs.rolling(d).sum() + _EPS)
        feats["VSUMN" + s] = vneg.rolling(d).sum() / (vabs.rolling(d).sum() + _EPS)
        feats["VSUMD" + s] = (vpos.rolling(d).sum() - vneg.rolling(d).sum()) \
            / (vabs.rolling(d).sum() + _EPS)

    df = pd.DataFrame(feats)
    return df


def _market_features(mkt_ohlcv: dict, dates: pd.DatetimeIndex):
    """63 tržnih značilk (3 indeksi × 21), poravnano na `dates`."""
    cols = {}
    for tk in MARKET_TICKERS:
        c = mkt_ohlcv[tk]["close"].reindex(dates).ffill()
        v = mkt_ohlcv[tk]["volume"].reindex(dates).ffill()
        amount = c * v
        ret = c / c.shift(1) - 1.0
        cols[f"{tk}_ret"] = ret
        for d in WINDOWS:
            cols[f"{tk}_rmean{d}"] = ret.rolling(d).mean()
            cols[f"{tk}_rstd{d}"] = ret.rolling(d).std()
            cols[f"{tk}_amean{d}"] = amount.rolling(d).mean() / (amount + _EPS)
            cols[f"{tk}_astd{d}"] = amount.rolling(d).std() / (amount + _EPS)
    m = pd.DataFrame(cols, index=dates)
    return m   # (D, 63)


# ----------------------------------------------------------------------------
# Torch Dataset, ki posnema qlib TSDataSampler (interfejs, ki ga base_model rabi)
# ----------------------------------------------------------------------------
try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:                      # pragma: no cover
    _TorchDataset = object


class _MasterDataset(_TorchDataset):
    """Vrne [N,T,F] presečno-dnevne bloke; get_index() vrne MultiIndex
    (datetime, instrument). F = 221 značilk + 1 oznaka = 222."""

    def __init__(self, cube, dates, tickers, samples, index):
        self.cube = cube          # np.float32 [D, N, 222]
        self.dates = dates
        self.tickers = tickers
        self.samples = samples    # list[(day_pos, stock_pos)]
        self._index = index       # pd.MultiIndex

    def get_index(self):
        return self._index

    def _win(self, day, st):
        return self.cube[day - (T_LOOKBACK - 1):day + 1, st, :]   # [T,222]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        if np.isscalar(k):
            day, st = self.samples[k]
            return self._win(day, st)
        return np.stack([self._win(*self.samples[i]) for i in k])


# ----------------------------------------------------------------------------
# gradnja podatkovnega kocke (cube) — deljena za train in inferenco
# ----------------------------------------------------------------------------
def _download_ohlcv(tickers, start, end):
    """Vrne dict[ticker] -> DataFrame[open,high,low,close,volume]."""
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="ticker")
    out = {}
    for tk in tickers:
        try:
            sub = raw[tk]                       # group_by="ticker" → vedno 2-nivo
            df = pd.DataFrame({
                "open": sub["Open"], "high": sub["High"], "low": sub["Low"],
                "close": sub["Close"], "volume": sub["Volume"],
            }).dropna(how="all")
            if len(df) > 60:
                out[tk] = df
        except Exception:
            continue
    return out


# ----------------------------------------------------------------------------
# javni API
# ----------------------------------------------------------------------------
def train_master_us(config, df_train, avail_tickers, lookahead):
    """Nauči ORIGINALNI MASTER na ameriških podatkih. Vrne registry dict."""
    import tempfile
    import torch
    from master import MASTERModel

    mc = config.get("return_estimator", {})
    tc = config.get("test_config", {})
    mcfg = config.get("evaluation", {}).get("master", {})

    train_start = mc.get("start_date", "2012-01-01")
    test_end = tc.get("end_date", "2022-01-01")
    tickers = list(avail_tickers)

    print(f"  [MASTER] prenos OHLCV ({len(tickers)} delnic + {len(MARKET_TICKERS)} "
          f"indeksov) {train_start}→{test_end} ...")
    ohlcv = _download_ohlcv(tickers, train_start, test_end)
    tickers = [t for t in tickers if t in ohlcv]
    mkt_raw = _download_ohlcv(MARKET_TICKERS, train_start, test_end)
    mkt_ohlcv = {tk: {"close": mkt_raw[tk]["close"], "volume": mkt_raw[tk]["volume"]}
                 for tk in MARKET_TICKERS if tk in mkt_raw}
    if len(mkt_ohlcv) < len(MARKET_TICKERS):
        raise RuntimeError(f"manjkajo tržni indeksi: {mkt_ohlcv.keys()}")

    # cube na CELOTNEM razponu; RobustZScoreNorm z median/MAD SAMO iz učnega
    # obdobja (brez leakage — test podatki ne vplivajo na normalizacijo).
    train_end = pd.Timestamp(mc.get("end_date", tc.get("start_date")))
    print(f"  [MASTER] gradnja Alpha158 (158) + tržnih značilk (63) ...")
    cube, dates, stats = _raw_then_norm(ohlcv, mkt_ohlcv, tickers, lookahead,
                                        train_end)

    D, N = cube.shape[0], cube.shape[1]

    # učni vzorci: dan >= T-1, label ni nan, znotraj učnega obdobja.
    # Presečni dnevi s premalo delnicami se PRESKOČIJO: MASTER-jev drop_extreme
    # (indices[k:-k], k=int(0.025·N)) vrne prazno pri N<20 → zscore(prazno)=nan →
    # nan gradient poškoduje ves model. Zahtevamo vsaj MIN_STOCKS delnic/dan.
    MIN_STOCKS = 25
    train_samples, train_idx = [], []
    valid_samples, valid_idx = [], []
    valid_start = train_end - pd.Timedelta(days=120)
    for di, dt in enumerate(dates):
        if di < T_LOOKBACK - 1:
            continue
        if dt > train_end:
            break
        day_list = []
        for si, tk in enumerate(tickers):
            if not np.isfinite(cube[di, si, 221]):
                continue
            if np.all(cube[di, si, :158] == 0):
                continue
            day_list.append((si, tk))
        if len(day_list) < MIN_STOCKS:
            continue
        for si, tk in day_list:
            if dt >= valid_start:
                valid_samples.append((di, si))
                valid_idx.append((dt, tk))
            else:
                train_samples.append((di, si))
                train_idx.append((dt, tk))

    def _mk(ds_samples, ds_idx):
        idx = pd.MultiIndex.from_tuples(ds_idx, names=["datetime", "instrument"])
        return _MasterDataset(cube, dates, tickers, ds_samples, idx)

    dl_train = _mk(train_samples, train_idx)
    dl_valid = _mk(valid_samples, valid_idx) if valid_samples else None
    print(f"  [MASTER] učni vzorci: {len(train_samples)}  "
          f"validacijski: {len(valid_samples)}  (N≈{N}/dan)")

    seed = int(mcfg.get("seed", 0))
    n_epoch = int(mcfg.get("epochs", 15))
    beta = float(mcfg.get("beta", 5))
    save_dir = tempfile.mkdtemp(prefix="master_us_")
    model = MASTERModel(
        d_feat=158, d_model=256, t_nhead=4, s_nhead=2,
        T_dropout_rate=0.5, S_dropout_rate=0.5, beta=beta,
        gate_input_start_index=158, gate_input_end_index=221,
        n_epochs=n_epoch, lr=float(mcfg.get("lr", 1e-5)), GPU=None, seed=seed,
        train_stop_loss_thred=float(mcfg.get("train_stop_loss_thred", -1.0)),
        save_path=save_dir, save_prefix="us",
    )
    print(f"  [MASTER] učenje ({n_epoch} epoh, beta={beta}, seed={seed}) ...")
    model.fit(dl_train, dl_valid)
    model.fitted = max(model.fitted, 0)   # zagotovi, da predict/inferenca deluje

    return {
        "_type": "master_us",
        "model": model,
        "cube": cube, "dates": dates, "tickers": tickers,
        "ohlcv": ohlcv, "mkt_ohlcv": mkt_ohlcv,
        "norm_stats": stats, "lookahead": lookahead,
        "device": model.device,
    }


def _raw_then_norm(ohlcv, mkt_ohlcv, tickers, lookahead, train_end):
    """Zgradi cube s surovimi značilkami, izračuna median/MAD IZ UČNIH vrstic,
    normira (RobustZScoreNorm, clip ±3, fillna 0). Vrne (cube, dates, (med,mad))."""
    all_dates = None
    for tk in tickers:
        idx = ohlcv[tk].index
        all_dates = idx if all_dates is None else all_dates.union(idx)
    dates = pd.DatetimeIndex(sorted(all_dates))
    D, N = len(dates), len(tickers)
    market = _market_features(mkt_ohlcv, dates)
    mkt_arr = market.values.astype(np.float32)

    cube = np.full((D, N, 222), np.nan, dtype=np.float32)
    for si, tk in enumerate(tickers):
        df = ohlcv[tk].reindex(dates)
        o, h, l, c, v = (df["open"], df["high"], df["low"],
                         df["close"], df["volume"])
        cube[:, si, :158] = _alpha158_one(o, h, l, c, v).values.astype(np.float32)
        cube[:, si, 158:221] = mkt_arr
        fwd = c.shift(-lookahead) / c.shift(-1) - 1.0
        cube[:, si, 221] = fwd.values.astype(np.float32)

    train_rows = dates <= train_end
    feat = cube[:, :, :221]
    flat_train = feat[train_rows].reshape(-1, 221)
    med = np.nanmedian(flat_train, axis=0)
    mad = np.nanmedian(np.abs(flat_train - med), axis=0)
    scale = 1.4826 * mad + _EPS
    feat = np.clip((feat - med) / scale, -3.0, 3.0)
    cube[:, :, :221] = np.nan_to_num(feat, nan=0.0)
    return cube, dates, (med, mad)


def get_mu_master_us(registry, df_hist, avail_tickers, lookahead):
    """Presečni μ (z-score) za trenutno okno. df_hist določa datum napovedi.
    Kavzalno: značilke uporabijo le podatke do df_hist.index[-1]."""
    import torch
    model = registry["model"]
    cube = registry["cube"]
    dates = registry["dates"]
    tickers = registry["tickers"]
    device = registry["device"]

    t_now = pd.Timestamp(df_hist.index[-1])
    pos = dates.searchsorted(t_now, side="right") - 1
    pos = int(min(max(pos, T_LOOKBACK - 1), len(dates) - 1))

    windows, cols = [], []
    for si, tk in enumerate(tickers):
        w = cube[pos - (T_LOOKBACK - 1):pos + 1, si, :221]   # [T,221]
        if w.shape[0] < T_LOOKBACK or np.all(cube[pos, si, :158] == 0):
            continue
        windows.append(w)
        cols.append(tk)
    if not windows:
        return np.zeros(len(avail_tickers), dtype=float)

    feat = torch.from_numpy(np.stack(windows)).float().to(device)   # [N,T,221]
    model.model.eval()
    with torch.no_grad():
        raw = model.model(feat).detach().cpu().numpy().ravel()      # [N]

    pred = pd.Series(raw, index=cols)
    # presečni z-score (kot pri drugih μ-only virih), poravnan na avail_tickers
    z = (pred - pred.mean()) / (pred.std() + _EPS)
    out = z.reindex(avail_tickers)
    out = out.fillna(0.0)
    return out.values.astype(float)
