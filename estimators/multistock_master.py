"""
estimators/multistock_master.py
===============================
MASTER-zvest cross-sectional multi-stock transformer kot return estimator za
walk-forward pipeline.

Znanstvene reference (za pisanje teze):
  • Arhitektura (market-guided cross-sectional attention):
      Li, T., Liu, Z., Shen, Y., Wang, X., Chen, H. & Zhu, S. (2024).
      "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting."
      Proc. AAAI 2024. arXiv:2312.15235.
  • Sorodni cross-sectional stock transformer (kontekst delnica↔trg):
      Yoo, J., Soun, Y., Park, Y.-c. & Kang, U. (2021). "Accurate Multivariate
      Stock Movement Prediction via Data-Axis Transformer with Multi-Level
      Contexts (DTML)." Proc. KDD 2021, 2037–2045.
  • Learning-to-rank izgubna funkcija (ListNet, cross-sectional rang donosov):
      Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F. & Li, H. (2007). "Learning to
      Rank: From Pairwise Approach to Listwise Approach." Proc. ICML 2007,
      129–136.

Zakaj tukaj (ne v experiments/signal_master.py): signal_master.py je HITER
IC-harness (dvostopenjska evalvacija na dveh oknih). Ta modul je PIPELINE
integracija — enak model, a s tremi razlikami, ki jih pipeline zahteva:

  1. ŠIROK učni univerzum: cross-sectional attention potrebuje bogate prereze
     (~195 imen), a pipeline poda le ~48 portfeljskih tikerjev. Modul si sam
     prenese širši univerzum (CANDIDATES ∪ portfelj) za TRENING; inferenca teče
     cross-stock nad 48 portfeljskimi tikerji (attention je permutacijsko
     ekvivariantna → poljuben N).
  2. VOLUMEN + TRG cache: model rabi log-volumen (per delnico) in tržne
     značilke (SPY ret + VIX + spread), a pipeline poda le Close cene. Ob
     TRENINGU prenesemo + shranimo volumen (portfeljski tikerji) in tržne serije
     čez cel razpon [train_start .. test_end]; inferenca reže cache do cur.
  3. SCALE-MATCH μ: model vrne cross-sectional SCORE (per-date-z rang), ne
     absolutnega donosa. Da se scenarija "Zgodovinski" in "Transformer"
     razlikujeta SAMO v μ-viru (kar izolira tezni prispevek), transformerjev μ
     afino reskaliramo, da se ujema s cross-sectional POVPREČJEM in STD
     istočasnega zgodovinskega (rolling sample mean) μ. Edina razlika ostane
     torej cross-sectional RANG delnic (transformer vs vzorčno povprečje).

Javni API (kliče run_benchmark.py + benchmark_core.py neposredno):
    reg = train_multistock(config, df_train, avail_tickers, lookahead)
    mu  = get_mu_multistock(reg, df_hist, avail_tickers, lookahead)
    cov = get_cov_multistock(reg, df_hist, avail_tickers, lookahead)

Model: 2 značilki (return + log-volumen) — PROVEN konfiguracija (dodatne
GKX-značilke izkazale regime-toksičnost v mean-rev oknu; glej memory /
experiments/signal_master.py). 3-seed ansambel.
"""
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# hiperparametri (usklajeni s proven signal_master.py)
H_DEFAULT   = 21
SEQ_LEN     = 45
N_CHUNKS, CHUNK = 5, 9                  # 5*9 = 45
TRAIN_STEP  = 2
EPOCHS, BATCH, LR, PATIENCE = 60, 16, 1e-3, 12
DMODEL, HEADS, DROPOUT, BETA = 32, 2, 0.2, 2.0
SEEDS = [42, 1, 7]
N_FEAT, N_MKT = 2, 3

# shared-representation covariance head (Part A)
# Privzete vrednosti; oba parametra sta config-vodena (return_estimator.n_factors,
# return_estimator.cov_lambda) → izbrana na VALIDACIJI, ne na testu.
N_FACTORS_DEFAULT = 8       # nizko-rangni faktorji za Σ = ββᵀ + diag(ψ)
COV_LAMBDA_DEFAULT = 0.5    # utež multi-task kovariančne NLL izgube
TOPW_ALPHA_DEFAULT = 0.25   # top-weighted listwise blend (Lever 1)

# širši univerzum pre-2010 large-capov za bogate cross-sectione (isti kot harness)
CANDIDATES = [
    "AAPL","MSFT","IBM","ORCL","CSCO","INTC","QCOM","TXN","ADI","MU","AMD","NVDA",
    "HPQ","GLW","ADBE","CRM","INTU","ADP","ACN","GOOGL","AMZN","EBAY","CMCSA",
    "DIS","VZ","T","TMUS",
    "JPM","BAC","WFC","C","GS","MS","USB","PNC","BK","AXP","COF","SCHW","BLK",
    "SPGI","CME","ICE","TRV","ALL","AIG","MET","PRU","AFL","CB","MMC","AON","PGR",
    "V","MA",
    "JNJ","PFE","MRK","ABT","BMY","AMGN","GILD","BIIB","LLY","TMO","DHR","MDT",
    "SYK","BDX","BSX","ISRG","CI","HUM","CNC","UNH","CVS","MCK","CAH","ZBH","BAX","A",
    "PG","KO","PEP","WMT","COST","MO","PM","CL","KMB","GIS","K","HSY","SYY",
    "KR","STZ","MKC","CLX","CHD","HRL","TSN","ADM","EL",
    "MCD","SBUX","NKE","HD","LOW","TGT","TJX","ROST","DG","DLTR","YUM","F","GM",
    "GPC","APTV","BBY","ORLY","AZO","MAR","HLT","GRMN",
    "GE","HON","MMM","CAT","DE","UNP","UPS","FDX","BA","LMT","RTX","NOC","GD",
    "EMR","ETN","ITW","PH","ROK","CMI","CSX","NSC","WM","RSG","PCAR","DOV","IR",
    "FAST","GWW","LUV","DAL",
    "XOM","CVX","COP","SLB","EOG","OXY","PSX","VLO","MPC","KMI","WMB","HAL","BKR",
    "DVN","HES","FANG",
    "LIN","APD","SHW","ECL","DD","DOW","NEM","FCX","NUE","VMC","MLM","PPG","IFF",
    "NEE","DUK","SO","D","AEP","EXC","XEL","SRE","PEG","ED","WEC","ES","PCG","EIX",
    "AMT","PLD","CCI","EQIX","PSA","O","SPG","WELL","VTR","AVB","EQR",
]


# model (MASTER-zvest, 2-feat)
class MasterLite(nn.Module):
    def __init__(self, n_feat, n_mkt, d_model, heads, seq_len, n_chunks,
                 n_factors=N_FACTORS_DEFAULT):
        super().__init__()
        self.n_chunks = n_chunks
        self.n_factors = n_factors
        self.in_proj = nn.Linear(n_feat, d_model)
        self.gate = nn.Linear(n_mkt, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        tenc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, DROPOUT,
                                          batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(tenc, 1)
        xenc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, DROPOUT,
                                          batch_first=True, norm_first=True)
        self.inter = nn.TransformerEncoder(xenc, 1)
        self.tagg = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        # shared-representation covariance head
        # Isti per-delnica embedding z → faktorske uteži β (nizko-rangni faktorji)
        # + idiosinkratska varianca ψ>0 → Σ = ββᵀ + diag(ψ) (SPD po konstrukciji;
        # neural factor model, FactorVAE/Deep-Risk-Model rod). μ-rang in Σ zdaj
        # izhajata iz ISTE reprezentacije → reši razkorak med modelom in optimizator.
        self.cov_norm = nn.LayerNorm(d_model)
        self.cov_beta = nn.Linear(d_model, n_factors)
        self.cov_psi = nn.Linear(d_model, 1)

    def embed(self, x, mkt):
        """Skupna per-delnica reprezentacija z (B,N,d) — pred glavo."""
        B, N, L, F = x.shape
        h = self.in_proj(x)
        d = h.shape[-1]
        g = torch.softmax(self.gate(mkt) / BETA, dim=-1) * d
        h = h * g[:, None, None, :]
        h = h.reshape(B * N, L, d) + self.pos
        h = self.temporal(h)
        h = h.reshape(B * N, self.n_chunks, CHUNK, d).mean(dim=2)
        h = h.reshape(B, N, self.n_chunks, d).permute(0, 2, 1, 3).reshape(B * self.n_chunks, N, d)
        h = self.inter(h)
        h = h.reshape(B, self.n_chunks, N, d).permute(0, 2, 1, 3)
        aw = torch.softmax(self.tagg(h).squeeze(-1), dim=-1)
        return (h * aw.unsqueeze(-1)).sum(dim=2)              # z: (B,N,d)

    def score(self, z):
        return self.head(z).squeeze(-1)

    def cov_params(self, z):
        """β (B,N,k) faktorske uteži in ψ (B,N)>0 idiosinkratska varianca."""
        c = self.cov_norm(z)
        beta = self.cov_beta(c)
        psi = torch.nn.functional.softplus(self.cov_psi(c)).squeeze(-1) + 1e-4
        return beta, psi

    def forward(self, x, mkt):
        return self.score(self.embed(x, mkt))


# utils
def rank_ic_loss(pred, tgt):
    pm = pred - pred.mean(dim=1, keepdim=True)
    tm = tgt - tgt.mean(dim=1, keepdim=True)
    pn = pm / (pm.std(dim=1, keepdim=True) + 1e-6)
    tn = tm / (tm.std(dim=1, keepdim=True) + 1e-6)
    return -(pn * tn).mean()


# LEVER 1: top-weighted listwise loss (ListNet, Cao et al. 2007)
# rank_ic_loss weights every cross-sectional rank EQUALLY → indifferent to the
# tail (the few big winners that drive Sharpe; Bessembinder 2018). ListNet top-1
# is top-weighted by construction: softmax(tgt) concentrates mass on the highest
# forward-return names, so the loss cares most about the winners. Blended 50/50
# with rank_ic_loss so global ordering (mean-reversion regime) stays intact and
# only tail emphasis is added. Only affects TRAINING gradients; inference/arch
# unchanged (zero deploy overhead).
TOPW_TAU = 1.0

def listnet_loss(pred, tgt, tau=TOPW_TAU):
    logp = torch.log_softmax(pred / tau, dim=1)
    q    = torch.softmax(tgt / tau, dim=1)
    return -(q * logp).sum(dim=1).mean()

def topw_loss(pred, tgt, alpha=TOPW_ALPHA_DEFAULT):
    # alpha=0 → pure rank-IC ranker (winner-concentration is handled at inference
    # by the Hedge μ-ensemble, estimators/mu_ensemble.py). Short-circuit so the
    # neutral base model pays no listnet cost. alpha>0 kept as an ablation lever.
    if alpha <= 0.0:
        return rank_ic_loss(pred, tgt)
    return (1.0 - alpha) * rank_ic_loss(pred, tgt) + alpha * listnet_loss(pred, tgt)


# shared-representation covariance NLL (Part A)
# Gaussov NLL realiziranih H-dnevnih donosov pod Σ = ββᵀ + diag(ψ). Woodbury +
# Cholesky → poceni za nizko-rangno+diagonalno strukturo (k≪N). To je MLE za
# Gaussa s časovno-spremenljivo kovarianco (dinamični faktorski model; FactorVAE,
# Gu-Kelly-Xiu rod). r = per-datum cross-sectional demean realiziranih fwd donosov.
def cov_nll(beta, psi, r):
    B, N, k = beta.shape
    Dinv = 1.0 / psi                                    # (B,N)
    bDinv = beta * Dinv.unsqueeze(-1)                   # (B,N,k)
    eye = torch.eye(k, device=beta.device).expand(B, k, k)
    M = eye + torch.einsum("bnk,bnj->bkj", bDinv, beta)  # I + βᵀD⁻¹β  (B,k,k)
    Dinv_r = Dinv * r                                   # (B,N)
    a = torch.einsum("bnk,bn->bk", beta, Dinv_r)        # βᵀD⁻¹r  (B,k)
    Lc = torch.linalg.cholesky(M)
    sol = torch.cholesky_solve(a.unsqueeze(-1), Lc).squeeze(-1)   # M⁻¹a
    quad = (r * Dinv_r).sum(-1) - (a * sol).sum(-1)     # rᵀΣ⁻¹r (Woodbury)
    logdet = torch.log(psi).sum(-1) + \
        2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (quad + logdet).mean()


def _download_market(dates, dl_start, dl_end):
    import yfinance as yf
    def series(tk):
        try:
            dd = yf.download(tk, start=dl_start, end=dl_end, auto_adjust=True, progress=False)
            s = dd["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s.reindex(dates).ffill().bfill()
        except Exception:
            return pd.Series(0.0, index=dates)
    spy = series("SPY"); vix = series("^VIX"); tnx = series("^TNX"); irx = series("^IRX")
    spy_v = spy.values.astype(np.float64)
    spy_ret = np.zeros(len(dates)); spy_ret[1:] = np.log(spy_v[1:] / spy_v[:-1] + 1e-12)
    return spy_ret, vix.values.astype(np.float64), (tnx - irx).values.astype(np.float64)


def per_date_z(Y):
    mu = np.nanmean(Y, axis=1, keepdims=True)
    sg = np.nanstd(Y, axis=1, keepdims=True) + 1e-7
    return ((Y - mu) / sg).astype(np.float32)


def size_demean(fwd_rows, size_rows, n_groups=3):
    out = np.array(fwd_rows, dtype=np.float64)
    for i in range(out.shape[0]):
        s = size_rows[i]; y = out[i]
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.sum() < n_groups * 2:
            out[i] = y - np.nanmedian(y)
            continue
        q = np.quantile(s[ok], np.linspace(0, 1, n_groups + 1)[1:-1])
        grp = np.digitize(s, q)
        for g in np.unique(grp[ok]):
            m = ok & (grp == g)
            out[i][m] = y[m] - np.median(y[m])
    return out.astype(np.float32)


# training
def _fit_seeds(Xt, Mt, Yt, Rt, Xv, Mv, Yv, Rv, ntr,
               alpha, cov_lambda, n_factors):
    """Natrenira 3-seed ansambel (vsak seed od začetka)."""
    states = []
    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = MasterLite(N_FEAT, N_MKT, DMODEL, HEADS, SEQ_LEN, N_CHUNKS, n_factors)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)
        idx = np.arange(ntr); best, best_state, wait = float("inf"), None, 0
        for ep in range(EPOCHS):
            model.train(); np.random.shuffle(idx)
            for s in range(0, ntr, BATCH):
                b = idx[s:s + BATCH]
                opt.zero_grad()
                z = model.embed(Xt[b], Mt[b])
                rloss = topw_loss(model.score(z), Yt[b], alpha)
                beta, psi = model.cov_params(z)
                closs = cov_nll(beta, psi, Rt[b])
                loss = rloss + cov_lambda * closs
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            # Zgodnje ustavljanje SAMO na μ (return) validaciji → μ-izbira NEspremenjena;
            # kovariančna glava je skupno-reprezentacijski regularizator (free rider).
            model.eval()
            with torch.no_grad():
                vl = topw_loss(model(Xv, Mv), Yv, alpha).item()
            sch.step(vl)
            if vl < best - 1e-5:
                best = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        states.append(best_state if best_state else model.state_dict())
        model.load_state_dict(states[-1]); model.eval()
        with torch.no_grad():
            b0, p0 = model.cov_params(model.embed(Xv, Mv))
            vnll = cov_nll(b0, p0, Rv).item()
        print(f"  [MASTER-lite] seed {seed:2d}: val_topw_loss={best:+.4f}  val_cov_nll={vnll:+.4f}")
    return states


def _assemble_registry(config, avail_tickers, lookahead,
                       close_all, vol_all, mkt_raw, cutoff):
    """Zgradi registry iz prenesenih surovih podatkov (close_all/vol_all/mkt_raw)
    pri danem `cutoff` (koliko zgodovine model vidi). Model se uči ENKRAT."""
    mc = config.get("return_estimator", {})
    H  = lookahead
    alpha      = float(mc.get("topw_alpha", TOPW_ALPHA_DEFAULT))
    cov_lambda = float(mc.get("cov_lambda", COV_LAMBDA_DEFAULT))
    n_factors  = int(mc.get("n_factors",  N_FACTORS_DEFAULT))

    # UČNI univerzum: imena s polno zgodovino do cutoff (za bogate prereze)
    tr_slice = close_all.index <= cutoff
    train_names = [c for c in close_all.columns
                   if close_all.loc[tr_slice, c].notna().all()
                   and vol_all.loc[tr_slice, c].notna().any()]
    ctr = close_all[train_names].ffill()
    vtr = vol_all[train_names].ffill().fillna(1.0)
    print(f"  [MASTER-lite] učni univerzum: {len(train_names)} imen × "
          f"{int(tr_slice.sum())} dni ≤ {cutoff.date()}")

    dates = ctr.index
    C = ctr.values.astype(np.float64); Vv = vtr.values.astype(np.float64)
    T, N = C.shape
    rr = np.zeros_like(C); rr[1:] = np.log(C[1:] / C[:-1] + 1e-12)
    lv = np.log(np.clip(Vv, 1.0, None))
    dv = np.log(np.clip(C * Vv, 1.0, None))
    spy_ret = mkt_raw["spy"].values; vix = mkt_raw["vix"].values; spr = mkt_raw["spr"].values
    tr_rows = (dates <= cutoff).values if hasattr(dates <= cutoff, "values") else (dates <= cutoff)

    def zc(a):
        mu = a[tr_rows].mean(0); sg = a[tr_rows].std(0) + 1e-7
        return (a - mu) / sg
    def zg(a):
        mu = a[tr_rows].mean(); sg = a[tr_rows].std() + 1e-7
        return (a - mu) / sg, float(mu), float(sg)

    rn, lvn = zc(rr), zc(lv)
    spyn, m_spy, s_spy = zg(spy_ret)
    vixn, m_vix, s_vix = zg(vix)
    sprn, m_spr, s_spr = zg(spr)
    mkt = np.stack([spyn, vixn, sprn], axis=1)
    feats = np.stack([rn, lvn], axis=-1).astype(np.float32)       # (T,N,2)
    fwd = np.full((T, N), np.nan); fwd[:T - H] = np.log(C[H:] / C[:T - H] + 1e-12)

    tr_P = np.array([p for p in range(SEQ_LEN, T - H, TRAIN_STEP)
                     if dates[p + H] <= cutoff], dtype=int)

    def gather_X(P):
        kidx = P[:, None] - SEQ_LEN + 1 + np.arange(SEQ_LEN)
        return np.transpose(feats[kidx], (0, 2, 1, 3)).astype(np.float32)  # (nP,N,L,2)
    def gather_M(P):
        kidx = P[:, None] - SEQ_LEN + 1 + np.arange(SEQ_LEN)
        return mkt[kidx].mean(axis=1).astype(np.float32)

    Xtr = gather_X(tr_P); Mtr = gather_M(tr_P)
    Ytr = per_date_z(size_demean(fwd[tr_P], dv[tr_P]))
    fwd_raw = fwd[tr_P]
    Rtr = (fwd_raw - np.nanmean(fwd_raw, axis=1, keepdims=True)).astype(np.float32)
    ok = (np.isfinite(Xtr).all(axis=(1, 2, 3)) & np.isfinite(Ytr).all(axis=1)
          & np.isfinite(Mtr).all(axis=1) & np.isfinite(Rtr).all(axis=1))
    Xtr, Mtr, Ytr, Rtr = Xtr[ok], Mtr[ok], Ytr[ok], Rtr[ok]
    n = len(Xtr); nval = max(BATCH, int(0.15 * n)); ntr = n - nval
    Xt = torch.tensor(Xtr[:ntr]); Mt = torch.tensor(Mtr[:ntr]); Yt = torch.tensor(Ytr[:ntr]); Rt = torch.tensor(Rtr[:ntr])
    Xv = torch.tensor(Xtr[ntr:]); Mv = torch.tensor(Mtr[ntr:]); Yv = torch.tensor(Ytr[ntr:]); Rv = torch.tensor(Rtr[ntr:])
    print(f"  [MASTER-lite] trening: {ntr} cross-sectionov (val {nval}), N={N}")

    states = _fit_seeds(Xt, Mt, Yt, Rt, Xv, Mv, Yv, Rv, ntr,
                        alpha, cov_lambda, n_factors)

    # per-asset statistika za PORTFELJSKE tikerje do cutoff (za inferenčno normo)
    ca = close_all.ffill(); va = vol_all.ffill().fillna(1.0)
    ptr = ca.index <= cutoff
    ret_mu = {}; ret_sig = {}; lv_mu = {}; lv_sig = {}
    for tk in avail_tickers:
        if tk in ca.columns:
            p = ca.loc[ptr, tk].values.astype(np.float64)
            r = np.diff(np.log(p + 1e-12)) if len(p) > 1 else np.array([0.0])
            lvt = np.log(np.clip(va.loc[ptr, tk].values.astype(np.float64), 1.0, None))
            ret_mu[tk] = float(np.nanmean(r)); ret_sig[tk] = float(np.nanstd(r) + 1e-7)
            lv_mu[tk] = float(np.nanmean(lvt)); lv_sig[tk] = float(np.nanstd(lvt) + 1e-7)
        else:
            ret_mu[tk] = 0.0; ret_sig[tk] = 1.0; lv_mu[tk] = 0.0; lv_sig[tk] = 1.0

    # market cache (normirana serija čez cel razpon) za inferenco
    mkt_z = np.stack([(mkt_raw["spy"].values - m_spy) / s_spy,
                      (mkt_raw["vix"].values - m_vix) / s_vix,
                      (mkt_raw["spr"].values - m_spr) / s_spr], axis=1)
    mkt_df = pd.DataFrame(mkt_z, index=ca.index, columns=["spy", "vix", "spr"])
    vol_port = va.reindex(columns=avail_tickers)                 # volumen cache za portfelj

    return {
        "_type":     "multistock",
        "_tickers":  avail_tickers,
        "_seq_len":  SEQ_LEN,
        "_has_cov":  True,
        "states":    states,
        "arch":      dict(n_feat=N_FEAT, n_mkt=N_MKT, d_model=DMODEL, heads=HEADS,
                          seq_len=SEQ_LEN, n_chunks=N_CHUNKS, n_factors=n_factors),
        "ret_mu": ret_mu, "ret_sig": ret_sig, "lv_mu": lv_mu, "lv_sig": lv_sig,
        "mkt_df": mkt_df, "vol_port": vol_port,
    }


def train_multistock(config: dict,
                     df_train: pd.DataFrame,
                     avail_tickers: list,
                     lookahead: int) -> dict:
    """Prenese širok univerzum, natrenira 3-seed MASTER-lite ansambel na zgodovini
    ≤ train_end, in shrani cache (volumen+trg) za inferenco. Vrne registry."""
    import yfinance as yf

    mc         = config.get("return_estimator", {})
    tc         = config.get("test_config", {})
    train_start = mc.get("start_date", "2015-01-01")
    train_end   = mc.get("end_date",   "2023-01-01")
    test_end    = tc.get("end_date",   "2025-01-01")

    alpha      = float(mc.get("topw_alpha", TOPW_ALPHA_DEFAULT))
    cov_lambda = float(mc.get("cov_lambda", COV_LAMBDA_DEFAULT))
    n_factors  = int(mc.get("n_factors",  N_FACTORS_DEFAULT))
    print(f"  [MASTER-lite] topw_alpha={alpha:.3f}  cov_lambda={cov_lambda:.3f}  n_factors={n_factors}")

    cutoff = pd.Timestamp(train_end)
    # razpon prenosa: buffer pred train_start za SEQ_LEN+returns, do test_end za inferenco
    dl_start = (pd.Timestamp(train_start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    dl_end   = (pd.Timestamp(test_end)    + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    if mc.get("candidates_universe", True):
        universe = sorted(set(CANDIDATES) | set(avail_tickers))
    else:
        universe = sorted(set(avail_tickers))
    print(f"  [MASTER-lite] Prenos {len(universe)} kandidatov (učni univerzum) + trg {dl_start}→{dl_end} ...")
    raw = yf.download(universe, start=dl_start, end=dl_end, auto_adjust=True, progress=False)
    close_all = raw["Close"].copy(); vol_all = raw["Volume"].copy()

    # tržne serije čez cel razpon (poravnane na close_all.index) za inferenco
    spy_ret, vix, spr = _download_market(close_all.index, dl_start, dl_end)
    mkt_raw = pd.DataFrame({"spy": spy_ret, "vix": vix, "spr": spr}, index=close_all.index)

    return _assemble_registry(config, avail_tickers, lookahead,
                              close_all, vol_all, mkt_raw, cutoff)


# inference
def _build_inputs(registry: dict, df_hist: pd.DataFrame,
                  avail_tickers: list):
    """Zgradi model vhode za portfeljske tikerje: Xt (1,N,L,2), Mt (1,3), valid.
    Skupno za μ (get_mu) in Σ (get_cov) inferenco → oba iz iste reprezentacije."""
    seq_len = registry.get("_seq_len", SEQ_LEN)
    cur = df_hist.index[-1]
    ret_mu = registry["ret_mu"]; ret_sig = registry["ret_sig"]
    lv_mu = registry["lv_mu"]; lv_sig = registry["lv_sig"]
    vol_port = registry["vol_port"]; mkt_df = registry["mkt_df"]

    close = df_hist[avail_tickers].ffill()
    vol = vol_port.reindex(close.index).ffill().fillna(1.0)

    X = np.zeros((len(avail_tickers), seq_len, N_FEAT), dtype=np.float32)
    valid = np.zeros(len(avail_tickers), dtype=bool)
    for i, tk in enumerate(avail_tickers):
        p = close[tk].values.astype(np.float64)
        if np.isfinite(p).sum() < seq_len + 2:
            continue
        r = np.diff(np.log(p + 1e-12))                           # dnevni log-return
        lvt = np.log(np.clip(vol[tk].values.astype(np.float64), 1.0, None))
        if len(r) < seq_len or len(lvt) < seq_len + 1:
            continue
        rn = (r[-seq_len:] - ret_mu[tk]) / ret_sig[tk]
        lvn = (lvt[-seq_len:] - lv_mu[tk]) / lv_sig[tk]
        X[i, :, 0] = rn; X[i, :, 1] = lvn
        valid[i] = True

    msub = mkt_df[mkt_df.index <= cur].tail(seq_len)
    M = msub.mean(axis=0).values.astype(np.float32) if len(msub) else np.zeros(N_MKT, np.float32)
    M = np.nan_to_num(M)
    Xt = torch.tensor(np.nan_to_num(X)).unsqueeze(0)             # (1,N,L,2)
    Mt = torch.tensor(M).unsqueeze(0)                            # (1,3)
    return Xt, Mt, valid


def _load_models(registry: dict):
    arch = registry["arch"]
    models = []
    for state in registry["states"]:
        m = MasterLite(arch["n_feat"], arch["n_mkt"], arch["d_model"],
                       arch["heads"], arch["seq_len"], arch["n_chunks"],
                       arch.get("n_factors", N_FACTORS_DEFAULT))
        m.load_state_dict(state); m.eval()
        models.append(m)
    return models


def get_mu_multistock(registry: dict,
                      df_hist: pd.DataFrame,
                      avail_tickers: list,
                      lookahead: int) -> np.ndarray:
    """Cross-stock inferenca nad portfeljem + 3-seed ansambel; μ scale-matchan
    na istočasni zgodovinski (rolling sample mean) μ."""
    Xt, Mt, valid = _build_inputs(registry, df_hist, avail_tickers)

    # 3-seed ansambel
    preds = []
    for model in _load_models(registry):
        with torch.no_grad():
            preds.append(model(Xt, Mt).squeeze(0).cpu().numpy())
    score = np.mean(preds, axis=0)                               # (N,) cross-sectional score

    # scale-match na istočasni zgodovinski μ (izolira SAMO μ-rang)
    seq_cfg = registry.get("_seq_len", SEQ_LEN)
    roll_window = max(seq_cfg, 21)
    hist_rets = df_hist[avail_tickers].pct_change().dropna().tail(roll_window)
    mu_hist = hist_rets.mean().values * lookahead
    m0 = float(np.nanmean(mu_hist)); s0 = float(np.nanstd(mu_hist))

    sc = score.copy()
    sc[~valid] = np.nan
    zc_mu = np.nanmean(sc); zc_sd = np.nanstd(sc) + 1e-9
    z = (sc - zc_mu) / zc_sd
    mu = z * s0 + m0
    mu[~valid] = m0                                              # nevtralno za nepokrite
    return np.nan_to_num(mu, nan=m0).astype(np.float64)


def get_cov_multistock(registry: dict,
                       df_hist: pd.DataFrame,
                       avail_tickers: list,
                       lookahead: int) -> np.ndarray:
    """Skupno-reprezentacijska Σ iz kovariančne glave: Σ = ββᵀ + diag(ψ) (SPD),
    3-seed povprečje. Enota = H-dnevna donosna kovarianca (ujema μ * lookahead in
    sample Σ = daily_cov * lookahead). Enak vhod kot get_mu → μ-rang in Σ iz iste
    reprezentacije. Fallback na zgodovinsko sample Σ, če model nima kov. glave."""
    N = len(avail_tickers)

    # Zgodovinska sample Σ (H-dnevna) — za fallback in za nepokrite tikerje.
    hist_rets = df_hist[avail_tickers].pct_change().dropna()
    if len(hist_rets) >= N + 2:
        sample_cov = np.cov(hist_rets.values, rowvar=False) * lookahead
    else:
        sample_cov = np.eye(N) * (float(np.nanvar(hist_rets.values)) + 1e-6) * lookahead

    if not registry.get("_has_cov", False):
        return _ensure_spd(sample_cov)

    Xt, Mt, valid = _build_inputs(registry, df_hist, avail_tickers)
    sigmas = []
    for model in _load_models(registry):
        with torch.no_grad():
            beta, psi = model.cov_params(model.embed(Xt, Mt))
        b = beta.squeeze(0).cpu().numpy()                       # (N,k)
        p = psi.squeeze(0).cpu().numpy()                        # (N,)
        sigmas.append(b @ b.T + np.diag(p))
    cov = np.mean(sigmas, axis=0)                               # (N,N) H-dnevna Σ

    # Nepokriti tikerji (premalo zgodovine): vzemi sample-Σ vrstico/stolpec.
    if not valid.all():
        bad = ~valid
        cov[bad, :] = sample_cov[bad, :]
        cov[:, bad] = sample_cov[:, bad]

    return _ensure_spd(cov)


def _ensure_spd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    w = np.linalg.eigvalsh(cov)
    if w[0] < eps:
        cov = cov + np.eye(cov.shape[0]) * (abs(w[0]) + eps)
    return cov
