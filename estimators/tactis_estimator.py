"""
estimators/tactis_estimator.py

TACTiS-2 (Ashok, Marcotte, Zantedeschi, Chapados, Drouin; ICLR 2024) kot CELOTEN
napovedni vir za OBA momenta — μ IN Σ hkrati.

Ideja (glej razpravo v tezi): transformer-attentional-copula model napove SKUPNO
(joint) porazdelitev H-dnevnih donosov vseh sredstev naenkrat. Iz vzorcev te
napovedane porazdelitve izračunamo:
    μ = vzorčno povprečje kumulativnega H-dnevnega donosa (po sredstvih),
    Σ = vzorčna kovarianca istih kumulativnih donosov (med sredstvi).
Nič se ne prevzame iz zgodovinske vzorčne ocene (kot pri scale-matchingu
transformerja MASTER-lite) — μ in Σ sta oba neposredna izhoda modela.

Uporabljamo OFF-THE-SHELF paket `tactis` (pip install tactis), model kličemo
neposredno prek razreda tactis.model.tactis.TACTiS (brez gluonts ovojnice).
Učenje je dvostopenjsko (curriculum iz izvirnika): najprej robne porazdelitve
(flow / marginal_logdet), nato kopula (copula_loss).

Registry vmesnik je usklajen z ostalimi estimatorji:
    train_tactis(config, df_train, avail_tickers, lookahead) -> registry (dict)
    get_musigma_tactis(registry, df_hist, avail_tickers, lookahead) -> (mu, cov)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from tactis.model.tactis import TACTiS


# ----------------------------------------------------------------------------- #
#  Gradnja mreže — hiperparametri po nizko-dimenzionalnem demu izvirnika
#  (demo/random_walk.ipynb), primerni za našo univerzo nekaj deset sredstev.
# ----------------------------------------------------------------------------- #
def _make_net(num_series: int, device: torch.device, hp: dict) -> TACTiS:
    emb  = int(hp.get("series_embedding_dim", 5))
    ienc = int(hp.get("input_encoder_layers", 3))
    a_layers = int(hp.get("attention_layers", 2))
    a_heads  = int(hp.get("attention_heads", 3))
    a_dim    = int(hp.get("attention_dim", 16))
    a_ff     = int(hp.get("attention_feedforward_dim", 16))
    temporal = {
        "attention_layers": a_layers,
        "attention_heads":  a_heads,
        "attention_dim":    a_dim,
        "attention_feedforward_dim": a_ff,
        "dropout": 0.0,
    }
    net = TACTiS(
        num_series=num_series,
        flow_series_embedding_dim=emb,
        copula_series_embedding_dim=emb,
        flow_input_encoder_layers=ienc,
        copula_input_encoder_layers=ienc,
        input_encoding_normalization=True,
        data_normalization="standardization",
        loss_normalization="series",
        positional_encoding={"dropout": 0.0},
        flow_temporal_encoder=dict(temporal),
        copula_temporal_encoder=dict(temporal),
        copula_decoder={
            "min_u": 0.01,
            "max_u": 0.99,
            "attentional_copula": {
                "attention_heads": a_heads,
                "attention_layers": a_layers,
                "attention_dim": a_dim,
                "mlp_layers": 2,
                "mlp_dim": 16,
                "resolution": 20,
            },
            "dsf_marginal": {
                "mlp_layers": 2,
                "mlp_dim": 8,
                "flow_layers": 2,
                "flow_hid_dim": 8,
            },
        },
    )
    return net.to(device)


def _returns_matrix(df: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    """
    [N, T] matrika dnevnih (enostavnih) donosov, poravnana na `tickers`.

    Manjkajoče vrednosti (sredstva s poznejšim IPO — TSLA 2010, META 2012 —
    nimajo donosov na začetku učnega obdobja) zapolnimo s PRESEČNIM POVPREČJEM
    donosov tistega dne (nevtralna imputacija: manjkajoče sredstvo se obnaša kot
    trg). dropna(how="any") bi učni niz skrčil na skupno presečišče (nekaj deset
    dni). Fill z 0 pa bi ustvaril konstantne (ničelne) pred-IPO odseke z ničelno
    varianco → TACTiS-ova per-serijska standardizacija deli z ~0 → numerični
    razpad. Presečno-povprečna imputacija ohrani pravokotno matriko čez celotno
    zgodovino BREZ degeneriranih odsekov. Blaga winsorizacija (±50 %) omeji
    ekstreme (npr. kriza 2008), da flow-veja ostane numerično stabilna.
    """
    rets = (df[tickers].pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .iloc[1:])                              # prva vrstica pct_change je vsa-NaN
    row_mean = rets.mean(axis=1)                    # presečno povprečje po dnevih (skipna)
    rets = rets.apply(lambda col: col.fillna(row_mean), axis=0)
    rets = rets.fillna(0.0)                         # dnevi brez katerega koli sredstva
    rets = rets.clip(-0.5, 0.5)                     # blaga winsorizacija
    return rets.values.T.astype(np.float32)          # [N, T]


def _sample_windows(data: np.ndarray, batch_size: int, hist_len: int,
                    pred_len: int, device: torch.device, rng: np.random.Generator):
    """Naključni (hist, pred) izseki iz [N, T] za en učni korak."""
    N, T = data.shape
    max_idx = T - (hist_len + pred_len)
    hist, pred = [], []
    for _ in range(batch_size):
        i = int(rng.integers(0, max_idx + 1))
        hist.append(data[:, i:i + hist_len])
        pred.append(data[:, i + hist_len:i + hist_len + pred_len])
    hist_value = torch.tensor(np.stack(hist), device=device)   # [B, N, hist]
    pred_value = torch.tensor(np.stack(pred), device=device)   # [B, N, pred]
    B = batch_size
    hist_time = torch.arange(0, hist_len, device=device)[None, :].expand(B, -1)
    pred_time = torch.arange(hist_len, hist_len + pred_len,
                             device=device)[None, :].expand(B, -1)
    return hist_time, hist_value, pred_time, pred_value


# ----------------------------------------------------------------------------- #
#  Učenje (dvostopenjski curriculum)
# ----------------------------------------------------------------------------- #
def train_tactis(config: dict, df_train: pd.DataFrame,
                 avail_tickers: list[str], lookahead: int) -> dict:
    tcfg = config.get("evaluation", {}).get("tactis", {}) or {}
    seq_len = int(config.get("return_estimator", {}).get("sequence_length", 45))
    hist_len = int(tcfg.get("hist_length", max(seq_len, 45)))
    pred_len = int(lookahead)

    device = torch.device(tcfg.get("device", "cpu"))
    seed   = int(tcfg.get("seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    e1  = int(tcfg.get("epochs_stage1", 8))
    e2  = int(tcfg.get("epochs_stage2", 15))
    nb  = int(tcfg.get("batches_per_epoch", 20))
    bs  = int(tcfg.get("batch_size", 32))

    data = _returns_matrix(df_train, avail_tickers)              # [N, T]
    N = data.shape[0]
    if data.shape[1] < hist_len + pred_len + 2:
        raise ValueError(f"premalo učnih dni za TACTiS: {data.shape[1]}")

    net = _make_net(N, device, tcfg)

    # --- Stopnja 1: robne porazdelitve (flow) ---
    net.set_stage(1)
    opt = torch.optim.RMSprop(net.parameters(),
                              lr=float(tcfg.get("lr_stage1", 1e-3)))
    net.train()
    for ep in range(e1):
        run = 0.0
        for _ in range(nb):
            ht, hv, pt, pv = _sample_windows(data, bs, hist_len, pred_len, device, rng)
            opt.zero_grad()
            _ = net.loss(ht, hv, pt, pv)
            loss = (-net.marginal_logdet).mean()
            loss.backward()
            opt.step()
            run += float(loss.item())
        print(f"  [TACTiS s1] epoch {ep+1}/{e1}  loss={run/nb:.4f}")

    # --- Stopnja 2: kopula ---
    net.set_stage(2)
    net.initialize_stage2()
    net.to(device)
    stage2_names = ["copula_series_encoder", "copula_time_encoding",
                    "copula_input_encoder", "copula_encoder", "decoder.copula"]
    params2 = [p for n, p in net.named_parameters()
               if any(s in n for s in stage2_names)]
    opt = torch.optim.RMSprop(params2, lr=float(tcfg.get("lr_stage2", 1e-4)))
    for ep in range(e2):
        run = 0.0
        for _ in range(nb):
            ht, hv, pt, pv = _sample_windows(data, bs, hist_len, pred_len, device, rng)
            opt.zero_grad()
            _ = net.loss(ht, hv, pt, pv)
            loss = net.copula_loss.mean()
            loss.backward()
            opt.step()
            run += float(loss.item())
        print(f"  [TACTiS s2] epoch {ep+1}/{e2}  loss={run/nb:.4f}")

    net.eval()
    return {
        "_type": "tactis",
        "net": net,
        "device": device,
        "hist_len": hist_len,
        "pred_len": pred_len,
        "num_samples": int(tcfg.get("num_samples", 500)),
        "tickers": list(avail_tickers),
        # Kako iz vzorcev izpeljemo momenta (privzeto oba izboljšava vključena):
        #   downside_cov  — Σ iz DOWNSIDE (semi)kovariance (kaznuje sočasne padce),
        #                   sled poravnana na polno kovarianco (ohrani raven tveganja).
        #   confidence_mu — μ-odkloni skalirani z zaupanjem (1/razpršenost vzorcev):
        #                   zanesljivejša sredstva dobijo večjo utež, razpršenost μ ostane.
        "downside_cov":  bool(tcfg.get("downside_cov", True)),
        "confidence_mu": bool(tcfg.get("confidence_mu", True)),
    }


# ----------------------------------------------------------------------------- #
#  Sklepanje: μ in Σ iz vzorcev napovedane skupne porazdelitve
# ----------------------------------------------------------------------------- #
def _spd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


def _moments_from_cum(cum: np.ndarray, downside_cov: bool, confidence_mu: bool):
    """
    Iz [N, S] kumulativnih H-dnevnih donosov vzorcev izpelji (μ, Σ).
      confidence_mu (št. 2): presečne odklone μ pomnožimo z zaupanjem
        (1/razpršenost vzorcev, normirano na povprečje 1); presečno razpršenost μ
        NATANČNO ohranimo (spremeni se le OBLIKA, ne skala).
      downside_cov (št. 1): Σ iz NEGATIVNIH odklonov vzorcev (d@d'/S, PSD po
        konstrukciji), sled poravnana na polno kovarianco (ohrani raven tveganja);
        tvegani člen w'Σw tako kaznuje sočasne PADCE, ne pa sočasne rasti.
    """
    S = cum.shape[1]
    mu_raw = cum.mean(axis=1)
    spread = cum.std(axis=1) + 1e-12
    mu_bar = float(mu_raw.mean())
    dev = mu_raw - mu_bar
    if confidence_mu:
        conf = (1.0 / spread); conf = conf / conf.mean()
        dev2 = dev * conf
        dev2 = dev2 - dev2.mean()
        dev2 = dev2 * (dev.std() / (dev2.std() + 1e-12))    # ohrani presečno razpršenost
        dev = dev2
    mu = (mu_bar + dev).astype(float)

    full = np.cov(cum)
    if downside_cov:
        d = np.minimum(cum - cum.mean(axis=1, keepdims=True), 0.0)
        semi = (d @ d.T) / float(S)
        tr_full, tr_semi = np.trace(full), np.trace(semi)
        if tr_semi > 1e-18:
            semi = semi * (tr_full / tr_semi)
        cov = _spd(semi).astype(float)
    else:
        cov = _spd(full).astype(float)
    return mu, cov


@torch.no_grad()
def _sample_cum(registry: dict, df_hist: pd.DataFrame,
                avail_tickers: list[str], lookahead: int) -> np.ndarray:
    """Vzorči skupno napovedano porazdelitev → [N, S] kumulativnih H-dnevnih donosov."""
    net = registry["net"]; device = registry["device"]
    hist_len = registry["hist_len"]; pred_len = int(lookahead)
    S = int(registry["num_samples"])
    data = _returns_matrix(df_hist, avail_tickers)
    hist = data[:, -hist_len:]
    hv = torch.tensor(hist[None, :, :], device=device)
    ht = torch.arange(0, hist_len, device=device)[None, :]
    pt = torch.arange(hist_len, hist_len + pred_len, device=device)[None, :]
    s = net.sample(S, ht, hv, pt)[0].cpu().numpy()          # [N, pred, S]
    return s.sum(axis=1)                                    # [N, S]


def get_musigma_tactis(registry: dict, df_hist: pd.DataFrame,
                       avail_tickers: list[str], lookahead: int):
    """(μ, Σ) po nastavitvah registryja (downside_cov / confidence_mu)."""
    cum = _sample_cum(registry, df_hist, avail_tickers, lookahead)
    return _moments_from_cum(cum, registry.get("downside_cov", True),
                             registry.get("confidence_mu", True))


def get_musigma_tactis_variants(registry: dict, df_hist: pd.DataFrame,
                                avail_tickers: list[str], lookahead: int) -> dict:
    """
    Parni A/B: iz ISTIH vzorcev (en sampling) izpelji vse 4 recepte za (μ, Σ).
    Ker delijo isti (naučen) model IN iste vzorce, se razlikujejo SAMO po receptu
    → primerjava je brez šuma iz ponovnega učenja.
      plain = povprečje μ + polna Σ
      ds    = + downside Σ (št. 1)
      cf    = + confidence μ (št. 2)
      both  = št. 1 + št. 2
    """
    cum = _sample_cum(registry, df_hist, avail_tickers, lookahead)
    return {
        "plain": _moments_from_cum(cum, False, False),
        "ds":    _moments_from_cum(cum, True,  False),
        "cf":    _moments_from_cum(cum, False, True),
        "both":  _moments_from_cum(cum, True,  True),
    }
