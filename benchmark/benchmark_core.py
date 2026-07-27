"""
benchmark/benchmark_core.py
============================
Walk-forward benchmark — samo orchestracija.
Return μ/Σ logika je v estimators/multistock_master.py; baseline Σ je
vzorčna kovarianca (_sample_cov spodaj).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Baseline kovarianca

def _sample_cov(returns: np.ndarray, lookahead: int, eps: float = 1e-8) -> np.ndarray:
    """Vzorčna kovariančna matrika, skalirana z lookahead (PD-popravljena)."""
    cov = np.cov(returns, rowvar=False)
    w = np.linalg.eigvalsh(cov)
    if w[0] < eps:
        cov = cov + np.eye(cov.shape[0]) * (abs(w[0]) + eps)
    return cov * lookahead


def _ledoit_wolf_cov(returns: np.ndarray, lookahead: int, eps: float = 1e-8) -> np.ndarray:
    """Ledoit-Wolf skrčena (shrinkage) kovarianca — MOČAN klasični Σ baseline.

    Referenca (za pisanje teze):
      Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional
      covariance matrices." Journal of Multivariate Analysis 88(2):365–411.

    Vzorčna kovarianca je v končnih vzorcih slabo pogojena (znano; zato obstaja
    shrinkage). Poštena primerjava mora modelsko Σ meriti proti TEJ, ne le proti
    vzorčni Σ. Skalirano z lookahead (ujema _sample_cov in μ * lookahead)."""
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(np.asarray(returns, dtype=float)).covariance_
    except Exception:
        cov = np.cov(returns, rowvar=False)   # fallback
    w = np.linalg.eigvalsh(cov)
    if w[0] < eps:
        cov = cov + np.eye(cov.shape[0]) * (abs(w[0]) + eps)
    return cov * lookahead


# C++ optimizer

def find_binary() -> str:
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    for d in [project_root,
              os.path.join(project_root, "portfolio_optimizers"),
              os.getcwd()]:
        for name in ("portfolio_optimizer", "portfolio_optimization"):
            p = os.path.join(d, name)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p
    for name in ("portfolio_optimizer", "portfolio_optimization"):
        found = shutil.which(name)
        if found:
            return found
    return os.path.join(project_root, "portfolio_optimizers", "portfolio_optimizer")


def run_optimizer(base_config: dict, tickers: list, algo: str,
                  mu: list, cov_flat: list,
                  cov_selection_flat: list = None,
                  seed: int = None,
                  risk_parameter: float = None,
                  eps: list = None,
                  delta: list = None,
                  binary: str = "") -> dict:
    cfg = deepcopy(base_config)
    cfg["run_settings"]["optimizer_type"] = algo
    if seed is not None:
        cfg["optimizer_config"]["common"]["seed"] = seed
    if risk_parameter is not None:
        cfg["optimizer_config"]["common"]["risk_parameter"] = float(risk_parameter)

    n      = len(tickers)
    bridge = {"n": n, "tickers": tickers, "mu": mu, "cov": cov_flat}
    # Per-asset εᵢ in δᵢ (Chang et al. 2000, Eq. 12)
    # Če nista podani, C++ fallbacka na globalen w_min/w_max
    if eps is not None:
        bridge["eps"] = eps
    if delta is not None:
        bridge["delta"] = delta
    if cov_selection_flat is not None:
        bridge["cov_selection"] = cov_selection_flat

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path    = os.path.join(tmpdir, "cfg.json")
        bridge_path = os.path.join(tmpdir, "bridge.json")
        cfg["run_settings"]["data_bridge_file"] = bridge_path
        with open(cfg_path,    "w") as f: json.dump(cfg, f)
        with open(bridge_path, "w") as f: json.dump(bridge, f)

        t0     = time.perf_counter()
        result = subprocess.run(
            [os.path.abspath(binary), cfg_path],
            capture_output=True, text=True, cwd=tmpdir
        )
        elapsed = time.perf_counter() - t0

    weights = np.zeros(n)
    for line in result.stdout.splitlines():
        parts  = line.strip().split()
        if len(parts) < 2: continue
        ticker = parts[0].rstrip(":")
        if ticker in tickers:
            try:
                weights[tickers.index(ticker)] = \
                    float("".join(parts[-2:]).replace("%", "")) / 100.0
            except ValueError:
                pass
    s = weights.sum()
    if s > 1e-10:
        weights /= s

    cov_mat = np.array(cov_flat).reshape(n, n)
    return {
        "weights":  weights,
        "time_sec": elapsed,
        "return":   float(np.dot(weights, mu)),
        "risk":     float(weights @ cov_mat @ weights),
    }


# Pomožne

def portfolio_daily_returns(weights: np.ndarray,
                             prices: pd.DataFrame,
                             tickers: list) -> np.ndarray:
    pct = prices.pct_change().dropna()
    w   = np.array([weights[tickers.index(t)] for t in pct.columns])
    w  /= w.sum() + 1e-15
    return pct.values @ w


def eq_k_weights(mu: np.ndarray, K: int) -> np.ndarray:
    n    = len(mu)
    K    = min(K, n)
    topk = np.argsort(mu)[::-1][:K]
    w    = np.zeros(n)
    w[topk] = 1.0 / K
    return w


# Metahevristike (rešene v C++), vsaka paper-faithful implementacija.
METAHEURISTICS = ["pso", "sa"]
# Vse "rešitve" ki jih beležimo/rišemo.
ALL_ALGOS = METAHEURISTICS


def _rank(x: np.ndarray) -> np.ndarray:
    """Povprečni rangi (za Spearman-ov IC brez scipy odvisnosti)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # popravek za vezi (average ranks)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum - 1) / 2.0
    return avg[inv]


def prediction_quality(mu_pred: np.ndarray, realized: np.ndarray) -> dict:
    """
    Spearman IC (rank korelacija) in directional accuracy napovedi μ proti
    realiziranim donosom čez okno. Ključni sanity-check kvalitete napovedi:
    če IC ≈ 0, noben optimizator ne more rešiti signala.
    """
    mu_pred  = np.asarray(mu_pred,  dtype=float)
    realized = np.asarray(realized, dtype=float)
    n = len(mu_pred)
    if n < 3 or np.allclose(mu_pred.std(), 0) or np.allclose(realized.std(), 0):
        return {"ic": np.nan, "dir_acc": np.nan}
    rp, rr = _rank(mu_pred), _rank(realized)
    ic = float(np.corrcoef(rp, rr)[0, 1])
    dir_acc = float(np.mean(np.sign(mu_pred) == np.sign(realized)))
    return {"ic": ic, "dir_acc": dir_acc}


# Walk-forward

def run_walkforward(config: dict,
                    avail_tickers: list,
                    df_train: pd.DataFrame,
                    df_test: pd.DataFrame,
                    market_prices=None,
                    test_days: int = 21,
                    windows: int = 5,
                    binary: str = "",
                    ml_registry: dict = None,
                    lstm_registry: dict = None,
                    tactis_registry: dict = None,
                    master_registry: dict = None,
                    tft_registry: dict = None) -> dict:
    """
    Walk-forward zanka. Za vsako okno:
      1. df_hist = df_train + df_test[:ts]
      2. mu_hist  = zgodovinski povprečni donos iz df_hist
      3. mu_trans = multistock_master.get_mu(df_hist)   ← drsno
      4. Σ: vzorčna kovarianca SAMO za Zgodovinski; Transformer pipeline
         uporabi modelsko kov. glavo (če obstaja, sicer vzorčno).
      5. Optimizacija × metahevristike × scenariji
         Scenariji parijo (μ-vir, Σ-vir) — vsak pipeline svojo domorodno Σ:
           Zgodovinski = (hist μ,     vzorčna Σ)
           Transformer = (trans μ,    modelska Σ)   [vzorčna, če ni kov. glave]
      6. Kvaliteta napovedi (IC, dir-acc) μ proti realiziranim donosom.
    """
    if ml_registry is not None:
        from estimators.simple_ml import get_mu_simple_ml
    if lstm_registry is not None:
        from estimators.lstm_mu import get_mu_lstm
    if tactis_registry is not None:
        from estimators.tactis_estimator import get_musigma_tactis
    if master_registry is not None:
        from estimators.master_us import get_mu_master_us
    if tft_registry is not None:
        from estimators.tft_mu import get_mu_tft
    from estimators.black_litterman import bl_mu
    from benchmark.benchmark_strategies import (gmv_weights, max_sharpe_weights,
                                                risk_parity_weights, hrp_weights)

    N         = len(avail_tickers)
    lookahead = config["run_settings"].get("lookahead_days", 21)
    base_seed = config["optimizer_config"]["common"]["seed"]
    K_base    = config["optimizer_config"]["common"]["cardinality_K"]
    w_min     = config["optimizer_config"]["common"].get("w_min", 0.02)
    w_max     = config["optimizer_config"]["common"].get("w_max", 0.30)

    # Scenariji: (ime, μ-vir, Σ-vir).
    # Osnovni klasični pipeline: Zgodovinski (vzorčno povprečje μ + vzorčna Σ).
    scenario_defs = [("Zgodovinski", "hist",  "baseline")]

    # TACTiS-2 (Ashok et al. 2024): CELOTEN transformerski pipeline, ki napove
    # OBA momenta (μ IN Σ) iz vzorcev skupne napovedane porazdelitve — brez
    # scale-matchinga na zgodovino. Σ-vir 'tactis' = vzorčna kovarianca vzorcev.
    if tactis_registry is not None:
        scenario_defs.append(("TACTiS", "tactis", "tactis"))
        print("  TACTiS pipeline: μ IN Σ iz skupne napovedane porazdelitve "
              "(vzorčno povprečje / kovarianca; nič prevzeto iz zgodovine)")


    # KLASIČNI BASELINE-i: vzorčna Σ (ista kot Zgodovinski) + alternativni μ.
    #   SimpleML        = (LightGBM faktorski μ, vzorčna Σ)
    #   BlackLitterman  = (klasičen BL posterior μ, vzorčna Σ)
    #   LSTM            = (LSTM cross-sectional μ, vzorčna Σ)
    sb_cfg = config.get("evaluation", {}).get("strong_baselines", {})
    strong_on = sb_cfg.get("enabled", True)
    if strong_on:
        if ml_registry is not None:
            scenario_defs.append(("SimpleML", "mlbase", "baseline"))
        scenario_defs.append(("BlackLitterman", "blacklitterman", "baseline"))
        if lstm_registry is not None:
            scenario_defs.append(("LSTM", "lstm", "baseline"))
        extras = []
        if ml_registry is not None:
            extras.append("SimpleML")
        extras.append("BlackLitterman")
        if lstm_registry is not None:
            extras.append("LSTM")
        print(f"  Baseline-i: vzorčna Σ + BlackLitterman μ"
              f"{' + LightGBM μ' if ml_registry is not None else ''}"
              f"{' + LSTM μ' if lstm_registry is not None else ''} "
              f"(scenariji: {', '.join(extras)})")

    # MASTER (Li et al. AAAI 2024) na ameriških delnicah — ORIGINALNI repo model
    # (Alpha158 + tržni gating). μ-only: presečni rang-signal, reskaliran na
    # zgodovinsko μ skalo; Σ = vzorčna (ista kot Zgodovinski → izolira μ-vir).
    if master_registry is not None:
        scenario_defs.append(("MASTER", "master", "baseline"))
        print("  MASTER pipeline: Alpha158 + tržni gating μ (presečni rang) → "
              "reskaliran na zgodovinsko skalo, vzorčna Σ")

    # TFT (Temporal Fusion Transformer, Lim et al. 2021) — off-the-shelf splošni
    # napovedni transformer (pytorch-forecasting). μ-only: napove H dnevnih donosov,
    # μ = vsota; presečni z-score → reskaliran na zgodovinsko skalo, vzorčna Σ.
    if tft_registry is not None:
        scenario_defs.append(("TFT", "tft", "baseline"))
        print("  TFT pipeline: splošni napovedni transformer (multi-step donosi) → "
              "reskaliran na zgodovinsko skalo, vzorčna Σ")

    scenario_names = [s[0] for s in scenario_defs]

    # Zgodovinski baseline: DRSEČE (rolling sample mean)
    # POŠTENA primerjava: zgodovinski μ/Σ se računata drseče iz ENAKEGA
    # nedavnega okna, kot ga transformer vidi pri inferenci. Edina razlika med
    # scenarijema je tako μ-PRESLIKAVA (vzorčno povprečje vs transformer), kar
    # izolira znanstveni prispevek "transformer > zgodovina". To je tudi dejanski
    # literaturni baseline (rolling sample mean μ), ki ga uporabljajo članki.
    print("  Zgodovinski baseline: DRSEČ (rolling sample mean — pošten literaturni baseline)")

    all_results         = {s: {a: [] for a in ALL_ALGOS} for s in scenario_names}
    # Klasične benchmark strategije (benchmark_strategies.py) izbrane za
    # izolacijo prednosti/slabosti hibrida:
    #   1/N, 1/N@K       — naivne (že prej)
    #   GMV, RiskParity  — samo Σ (brez μ) → vrednost transformer-μ
    #   Markowitz_*      — isti μ,Σ, brez kardinalnosti → cena omejitve+metahev.
    #   Market           — pasivni indeks (SPY buy&hold), ~0 turnover
    baseline_keys = ["1/N", "1/N@K_zgod",
                     "GMV", "RiskParity", "HRP", "Markowitz_zgod"]
    if market_prices is not None:
        baseline_keys.append("Market")
    baseline_results    = {k: [] for k in baseline_keys}
    baseline_weights    = {k: [] for k in baseline_keys if k != "Market"}
    all_results_weights = {s: {a: [] for a in ALL_ALGOS} for s in scenario_names}
    # Per-window μ forecasts za Hedge ensemble — shranimo za vsak model in okno
    per_window_forecasts = {}  # key=(window_idx, model_name), value=μ_array
    per_window_realized = {}   # key=window_idx, value=realized_returns
    per_window_cov = {}        # key=window_idx, value=cov_matrix — za Hedge re-optimization
    per_window_test_prices = {} # key=window_idx, value=df_test_window — za Hedge daily rets
    # Kvaliteta napovedi μ (IC, dir-acc) po scenariju/oknu.
    pred_quality        = {"Zgodovinski": []}
    if ml_registry is not None:
        pred_quality["SimpleML"] = []
    if strong_on:
        pred_quality["BlackLitterman"] = []
    if lstm_registry is not None:
        pred_quality["LSTM"] = []
    if tactis_registry is not None:
        pred_quality["TACTiS"] = []
    if master_registry is not None:
        pred_quality["MASTER"] = []
    if tft_registry is not None:
        pred_quality["TFT"] = []

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD  ({windows} oken × {test_days} dni)")
    print(f"  Trening: {df_train.index[0].date()} → {df_train.index[-1].date()}")
    print(f"  Test:    {df_test.index[0].date()} → {df_test.index[-1].date()}")
    print(f"{'='*65}")

    for w_idx in range(windows):
        ts = w_idx * test_days
        te = ts + test_days
        if te > len(df_test):
            continue

        test_p  = df_test.iloc[ts:te]
        df_hist = pd.concat([df_train, df_test.iloc[:ts]]) if ts > 0 else df_train

        hist_rets = df_hist.pct_change().dropna()
        hist_rets_cov = hist_rets
        hist_rets_mu  = hist_rets

        # Poštena primerjava: zgodovinski μ iz drsnega okna
        # Transformer pri inferenci vidi zadnjih sequence_length cen.
        # Zgodovinski μ mora videti enako dolgo preteklost — sicer ima
        # transformer informacijsko prednost (vidi aktualni momentum,
        # zgodovinski pa samo statično povprečje 2010-2022).
        # Rešitev: rolling window enake dolžine kot transformer kontekst.
        # estimation_window_days (evaluation): eksplicitno drsno okno za oceno
        # zgodovinskih momentov (μ IN Σ) — omogoča zvesto prevzem protokola drugih
        # študij (npr. Kaucic et al. 2022: 60-mesečno okno ≈ 1260 dni; DeMiguel in
        # sod. 2009: 120-mesečno). Če je 0 (privzeto), ostane staro obnašanje:
        # μ iz drsnega okna dolžine sequence_length, Σ iz celotne zgodovine.
        seq_len      = config.get("return_estimator", {}).get("sequence_length", 15)
        est_window   = int(config.get("evaluation", {}).get("estimation_window_days", 0))
        roll_window  = est_window if est_window > 0 else max(seq_len, 21)
        hist_rets_mu = hist_rets_mu.tail(roll_window)
        mu_hist_win  = hist_rets_mu.mean().values * lookahead
        # Skala-sidro za cross-sectional ML/LSTM z-score μ = ZGODOVINSKI μ
        # (MASTER-lite odstranjen kot prejšnje sidro). std/mean zgodovinskega μ
        # dasta absolutno raven, tako da se med scenariji razlikuje le OBLIKA.
        hist_mu_std  = float(mu_hist_win.std()) + 1e-12
        hist_mu_mean = float(mu_hist_win.mean())
        # Σ iz istega drsnega okna kot μ, če je est_window podan; sicer cela zgodovina.
        cov_rets = hist_rets_cov.tail(est_window) if est_window > 0 else hist_rets_cov
        # Baseline Σ (vzorčna kovarianca, H-dnevna) — enkrat na okno.
        cov_by_src = {"baseline": _sample_cov(cov_rets.values, lookahead)}
        # Ledoit-Wolf skrčena Σ — MOČAN klasičen Σ baseline (samo za strong-baseline scen.).
        if strong_on:
            cov_by_src["shrinkage"] = _ledoit_wolf_cov(cov_rets.values, lookahead)
        mu_by_src  = {"hist": mu_hist_win}
        # TACTiS-2 skupni pipeline: μ IN Σ iz vzorcev napovedane porazdelitve.
        # Oba momenta na H-dnevni skali (kot ostali viri).
        # Rescaliramo μ na isto skalo kot MASTER/TFT za smiselno primerjavo.
        if tactis_registry is not None:
            mu_tac, cov_tac = get_musigma_tactis(
                tactis_registry, df_hist, avail_tickers, lookahead)
            # Rescale TACTiS μ na hist_mu scale (kao MASTER/TFT) — standardiziraj, potem rescaliraj
            mu_tac_z = (mu_tac - mu_tac.mean()) / (mu_tac.std() + 1e-12)
            mu_by_src["tactis"]  = mu_tac_z * hist_mu_std + hist_mu_mean
            cov_by_src["tactis"] = cov_tac
        # LightGBM faktorski μ (preprost ML baseline). Model odda cross-sectional
        # z-score; poravnamo na ZGODOVINSKO μ mean/std → ista absolutna skala pri
        # optimizatorju, razlikuje se le cross-sectional OBLIKA.
        if ml_registry is not None:
            mu_ml_z = get_mu_simple_ml(ml_registry, df_hist, avail_tickers, lookahead)
            mu_by_src["mlbase"] = mu_ml_z * hist_mu_std + hist_mu_mean
        # Black-Litterman posterior μ (klasičen): ravnovesni prior (enakomeren
        # referenčni portfelj) + zgodovinski drseči μ kot pogledi, čez Ledoit-Wolf Σ.
        # Že v DONOSNI skali (kot views) → NE reskaliramo (za razliko od ML/LSTM).
        if strong_on:
            mu_by_src["blacklitterman"] = bl_mu(
                cov_by_src["shrinkage"], views=mu_hist_win)
        # LSTM cross-sectional μ (sekvenčni baseline): z-score, poravnan na
        # ZGODOVINSKO μ mean/std (ista absolutna skala kot SimpleML).
        if lstm_registry is not None:
            mu_lstm_z = get_mu_lstm(lstm_registry, df_hist, avail_tickers, lookahead)
            mu_by_src["lstm"] = mu_lstm_z * hist_mu_std + hist_mu_mean
        # MASTER μ (presečni rang, z-score) → poravnan na ZGODOVINSKO μ mean/std
        # (ista absolutna skala kot SimpleML/LSTM; le presečna OBLIKA se razlikuje).
        if master_registry is not None:
            mu_master_z = get_mu_master_us(master_registry, df_hist,
                                           avail_tickers, lookahead)
            mu_by_src["master"] = mu_master_z * hist_mu_std + hist_mu_mean
        # TFT μ (multi-step vsota napovedanih dnevnih donosov, z-score) → poravnan
        # na ZGODOVINSKO μ mean/std (ista absolutna skala kot MASTER/SimpleML/LSTM).
        if tft_registry is not None:
            mu_tft_z = get_mu_tft(tft_registry, df_hist, avail_tickers, lookahead)
            mu_by_src["tft"] = mu_tft_z * hist_mu_std + hist_mu_mean
        win_seed = base_seed + w_idx * 100

        base_P = float(config["optimizer_config"]["common"]["risk_parameter"])

        mu_top = mu_by_src.get("tactis", mu_hist_win)
        top3 = sorted(zip(avail_tickers, mu_top), key=lambda x: -x[1])[:3]
        print(f"\n  Okno {w_idx+1}/{windows}  "
              f"({df_test.index[ts].date()} → {df_test.index[te-1].date()})  "
              f"[hist: {len(hist_rets)} dni]  μ_top3: {top3}")

        # Benchmark strategije
        # Naivne (1/N, 1/N@K) + klasične (GMV, RiskParity, Markowitz) na isti
        # baseline Σ; Markowitz per-scenarij z ustreznim μ (pošteni μ-primerjava).
        cov_base = cov_by_src["baseline"]
        bench_w = {
            "1/N":            np.ones(N) / N,
            "1/N@K_zgod":     eq_k_weights(mu_hist_win,  K_base),
            "GMV":            gmv_weights(cov_base),
            "RiskParity":     risk_parity_weights(cov_base),
            "HRP":            hrp_weights(cov_base),
            "Markowitz_zgod": max_sharpe_weights(mu_hist_win,  cov_base),
        }
        for key, w_b in bench_w.items():
            baseline_results[key].append(
                portfolio_daily_returns(w_b, test_p, avail_tickers))
            baseline_weights[key].append(np.asarray(w_b, dtype=float))

        # Market (SPY) buy&hold — pasivni indeks, poravnan na testno okno.
        if market_prices is not None:
            mkt = market_prices.reindex(test_p.index).ffill()
            mkt_rets = mkt.pct_change().dropna().values.astype(float).ravel()
            baseline_results["Market"].append(mkt_rets)

        # Kvaliteta napovedi μ proti realiziranim donosom čez okno
        realized = (test_p.pct_change().dropna() + 1.0).prod().values - 1.0
        # Shrani per-window μ forecasts in realized za Hedge ensemble
        # VSE μ so že na isti skali (hist_mu_mean/std) za smiselno tehtano povprečje
        per_window_realized[w_idx] = realized
        per_window_test_prices[w_idx] = test_p
        per_window_cov[w_idx] = cov_by_src  # shrani vse Σ vire za Hedge re-optimization
        if master_registry is not None:
            per_window_forecasts[(w_idx, "MASTER")] = mu_by_src["master"]
        if tft_registry is not None:
            per_window_forecasts[(w_idx, "TFT")] = mu_by_src["tft"]
        if tactis_registry is not None:
            per_window_forecasts[(w_idx, "TACTiS")] = mu_by_src["tactis"]
        pred_quality["Zgodovinski"].append(prediction_quality(mu_hist_win,  realized))
        if ml_registry is not None:
            pred_quality["SimpleML"].append(
                prediction_quality(mu_by_src["mlbase"], realized))
        if strong_on:
            pred_quality["BlackLitterman"].append(
                prediction_quality(mu_by_src["blacklitterman"], realized))
        if lstm_registry is not None:
            pred_quality["LSTM"].append(
                prediction_quality(mu_by_src["lstm"], realized))
        if tactis_registry is not None:
            pred_quality["TACTiS"].append(
                prediction_quality(mu_by_src["tactis"], realized))
        if master_registry is not None:
            pred_quality["MASTER"].append(
                prediction_quality(mu_by_src["master"], realized))
        if tft_registry is not None:
            pred_quality["TFT"].append(
                prediction_quality(mu_by_src["tft"], realized))

        # Scenariji: (μ-vir, Σ-vir) × metahevristike
        for scen_name, mu_src, cov_src in scenario_defs:
            mu_win  = mu_by_src[mu_src]
            cov_win = cov_by_src[cov_src]
            for algo in METAHEURISTICS:
                res    = run_optimizer(config, avail_tickers, algo,
                                       mu_win.tolist(), cov_win.flatten().tolist(),
                                       risk_parameter=base_P,
                                       seed=win_seed, binary=binary)
                d_rets = portfolio_daily_returns(res["weights"], test_p, avail_tickers)
                all_results[scen_name][algo].append(d_rets)
                all_results_weights[scen_name][algo].append(res["weights"])
                cum = np.prod(1 + d_rets) - 1
                print(f"    [{scen_name[:4]}] {algo:10s} | "
                      f"{cum*100:+6.2f}% | {res['time_sec']:.1f}s")

    return {
        "all_results":         all_results,
        "baseline_results":    baseline_results,
        "baseline_weights":    baseline_weights,
        "all_results_weights": all_results_weights,
        "prediction_quality":  pred_quality,
        "scenario_names":      scenario_names,
        "per_window_forecasts": per_window_forecasts,
        "per_window_realized":  per_window_realized,
        "per_window_cov":       per_window_cov,
        "per_window_test_prices": per_window_test_prices,
    }