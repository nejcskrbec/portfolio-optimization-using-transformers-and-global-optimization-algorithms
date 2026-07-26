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
METAHEURISTICS = ["pso", "sa", "ga"]
# Vse "rešitve" ki jih beležimo/rišemo — vključno z near-exact referenco
# (rešeno v Pythonu, benchmark/near_exact.py) za merjenje optimality gap.
ALL_ALGOS = METAHEURISTICS + ["near_exact"]


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
                    return_model_registry: dict,
                    market_prices=None,
                    test_days: int = 21,
                    windows: int = 5,
                    binary: str = "",
                    ml_registry: dict = None,
                    lstm_registry: dict = None) -> dict:
    """
    Walk-forward zanka. Za vsako okno:
      1. df_hist = df_train + df_test[:ts]
      2. mu_hist  = zgodovinski povprečni donos iz df_hist
      3. mu_trans = multistock_master.get_mu(df_hist)   ← drsno
      4. Σ: vzorčna kovarianca SAMO za Zgodovinski; Transformer & Ansambel
         uporabita modelsko kov. glavo (če obstaja, sicer vzorčno).
      5. Optimizacija × {metahevristike + near-exact} × scenariji
         Scenariji parijo (μ-vir, Σ-vir) — vsak pipeline svojo domorodno Σ:
           Zgodovinski = (hist μ,     vzorčna Σ)
           Transformer = (trans μ,    modelska Σ)   [vzorčna, če ni kov. glave]
           Ansambel    = (Hedge μ,    modelska Σ)   [vzorčna, če ni kov. glave]
      6. Kvaliteta napovedi (IC, dir-acc) μ proti realiziranim donosom.
    """
    from estimators.multistock_master import (get_mu_multistock as get_mu,
                                               get_cov_multistock as get_cov)
    if ml_registry is not None:
        from estimators.simple_ml import get_mu_simple_ml
    if lstm_registry is not None:
        from estimators.lstm_mu import get_mu_lstm
    from estimators.black_litterman import bl_mu
    from benchmark.near_exact import solve_near_exact
    from benchmark.benchmark_strategies import (gmv_weights, max_sharpe_weights,
                                                risk_parity_weights, hrp_weights)

    N         = len(avail_tickers)
    lookahead = config["run_settings"].get("lookahead_days", 21)
    base_seed = config["optimizer_config"]["common"]["seed"]
    K_base    = config["optimizer_config"]["common"]["cardinality_K"]
    w_min     = config["optimizer_config"]["common"].get("w_min", 0.02)
    w_max     = config["optimizer_config"]["common"].get("w_max", 0.30)
    ne_cfg    = config.get("near_exact", {})
    ne_on     = ne_cfg.get("enabled", True)

    # Scenariji: (ime, μ-vir, Σ-vir). Σ-vir "transformer" iz SKUPNO-REPREZENTACIJSKE
    # kovariančne glave return modela (multistock). To reši razkorak "optimizator
    # uporablja drugo Σ kot jo model implicira": μ-rang in Σ izhajata iz iste
    # MASTER-lite reprezentacije.
    shared_cov_on = (return_model_registry.get("_type") == "multistock"
                     and return_model_registry.get("_has_cov", False))
    # Σ-viri po scenariju: vzorčna Σ SAMO za Zgodovinski (klasičen baseline).
    # Transformer & Ansambel dobita modelsko (skupno-reprezentacijsko kov. glavo)
    # če obstaja — sicer padeta nazaj na vzorčno Σ. Vsak "pipeline" torej uporabi
    # svojo domorodno Σ: hist=vzorčna, transformer=modelska, ansambel=modelska.
    # Scenario tuple = (ime, μ-vir, Σ-vir, run_near_exact). Near-exact (počasen,
    # ~15s/okno) računamo SAMO za 3 osrednje scenarije (optimality gap); dodatni
    # "močni baseline" scenariji tečejo le metahevristike (dovolj za primerjavo).
    trans_cov = "transformer" if shared_cov_on else "baseline"
    scenario_defs = [("Zgodovinski", "hist",  "baseline", True),
                     ("Transformer", "trans", trans_cov,  True)]
    if shared_cov_on:
        print("  Σ-vir 'transformer' = skupno-reprezentacijska kovariančna glava (MASTER-lite)")

    # Ansambel μ: spletna forecast-combination (Hedge)
    # Nadomesti krhki enojni winner-tilt hiperparameter (topw_alpha) s
    # KOMBINACIJO μ-pogledov (zgodovinski + nevtralni transformer pod več γ
    # signed-power nagibi), katere uteži se prilagajajo vsako okno iz realizirane
    # uspešnosti ekspertov. Nič ni fitano na testne režime; η iz teorije
    # (Cesa-Bianchi & Lugosi). Regret meja → kombinacija ne more zaostati za
    # najboljšim ekspertom (vključno zgodovinskim baselineom) → "robustno proti
    # zgodovini" v VSAKEM režimu. Glej estimators/mu_ensemble.py.
    ens_cfg     = config.get("evaluation", {}).get("mu_ensemble", {})
    ensemble_on = ens_cfg.get("enabled", True)
    mu_ens      = None
    if ensemble_on:
        from estimators.mu_ensemble import HedgeMuEnsemble
        mu_ens = HedgeMuEnsemble(
            gamma_grid=tuple(ens_cfg.get("gamma_grid", (0.5, 1.0, 2.0))),
            n_windows=windows, topk=K_base, eta=ens_cfg.get("eta", None))
        ens_cov = "transformer" if shared_cov_on else "baseline"
        scenario_defs.append(("Ansambel", "ensemble", ens_cov, True))
        print(f"  Ansambel μ: Hedge forecast-combination "
              f"(eksperti={mu_ens.names}, η={mu_ens.eta:.3f}, Σ-vir={ens_cov})")

    # MOČNI KLASIČNI BASELINE-i (odgovor na recenzenta)
    # Vzorčna Σ je šibek Σ baseline (Ledoit & Wolf 2004); zgodovinsko povprečje
    # je šibek μ baseline. Dodamo (config: evaluation.strong_baselines.enabled):
    #   Zgodovinski-LW  = (zgodovinski μ, Ledoit-Wolf Σ)  → MOČAN klasičen pipeline
    #   Transformer-LWΣ = (transformer μ, Ledoit-Wolf Σ)  → izolira modelsko kov. glavo
    #                                                        (model Σ vs shrinkage Σ)
    #   SimpleML        = (LightGBM faktorski μ, Ledoit-Wolf Σ) → izolira μ
    #                                                        (transformer μ vs preprost ML μ)
    #   BlackLitterman  = (klasičen BL posterior μ, Ledoit-Wolf Σ) → predlagana
    #                                                        klasična primerjava (proposal)
    #   LSTM            = (LSTM cross-sectional μ, Ledoit-Wolf Σ) → predlagana
    #                                                        sekvenčno-modelska primerjava
    #                                                        (arhitektura: LSTM vs transformer)
    # Vse tečejo le metahevristike (brez near-exact) da omejimo dodaten čas.
    sb_cfg = config.get("evaluation", {}).get("strong_baselines", {})
    strong_on = sb_cfg.get("enabled", True)
    if strong_on:
        scenario_defs.append(("Zgodovinski-LW",  "hist",  "shrinkage", False))
        scenario_defs.append(("Transformer-LWΣ", "trans", "shrinkage", False))
        if ml_registry is not None:
            scenario_defs.append(("SimpleML", "mlbase", "shrinkage", False))
        # Black-Litterman: sproti na okno (brez učenja) → vedno na voljo.
        scenario_defs.append(("BlackLitterman", "blacklitterman", "shrinkage", False))
        if lstm_registry is not None:
            scenario_defs.append(("LSTM", "lstm", "shrinkage", False))
        extras = ["Zgodovinski-LW", "Transformer-LWΣ"]
        if ml_registry is not None:
            extras.append("SimpleML")
        extras.append("BlackLitterman")
        if lstm_registry is not None:
            extras.append("LSTM")
        print(f"  Močni baseline-i: Ledoit-Wolf Σ + BlackLitterman μ"
              f"{' + LightGBM μ' if ml_registry is not None else ''}"
              f"{' + LSTM μ' if lstm_registry is not None else ''} "
              f"(scenariji: {', '.join(extras)})")

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
    baseline_keys = ["1/N", "1/N@K_zgod", "1/N@K_trans",
                     "GMV", "RiskParity", "HRP", "Markowitz_zgod", "Markowitz_trans"]
    if market_prices is not None:
        baseline_keys.append("Market")
    baseline_results    = {k: [] for k in baseline_keys}
    baseline_weights    = {k: [] for k in baseline_keys if k != "Market"}
    all_results_weights = {s: {a: [] for a in ALL_ALGOS} for s in scenario_names}
    # Kvaliteta napovedi μ (IC, dir-acc) po scenariju/oknu.
    pred_quality        = {"Zgodovinski": [], "Transformer": []}
    if mu_ens is not None:
        pred_quality["Ansambel"] = []
    if ml_registry is not None:
        pred_quality["SimpleML"] = []
    if strong_on:
        pred_quality["BlackLitterman"] = []
    if lstm_registry is not None:
        pred_quality["LSTM"] = []

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
        mu_trans_win = get_mu(return_model_registry, df_hist, avail_tickers, lookahead)
        # Σ iz istega drsnega okna kot μ, če je est_window podan; sicer cela zgodovina.
        cov_rets = hist_rets_cov.tail(est_window) if est_window > 0 else hist_rets_cov
        # Baseline Σ (vzorčna kovarianca, H-dnevna) — enkrat na okno.
        cov_by_src = {"baseline": _sample_cov(cov_rets.values, lookahead)}
        # Ledoit-Wolf skrčena Σ — MOČAN klasičen Σ baseline (samo za strong-baseline scen.).
        if strong_on:
            cov_by_src["shrinkage"] = _ledoit_wolf_cov(cov_rets.values, lookahead)
        # Skupno-reprezentacijska Σ iz return modela (multistock kov. glava)
        # ista df_hist kot μ, ista reprezentacija → uskladi model in optimizator.
        if shared_cov_on:
            cov_shared = get_cov(return_model_registry, df_hist, avail_tickers, lookahead)
            if cov_shared is not None:
                cov_by_src["transformer"] = cov_shared
        mu_by_src  = {"hist": mu_hist_win, "trans": mu_trans_win}
        # LightGBM faktorski μ (preprost ML baseline). Model odda cross-sectional
        # z-score; poravnamo na transformer μ mean/std → ista skala pri optimizatorju,
        # samo cross-sectional OBLIKA se razlikuje (kot pri ansamblu).
        if ml_registry is not None:
            mu_ml_z = get_mu_simple_ml(ml_registry, df_hist, avail_tickers, lookahead)
            mu_by_src["mlbase"] = mu_ml_z * mu_trans_win.std() + mu_trans_win.mean()
        # Black-Litterman posterior μ (klasičen): ravnovesni prior (enakomeren
        # referenčni portfelj) + zgodovinski drseči μ kot pogledi, čez Ledoit-Wolf Σ.
        # Že v DONOSNI skali (kot views) → NE reskaliramo (za razliko od ML/LSTM).
        if strong_on:
            mu_by_src["blacklitterman"] = bl_mu(
                cov_by_src["shrinkage"], views=mu_hist_win)
        # LSTM cross-sectional μ (sekvenčni baseline): z-score, poravnan na
        # transformer μ mean/std (ista skala pri optimizatorju kot SimpleML/ansambel).
        if lstm_registry is not None:
            mu_lstm_z = get_mu_lstm(lstm_registry, df_hist, avail_tickers, lookahead)
            mu_by_src["lstm"] = mu_lstm_z * mu_trans_win.std() + mu_trans_win.mean()
        # Ansambel μ (Hedge kombinacija) — uteži temeljijo SAMO na preteklih
        # oknih (spletna posodobitev spodaj po realizaciji) → kavzalno.
        if mu_ens is not None:
            mu_by_src["ensemble"] = mu_ens.combine(mu_hist_win, mu_trans_win)
            w_str = ", ".join(f"{n}={w:.2f}"
                              for n, w in zip(mu_ens.names, mu_ens.w))
            print(f"  Ansambel uteži: {w_str}")

        win_seed = base_seed + w_idx * 100

        # Dinamični risk_parameter (regime-switching)
        #
        # Referenca (za pisanje teze):
        #   Ang, A. & Bekaert, G. (2004). "How Regimes Affect Asset Allocation."
        #   Financial Analysts Journal 60(2):86–99.
        #
        # Ang & Bekaert (2004) pokažejo da investitor ki ignorira volatilnostne režime
        # drži preveč delnic v visokovolat. (bear) režimu. V bull (nizka vol) režimu
        # ima mean-variance frontier višji Sharpe → zaupaj bolj transformer μ (višji P).
        # V bear (visoka vol) režimu μ napovedi manj zanesljive → nižji P.
        #
        # Implementacija po dvostopenjskem modelu:
        #   Volatilnostni prag: VIX ≈ 20 → 20/sqrt(252) ≈ 1.26% dnevna vol
        #   Nizka vol (bull):  σ_21d < 1.0% → P_high = base_P + Δ
        #   Visoka vol (bear): σ_21d > 1.5% → P_low  = base_P - Δ
        #   Δ = 0.10 (10pp prilagoditev — po Ang & Bekaert magnitude of effect)
        #
        # Namesto diskretnega 2-stanjskega modela uporabimo zvezno interpolacijo:
        #   P(σ) = base_P + Δ * (σ_low - σ_21d) / (σ_high - σ_low)
        # ki je zvezna in daje enake rezultate kot 2-stanjski model na mejah.
        base_P   = config["optimizer_config"]["common"]["risk_parameter"]
        sigma_21 = hist_rets.tail(21).std().mean()   # 21-dnevna realizirana vol (dnevna)
        sigma_lo = 0.010   # bull prag: 1.0% dnevna ≈ VIX 15.9
        sigma_hi = 0.015   # bear prag: 1.5% dnevna ≈ VIX 23.8
        delta_P  = 0.10
        regime_scale = np.clip((sigma_hi - sigma_21) / (sigma_hi - sigma_lo), 0.0, 1.0)
        dynamic_P    = base_P - delta_P + 2 * delta_P * regime_scale
        dynamic_P    = float(np.clip(dynamic_P, 0.50, 0.95))
        regime_label = "bull" if sigma_21 < sigma_lo else ("bear" if sigma_21 > sigma_hi else "mid")
        print(f"  σ_21d={sigma_21*100:.2f}%  regime={regime_label}  P={dynamic_P:.3f}")

        top3 = sorted(zip(avail_tickers, mu_trans_win), key=lambda x: -x[1])[:3]
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
            "1/N@K_trans":    eq_k_weights(mu_trans_win, K_base),
            "GMV":            gmv_weights(cov_base),
            "RiskParity":     risk_parity_weights(cov_base),
            "HRP":            hrp_weights(cov_base),
            "Markowitz_zgod": max_sharpe_weights(mu_hist_win,  cov_base),
            "Markowitz_trans":max_sharpe_weights(mu_trans_win, cov_base),
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
        pred_quality["Zgodovinski"].append(prediction_quality(mu_hist_win,  realized))
        pred_quality["Transformer"].append(prediction_quality(mu_trans_win, realized))
        if mu_ens is not None:
            pred_quality["Ansambel"].append(
                prediction_quality(mu_by_src["ensemble"], realized))
            mu_ens.update(realized)   # spletna posodobitev uteži za NASLEDNJE okno
        if ml_registry is not None:
            pred_quality["SimpleML"].append(
                prediction_quality(mu_by_src["mlbase"], realized))
        if strong_on:
            pred_quality["BlackLitterman"].append(
                prediction_quality(mu_by_src["blacklitterman"], realized))
        if lstm_registry is not None:
            pred_quality["LSTM"].append(
                prediction_quality(mu_by_src["lstm"], realized))

        # Scenariji: (μ-vir, Σ-vir) × {metahevristike + near-exact}
        for scen_name, mu_src, cov_src, ne_this in scenario_defs:
            mu_win  = mu_by_src[mu_src]
            cov_win = cov_by_src[cov_src]
            run_ne  = ne_on and ne_this
            warm_supports = []   # supporti metahevristik → ogrej near-exact
            for algo in METAHEURISTICS:
                res    = run_optimizer(config, avail_tickers, algo,
                                       mu_win.tolist(), cov_win.flatten().tolist(),
                                       risk_parameter=dynamic_P,
                                       seed=win_seed, binary=binary)
                d_rets = portfolio_daily_returns(res["weights"], test_p, avail_tickers)
                all_results[scen_name][algo].append(d_rets)
                all_results_weights[scen_name][algo].append(res["weights"])
                warm_supports.append(np.nonzero(res["weights"] > 1e-6)[0])
                cum = np.prod(1 + d_rets) - 1
                print(f"    [{scen_name[:4]}] {algo:10s} | "
                      f"{cum*100:+6.2f}% | {res['time_sec']:.1f}s")

            # Near-exact referenca (isti μ, Σ, P) → optimality gap.
            if run_ne:
                ne = solve_near_exact(
                    mu_win, cov_win, K_base, w_min, w_max, dynamic_P,
                    enumerate_max=ne_cfg.get("enumerate_max", 200000),
                    restarts=ne_cfg.get("restarts", 8),
                    time_limit_sec=ne_cfg.get("time_limit_sec", 15.0),
                    warm_starts=warm_supports, seed=win_seed)
                ne_w = ne["weights"]
            else:
                ne_w = np.zeros(N)
            ne_rets = portfolio_daily_returns(ne_w, test_p, avail_tickers) \
                      if ne_w.sum() > 1e-9 else np.zeros(len(test_p) - 1)
            all_results[scen_name]["near_exact"].append(ne_rets)
            all_results_weights[scen_name]["near_exact"].append(ne_w)
            if run_ne:
                tag = "exact" if ne.get("exact") else "near"
                print(f"    [{scen_name[:4]}] {'near_exact':10s} | "
                      f"obj={ne['objective']:+.5f} ({tag}, "
                      f"{ne['n_supports_evaluated']} supp.)")

    return {
        "all_results":         all_results,
        "baseline_results":    baseline_results,
        "baseline_weights":    baseline_weights,
        "all_results_weights": all_results_weights,
        "prediction_quality":  pred_quality,
        "scenario_names":      scenario_names,
    }