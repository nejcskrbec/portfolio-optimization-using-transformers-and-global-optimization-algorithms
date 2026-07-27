#!/usr/bin/env python3
"""
run_benchmark.py
=================
Uporaba:
  python run_benchmark.py config.json
"""

import argparse
import json
import os
import sys
import time

import yfinance as yf

# Skripta živi v benchmark/ → repo koren je nadrejeni imenik (na sys.path
# zato, da deluje absolutni uvoz `from benchmark.… import …`).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmark.benchmark_core   import find_binary, run_walkforward, portfolio_daily_returns, run_optimizer
from benchmark.benchmark_report import (plot_combined_walkforward,
                                         plot_all_algorithms_grid,
                                         plot_weights_grid,
                                         print_summary_table, export_csv,
                                         print_prediction_quality,
                                         print_turnover_and_costs,
                                         print_scenario_significance,
                                         print_efficiency_ranking,
                                         RESULTS_DIR)


def _load_optimizer_config(config: dict) -> dict:
    """Zlij solver hiperparametre (portfolio_optimizers/config.json) s problemskimi
    omejitvami (benchmark config, blok 'portfolio') → optimizer_config, kot ga
    pričakujeta benchmark_core in C++ most. Ločena lastnika: solver knobi so v
    portfolio_optimizers/config.json, problem (K, w_min, w_max, risk) v benchmark configu."""
    with open(os.path.join(ROOT, "portfolio_optimizers", "config.json")) as f:
        solver = json.load(f)
    pf = config["portfolio"]
    common = {
        "population_size": solver["common"]["population_size"],
        "num_generations": solver["common"]["num_generations"],
        "seed":            solver["common"]["seed"],
        "cardinality_K":   pf["cardinality_K"],
        "w_min":           pf["w_min"],
        "w_max":           pf["w_max"],
        "risk_parameter":  pf["risk_parameter"],
    }
    return {"common": common, "pso": solver["pso"],
            "sa": solver["sa"], "ga": solver["ga"]}


def run_everything(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    # Config struktura
    # tickers          → config["tickers"]
    # return_estimator → config["return_estimator"]  (MASTER-lite hiperparametri)
    # test_config      → config["test_config"]
    # portfolio        → config["portfolio"]  (problemske omejitve: K, w_min/max, risk)
    # optimizer_config → zlit iz portfolio_optimizers/config.json + portfolio blok
    # lookahead_days   → izračunan dinamično iz test_config
    # Baseline Σ = vzorčna kovarianca (benchmark_core._sample_cov) — brez configa.
    config["optimizer_config"] = _load_optimizer_config(config)
    tc = config.get("test_config", {})
    mc = config.get("return_estimator", {})

    tickers        = config["tickers"]
    train_start_dt = mc.get("start_date", "2015-01-01")
    train_end_dt   = mc.get("end_date",   "2023-01-01")
    test_start_dt  = tc.get("start_date", "2023-01-01")
    test_end_dt    = tc.get("end_date",   "2025-01-01")
    windows        = tc.get("windows", 5)

    # Dinamični lookahead: test_days = bdate_range / windows → ratio = 1×
    import pandas as _pd
    _t0 = _pd.Timestamp(test_start_dt)
    _t1 = _pd.Timestamp(test_end_dt)
    total_days = len(_pd.bdate_range(_t0, _t1))
    test_days  = max(1, total_days // windows)

    # Injiciraj lookahead nazaj v config za vse downstream komponente
    config["run_settings"] = config.get("run_settings", {})
    config["run_settings"]["lookahead_days"] = test_days
    config["run_settings"]["tickers"]        = tickers
    print(f"  Dinamični lookahead: test_days={test_days}d  "
          f"(windows={windows}, ratio=1×)")

    # Podatki
    print(f"\n{'='*65}\n  NALAGANJE PODATKOV\n{'='*65}")
    df_train = yf.download(tickers, start=train_start_dt, end=train_end_dt,
                           auto_adjust=True, progress=False)["Close"]
    df_train = df_train.ffill().dropna(axis=1, thresh=100)
    avail_tickers = list(df_train.columns)

    df_test = yf.download(avail_tickers, start=test_start_dt, end=test_end_dt,
                          auto_adjust=True, progress=False)["Close"]
    df_test = df_test.ffill().dropna(axis=1, thresh=100)
    avail_tickers = [t for t in avail_tickers if t in df_test.columns]
    df_train = df_train[avail_tickers]
    df_test  = df_test[avail_tickers]
    print(f"  Tikerji: {len(avail_tickers)}  "
          f"Trening: {df_train.index[0].date()} → {df_train.index[-1].date()}  "
          f"Test: {df_test.index[0].date()} → {df_test.index[-1].date()}")

    # Tržni indeks (pasivni benchmark, buy&hold)
    market_ticker = config.get("evaluation", {}).get("market_ticker", "SPY")
    market_prices = None
    try:
        _mkt = yf.download(market_ticker, start=test_start_dt, end=test_end_dt,
                           auto_adjust=True, progress=False)["Close"]
        market_prices = _mkt.ffill()
        if hasattr(market_prices, "columns"):           # DataFrame → Series
            market_prices = market_prices.iloc[:, 0]
        print(f"  Tržni indeks: {market_ticker} "
              f"({market_prices.index[0].date()} → {market_prices.index[-1].date()})")
    except Exception as e:
        print(f"  [opozorilo] Tržni indeks '{market_ticker}' ni na voljo: {e}")

    # Napovedni transformer je TACTiS-2 (spodaj); MASTER-lite je odstranjen.
    # Baseline Σ (vzorčna kovarianca) se računa drseče znotraj walk-forwarda.
    print(f"\n{'='*65}\n  NAPOVEDNI MODELI\n{'='*65}")

    timing = {}

    # Močan ML μ-baseline (LightGBM) — pošten primerjalni μ (gated)
    ml_registry = None
    if config.get("evaluation", {}).get("strong_baselines", {}).get("enabled", True):
        try:
            from estimators.simple_ml import train_simple_ml
            print("  [SimpleML] LightGBM faktorski μ-baseline (Gu-Kelly-Xiu 2020)...")
            t0 = time.time()
            ml_registry = train_simple_ml(config, df_train, avail_tickers,
                                          config["run_settings"]["lookahead_days"])
            timing["train_simpleml"] = time.time() - t0
            print(f"           → done in {timing['train_simpleml']:.1f}s")
        except Exception as e:
            print(f"  [SimpleML] preskočen ({type(e).__name__}: {e})")
            ml_registry = None

    # Sekvenčni LSTM μ-baseline (predlagana arhitekturna primerjava, gated)
    lstm_registry = None
    if config.get("evaluation", {}).get("strong_baselines", {}).get("enabled", True):
        try:
            from estimators.lstm_mu import train_lstm
            print("  [LSTM] Sekvenčni LSTM μ-baseline (Hochreiter & Schmidhuber 1997)...")
            t0 = time.time()
            lstm_registry = train_lstm(config, df_train, avail_tickers,
                                       config["run_settings"]["lookahead_days"])
            timing["train_lstm"] = time.time() - t0
            print(f"         → done in {timing['train_lstm']:.1f}s")
        except Exception as e:
            print(f"  [LSTM] preskočen ({type(e).__name__}: {e})")
            lstm_registry = None

    # TACTiS-2 skupni μ+Σ napovednik (gated: evaluation.tactis.enabled)
    # Transformer-attentional-copula (Ashok et al. 2024) — napove OBA momenta
    # iz skupne porazdelitve; alternativa MASTER-lite + scale-matchingu.
    tactis_registry = None
    if config.get("evaluation", {}).get("tactis", {}).get("enabled", False):
        try:
            from estimators.tactis_estimator import train_tactis
            print("  [TACTiS] TACTiS-2 skupni μ+Σ napovednik (Ashok et al. 2024)...")
            t0 = time.time()
            tactis_registry = train_tactis(config, df_train, avail_tickers,
                                           config["run_settings"]["lookahead_days"])
            timing["train_tactis"] = time.time() - t0
            print(f"          → done in {timing['train_tactis']:.1f}s")
        except Exception as e:
            print(f"  [TACTiS] preskočen ({type(e).__name__}: {e})")
            tactis_registry = None

    # MASTER na amerikai delnicah (gated: evaluation.master.enabled)
    # ORIGINALNI repo model (SJTU-DMTai/MASTER, Li et al. AAAI 2024), Alpha158 +
    # tržni gating zgrajena iz yfinance OHLCV (brez Qlib/CSI, teče na torch 2.x).
    master_registry = None
    if config.get("evaluation", {}).get("master", {}).get("enabled", False):
        try:
            from estimators.master_us import train_master_us
            print("  [MASTER] MASTER μ-napovednik na ameriških delnicah "
                  "(Li et al. AAAI 2024, originalni repo)...")
            t0 = time.time()
            master_registry = train_master_us(config, df_train, avail_tickers,
                                               config["run_settings"]["lookahead_days"])
            timing["train_master"] = time.time() - t0
            print(f"          → done in {timing['train_master']:.1f}s")
        except Exception as e:
            import traceback as _tb
            print(f"  [MASTER] preskočen ({type(e).__name__}: {e})")
            _tb.print_exc()
            master_registry = None

    # TFT (Temporal Fusion Transformer, Lim et al. 2021) μ-napovednik (gated:
    # evaluation.tft.enabled) — off-the-shelf pytorch-forecasting, splošni
    # napovedni transformer (μ-only, tretji transformer poleg TACTiS/MASTER).
    tft_registry = None
    if config.get("evaluation", {}).get("tft", {}).get("enabled", False):
        try:
            from estimators.tft_mu import train_tft
            print("  [TFT] Temporal Fusion Transformer μ-napovednik "
                  "(Lim et al. 2021, pytorch-forecasting)...")
            t0 = time.time()
            tft_registry = train_tft(config, df_train, avail_tickers,
                                     config["run_settings"]["lookahead_days"])
            timing["train_tft"] = time.time() - t0
            print(f"       → done in {timing['train_tft']:.1f}s")
        except Exception as e:
            import traceback as _tb
            print(f"  [TFT] preskočen ({type(e).__name__}: {e})")
            _tb.print_exc()
            tft_registry = None

    # Walk-forward
    binary = find_binary()
    print(f"\n  Optimizator: {binary}")
    t_wf_start = time.time()
    results = run_walkforward(
        config                = config,
        avail_tickers         = avail_tickers,
        df_train              = df_train,
        df_test               = df_test,
        market_prices         = market_prices,
        test_days             = test_days,
        windows               = windows,
        binary                = binary,
        ml_registry           = ml_registry,
        lstm_registry         = lstm_registry,
        tactis_registry       = tactis_registry,
        master_registry       = master_registry,
        tft_registry          = tft_registry,
    )
    timing["walkforward_total"] = time.time() - t_wf_start
    results["timing"] = timing
    print(f"\n  Časi: {', '.join(f'{k}={v:.0f}s' for k,v in sorted(timing.items()))}")

    # Hedge ensemble REMOVED (2026-08-25): equal-weight Hedge shows <0.1bp IC advantage over adaptive.
    # Files hedge_ensemble.py and mu_ensemble.py retained for reference/future work.

    ar  = results["all_results"]
    br  = results["baseline_results"]
    bw  = results.get("baseline_weights", {})
    arw = results["all_results_weights"]
    pq  = results.get("prediction_quality", {})

    # period_tag = obdobje testa; opcijski config["run_tag"] ga razširi, da
    # dva configa z ISTIM testnim obdobjem (npr. mega-cap vs divuniverse 2019–22)
    # ne pišeta čez iste artefakte (pickle/plots/CSV).
    period_tag = (f"{df_test.index[0].strftime('%Y%m')}_"
                  f"{df_test.index[-1].strftime('%Y%m')}")
    run_tag = config.get("run_tag", "")
    if run_tag:
        period_tag = f"{period_tag}_{run_tag}"

    # Trajno shrani surove rezultate PRED risanjem
    # Walk-forward je drag (~40 min): napaka v risanju/izvozu NE sme zavreči
    # izračuna. Najprej pickle, nato vsak izhodni korak posebej ovit v try/except.
    import pickle, traceback
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _dump = os.path.join(RESULTS_DIR, f"results_{period_tag}.pkl")
    with open(_dump, "wb") as _f:
        pickle.dump(results, _f)
    print(f"\n  Surovi rezultati shranjeni → {_dump}")

    # Izhodi
    print(f"\n{'='*65}\n  GRAFI & IZVOZ\n{'='*65}")
    tc_bps = config.get("evaluation", {}).get("transaction_cost_bps", 10.0)
    stat_test = config.get("evaluation", {}).get("stat_test", "wilcoxon")

    def _safe(fn, *a, **k):
        try:
            fn(*a, **k)
        except Exception:
            print(f"  [OPOZORILO] {getattr(fn, '__name__', fn)} spodletel — "
                  f"nadaljujem (rezultati so v {_dump}):")
            traceback.print_exc()

    _safe(plot_combined_walkforward, ar, br, df_test, test_days, period_tag)
    _safe(plot_all_algorithms_grid, ar, br, df_test, test_days, period_tag)
    _safe(plot_weights_grid, arw, df_test, test_days, avail_tickers, period_tag)
    _safe(print_summary_table, ar, br, df_test, test_days, tc_bps=tc_bps,
          weights=arw, baseline_weights=bw)
    _safe(print_prediction_quality, pq)
    _safe(print_turnover_and_costs, arw, ar, df_test, test_days, tc_bps=tc_bps)
    _safe(print_scenario_significance, ar, df_test, test_days, method=stat_test)
    _safe(export_csv, ar, br, arw, df_test, test_days, avail_tickers, period_tag,
          prediction_quality=pq, tc_bps=tc_bps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Pot do config.json")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Napaka: '{args.config}' ne obstaja!")
        sys.exit(1)
    run_everything(args.config)