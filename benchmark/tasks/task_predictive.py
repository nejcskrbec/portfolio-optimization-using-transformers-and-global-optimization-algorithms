#!/usr/bin/env python3
"""Predictive IC/RankIC benchmark (qlib-style cross-sectional evaluation)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from benchmark.utils.benchmark_utils import (
    ROOT, BENCH, CFG, CONFIGS, _load_optimizer_config,
    _print_forecast, _print_portfolio, _dry, _run_named_literature,
)


def task_predictive(argv=None):
    #!/usr/bin/env python3
    """
    STANDARDNI ZUNANJI qlib-stil napovedni benchmark.
    =================================================

    Napovedne modele (MASTER / TFT / PatchTST + zgodovinsko drsece povprecje) ovrednotimo po
    KANONICNEM presecnem protokolu, po katerem porocajo MASTER (Li in sod. 2024),
    DTML (Yoo in sod. 2021) in qlib: na sirokem indeksu (S&P 500) *vsak trgovalni
    dan* napovemo presecni signal za vse clane, ga koreliramo z realiziranim
    naprejnjim donosom in porocamo standardno cetverko

        IC        -- povprecna dnevna presecna Pearsonova korelacija napoved<->donos
        ICIR      -- IC information ratio = mean(IC_t) / std(IC_t)
        RankIC    -- povprecna dnevna Spearmanova (rang) korelacija
        RankICIR  -- mean(RankIC_t) / std(RankIC_t)

    ter dodatno LONG-SHORT decilni portfelj (dnevno rebalansiran: dolgo zgornji
    decil napovedi, kratko spodnji), z anualiziranim donosom in Sharpom -- t.i.
    qlib "GroupReturn". Vse mere so cisto napovedne (neodvisne od CCMV-optimizatorja).

    To je standardni zunanji napovedni benchmark, uporabljen v magistrski: sirok
    indeksni univerzum + dnevna presecna IC/RankIC.

    Uporaba (v okolju `magistrska`):
        python -u benchmark/run.py predictive

    Izhod: konzolna tabela + CSV v test_results/qlib_ic_benchmark.csv.
    """
    import os
    import sys
    import json
    import numpy as np
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RESULTS_DIR = os.path.join(ROOT, "test_results")
    ANN = 252.0


    # ----------------------------------------------------------------------------
    #  Presecne mere (en dan)
    # ----------------------------------------------------------------------------
    def _rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        return r


    def _ic_day(pred, real, rank=False):
        m = np.isfinite(pred) & np.isfinite(real)
        p, r = pred[m], real[m]
        if len(p) < 5 or p.std() < 1e-12 or r.std() < 1e-12:
            return np.nan
        if rank:
            p, r = _rank(p), _rank(r)
        return float(np.corrcoef(p, r)[0, 1])


    def _sharpe(daily, ann=ANN):
        d = np.asarray(daily, dtype=float)
        d = d[np.isfinite(d)]
        if len(d) < 2 or d.std(ddof=1) < 1e-12:
            return np.nan
        return float(d.mean() / d.std(ddof=1) * np.sqrt(ann))


    def _long_short_day(pred, fwd1, q):
        """Dnevni long-short donos (dolgo zgornji, kratko spodnji decil), enako utezen."""
        m = np.isfinite(pred) & np.isfinite(fwd1)
        p, r = pred[m], fwd1[m]
        n = len(p)
        if n < 10:
            return np.nan
        k = max(1, int(np.floor(q * n)))
        order = np.argsort(p)
        short = r[order[:k]].mean()
        long_ = r[order[-k:]].mean()
        return float(long_ - short)


    # ----------------------------------------------------------------------------
    def _load_universe(path, date_col):
        df = pd.read_csv(os.path.join(ROOT, path))
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        return df.ffill()


    def _enabled(config, key):
        return bool(config.get("evaluation", {}).get(key, {}).get("enabled", False))


    def run(config_path):
        with open(config_path) as f:
            config = json.load(f)
        pc = config.get("predictive", {})
        mc = config.get("return_estimator", {})
        tc = config.get("test_config", {})

        H = int(pc.get("horizon_days", 5))
        q = float(pc.get("long_short_decile", 0.1))
        train_end = pd.Timestamp(mc.get("end_date", "2020-12-30"))
        test_start = pd.Timestamp(tc.get("start_date", "2021-01-01"))
        test_end = pd.Timestamp(tc.get("end_date", "2021-12-29"))
        config["run_settings"] = {"lookahead_days": H}

        print("=" * 72)
        print("  qlib-STIL NAPOVEDNI BENCHMARK  (dnevna presecna IC na sirokem indeksu)")
        print("=" * 72)
        prices = _load_universe(pc["universe_csv"], pc.get("date_column", "Date"))

        # Neobvezna omejitev univerzuma (predictive.max_universe): izberi N najbolj
        # popolnih serij. Uporabno je za hitre/smoke eksperimente; pri glavnem qlib
        # benchmarku lahko vsi trije transformerji tecejo na istem sirokem univerzumu.
        cap = int(pc.get("max_universe", 0))
        if cap and cap < prices.shape[1]:
            completeness = prices.notna().sum().sort_values(ascending=False)
            keep = sorted(completeness.index[:cap])
            prices = prices[keep]

        tickers = list(prices.columns)
        df_train = prices.loc[:train_end]
        print(f"  Univerzum: {len(tickers)} clanov  |  horizont H={H}d  |  "
              f"train -> {train_end.date()}  test {test_start.date()}..{test_end.date()}"
              + (f"  (omejeno na {cap} najbolj popolnih)" if cap else ""))

        # --- Ucenje (enkrat, z diskom cache) --------------------------------------
        import pickle
        from pathlib import Path as _Path
        CACHE_DIR = _Path(ROOT) / "test_results" / "predictive_cache"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        def _cache_path(name):
            return CACHE_DIR / f"{name}_registry.pkl"

        # PatchTST pickles reference the `layers` package inside its source tree
        _PATCHTST_SRC = str(_Path(ROOT) / "estimators" / "patchtst_official" / "PatchTST_supervised")

        def _load_or_train(name, train_fn, get_fn, *train_args):
            cp = _cache_path(name)
            if cp.exists():
                print(f"\n  [{name}] nalozitev iz cache: {cp}")
                if name == "PatchTST" and _PATCHTST_SRC not in sys.path:
                    sys.path.insert(0, _PATCHTST_SRC)
                with open(cp, "rb") as f:
                    reg = pickle.load(f)
                return reg, get_fn
            print(f"\n  Ucenje {name} ...")
            reg = train_fn(*train_args)
            with open(cp, "wb") as f:
                pickle.dump(reg, f)
            print(f"  [{name}] shranjen v cache: {cp}")
            return reg, get_fn

        registries = {}
        if _enabled(config, "master"):
            from estimators.master_us import train_master_us, get_mu_master_us
            registries["MASTER"] = _load_or_train(
                "MASTER", train_master_us, get_mu_master_us,
                config, df_train, tickers, H)
        if _enabled(config, "tft"):
            from estimators.tft_mu import train_tft, get_mu_tft
            registries["TFT"] = _load_or_train(
                "TFT", train_tft, get_mu_tft,
                config, df_train, tickers, H)
        if _enabled(config, "patchtst"):
            from estimators.patchtst_mu import train_patchtst, get_mu_patchtst
            registries["PatchTST"] = _load_or_train(
                "PatchTST", train_patchtst, get_mu_patchtst,
                config, df_train, tickers, H)

        seq_len = int(mc.get("sequence_length", 45))
        roll = max(seq_len, 21)
        sources = ["Zgodovinski"] + list(registries.keys())

        # --- Dnevna presecna zanka -------------------------------------------------
        dates = prices.index
        test_dates = [d for d in dates if test_start <= d <= test_end]
        ic = {s: [] for s in sources}          # dnevni IC
        ric = {s: [] for s in sources}         # dnevni RankIC
        ls = {s: [] for s in sources}          # dnevni long-short donos
        px = prices.values
        pos_of = {d: i for i, d in enumerate(dates)}

        print(f"\n  Dnevna inferenca ({len(test_dates)} dni) ...")
        for n_done, d in enumerate(test_dates):
            i = pos_of[d]
            if i + H >= len(dates):
                break
            cur = px[i]
            fwdH = px[i + H] / cur - 1.0                 # naprejnji H-dnevni donos
            fwd1 = px[i + 1] / cur - 1.0                 # naprejnji 1-dnevni (za long-short)
            df_hist = prices.iloc[:i + 1]

            # Zgodovinski drseci mu.
            hist_rets = df_hist.pct_change().tail(roll)
            mu_hist = hist_rets.mean().values * H
            preds = {"Zgodovinski": mu_hist}
            for name, (reg, get_mu) in registries.items():
                preds[name] = np.asarray(get_mu(reg, df_hist, tickers, H), dtype=float)

            for s in sources:
                ic[s].append(_ic_day(preds[s], fwdH, rank=False))
                ric[s].append(_ic_day(preds[s], fwdH, rank=True))
                ls[s].append(_long_short_day(preds[s], fwd1, q))
            if (n_done + 1) % 25 == 0:
                print(f"    {n_done + 1:3d}/{len(test_dates)}  "
                      + "  ".join(f"{s}:RankIC~{np.nanmean(ric[s]):+.3f}" for s in sources))

        # --- Agregacija ------------------------------------------------------------
        def _mir(a):
            a = np.array([v for v in a if np.isfinite(v)], dtype=float)
            if len(a) < 2:
                return np.nan, np.nan
            return float(a.mean()), float(a.mean() / (a.std(ddof=1) + 1e-12))

        rows = []
        for s in sources:
            m_ic, icir = _mir(ic[s])
            m_ric, ricir = _mir(ric[s])
            ls_arr = np.array([v for v in ls[s] if np.isfinite(v)], dtype=float)
            rows.append({
                "vir": s, "IC": m_ic, "ICIR": icir, "RankIC": m_ric, "RankICIR": ricir,
                "LS_ann_ret": float(ls_arr.mean() * ANN) if len(ls_arr) else np.nan,
                "LS_Sharpe": _sharpe(ls_arr), "n_dni": int(np.isfinite(ic[s]).sum()),
            })

        # --- Izpis -----------------------------------------------------------------
        print("\n" + "=" * 72)
        print("  REZULTATI  (qlib standard; visje IC/ICIR/RankIC/RankICIR = boljse)")
        print("=" * 72)
        hdr = (f"  {'vir':<14}{'IC':>8}{'ICIR':>8}{'RankIC':>9}{'RankICIR':>10}"
               f"{'LS ann%':>9}{'LS Sharpe':>11}")
        print(hdr + "\n  " + "-" * (len(hdr) - 2))
        for r in rows:
            print(f"  {r['vir']:<14}{r['IC']:>8.4f}{r['ICIR']:>8.2f}{r['RankIC']:>9.4f}"
                  f"{r['RankICIR']:>10.2f}{r['LS_ann_ret'] * 100:>9.2f}"
                  f"{r['LS_Sharpe']:>11.2f}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "qlib_ic_benchmark.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  CSV -> {out}")
        return rows

    return run(str(CFG / "config_qlib_ic.json"))

