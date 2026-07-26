"""
benchmark/significance_pooled.py
================================
Bolje-močni testi statistične značilnosti (odgovor na "kako dokazati značilnost"
pri ~30–48 oknih na režim, kjer posamezni režim nima moči):

  #1 SIGNAL-NIVO (pooled rank-IC): namesto ~40 okenskih Sharpov testiramo NAPOVED
     samo. Za vsak scenarij zložimo per-okno rank-IC serijo ČEZ VSE režime (n ~250)
     in poročamo povprečni IC z Newey-West (HAC) t-statistiko (informacijski
     količnik). Nato PARNI IC-razlike (Transformer − baseline) po oknu, zložene
     čez režime → neposreden test "transformerjeva napoved je bolj napovedljiva".
       Reference: Newey & West (1987) HAC; Grinold & Kahn (IC information ratio);
       Diebold & Mariano (1995) forecast-comparison logika.

  #2 PORTFELJ-NIVO (pooled per-okno donos): per-okno kumulativni donos vsakega
     scenarija, RAZLIKA (Transformer − baseline) zložena čez vse režime → en test
     z n ~245 namesto šest šibkih. t-test + Wilcoxon + predznačni test + režimsko
     grupiran (cluster-robust) SE.

Uporablja SAMO že-piklane rezultate (test_results/results_*.pkl) — nič ne požene
znova. Zaženi iz repo roota:  python benchmark/significance_pooled.py
"""
import csv
import os
import pickle

import numpy as np
from scipy import stats

# zbrane vrstice za CSV/markdown izvoz (napolnijo test1/test2)
_IC_ROWS = []        # signal-nivo (a)+(b)
_RET_ROWS = []       # portfelj-nivo (#2), po algoritmih
_SHARPE_ROWS = []    # #3 Sharpe-difference test
_DSR_ROWS = []       # #4 Deflated Sharpe Ratio
_FDR_ROWS = []       # #5 Benjamini-Hochberg FDR

EULER = 0.5772156649015329   # Euler-Mascheroni (za DSR SR0)
ANN = np.sqrt(252.0)         # anualizacija Sharpa (dnevni → letni)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "test_results")

# pkl datoteka → berljivo ime režima
REGIME_MAP = {
    "results_200701_201012_gfc2008.pkl":     "GFC-2008",
    "results_201301_201712.pkl":             "2013-2017",
    "results_201906_202112_covid2020.pkl":   "COVID-2020",
    "results_202201_202412.pkl":             "2022-2025",
    "results_201901_202212_divuniverse.pkl": "divuniverse",
    "results_201901_202212.pkl":             "headline-megacap",
}

# glavni par-primerjave (A vs B) za teze
BASELINES = ["LSTM", "BlackLitterman", "Zgodovinski", "SimpleML"]
ALGO = "pso"   # reprezentativna metahevristika (kot v poročilih)


def _nw_tstat(x, L=None):
    """Newey-West (HAC, Bartlett) t-statistika za H0: mean(x)=0.
    Regresija na konstanto → dolgoročna varianca povprečja."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 or np.allclose(x.std(), 0):
        return np.nan, np.nan, n
    mu = x.mean()
    e = x - mu
    if L is None:
        L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))   # standardni avtomat. lag
    L = max(0, min(L, n - 1))
    g0 = np.mean(e * e)
    lrv = g0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)                              # Bartlett utež
        gl = np.mean(e[l:] * e[:-l])
        lrv += 2.0 * w * gl
    lrv = max(lrv, 1e-18)
    se = np.sqrt(lrv / n)
    t = mu / se
    p = 2.0 * stats.t.sf(abs(t), df=n - 1)
    return t, p, n


def _cluster_tstat(vals, groups):
    """Po-režimu grupiran (cluster-robust) t za povprečje: povprečje skupinskih
    povprečij, SE iz medskupinske variance (G majhen → t z G-1 df)."""
    gmeans = [np.mean(v) for v in groups.values() if len(v)]
    gmeans = np.asarray(gmeans, dtype=float)
    G = len(gmeans)
    if G < 2:
        return np.nan, np.nan, G
    m = gmeans.mean()
    se = gmeans.std(ddof=1) / np.sqrt(G)
    if se < 1e-18:
        return np.nan, np.nan, G
    t = m / se
    p = 2.0 * stats.t.sf(abs(t), df=G - 1)
    return t, p, G


# pooled dnevni donosi (za Sharpe-testa)
def _pool_daily(regimes, scen, algo):
    """Zloži DNEVNE donose scenarija čez vse režime v eno serijo."""
    out = []
    for d in regimes.values():
        ar = d.get("all_results", {})
        if scen in ar and algo in ar[scen]:
            for w in ar[scen][algo]:
                out.append(np.asarray(w, dtype=float))
    return np.concatenate(out) if out else None


def _pool_daily_pair(regimes, sa, sb, algo):
    """Poravnano (dan-za-dan, isto okno) zloži dnevne donose dveh scenarijev."""
    aa, bb = [], []
    for d in regimes.values():
        ar = d.get("all_results", {})
        if sa in ar and sb in ar and algo in ar[sa] and algo in ar[sb]:
            wa, wb = ar[sa][algo], ar[sb][algo]
            for i in range(min(len(wa), len(wb))):
                ra = np.asarray(wa[i], float); rb = np.asarray(wb[i], float)
                k = min(len(ra), len(rb))
                aa.append(ra[:k]); bb.append(rb[:k])
    if not aa:
        return None, None
    return np.concatenate(aa), np.concatenate(bb)


def _hac_cov(V, L=None):
    """Newey-West (Bartlett) dolgoročna kovarianca vrstic matrike V (T×k)."""
    V = np.asarray(V, float)
    T = V.shape[0]
    Vc = V - V.mean(0)
    if L is None:
        L = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    S = (Vc.T @ Vc) / T
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = (Vc[l:].T @ Vc[:-l]) / T
        S += w * (G + G.T)
    return S


# #3  Sharpe-difference test (Ledoit-Wolf 2008 / Memmel 2003, HAC delta)
def _sharpe_diff_test(ra, rb):
    """H0: SR_a = SR_b. Delta-metoda z Newey-West HAC kovarianco momentov
    (ma,mb,E[a²],E[b²]); anualiziran ΔSharpe, a test je invarianten na skalo."""
    ra = np.asarray(ra, float); rb = np.asarray(rb, float)
    T = len(ra)
    ma, mb = ra.mean(), rb.mean()
    ga, gb = (ra * ra).mean(), (rb * rb).mean()
    va, vb = ga - ma * ma, gb - mb * mb
    sr_a, sr_b = ma / np.sqrt(va), mb / np.sqrt(vb)
    diff = sr_a - sr_b
    grad = np.array([ga / va ** 1.5, -gb / vb ** 1.5,
                     -0.5 * ma / va ** 1.5, 0.5 * mb / vb ** 1.5])
    S = _hac_cov(np.column_stack([ra, rb, ra * ra, rb * rb]))
    var = float(grad @ S @ grad) / T
    se = np.sqrt(max(var, 1e-18))
    z = diff / se
    p = 2.0 * stats.norm.sf(abs(z))
    return sr_a * ANN, sr_b * ANN, diff * ANN, z, p, T


def _sb_indices(T, meanblock, rng):
    """Politis-Romano stacionarni bootstrap indeksi (geometrijske bloke)."""
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(T)
    newblk = rng.random(T) < (1.0 / meanblock)
    for i in range(1, T):
        idx[i] = rng.integers(T) if newblk[i] else (idx[i - 1] + 1) % T
    return idx


def _sharpe_diff_boot_p(ra, rb, diff_obs_ann, B=1000, meanblock=10, seed=0):
    """Robustnostni p za ΔSharpe: stacionarni bootstrap, dvostranski, recentriran."""
    rng = np.random.default_rng(seed)
    ra = np.asarray(ra, float); rb = np.asarray(rb, float)
    T = len(ra)
    diffs = np.empty(B)
    for b in range(B):
        ix = _sb_indices(T, meanblock, rng)
        a, c = ra[ix], rb[ix]
        ma, mb = a.mean(), c.mean()
        va, vb = (a * a).mean() - ma * ma, (c * c).mean() - mb * mb
        diffs[b] = (ma / np.sqrt(va) - mb / np.sqrt(vb)) * ANN
    centered = diffs - diffs.mean()
    return float(np.mean(np.abs(centered) >= abs(diff_obs_ann)))


def test3_sharpe_diff(regimes, algo=ALGO):
    print("\n" + "=" * 82)
    print(f"  #3  SHARPE-DIFFERENCE  —  pooled dnevni donosi, ΔSharpe (letni), "
          f"HAC + bootstrap (algo={algo})")
    print("=" * 82)
    print(f"  {'Primerjava':<24}{'SR_T':>7}{'SR_B':>7}{'ΔSR':>8}{'z':>7}"
          f"{'HAC p':>9}{'boot p':>9}{'T dni':>8}")
    print("  " + "-" * 78)
    for base in BASELINES:
        ra, rb = _pool_daily_pair(regimes, "Transformer", base, algo)
        if ra is None:
            continue
        sr_a, sr_b, diff, z, p, T = _sharpe_diff_test(ra, rb)
        bp = _sharpe_diff_boot_p(ra, rb, diff)
        star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"  {'Transf − ' + base:<24}{sr_a:>7.2f}{sr_b:>7.2f}{diff:>8.2f}"
              f"{z:>7.2f}{p:>9.4f}{bp:>9.4f}{T:>8}  {star}")
        _SHARPE_ROWS.append({"algo": algo, "vs": base, "sr_transf": sr_a,
                             "sr_base": sr_b, "delta_sr": diff, "z": z,
                             "hac_p": p, "boot_p": bp, "T": T})
    print("\n  ΔSR = letni Sharpe(Transformer) − Sharpe(baseline) na zloženih dnevnih "
          "donosih;\n  HAC p = Ledoit-Wolf/Memmel delta-metoda (Newey-West), boot p = "
          "stacionarni bootstrap (1000×).")


# #4  Deflated Sharpe Ratio (Bailey & López de Prado 2014)
def _dsr(returns, trial_sharpes):
    r = np.asarray(returns, float)
    T = len(r)
    sd = r.std(ddof=1)
    sr = r.mean() / sd                              # ne-anualiziran (per-dan)
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))     # ne-presežna (Pearson)
    ts = np.asarray(trial_sharpes, float)
    ts = ts[np.isfinite(ts)]
    N = len(ts)
    V = np.var(ts, ddof=1)
    z1 = stats.norm.ppf(1.0 - 1.0 / N)
    z2 = stats.norm.ppf(1.0 - 1.0 / (N * np.e))
    sr0 = np.sqrt(V) * ((1.0 - EULER) * z1 + EULER * z2)
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))
    return sr * ANN, sr0 * ANN, dsr, T, N


def _all_trial_sharpes(regimes):
    """Ne-anualizirani Sharpi vsake (scenarij×algo) strategije na zloženih dnevnih
    donosih — ocena 'števila poskusov' + disperzije za SR0."""
    algos = ["pso", "sa", "ga"]
    scens = ["Transformer", "Ansambel", "LSTM", "SimpleML", "BlackLitterman",
             "Zgodovinski", "Zgodovinski-LW", "Transformer-LWΣ"]
    out = {}
    for s in scens:
        for a in algos:
            r = _pool_daily(regimes, s, a)
            if r is not None and r.std(ddof=1) > 0:
                out[(s, a)] = r.mean() / r.std(ddof=1)
    return out


def test4_deflated_sharpe(regimes):
    print("\n" + "=" * 82)
    print("  #4  DEFLATED SHARPE RATIO  —  popravek na izbiro (Bailey & LdP 2014)")
    print("=" * 82)
    trials = _all_trial_sharpes(regimes)
    ts_vals = list(trials.values())
    N = len(ts_vals)
    print(f"  Poskusov (scenarij×algo, pooled): N={N}   "
          f"(SR0 = pričakovani MAX Sharpe pod H0 čez N poskusov)")
    print(f"  {'Strategija (pso)':<24}{'SR (let.)':>10}{'SR0 (let.)':>11}"
          f"{'DSR=P(SR>0)':>13}{'T dni':>8}")
    print("  " + "-" * 78)
    for scen in ["Transformer", "Ansambel", "Zgodovinski", "LSTM",
                 "BlackLitterman", "SimpleML"]:
        r = _pool_daily(regimes, scen, ALGO)
        if r is None:
            continue
        sr, sr0, dsr, T, Nn = _dsr(r, ts_vals)
        mark = "✓ >0.95" if dsr > 0.95 else ("~0.90" if dsr > 0.90 else "")
        print(f"  {scen:<24}{sr:>10.2f}{sr0:>11.2f}{dsr:>13.3f}{T:>8}  {mark}")
        _DSR_ROWS.append({"scenario": scen, "sr_ann": sr, "sr0_ann": sr0,
                          "dsr": dsr, "T": T, "n_trials": Nn})
    print("\n  DSR > 0.95 ⇒ Sharpe ostane značilno > 0 tudi PO popravku na to, da smo "
          "preizkusili N strategij.")


# #5  Benjamini-Hochberg FDR čez vse parne p (iz #2)
def _bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


def test5_fdr():
    print("\n" + "=" * 82)
    print("  #5  BENJAMINI-HOCHBERG FDR  —  popravek čez vse parne teste (#2, "
          "4 algo × 4 baseline)")
    print("=" * 82)
    if not _RET_ROWS:
        print("  (ni zbranih p-vrednosti — zaženi #2 najprej)")
        return
    labels = [f"{r['algo']}: Transf−{r['vs']}" for r in _RET_ROWS]
    praw = [r["ttest_p"] for r in _RET_ROWS]
    padj = _bh(praw)
    print(f"  {'Test':<26}{'raw p':>9}{'BH q':>9}{'FDR<0.05':>10}{'FDR<0.10':>10}")
    print("  " + "-" * 70)
    for lab, pr, pa in sorted(zip(labels, praw, padj), key=lambda x: x[2]):
        print(f"  {lab:<26}{pr:>9.4f}{pa:>9.4f}"
              f"{('  ✓' if pa < 0.05 else '  ·'):>10}"
              f"{('  ✓' if pa < 0.10 else '  ·'):>10}")
        _FDR_ROWS.append({"test": lab, "raw_p": pr, "bh_q": pa,
                          "sig_005": pa < 0.05, "sig_010": pa < 0.10})
    n_pass = int(np.sum(padj < 0.05))
    print(f"\n  {n_pass}/{len(padj)} testov preživi FDR<0.05 (Benjamini-Hochberg).")


def load_regimes():
    out = {}
    for fn, name in REGIME_MAP.items():
        p = os.path.join(RESULTS_DIR, fn)
        if os.path.isfile(p):
            out[name] = pickle.load(open(p, "rb"))
    return out


def _win_cum(daily_list):
    """per-okno kumulativni donos iz seznama dnevnih-donos arrayev."""
    return np.array([np.prod(1.0 + np.asarray(r, dtype=float)) - 1.0
                     for r in daily_list], dtype=float)


# #1  SIGNAL-NIVO: pooled rank-IC
def test1_pooled_ic(regimes):
    print("\n" + "=" * 82)
    print("  #1  SIGNAL-NIVO  —  pooled rank-IC čez vse režime (Newey-West HAC)")
    print("=" * 82)

    # (a) povprečni IC + HAC t za vsak scenarij (zložen čez režime)
    scen_ic = {}
    for name, d in regimes.items():
        pq = d.get("prediction_quality", {})
        for scen, wins in pq.items():
            ics = [w["ic"] for w in wins if np.isfinite(w.get("ic", np.nan))]
            scen_ic.setdefault(scen, {})[name] = np.asarray(ics, dtype=float)

    print("\n  (a) Povprečni IC (pooled), Newey-West t, in po-režimu grupiran t:")
    print(f"  {'Scenarij':<16}{'mean IC':>9}{'NW t':>8}{'NW p':>9}"
          f"{'clust t':>9}{'clust p':>9}{'N oken':>8}")
    print("  " + "-" * 78)
    for scen in ["Transformer", "Ansambel", "LSTM", "SimpleML",
                 "BlackLitterman", "Zgodovinski"]:
        if scen not in scen_ic:
            continue
        groups = scen_ic[scen]
        pooled = np.concatenate([groups[r] for r in groups])
        t, p, n = _nw_tstat(pooled)
        ct, cp, G = _cluster_tstat(pooled, groups)
        star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"  {scen:<16}{pooled.mean():>9.4f}{t:>8.2f}{p:>9.4f}"
              f"{ct:>9.2f}{cp:>9.4f}{n:>8}  {star}")
        _IC_ROWS.append({"kind": "level", "scenario": scen, "vs": "",
                         "mean_ic": pooled.mean(), "nw_t": t, "nw_p": p,
                         "cluster_t": ct, "cluster_p": cp, "n": n})

    # (b) PARNI IC-razlike Transformer − baseline, po oknu, pooled čez režime
    print("\n  (b) Parne IC-razlike  Transformer − baseline  (pooled po oknu):")
    print(f"  {'Primerjava':<24}{'Δmean IC':>10}{'NW t':>8}{'NW p':>9}"
          f"{'Wilcox p':>10}{'%T>B':>7}{'N':>6}")
    print("  " + "-" * 78)
    T = scen_ic.get("Transformer", {})
    for base in BASELINES:
        B = scen_ic.get(base, {})
        diffs = []
        for r in T:
            if r in B:
                m = min(len(T[r]), len(B[r]))
                diffs.append(T[r][:m] - B[r][:m])
        if not diffs:
            continue
        dd = np.concatenate(diffs)
        t, p, n = _nw_tstat(dd)
        try:
            _, wp = stats.wilcoxon(dd)
        except Exception:
            wp = np.nan
        win = float(np.mean(dd > 0)) * 100
        star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"  {'Transf − ' + base:<24}{dd.mean():>10.4f}{t:>8.2f}{p:>9.4f}"
              f"{wp:>10.4f}{win:>7.0f}{n:>6}  {star}")
        _IC_ROWS.append({"kind": "paired", "scenario": "Transformer", "vs": base,
                         "mean_ic": dd.mean(), "nw_t": t, "nw_p": p,
                         "cluster_t": "", "cluster_p": "", "n": n,
                         "wilcoxon_p": wp, "pct_T_gt_B": win})


# #2  PORTFELJ-NIVO: pooled per-okno donos
def test2_pooled_returns(regimes, algo=ALGO):
    print("\n" + "=" * 82)
    print(f"  #2  PORTFELJ-NIVO  —  pooled per-okno kum. donos, razlika čez režime "
          f"(algo={algo})")
    print("=" * 82)
    print(f"  {'Primerjava':<24}{'Δmean%':>9}{'t p':>9}{'NW p':>9}"
          f"{'Wilcox p':>10}{'sign p':>9}{'clust p':>9}{'N':>6}")
    print("  " + "-" * 78)

    # zberi per-okno kum. donose po scenariju/režimu
    def cums(scen):
        g = {}
        for name, d in regimes.items():
            ar = d.get("all_results", {})
            if scen in ar and algo in ar[scen]:
                g[name] = _win_cum(ar[scen][algo])
        return g

    T = cums("Transformer")
    for base in BASELINES:
        B = cums(base)
        diffs, groups = [], {}
        for r in T:
            if r in B:
                m = min(len(T[r]), len(B[r]))
                dr = T[r][:m] - B[r][:m]
                diffs.append(dr); groups[r] = dr
        if not diffs:
            continue
        dd = np.concatenate(diffs)
        _, tp = stats.ttest_1samp(dd, 0.0)
        _, nwp, n = _nw_tstat(dd)
        try:
            _, wp = stats.wilcoxon(dd)
        except Exception:
            wp = np.nan
        npos = int(np.sum(dd > 0)); ntot = int(np.sum(dd != 0))
        sp = stats.binomtest(npos, ntot, 0.5).pvalue if ntot else np.nan
        _, cp, _ = _cluster_tstat(dd, groups)
        star = "***" if tp < 0.01 else ("**" if tp < 0.05 else ("*" if tp < 0.1 else ""))
        print(f"  {'Transf − ' + base:<24}{dd.mean()*100:>9.2f}{tp:>9.4f}{nwp:>9.4f}"
              f"{wp:>10.4f}{sp:>9.4f}{cp:>9.4f}{n:>6}  {star}")
        _RET_ROWS.append({"algo": algo, "scenario": "Transformer", "vs": base,
                          "mean_diff_pct": dd.mean() * 100, "ttest_p": tp,
                          "nw_p": nwp, "wilcoxon_p": wp, "sign_p": sp,
                          "cluster_p": cp, "n": n})
    print("\n  *** p<0.01  ** p<0.05  * p<0.1   |   NW=Newey-West HAC, "
          "clust=po-režimu grupiran, sign=predznačni (binom) test")


def _export(regimes):
    """Zapiši pooled rezultate v CSV + markdown (za tezo) v test_results/."""
    outdir = os.path.abspath(RESULTS_DIR)
    ic_csv = os.path.join(outdir, "significance_pooled_signal.csv")
    ret_csv = os.path.join(outdir, "significance_pooled_returns.csv")
    sharpe_csv = os.path.join(outdir, "significance_pooled_sharpe.csv")
    dsr_csv = os.path.join(outdir, "significance_pooled_dsr.csv")
    fdr_csv = os.path.join(outdir, "significance_pooled_fdr.csv")
    md = os.path.join(outdir, "significance_pooled.md")

    if _IC_ROWS:
        keys = ["kind", "scenario", "vs", "mean_ic", "nw_t", "nw_p",
                "cluster_t", "cluster_p", "wilcoxon_p", "pct_T_gt_B", "n"]
        with open(ic_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in _IC_ROWS:
                w.writerow({k: r.get(k, "") for k in keys})
    if _RET_ROWS:
        keys = ["algo", "scenario", "vs", "mean_diff_pct", "ttest_p", "nw_p",
                "wilcoxon_p", "sign_p", "cluster_p", "n"]
        with open(ret_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in _RET_ROWS:
                w.writerow({k: r.get(k, "") for k in keys})
    if _SHARPE_ROWS:
        keys = ["algo", "vs", "sr_transf", "sr_base", "delta_sr", "z",
                "hac_p", "boot_p", "T"]
        with open(sharpe_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in _SHARPE_ROWS:
                w.writerow({k: r.get(k, "") for k in keys})
    if _DSR_ROWS:
        keys = ["scenario", "sr_ann", "sr0_ann", "dsr", "T", "n_trials"]
        with open(dsr_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in _DSR_ROWS:
                w.writerow({k: r.get(k, "") for k in keys})
    if _FDR_ROWS:
        keys = ["test", "raw_p", "bh_q", "sig_005", "sig_010"]
        with open(fdr_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in _FDR_ROWS:
                w.writerow({k: r.get(k, "") for k in keys})

    # markdown povzetek (pso reprezentativno + signal-nivo)
    with open(md, "w") as f:
        f.write("# Pooled significance across regimes\n\n")
        f.write(f"Regimes pooled ({len(regimes)}): "
                f"{', '.join(regimes.keys())}\n\n")
        f.write("## #1 Signal level — pooled rank-IC (mean IC / t)\n\n")
        f.write("| Scenario | mean IC | Newey-West t | NW p | cluster-t | cluster-p | N |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in _IC_ROWS:
            if r["kind"] != "level":
                continue
            f.write(f"| {r['scenario']} | {r['mean_ic']:+.4f} | {r['nw_t']:.2f} | "
                    f"{r['nw_p']:.4f} | {r['cluster_t']:.2f} | {r['cluster_p']:.4f} | {r['n']} |\n")
        f.write("\n## #2 Portfolio level — pooled per-window return diff "
                "(algo=pso)\n\n")
        f.write("| Comparison | Δ/window% | t-test p | Wilcoxon p | sign p | cluster-p | N |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in _RET_ROWS:
            if r["algo"] != ALGO:
                continue
            f.write(f"| Transf − {r['vs']} | {r['mean_diff_pct']:+.2f} | "
                    f"{r['ttest_p']:.4f} | {r['wilcoxon_p']:.4f} | {r['sign_p']:.4f} | "
                    f"{r['cluster_p']:.4f} | {r['n']} |\n")
        f.write("\n_Per-window pool treats non-overlapping monthly windows as "
                "~independent (n≈254, well-powered); cluster-p treats each regime "
                "as one unit (6 clusters, most conservative). Report both._\n")

        if _SHARPE_ROWS:
            f.write("\n## #3 Sharpe-difference test (pooled daily, algo=pso)\n\n")
            f.write("Ledoit-Wolf (2008) / Memmel (2003) HAC delta method + "
                    "stationary bootstrap. ΔSR annualized.\n\n")
            f.write("| Comparison | SR Transf | SR base | ΔSR | z | HAC p | boot p | T |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in _SHARPE_ROWS:
                if r["algo"] != ALGO:
                    continue
                f.write(f"| Transf − {r['vs']} | {r['sr_transf']:.2f} | "
                        f"{r['sr_base']:.2f} | {r['delta_sr']:+.2f} | {r['z']:.2f} | "
                        f"{r['hac_p']:.4f} | {r['boot_p']:.4f} | {r['T']} |\n")

        if _DSR_ROWS:
            f.write("\n## #4 Deflated Sharpe Ratio (Bailey & López de Prado 2014)\n\n")
            nt = _DSR_ROWS[0]["n_trials"]
            f.write(f"Selection-adjusted over N={nt} trials (scenario×algo). "
                    f"DSR>0.95 ⇒ Sharpe stays significantly >0 after correcting "
                    f"for the search. SR annualized.\n\n")
            f.write("| Strategy (pso) | SR (ann) | SR0 (ann) | DSR=P(SR>0) | T |\n")
            f.write("|---|---|---|---|---|\n")
            for r in _DSR_ROWS:
                f.write(f"| {r['scenario']} | {r['sr_ann']:.2f} | {r['sr0_ann']:.2f} "
                        f"| {r['dsr']:.3f} | {r['T']} |\n")

        if _FDR_ROWS:
            f.write("\n## #5 Benjamini-Hochberg FDR (across all #2 pairwise tests)\n\n")
            npass = sum(1 for r in _FDR_ROWS if r["sig_005"])
            f.write(f"{npass}/{len(_FDR_ROWS)} of the 4-algo × 4-baseline return "
                    f"tests survive FDR<0.05.\n\n")
            f.write("| Test | raw p | BH q | FDR<0.05 |\n")
            f.write("|---|---|---|---|\n")
            for r in sorted(_FDR_ROWS, key=lambda x: x["bh_q"]):
                f.write(f"| {r['test']} | {r['raw_p']:.4f} | {r['bh_q']:.4f} | "
                        f"{'✓' if r['sig_005'] else '·'} |\n")

    outs = [ic_csv, ret_csv]
    if _SHARPE_ROWS: outs.append(sharpe_csv)
    if _DSR_ROWS: outs.append(dsr_csv)
    if _FDR_ROWS: outs.append(fdr_csv)
    outs.append(md)
    print("\n  Izvoz:\n    " + "\n    ".join(outs))


def main():
    regimes = load_regimes()
    print(f"\nNaloženi režimi ({len(regimes)}): {list(regimes.keys())}")
    test1_pooled_ic(regimes)
    test2_pooled_returns(regimes, ALGO)
    # robustnost: ponovi #2 za ostale metahevristike
    for a in ["sa", "ga"]:
        test2_pooled_returns(regimes, a)
    test3_sharpe_diff(regimes, ALGO)
    test4_deflated_sharpe(regimes)
    test5_fdr()
    _export(regimes)
    print()


if __name__ == "__main__":
    main()
