"""
benchmark/benchmark_strategies.py
=================================
Klasične benchmark strategije za historični walk-forward. Izbrane tako, da
IZOLIRAJO prednosti/slabosti hibridnega pristopa (transformer μ + kardinalno
omejena metahevristika):

  • GMV  (Global Minimum Variance) — uporablja SAMO Σ, brez μ.
    Klasičen rezultat (DeMiguel, Garlappi & Uppal 2009): napaka ocene μ pogosto
    naredi mean-variance slabši od strategij ki μ ignorirajo. Če hibrid premaga
    GMV, transformer-μ doda vrednost; sicer je to njegova slabost.

  • Markowitz Max-Sharpe (tangentni) — long-only, BREZ kardinalne omejitve,
    ISTI μ in Σ kot dobijo optimizatorji, rešeno konveksno na vseh N sredstvih.
    Izolira CENO kardinalne omejitve + metahevristike vs. klasični optimum.

  • Risk parity (ERC, Maillard et al. 2010) — enak prispevek k tveganju, samo Σ.
    Pokaže ali je upravljanje tveganja metahevristike konkurenčno.

Vse vrnejo w (N,), long-only, Σw=1. Uporabljajo scipy (tranzitivna odvisnost);
brez scipy padejo nazaj na analitične približke.

Znanstvene reference (za pisanje teze):
  • Mean-variance / tangentni (max-Sharpe) portfelj:
      Markowitz, H. (1952). "Portfolio Selection." Journal of Finance
      7(1):77–91.
  • Estimation-error argument (naivni 1/N pogosto premaga mean-variance):
      DeMiguel, V., Garlappi, L. & Uppal, R. (2009). "Optimal Versus Naive
      Diversification: How Inefficient Is the 1/N Portfolio Strategy?"
      Review of Financial Studies 22(5):1915–1953.
  • Risk parity / equal risk contribution (ERC):
      Maillard, S., Roncalli, T. & Teïletche, J. (2010). "The Properties of
      Equally Weighted Risk Contribution Portfolios." Journal of Portfolio
      Management 36(4):60–70.
  • Hierarchical Risk Parity (HRP) — kanonična moderna Σ-only referenca:
      López de Prado, M. (2016). "Building Diversified Portfolios that
      Outperform Out of Sample." Journal of Portfolio Management 42(4):59–69.
"""

import numpy as np

try:
    from scipy.optimize import minimize as _minimize
    _HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY = False

try:
    from scipy.cluster.hierarchy import linkage as _linkage
    from scipy.spatial.distance import squareform as _squareform
    _HAVE_HIER = True
except Exception:                                    # pragma: no cover
    _HAVE_HIER = False


def _psd_regularize(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rahla diagonalna regularizacija za numerično stabilnost inverza/QP."""
    n = cov.shape[0]
    return cov + eps * np.eye(n) * (np.trace(cov) / max(n, 1))


def gmv_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """Global Minimum Variance: min wᵀΣw  s.t. Σw=1 (in w≥0 če long_only)."""
    n = cov.shape[0]
    S = _psd_regularize(cov)
    if long_only and _HAVE_SCIPY:
        w0   = np.full(n, 1.0 / n)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0,
                 "jac": lambda w: np.ones(n)},)
        res  = _minimize(lambda w: w @ S @ w, w0, jac=lambda w: 2.0 * S @ w,
                         method="SLSQP", bounds=[(0.0, 1.0)] * n,
                         constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
        w = np.clip(res.x, 0.0, None)
        s = w.sum()
        return w / s if s > 1e-12 else w0
    # Analitičen (dovoljuje kratke pozicije): w = Σ⁻¹1 / (1ᵀΣ⁻¹1)
    inv = np.linalg.pinv(S)
    ones = np.ones(n)
    w = inv @ ones
    w = w / (ones @ w)
    if long_only:
        w = np.clip(w, 0.0, None); w /= w.sum() + 1e-15
    return w


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray,
                       rf: float = 0.0, long_only: bool = True) -> np.ndarray:
    """
    Markowitz tangentni (max-Sharpe) portfelj, long-only, brez kardinalnosti.
    Rešeno prek min wᵀΣw s.t. (μ−rf)ᵀw = 1, w≥0, nato normalizacija Σw=1.
    Če noben μ ni nad rf (neizvedljivo), pade nazaj na GMV.
    """
    n  = cov.shape[0]
    S  = _psd_regularize(cov)
    ex = np.asarray(mu, float) - rf
    if np.all(ex <= 1e-12):
        return gmv_weights(cov, long_only)
    if long_only and _HAVE_SCIPY:
        w0   = np.clip(ex, 0, None); w0 = w0 / (w0.sum() + 1e-15)
        cons = ({"type": "eq", "fun": lambda w: ex @ w - 1.0,
                 "jac": lambda w: ex},)
        res  = _minimize(lambda w: w @ S @ w, w0, jac=lambda w: 2.0 * S @ w,
                         method="SLSQP", bounds=[(0.0, None)] * n,
                         constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
        w = np.clip(res.x, 0.0, None)
        s = w.sum()
        return w / s if s > 1e-12 else w0
    # Analitičen: w ∝ Σ⁻¹(μ−rf)
    inv = np.linalg.pinv(S)
    w = inv @ ex
    if long_only:
        w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if abs(s) > 1e-12 else gmv_weights(cov, long_only)


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """
    Equal Risk Contribution (Maillard et al. 2010) prek konveksne formulacije
    (Spinu 2013):  min ½wᵀΣw − (1/N)Σ ln(wᵢ),  w>0,  nato normalizacija.
    Vsak asset prispeva enak delež k skupnemu tveganju.
    """
    n = cov.shape[0]
    S = _psd_regularize(cov)
    if _HAVE_SCIPY:
        def obj(w):  return 0.5 * (w @ S @ w) - (1.0 / n) * np.sum(np.log(w))
        def grad(w): return S @ w - (1.0 / n) / w
        w0  = np.full(n, 1.0 / n)
        res = _minimize(obj, w0, jac=grad, method="L-BFGS-B",
                        bounds=[(1e-8, None)] * n,
                        options={"maxiter": 500, "ftol": 1e-14})
        w = np.clip(res.x, 1e-12, None)
        return w / w.sum()
    # Fallback: inverzna volatilnost (približek ERC pri diagonalni Σ).
    iv = 1.0 / np.sqrt(np.maximum(np.diag(S), 1e-12))
    return iv / iv.sum()


def _hrp_quasi_diag(link: np.ndarray, n: int) -> list:
    """Vrstni red listov dendrograma (kvazi-diagonalizacija). Koren = 2n−2."""
    link = link.astype(int)

    def expand(node):
        if node < n:
            return [node]
        row = link[node - n]
        return expand(row[0]) + expand(row[1])

    return expand(2 * n - 2)


def _hrp_cluster_var(S: np.ndarray, items: list) -> float:
    """Varianca inverzno-variančnega (IVP) portfelja znotraj klastra."""
    sub = S[np.ix_(items, items)]
    ivp = 1.0 / np.maximum(np.diag(sub), 1e-12)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """
    Hierarchical Risk Parity (López de Prado 2016). Klastrira sredstva po
    korelacijski razdalji dᵢⱼ=√((1−ρᵢⱼ)/2), kvazi-diagonalizira Σ in z
    rekurzivno bisekcijo razporeja uteži obratno sorazmerno z varianco klastra.
    Samo Σ (brez μ), long-only, Σw=1. Brez scipy pade na inverzno-variančni
    približek.
    """
    n = cov.shape[0]
    S = _psd_regularize(cov)
    if n <= 1:
        return np.ones(max(n, 1))
    if not (_HAVE_SCIPY and _HAVE_HIER):
        iv = 1.0 / np.maximum(np.diag(S), 1e-12)
        return iv / iv.sum()

    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    corr = np.clip(S / np.outer(d, d), -1.0, 1.0)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = _linkage(_squareform(dist, checks=False), method="single")
    sort_ix = _hrp_quasi_diag(link, n)

    w = np.ones(n)
    clusters = [sort_ix]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            c0, c1 = c[:half], c[half:]
            v0, v1 = _hrp_cluster_var(S, c0), _hrp_cluster_var(S, c1)
            alpha = 1.0 - v0 / (v0 + v1 + 1e-15)
            for i in c0:
                w[i] *= alpha
            for i in c1:
                w[i] *= (1.0 - alpha)
            nxt += [c0, c1]
        clusters = nxt

    s = w.sum()
    return w / s if s > 1e-15 else np.full(n, 1.0 / n)
