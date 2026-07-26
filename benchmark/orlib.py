"""
benchmark/orlib.py
==================
OR-Library standardni benchmark za kardinalno-omejeno mean-variance (CCMV)
optimizacijo — de-facto referenca v literaturi (Chang et al. 2000; Crama &
Schyns 2003; Cura 2009; Woodside-Oriakhi et al. 2011).

Podatki (Beasley OR-Library, port1–port5):
    port1  Hang Seng (HK)      N=31
    port2  DAX 100 (DE)        N=85
    port3  FTSE 100 (UK)       N=89
    port4  S&P 100 (US)        N=98
    port5  Nikkei (JP)         N=225

Vsak instance ima priloženo NEomejeno efficient frontier (portefX.txt) —
2000 (donos, varianca) točk brez kardinalnostne omejitve. Standardna metrika
kvalitete hevristike je odstopanje njene (omejene) fronte od te reference.

Standardne nastavitve iz literature (Chang et al. 2000):
    K = 10,  εᵢ = 0.01,  δᵢ = 1.0,  ~50 λ točk za trasiranje fronte.

Format port fajla:
    vrstica 1        : N
    vrstice 2..N+1   : μᵢ  σᵢ         (pričakovani donos, std. odklon)
    preostanek       : i  j  ρᵢⱼ      (1-based, zgornji trikotnik z diagonalo)
  → Σᵢⱼ = ρᵢⱼ · σᵢ · σⱼ

Format portef fajla:
    vsaka vrstica    : donos  varianca   (točke NEomejene fronte)
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "orlib_data")

# Meta za izpise/tabele (ime indeksa in N po Chang et al. 2000).
DATASETS = {
    "port1": {"name": "Hang Seng", "N": 31},
    "port2": {"name": "DAX 100",   "N": 85},
    "port3": {"name": "FTSE 100",  "N": 89},
    "port4": {"name": "S&P 100",   "N": 98},
    "port5": {"name": "Nikkei",    "N": 225},
}


def load_orlib(name: str, data_dir: str = DATA_DIR):
    """Naloži port instance → (mu (N,), cov (N,N), sigma (N,))."""
    path = os.path.join(data_dir, f"{name}.txt")
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N  = int(next(it))
    mu    = np.zeros(N)
    sigma = np.zeros(N)
    for i in range(N):
        mu[i]    = float(next(it))
        sigma[i] = float(next(it))
    corr = np.zeros((N, N))
    # Preostanek: trojice i j ρ (1-based, zgornji trikotnik z diagonalo).
    while True:
        try:
            i = int(next(it)) - 1
            j = int(next(it)) - 1
            r = float(next(it))
        except StopIteration:
            break
        corr[i, j] = r
        corr[j, i] = r
    cov = corr * np.outer(sigma, sigma)
    return mu, cov, sigma


def load_frontier(name: str, data_dir: str = DATA_DIR) -> np.ndarray:
    """Naloži NEomejeno efficient frontier → array (M, 2) = (donos, varianca),
    urejeno naraščajoče po donosu."""
    path = os.path.join(data_dir, f"portef{name[-1]}.txt")
    pts = np.loadtxt(path)
    ret, var = pts[:, 0], pts[:, 1]
    order = np.argsort(ret)
    return np.column_stack([ret[order], var[order]])


# Chang et al. (2000) standardna metrika napake fronte

def _interp_var_at_return(frontier: np.ndarray, r: float) -> float:
    """Varianca na referenčni fronti pri danem donosu r (linearna interpolacija)."""
    ret, var = frontier[:, 0], frontier[:, 1]
    if r <= ret[0]:   return float(var[0])
    if r >= ret[-1]:  return float(var[-1])
    return float(np.interp(r, ret, var))


def _interp_return_at_var(frontier: np.ndarray, v: float) -> float:
    """Donos na referenčni fronti pri dani varianci v. Referenčna fronta je
    monotona (donos ↑ ⇒ varianca ↑), zato lahko interpoliramo po varianci."""
    ret, var = frontier[:, 0], frontier[:, 1]
    # Zagotovi naraščajočo varianco za np.interp.
    o = np.argsort(var)
    var_s, ret_s = var[o], ret[o]
    if v <= var_s[0]:   return float(ret_s[0])
    if v >= var_s[-1]:  return float(ret_s[-1])
    return float(np.interp(v, var_s, ret_s))


def frontier_error(points: np.ndarray, frontier: np.ndarray) -> dict:
    """
    Standardna Chang et al. (2000) odstotna napaka omejene fronte glede na
    NEomejeno referenčno fronto.

    Za vsako hevristično točko (rᵢ, vᵢ):
      • variančna napaka  = 100·(vᵢ − v*(rᵢ)) / v*(rᵢ)   pri fiksnem donosu
      • donosna napaka    = 100·(r*(vᵢ) − rᵢ) / r*(vᵢ)    pri fiksni varianci
    Standardna napaka točke = min(variančna, donosna) — najbližja razdalja do
    reference. Vrne agregate čez vse točke (mean/median/min/max/std).

    points: (M, 2) = (donos, varianca) hevristične (omejene) fronte.
    """
    if len(points) == 0:
        return {k: np.nan for k in ("mean", "median", "min", "max", "std", "n")}
    errs = []
    for r, v in points:
        v_star = _interp_var_at_return(frontier, r)
        r_star = _interp_return_at_var(frontier, v)
        var_err = 100.0 * (v - v_star) / abs(v_star) if v_star != 0 else np.nan
        ret_err = 100.0 * (r_star - r) / abs(r_star) if r_star != 0 else np.nan
        cand = [e for e in (var_err, ret_err) if not np.isnan(e)]
        if not cand:
            continue
        # Napaka je nenegativna (omejena fronta ne more premagati neomejene);
        # numerični šum lahko da rahlo negativno → clip na 0.
        errs.append(max(0.0, min(cand)))
    errs = np.array(errs)
    if len(errs) == 0:
        return {k: np.nan for k in ("mean", "median", "min", "max", "std", "n")}
    return {
        "mean":   float(errs.mean()),
        "median": float(np.median(errs)),
        "min":    float(errs.min()),
        "max":    float(errs.max()),
        "std":    float(errs.std(ddof=1)) if len(errs) > 1 else 0.0,
        "n":      int(len(errs)),
    }
