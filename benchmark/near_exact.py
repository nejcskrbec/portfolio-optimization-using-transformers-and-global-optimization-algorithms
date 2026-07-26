"""
benchmark/near_exact.py
========================
Near-exact rešitev kardinalno-omejenega mean-variance (CCMV) problema —
referenca za merjenje optimalnostne vrzeli (optimality gap) metahevristik.

Problem (isti kot v C++ optimizatorjih):
    max_w  P·μᵀw − (1−P)·wᵀΣw
    s.t.   Σ wᵢ = 1
           |supp(w)| = K
           εᵢ ≤ wᵢ ≤ δᵢ   za i ∈ supp(w)

Ključna ideja: ob FIKSNEM naboru (support) je problem navaden konveksni
box-omejeni QP s K spremenljivkami → rešljiv do optimuma. Preostane le
kombinatorika izbire supporta:

  • Če je C(N,K) majhen (≤ enumerate_max) → EKSAKTNA enumeracija vseh
    supportov (globalni optimum CCMV).
  • Sicer → multi-start lokalno iskanje z 1-swap sosedščino, po želji
    ogreto (warm-started) s supporti metahevrističnih rešitev. Rezultat je
    močan lokalni optimum ("near-exact"), ki daje tesno zgornjo mejo.

Odvisnosti: numpy + scipy (scipy je tranzitivna odvisnost sklearn/LW).
Če scipy ni na voljo, pade nazaj na analitično enako-utežno rešitev na
najboljšem supportu.

Znanstvene reference (za pisanje teze):
  • CCMV problem in standardna optimality-gap metrika metahevristik:
      Chang, T.-J., Meade, N., Beasley, J.E. & Sharaiha, Y.M. (2000).
      "Heuristics for Cardinality Constrained Portfolio Optimisation."
      Computers & Operations Research 27(13):1271–1302.
  • NP-težavnost kardinalno-omejenega mean-variance portfelja (utemeljitev,
    zakaj eksaktna enumeracija le za majhen C(N,K), sicer lokalno iskanje):
      Bienstock, D. (1996). "Computational Study of a Family of Mixed-Integer
      Quadratic Programming Problems." Mathematical Programming 74(2):121–140.
"""

import itertools
import math
import time

import numpy as np

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    _HAVE_SCIPY = False


def _solve_support_qp(mu, cov, support, P, eps, delta):
    """
    Reši box-omejeni QP na fiksnem supportu.
    Vrne (w_full (N,), objective) kjer je objective P·μw − (1−P)·wΣw.
    """
    idx = list(support)
    k   = len(idx)
    m   = mu[idx]
    S   = cov[np.ix_(idx, idx)]
    lo  = np.array([eps[i]   for i in idx])
    hi  = np.array([delta[i] for i in idx])

    N = len(mu)
    w_full = np.zeros(N)

    # Neizvedljivo (vsote spodnjih/zgornjih mej ne oklepajo 1) → preskoči.
    if lo.sum() > 1.0 + 1e-9 or hi.sum() < 1.0 - 1e-9:
        return w_full, -np.inf

    def neg_obj(w):
        return float((1.0 - P) * (w @ S @ w) - P * (m @ w))

    def neg_grad(w):
        return (1.0 - P) * (2.0 * S @ w) - P * m

    if _HAVE_SCIPY:
        w0  = np.clip(np.full(k, 1.0 / k), lo, hi)
        s0  = w0.sum()
        if s0 > 0:
            w0 = lo + (w0 - lo) * (1.0 - lo.sum()) / max(s0 - lo.sum(), 1e-12)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0,
                 "jac": lambda w: np.ones(k)},)
        bnds = list(zip(lo, hi))
        import warnings
        with warnings.catch_warnings():
            # SLSQP med line-search občasno oceni točko izven meja in jo
            # obreže — nenevarno, a hrupno; utišamo.
            warnings.simplefilter("ignore", RuntimeWarning)
            res = _scipy_minimize(neg_obj, w0, jac=neg_grad, method="SLSQP",
                                  bounds=bnds, constraints=cons,
                                  options={"maxiter": 200, "ftol": 1e-12})
        w = np.clip(res.x, lo, hi)
        w = w / w.sum() if w.sum() > 0 else w0
    else:                                            # pragma: no cover
        # Fallback: enako-utežno znotraj mej.
        w = np.clip(np.full(k, 1.0 / k), lo, hi)
        w = w / w.sum()

    w_full[idx] = w
    obj = float(P * (mu[idx] @ w) - (1.0 - P) * (w @ S @ w))
    return w_full, obj


def _screen_scores(mu, cov):
    """μ/σ (Sharpe-proxy) score za začetni support in urejanje kandidatov."""
    sig = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return mu / (sig + 1e-15)


def solve_near_exact(mu, cov, K, w_min, w_max, P,
                     eps=None, delta=None,
                     enumerate_max=200000, restarts=8,
                     time_limit_sec=15.0, warm_starts=None, seed=0):
    """
    Vrne dict: {weights, objective, exact (bool), n_supports_evaluated}.

    warm_starts: opcijski seznam index-množic (supportov) metahevrističnih
                 rešitev — uporabljen kot dodatna startna točka lokalnega
                 iskanja (izboljša kvaliteto near-exact reference).
    """
    mu  = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    N   = len(mu)
    K   = min(K, N)
    eps   = np.full(N, w_min) if eps   is None else np.asarray(eps, dtype=float)
    delta = np.full(N, w_max) if delta is None else np.asarray(delta, dtype=float)
    rng   = np.random.default_rng(seed)

    n_comb = math.comb(N, K)
    t0 = time.perf_counter()

    best_w, best_obj, n_eval = np.zeros(N), -np.inf, 0

    # Eksaktna enumeracija
    if n_comb <= enumerate_max:
        for support in itertools.combinations(range(N), K):
            w, obj = _solve_support_qp(mu, cov, support, P, eps, delta)
            n_eval += 1
            if obj > best_obj:
                best_obj, best_w = obj, w
            if time.perf_counter() - t0 > time_limit_sec:
                # Preveč — nadaljuj kot heuristika od najboljšega doslej.
                return _local_search(mu, cov, K, P, eps, delta, best_w, best_obj,
                                     n_eval, rng, restarts, time_limit_sec, t0,
                                     warm_starts, exact=False)
        return {"weights": best_w, "objective": best_obj,
                "exact": True, "n_supports_evaluated": n_eval}

    # Heuristično multi-start lokalno iskanje
    return _local_search(mu, cov, K, P, eps, delta, best_w, best_obj,
                         n_eval, rng, restarts, time_limit_sec, t0,
                         warm_starts, exact=False)


def _local_search(mu, cov, K, P, eps, delta, best_w, best_obj, n_eval,
                  rng, restarts, time_limit_sec, t0, warm_starts, exact):
    N = len(mu)
    scores = _screen_scores(mu, cov)
    ranked = list(np.argsort(scores)[::-1])

    # Startni supporti: top-K po score, warm-starts, in naključni.
    starts = [frozenset(ranked[:K])]
    for ws in (warm_starts or []):
        s = frozenset(int(i) for i in ws)
        if len(s) == K:
            starts.append(s)
    while len(starts) < restarts + 1 + len(warm_starts or []):
        starts.append(frozenset(rng.choice(N, size=K, replace=False).tolist()))

    seen = set()
    for start in starts:
        if time.perf_counter() - t0 > time_limit_sec:
            break
        cur = set(start)
        w, obj = _solve_support_qp(mu, cov, cur, P, eps, delta); n_eval += 1
        improved = True
        while improved:
            improved = False
            if time.perf_counter() - t0 > time_limit_sec:
                break
            in_assets  = list(cur)
            out_assets = [i for i in range(N) if i not in cur]
            # 1-swap sosedščina; kandidate zunaj omejimo na najboljše po score.
            out_cand = sorted(out_assets, key=lambda i: -scores[i])[:max(10, 2 * K)]
            for a_in in in_assets:
                for a_out in out_cand:
                    trial = frozenset((cur - {a_in}) | {a_out})
                    if trial in seen:
                        continue
                    seen.add(trial)
                    tw, tobj = _solve_support_qp(mu, cov, trial, P, eps, delta)
                    n_eval += 1
                    if tobj > obj + 1e-12:
                        cur, w, obj = set(trial), tw, tobj
                        improved = True
                        break
                if improved:
                    break
        if obj > best_obj:
            best_obj, best_w = obj, w

    return {"weights": best_w, "objective": best_obj,
            "exact": exact, "n_supports_evaluated": n_eval}
