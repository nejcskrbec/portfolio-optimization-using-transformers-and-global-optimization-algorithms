"""Online forecast-combination of cross-sectional μ 'experts' (Hedge).

Combines the per-window μ forecasts of several models into one blended μ whose
weights adapt each walk-forward window from the experts' own realized
performance — no parameter is fit on the test regimes and the mixing rate η is
set from theory. Here the experts are the kept transformer pipelines plus the
historical baseline (e.g. {hist, MASTER, TFT, TACTiS}).

Combination is the multiplicative-weights / Hedge algorithm (Freund & Schapire
1997; Cesa-Bianchi & Lugosi 2006). It carries an O(√(T ln K)) regret bound, so
the blend cannot trail the best single expert — which *includes* the historical
baseline — by more than that in ANY regime. This "near-best in every regime"
robustness, applied to combining machine-learning return forecasts into a
portfolio, follows Remlinger et al. (2023), who show online expert aggregation
of ML return forecasts yields a higher portfolio Sharpe with similar turnover.

Znanstvene reference (za pisanje teze):
  • Hedge / multiplicative-weights + regret bound:
      Freund, Y. & Schapire, R.E. (1997). "A Decision-Theoretic Generalization
      of On-Line Learning and an Application to Boosting." JCSS 55(1):119–139.
  • Napovedovanje z nasvetom ekspertov (okvir + regret):
      Cesa-Bianchi, N. & Lugosi, G. (2006). "Prediction, Learning, and Games."
      Cambridge University Press.
  • Ekspertna agregacija napovedi donosov v finance (domenska podpora):
      Remlinger, C., Brière, M., Alasseur, C. & Mikael, J. (2023). "Expert
      aggregation for financial forecasting." J. of Finance and Data Science.
"""
import numpy as np


def _standardize(v):
    v = np.asarray(v, dtype=float)
    m = np.nanmean(v)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(v)
    return (v - m) / sd


class HedgeMuEnsemble:
    """Hedge combination of an arbitrary set of named μ-experts.

    expert_keys : list[str]  keys into the per-window ``mu_by_src`` dict, e.g.
                             ["hist", "master", "tft", "tactis"].
    """
    def __init__(self, expert_keys, n_windows=None, topk=10, eta=None):
        self.keys = list(expert_keys)
        self.K = len(self.keys)
        self.topk = int(topk)
        # Theory-optimal Hedge rate (Cesa-Bianchi & Lugosi): η = sqrt(8 ln K / T).
        T = max(int(n_windows) if n_windows else 8, 2)
        self.eta = float(eta) if eta is not None else np.sqrt(8.0 * np.log(self.K) / T)
        self.w = np.ones(self.K) / self.K
        self._last_experts = None       # (K, N) standardized signals of last window
        self.weight_history = []        # (K,) weight vectors actually used per window

    def combine(self, mu_by_src, scale_ref):
        """Blend the standardized expert μ's with current weights, then rescale
        to the mean/std of ``scale_ref`` (historical μ) so the absolute level and
        the P/Σ trade-off match the other μ-only scenarios; only the
        cross-sectional SHAPE differs between scenarios."""
        # varno: če ekspertni ključ manjka (npr. TACTiS v A/B načinu odda tac_plain),
        # uporabi razumen nadomestek, sicer nevtralen (ničelni) signal.
        def _get(k):
            if k in mu_by_src: return mu_by_src[k]
            if k == "tactis" and "tac_plain" in mu_by_src: return mu_by_src["tac_plain"]
            return np.zeros_like(np.asarray(scale_ref, dtype=float))
        E = np.vstack([_standardize(_get(k)) for k in self.keys])   # (K, N)
        self._last_experts = E
        self.weight_history.append(self.w.copy())
        blended = _standardize(self.w @ E)
        ref = np.asarray(scale_ref, dtype=float)
        return blended * (np.nanstd(ref) + 1e-12) + np.nanmean(ref)

    def update(self, realized):
        """After the window resolves, score each expert by its realized top-K
        equal-weight return and apply the Hedge multiplicative update. Causal:
        weights for window w depend only on windows < w."""
        if self._last_experts is None:
            return
        E, self._last_experts = self._last_experts, None
        r = np.asarray(realized, dtype=float)
        gains = np.array([self._topk_return(sig, r) for sig in E])
        lo, hi = np.nanmin(gains), np.nanmax(gains)
        if hi - lo < 1e-12:
            return                       # no information → weights unchanged
        loss = 1.0 - (gains - lo) / (hi - lo)   # best expert → 0 loss, in [0,1]
        self.w = self.w * np.exp(-self.eta * loss)
        self.w = self.w / self.w.sum()

    def _topk_return(self, signal, realized):
        finite = np.isfinite(signal)
        k = min(self.topk, int(finite.sum()))
        if k <= 0:
            return 0.0
        idx = np.argsort(np.where(finite, -signal, np.inf))[:k]
        vals = realized[idx]
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else 0.0
