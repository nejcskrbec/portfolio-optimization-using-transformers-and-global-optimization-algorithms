"""Online forecast-combination of cross-sectional μ 'experts' (Hedge).

Replaces the single, regime-fragile winner-tilt hyperparameter (``topw_alpha``)
with a *combination* of μ views whose weights adapt each walk-forward window
from the experts' own realized performance — no parameter is fit on the test
regimes and the mixing rate η is set from theory.

Experts (all causal, all cheap re-weightings of quantities already computed
per window):
  • ``hist``      — historical rolling-mean μ (the literature baseline itself)
  • ``trans_gγ``  — the *neutral* transformer μ under a signed-power tilt
                    ``sign(z)·|z|^γ`` for each γ in a fixed grid. γ<1 flattens
                    (diversify), γ=1 is neutral, γ>1 sharpens the winners
                    (the job the old topw_alpha did, now a dial, not a bake-in).

Combination is the multiplicative-weights / Hedge algorithm (Freund &
Schapire 1997; Cesa-Bianchi & Lugosi 2006). It carries an O(√(T ln K)) regret
bound, so the blend cannot trail the best single expert — which *includes* the
historical baseline — by more than that in ANY regime. This is the "bulletproof
vs history" property: worst case it degrades to a robust forecast average
(itself hard to beat under structural breaks — Wang & Hyndman 2023; Fed
FEDS 2007-42), best case it adapts toward whichever view the current regime
rewards.

Znanstvene reference (za pisanje teze):
  • Hedge / multiplicative-weights + regret bound:
      Freund, Y. & Schapire, R.E. (1997). "A Decision-Theoretic Generalization
      of On-Line Learning and an Application to Boosting." Journal of Computer
      and System Sciences 55(1):119–139.
  • Prediction with expert advice (splošni okvir + regret analiza):
      Cesa-Bianchi, N. & Lugosi, G. (2006). "Prediction, Learning, and Games."
      Cambridge University Press.
"""
import numpy as np


def _standardize(v):
    v = np.asarray(v, dtype=float)
    m = np.nanmean(v)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(v)
    return (v - m) / sd


def _signed_power(z, gamma):
    return np.sign(z) * np.abs(z) ** gamma


class HedgeMuEnsemble:
    def __init__(self, gamma_grid=(0.5, 1.0, 2.0), n_windows=None,
                 topk=10, eta=None):
        self.gammas = tuple(float(g) for g in gamma_grid)
        self.names = ["hist"] + [f"trans_g{g:g}" for g in self.gammas]
        self.K = len(self.names)
        self.topk = int(topk)
        # Theory-optimal Hedge rate (Cesa-Bianchi & Lugosi): η = sqrt(8 ln K / T).
        # Set from the known horizon, never tuned on outcomes.
        T = max(int(n_windows) if n_windows else 8, 2)
        self.eta = float(eta) if eta is not None else np.sqrt(8.0 * np.log(self.K) / T)
        self.w = np.ones(self.K) / self.K
        self._last_experts = None      # (K, N) standardized signals of last window
        self.weight_history = []       # list of (K,) weight vectors actually used

    def experts(self, mu_hist, mu_trans):
        z_h = _standardize(mu_hist)
        z_t = _standardize(mu_trans)
        rows = [z_h] + [_standardize(_signed_power(z_t, g)) for g in self.gammas]
        return np.vstack(rows)          # (K, N)

    def combine(self, mu_hist, mu_trans):
        """Blend experts with the current weights and return a μ on the SAME
        scale as ``mu_trans`` — so the risk parameter P and Σ trade-off behave
        exactly as in the Transformer scenario and ONLY the cross-sectional
        shape differs."""
        E = self.experts(mu_hist, mu_trans)
        self._last_experts = E
        self.weight_history.append(self.w.copy())
        blended = _standardize(self.w @ E)
        t = np.asarray(mu_trans, dtype=float)
        return blended * np.nanstd(t) + np.nanmean(t)

    def update(self, realized):
        """After the window resolves, score each expert by its realized top-K
        equal-weight return and apply the Hedge multiplicative update. Uses only
        past outcomes → the weights for window w depend on windows <w only."""
        if self._last_experts is None:
            return
        E, self._last_experts = self._last_experts, None
        r = np.asarray(realized, dtype=float)
        gains = np.array([self._topk_return(sig, r) for sig in E])
        lo, hi = np.nanmin(gains), np.nanmax(gains)
        if hi - lo < 1e-12:
            return                      # no information this round → weights unchanged
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
