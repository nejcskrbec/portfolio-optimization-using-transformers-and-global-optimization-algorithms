"""
estimators/black_litterman.py
=============================
Klasičen Black-Litterman (He & Litterman 1999; Black & Litterman 1992) μ-baseline
— odgovarja na predlagano primerjalno metodo iz teme magistrske naloge (proposal
eksplicitno navaja Black-Littermanov model kot primerjavo).

Zasnova (namenoma KLASIČNA, brez transformerja → poštena "močna klasična" primerjava):
  • Ravnovesni prior Π = δ·Σ·w_ref  (reverse-optimization; He & Litterman 1999).
    Referenčni portfelj w_ref = enakomerne uteži (1/N). Enakomerni referenčni
    portfelj je KAVZALEN in ne potrebuje point-in-time tržnih kapitalizacij (te za
    zgodovinski backtest 2007–2010 niso zanesljivo dosegljive → statične današnje
    kapitalizacije bi vnesle look-ahead). Prior je tako "ravnovesje glede na
    tveganje": zahtevani donos ∝ prispevku k tveganju.
  • Pogledi (views) Q = zgodovinski drseči μ (klasična vzorčna ocena), P = I
    (absolutni pogled na vsako delnico). Ω = diag(τ·PΣPᵀ) (He & Litterman
    standard: negotovost pogleda ∝ prior varianci).
  • Posterior (klasična BL formula):
        μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
    → shrinkage med ravnovesnim priorjem in vzorčnimi pogledi. To je natanko
    "sofisticiran klasičen investitor", ki popravi nestabilnost čiste povprečje-
    varianca ocene (Black & Litterman 1992; Bessler et al. 2017).

Reference (za pisanje teze):
  Black, F. & Litterman, R. (1992). "Global Portfolio Optimization."
    Financial Analysts Journal 48(5):28–43.
  He, G. & Litterman, R. (1999). "The Intuition Behind Black-Litterman Model
    Portfolios." Goldman Sachs Investment Management Research.

API (računa se sproti na okno; NI učenja/registrija):
  bl_mu(cov, views, delta=2.5, tau=0.05, w_ref=None) → np.ndarray (N,)  [donosna skala]
"""

import numpy as np


def bl_mu(cov: np.ndarray, views: np.ndarray,
          delta: float = 2.5, tau: float = 0.05,
          w_ref: np.ndarray = None) -> np.ndarray:
    """Klasičen Black-Litterman posteriorni μ (donosna skala, kot views).

    cov   : (N,N) kovariančna matrika (ista H-dnevna skala kot views/optimizator).
    views : (N,) absolutni pogledi Q (npr. zgodovinski drseči μ). P = I.
    delta : koeficient nenaklonjenosti tveganju (He & Litterman standard ≈ 2.5).
    tau   : skala negotovosti priorja (standard 0.025–0.05).
    w_ref : (N,) referenčne uteži za ravnovesni prior (privzeto enakomerne 1/N).
    """
    cov = np.asarray(cov, dtype=float)
    N = cov.shape[0]
    if w_ref is None:
        w_ref = np.ones(N) / N
    w_ref = np.asarray(w_ref, dtype=float).reshape(N)
    Q = np.asarray(views, dtype=float).reshape(N)
    Q = np.where(np.isfinite(Q), Q, np.nanmean(Q[np.isfinite(Q)]) if np.isfinite(Q).any() else 0.0)

    pi = delta * (cov @ w_ref)                       # ravnovesni prior Π
    tauS = tau * cov
    # P = I → PᵀΩ⁻¹P = Ω⁻¹, PᵀΩ⁻¹Q = Ω⁻¹Q; Ω = diag(τ·Σ) (He-Litterman).
    omega_diag = np.clip(np.diag(tauS), 1e-12, None)
    Om_inv = np.diag(1.0 / omega_diag)
    tauS_inv = np.linalg.pinv(tauS)
    A = tauS_inv + Om_inv
    b = tauS_inv @ pi + Om_inv @ Q
    try:
        mu_bl = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        mu_bl = np.linalg.pinv(A) @ b
    return mu_bl
