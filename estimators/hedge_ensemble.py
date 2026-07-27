"""
hedge_ensemble.py
=================

Dynamic Hedge / multiplicative-weights ensemble for combining
MASTER + TFT + TACTiS expected-return (mu) forecasts.

Reference
---------
Freund, Y. & Schapire, R. E. (1997).
"A Decision-Theoretic Generalization of On-Line Learning and
an Application to Boosting."
Journal of Computer and System Sciences, 55(1), 119-139.

Core Hedge update
-----------------
For expert i at time t:

    w_{i,t+1} ∝ w_{i,t} * exp(-eta * L_{i,t})

where losses must be bounded.

For this application, forecast quality is measured using cross-sectional
Spearman rank correlation (Information Coefficient, IC):

    IC_{i,t} = Spearman(mu_hat_{i,t}, r_{t+1})

Since IC is in [-1, 1], it is converted to a bounded loss in [0, 1]:

    L_{i,t} = (1 - IC_{i,t}) / 2

Thus:
    IC = +1  -> loss = 0.0   (perfect ranking)
    IC =  0  -> loss = 0.5   (no ranking information)
    IC = -1  -> loss = 1.0   (perfectly reversed ranking)

Important
---------
At every walk-forward window:

    1. Use current weights w_t to construct the ensemble forecast.
    2. Construct / evaluate the portfolio using that forecast.
    3. Observe realized returns.
    4. Compute each expert's loss.
    5. Update weights to w_{t+1}.

Therefore, realized returns from window t NEVER affect the ensemble
prediction for window t. They only affect subsequent windows.

The Spearman-IC loss is an application-specific choice for this thesis;
it is not prescribed by Freund & Schapire.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Hashable, Tuple

import numpy as np
from scipy.stats import spearmanr


Array = np.ndarray


class HedgeEnsemble:
    """
    Online Hedge ensemble for K forecasting experts.

    The experts produce cross-sectional expected-return forecasts mu_hat.
    Their forecasts are dynamically combined using multiplicative weights.

    Parameters
    ----------
    expert_names : list[str]
        Names of experts, e.g. ["MASTER", "TFT", "TACTiS"].

    T : int, optional
        Number of online update windows.

        If eta is not supplied, T is used to obtain a theoretically
        motivated learning rate.

    eta : float, optional
        Learning rate.

        If supplied, this value is used directly.

        If omitted, we use the Freund-Schapire-style parameterization

            beta = 1 / (1 + sqrt(2 ln(K) / T))

        and convert it to the exponential-update parameter

            eta = -ln(beta).

    initial_weights : array-like, optional
        Initial expert weights.

        If omitted, experts start uniformly:

            w_i = 1/K.
    """

    def __init__(
        self,
        expert_names: List[str],
        T: Optional[int] = None,
        eta: Optional[float] = None,
        initial_weights: Optional[Array] = None,
        fixed_weights: bool = False,
    ):
        if len(expert_names) < 2:
            raise ValueError("Hedge requires at least two experts.")

        if len(set(expert_names)) != len(expert_names):
            raise ValueError("Expert names must be unique.")

        self.expert_names = list(expert_names)
        self.K = len(self.expert_names)
        self.T = T
        self.fixed_weights = bool(fixed_weights)

        # --------------------------------------------------------------
        # Initial weights
        # --------------------------------------------------------------

        if initial_weights is None:
            self.weights = np.ones(self.K, dtype=float) / self.K

        else:
            initial_weights = np.asarray(initial_weights, dtype=float)

            if initial_weights.shape != (self.K,):
                raise ValueError(
                    f"initial_weights must have shape ({self.K},), "
                    f"got {initial_weights.shape}."
                )

            if not np.all(np.isfinite(initial_weights)):
                raise ValueError("Initial weights must be finite.")

            if np.any(initial_weights < 0):
                raise ValueError("Initial weights cannot be negative.")

            total = initial_weights.sum()

            if total <= 0:
                raise ValueError("Initial weights must sum to > 0.")

            self.weights = initial_weights / total

        # --------------------------------------------------------------
        # Learning rate / fixed-weight mode
        # --------------------------------------------------------------

        if self.fixed_weights:
            # Equal/static-weight baseline: no multiplicative update.
            # eta=0 and beta=1 make the no-update semantics explicit.
            if eta is not None:
                raise ValueError(
                    "eta must be omitted when fixed_weights=True."
                )
            self.eta = 0.0
            self.beta = 1.0

        elif eta is not None:

            if not np.isfinite(eta) or eta <= 0:
                raise ValueError("eta must be a finite positive number.")

            self.eta = float(eta)
            self.beta = float(np.exp(-self.eta))

        else:

            if T is None:
                raise ValueError(
                    "Provide either T or eta. "
                    "T is required for the automatic learning rate."
                )

            if T <= 0:
                raise ValueError("T must be positive.")

            self.beta = 1.0 / (
                1.0 + np.sqrt(2.0 * np.log(self.K) / float(T))
            )

            self.eta = -np.log(self.beta)

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------

        # weights used before any observations
        self.weight_history = [self.weights.copy()]

        self.loss_history: Dict[str, List[float]] = {
            name: [] for name in self.expert_names
        }

        self.ic_history: Dict[str, List[float]] = {
            name: [] for name in self.expert_names
        }

        self.window_history: List[Hashable] = []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_forecasts(
        self,
        forecasts: Dict[str, Array],
        realized: Optional[Array] = None,
    ) -> Dict[str, Array]:
        """
        Validate that every expert has exactly one finite 1-D forecast.

        All experts must be available in every window. Missing experts are
        treated as an error rather than silently receiving zero loss.
        """

        missing = [
            name
            for name in self.expert_names
            if name not in forecasts
        ]

        if missing:
            raise ValueError(
                f"Missing forecasts for experts: {missing}"
            )

        validated: Dict[str, Array] = {}

        expected_length = None

        for name in self.expert_names:

            arr = np.asarray(forecasts[name], dtype=float)

            if arr.ndim != 1:
                raise ValueError(
                    f"{name} forecast must be 1-D; "
                    f"got shape {arr.shape}."
                )

            if len(arr) < 2:
                raise ValueError(
                    f"{name} forecast must contain at least 2 assets."
                )

            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} forecast contains NaN or infinite values."
                )

            if expected_length is None:
                expected_length = len(arr)

            elif len(arr) != expected_length:
                raise ValueError(
                    "All expert forecasts must have the same number "
                    "of assets."
                )

            validated[name] = arr

        if realized is not None:

            realized_arr = np.asarray(realized, dtype=float)

            if realized_arr.ndim != 1:
                raise ValueError(
                    f"realized must be 1-D; got {realized_arr.shape}."
                )

            if len(realized_arr) != expected_length:
                raise ValueError(
                    "realized returns and forecasts must contain "
                    "the same number of assets."
                )

            if not np.all(np.isfinite(realized_arr)):
                raise ValueError(
                    "realized returns contain NaN or infinite values."
                )

        return validated

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        forecasts: Dict[str, Array],
        realized: Array,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute each expert's bounded Hedge loss.

        We use cross-sectional Spearman IC:

            IC = Spearman(mu_hat, realized_returns)

        and transform it to:

            loss = (1 - IC) / 2

        which lies in [0, 1].

        Returns
        -------
        losses : dict[str, float]
            Hedge losses in [0, 1].

        ics : dict[str, float]
            Raw Spearman IC values in [-1, 1].
        """

        forecasts = self._validate_forecasts(
            forecasts,
            realized=realized,
        )

        realized = np.asarray(realized, dtype=float)

        losses: Dict[str, float] = {}
        ics: Dict[str, float] = {}

        for name in self.expert_names:

            mu_pred = forecasts[name]

            # scipy's spearmanr correctly handles tied ranks.
            result = spearmanr(mu_pred, realized)

            ic = float(result.statistic)

            # Constant forecasts / returns can make Spearman undefined.
            # Treat undefined IC as no predictive ranking information.
            if not np.isfinite(ic):
                ic = 0.0

            # Numerical safety.
            ic = float(np.clip(ic, -1.0, 1.0))

            # Convert [-1, 1] IC to [0, 1] Hedge loss.
            loss = (1.0 - ic) / 2.0

            losses[name] = float(loss)
            ics[name] = ic

        return losses, ics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        forecasts: Dict[str, Array],
    ) -> Array:
        """
        Produce ensemble mu using CURRENT weights.

        This must be called BEFORE update() for the current window.

        ensemble_mu =
            sum_i w_i,t * mu_hat_i,t

        Returns
        -------
        numpy.ndarray
            Weighted expected-return forecast.
        """

        forecasts = self._validate_forecasts(forecasts)

        first_name = self.expert_names[0]

        ensemble = np.zeros_like(
            forecasts[first_name],
            dtype=float,
        )

        for i, name in enumerate(self.expert_names):
            ensemble += self.weights[i] * forecasts[name]

        return ensemble

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        forecasts: Dict[str, Array],
        realized: Array,
        window_id: Optional[Hashable] = None,
    ) -> Dict[str, float]:
        """
        Update expert weights AFTER realizing window-t returns.

        Hedge:

            w_i <- w_i * exp(-eta * loss_i)

        followed by normalization.

        Parameters
        ----------
        forecasts : dict
            Forecasts that were made for the just-completed window.

        realized : ndarray
            Realized asset returns in that window.

        window_id : optional
            Identifier stored for diagnostics.

        Returns
        -------
        dict
            New normalized weights.
        """

        losses, ics = self.compute_loss(
            forecasts=forecasts,
            realized=realized,
        )

        loss_vector = np.array(
            [losses[name] for name in self.expert_names],
            dtype=float,
        )

        # --------------------------------------------------------------
        # Weight update
        # --------------------------------------------------------------

        if not self.fixed_weights:
            # Standard Hedge:
            #
            #   w_i <- w_i * exp(-eta * L_i)
            #
            # Performed in log space for numerical stability.
            log_weights = np.log(self.weights)
            log_weights -= self.eta * loss_vector

            # Remove common offset before exponentiation.
            log_weights -= np.max(log_weights)

            new_weights = np.exp(log_weights)
            total = new_weights.sum()

            if not np.isfinite(total) or total <= 0:
                raise FloatingPointError(
                    "Hedge weight normalization failed."
                )

            self.weights = new_weights / total

        # In fixed_weights mode the forecast is still evaluated and the
        # expert IC/loss histories are still recorded, but weights remain
        # exactly at their initial values. With the default initialization
        # this is the 1/K equal-weight ensemble.

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------

        for name in self.expert_names:
            self.loss_history[name].append(losses[name])
            self.ic_history[name].append(ics[name])

        self.weight_history.append(self.weights.copy())

        if window_id is not None:
            self.window_history.append(window_id)

        return self.get_weights_dict()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_weights(self) -> Array:
        """Return current weights as NumPy array."""
        return self.weights.copy()

    def get_weights_dict(self) -> Dict[str, float]:
        """Return current weights by expert name."""

        return {
            name: float(self.weights[i])
            for i, name in enumerate(self.expert_names)
        }

    def get_weight_history(self) -> Array:
        """
        Return weight history.

        Shape:
            (num_updates + 1, K)

        Row 0 contains the initial weights.
        """

        return np.asarray(
            self.weight_history,
            dtype=float,
        )

    def get_loss_history(self) -> Dict[str, List[float]]:
        """Return copy of loss history."""

        return {
            name: values.copy()
            for name, values in self.loss_history.items()
        }

    def get_ic_history(self) -> Dict[str, List[float]]:
        """Return copy of IC history."""

        return {
            name: values.copy()
            for name, values in self.ic_history.items()
        }


# ======================================================================
# WALK-FORWARD HELPER
# ======================================================================


def build_hedge_forecasts(
    per_window_forecasts: Dict[Tuple[Hashable, str], Array],
    per_window_realized: Dict[Hashable, Array],
    expert_names: Optional[List[str]] = None,
    T: Optional[int] = None,
    eta: Optional[float] = None,
    volatility_adaptive: bool = False,
    weighting_mode: str = "hedge",
):
    """
    Replay a complete walk-forward experiment with Hedge.

    Parameters
    ----------
    per_window_forecasts : dict
        Mapping:

            (window_idx, expert_name) -> mu forecast

        Example:

            {
                (0, "MASTER"): np.array(...),
                (0, "TFT"): np.array(...),
                (0, "TACTiS"): np.array(...),
                (1, "MASTER"): np.array(...),
                ...
            }

    per_window_realized : dict
        Mapping:

            window_idx -> realized asset returns

    expert_names : list[str], optional
        Default:

            ["MASTER", "TFT", "TACTiS"]

    T : int, optional
        Horizon used for the automatic Hedge learning rate.

        Defaults to number of walk-forward windows.

    eta : float, optional
        Manual learning rate. If supplied, overrides automatic choice.

    volatility_adaptive : bool, optional
        If True, scale η based on regime volatility.
        Low volatility (calm markets) -> higher η (faster adaptation).
        Default: False (use constant η).

    weighting_mode : {"hedge", "equal"}, optional
        "hedge" uses the dynamic multiplicative-weights update.
        "equal" freezes weights at 1/K for every window while preserving
        identical prediction/evaluation timing and diagnostics.
        Default: "hedge" for backward compatibility.

    Returns
    -------
    ensemble_forecasts : dict
        Mapping:

            window_idx -> ensemble mu forecast

        IMPORTANT: each forecast was generated BEFORE observing that
        window's realized returns.

    hedge : HedgeEnsemble
        Ensemble object containing all diagnostics. In weighting_mode="equal"
        its weights remain fixed at 1/K for the complete OOS replay.

    weights_used : dict
        Mapping:

            window_idx -> weights used to create that window's forecast

        This is useful for tables/plots in the thesis.
    """

    if expert_names is None:
        expert_names = [
            "MASTER",
            "TFT",
            "TACTiS",
        ]

    if not per_window_realized:
        raise ValueError(
            "per_window_realized cannot be empty."
        )

    window_ids = sorted(per_window_realized.keys())

    if T is None:
        T = len(window_ids)

    weighting_mode = str(weighting_mode).lower().strip()
    if weighting_mode not in {"hedge", "equal"}:
        raise ValueError(
            "weighting_mode must be either 'hedge' or 'equal'."
        )

    fixed_weights = weighting_mode == "equal"

    if fixed_weights and eta is not None:
        raise ValueError(
            "eta is not used by the equal-weight ensemble; omit eta."
        )

    if fixed_weights and volatility_adaptive:
        raise ValueError(
            "volatility_adaptive is not applicable to equal weights."
        )

    # Detect regime volatility for adaptive η
    if volatility_adaptive and eta is None:
        # Compute realized volatilities across all windows
        realized_vols = []
        for window_id in window_ids:
            ret = np.asarray(per_window_realized[window_id], dtype=float)
            vol = float(np.std(ret))
            realized_vols.append(vol)

        mean_vol = float(np.mean(realized_vols))
        median_vol = float(np.median(realized_vols))

        # Scale factor: lower volatility -> higher η (faster adaptation)
        # Use median/current_vol as proxy; if vol < median, scale up
        vol_scaling = median_vol / (mean_vol + 1e-8)  # Ratio; calm regimes have vol_scaling > 1

        # Compute base η from T and scale it
        K = len(expert_names)
        beta_base = 1.0 / (1.0 + np.sqrt(2.0 * np.log(K) / float(T)))
        eta_base = -np.log(beta_base)

        # Scale η: higher vol_scaling -> higher η (calm regime adapts faster)
        scaled_eta = float(eta_base * vol_scaling)
    else:
        scaled_eta = eta

    hedge = HedgeEnsemble(
        expert_names=expert_names,
        T=T,
        eta=scaled_eta,
        fixed_weights=fixed_weights,
    )

    ensemble_forecasts: Dict[Hashable, Array] = {}
    weights_used: Dict[Hashable, Dict[str, float]] = {}

    # ==============================================================
    # STRICT WALK-FORWARD LOOP
    # ==============================================================

    for window_id in window_ids:

        # ----------------------------------------------------------
        # 1. Collect forecasts made BEFORE this window was realized
        # ----------------------------------------------------------

        forecasts_w: Dict[str, Array] = {}

        for name in expert_names:

            key = (window_id, name)

            if key not in per_window_forecasts:
                raise KeyError(
                    f"Missing forecast for window={window_id}, "
                    f"expert={name}."
                )

            forecasts_w[name] = np.asarray(
                per_window_forecasts[key],
                dtype=float,
            )

        # ----------------------------------------------------------
        # 2. Record weights available at START of window t
        # ----------------------------------------------------------

        weights_used[window_id] = hedge.get_weights_dict()

        # ----------------------------------------------------------
        # 3. Make ensemble prediction using ONLY past information
        # ----------------------------------------------------------

        ensemble_mu = hedge.predict(forecasts_w)

        ensemble_forecasts[window_id] = ensemble_mu

        # ----------------------------------------------------------
        # 4. NOW observe realized returns for this window
        # ----------------------------------------------------------

        realized_w = np.asarray(
            per_window_realized[window_id],
            dtype=float,
        )

        # ----------------------------------------------------------
        # 5. Update weights for NEXT window
        # ----------------------------------------------------------

        hedge.update(
            forecasts=forecasts_w,
            realized=realized_w,
            window_id=window_id,
        )

    return ensemble_forecasts, hedge, weights_used


# ======================================================================
# EXAMPLE
# ======================================================================


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    n_windows = 20
    n_assets = 50

    expert_names = [
        "MASTER",
        "TFT",
        "TACTiS",
    ]

    per_window_forecasts = {}
    per_window_realized = {}

    for t in range(n_windows):

        # Example realized future returns
        realized = rng.normal(
            loc=0.0,
            scale=0.02,
            size=n_assets,
        )

        per_window_realized[t] = realized

        # ----------------------------------------------------------
        # Dummy forecasts ONLY for demonstration.
        #
        # Replace these with your actual saved walk-forward mu arrays.
        # ----------------------------------------------------------

        master = (
            0.40 * realized
            + rng.normal(0, 0.02, n_assets)
        )

        tft = (
            0.30 * realized
            + rng.normal(0, 0.02, n_assets)
        )

        tactis = (
            0.20 * realized
            + rng.normal(0, 0.02, n_assets)
        )

        per_window_forecasts[(t, "MASTER")] = master
        per_window_forecasts[(t, "TFT")] = tft
        per_window_forecasts[(t, "TACTiS")] = tactis

    # --------------------------------------------------------------
    # Build strict online ensemble
    # --------------------------------------------------------------

    ensemble_forecasts, hedge, weights_used = (
        build_hedge_forecasts(
            per_window_forecasts=per_window_forecasts,
            per_window_realized=per_window_realized,
            expert_names=expert_names,
            # weighting_mode="equal",  # uncomment for fixed 1/K weights
        )
    )

    print("\nLearning rate:")
    print(f"eta  = {hedge.eta:.6f}")
    print(f"beta = {hedge.beta:.6f}")

    print("\nFinal weights:")
    for name, weight in hedge.get_weights_dict().items():
        print(f"{name:8s}: {weight:.4f}")

    print("\nWeights used by window:")

    for window_id in sorted(weights_used):

        weights = weights_used[window_id]

        formatted = ", ".join(
            f"{name}={weights[name]:.3f}"
            for name in expert_names
        )

        print(
            f"Window {window_id:>3}: "
            f"{formatted}"
        )

    print("\nFinal IC means:")

    for name, values in hedge.get_ic_history().items():

        mean_ic = np.mean(values)

        print(
            f"{name:8s}: "
            f"{mean_ic:.4f}"
        )