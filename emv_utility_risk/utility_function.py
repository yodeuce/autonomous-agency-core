"""
FILE 15: utility_function.py
PURPOSE: Adjusts EMV for preferences and risk
ROLE: Advanced agents do not optimize raw EMV
SPEC: CARBON[6] Technical Architecture Specification v1.0.0

Available Utility Functions (CARBON[6] §5.2):
    | Function      | Formula                              | Risk Attitude            |
    |--------------|--------------------------------------|--------------------------|
    | Linear       | U(x) = x                            | Risk Neutral             |
    | Logarithmic  | U(x) = ln(x + 1)                    | Risk Averse (decreasing) |
    | Exponential  | U(x) = 1 - e^(-αx)                  | Risk Averse (constant)   |
    | Power (CRRA) | U(x) = x^(1-ρ) / (1-ρ)             | Risk Averse (configurable)|
    | Prospect     | U(x) = x^α if x≥0, -λ(-x)^β if x<0 | Loss Averse              |

Includes:
- Risk aversion
- Loss penalties
- Tail-risk weighting
- Non-linear utility curves
- Adaptive utility (context-aware risk aversion)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class UtilityType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POWER = "power"
    LOGARITHMIC = "logarithmic"
    PROSPECT_THEORY = "prospect_theory"


@dataclass
class UtilityConfig:
    """Configuration for the utility function."""
    utility_type: UtilityType = UtilityType.PROSPECT_THEORY
    risk_aversion: float = 0.5  # 0 = risk neutral, 1 = very risk averse
    loss_aversion: float = 2.25  # Losses weighed this much more than gains
    reference_point: float = 0.0  # Reference point for gains/losses
    tail_risk_weight: float = 1.5  # Extra weight on extreme downside
    tail_risk_percentile: float = 0.05  # What counts as "tail"


class UtilityFunction:
    """
    Transforms raw EMV values into utility values that account for
    risk preferences, loss aversion, and non-linear value perception.
    """

    def __init__(self, config: UtilityConfig | None = None):
        self.config = config or UtilityConfig()

    def compute_utility(self, value: float) -> float:
        """
        Compute utility for a single value.

        Args:
            value: Raw monetary/economic value

        Returns:
            Utility-adjusted value
        """
        t = self.config.utility_type

        if t == UtilityType.LINEAR:
            return self._linear(value)
        elif t == UtilityType.EXPONENTIAL:
            return self._exponential(value)
        elif t == UtilityType.POWER:
            return self._power(value)
        elif t == UtilityType.LOGARITHMIC:
            return self._logarithmic(value)
        elif t == UtilityType.PROSPECT_THEORY:
            return self._prospect_theory(value)
        else:
            return value

    def compute_expected_utility(
        self,
        outcomes: list[dict[str, float]],
    ) -> float:
        """
        Compute expected utility over a set of outcomes.

        Args:
            outcomes: List of {probability, payoff} dicts

        Returns:
            Expected utility value
        """
        total = 0.0
        for outcome in outcomes:
            prob = outcome["probability"]
            payoff = outcome["payoff"]
            utility = self.compute_utility(payoff)

            # Apply probability weighting (Prospect Theory style)
            weighted_prob = self._weight_probability(prob, payoff)
            total += weighted_prob * utility

        return total

    def apply_loss_penalty(self, value: float) -> float:
        """Apply asymmetric loss penalty."""
        ref = self.config.reference_point
        if value < ref:
            loss = ref - value
            return ref - loss * self.config.loss_aversion
        return value

    def apply_tail_risk_weighting(
        self, outcomes: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """
        Amplify the weight of tail-risk (extreme downside) outcomes.

        Returns:
            Modified outcomes with adjusted probabilities
        """
        if not outcomes:
            return outcomes

        payoffs = sorted([o["payoff"] for o in outcomes])
        tail_threshold_idx = max(1, int(len(payoffs) * self.config.tail_risk_percentile))
        tail_threshold = payoffs[tail_threshold_idx - 1]

        adjusted = []
        extra_weight = 0.0

        for o in outcomes:
            new_o = dict(o)
            if o["payoff"] <= tail_threshold:
                boost = self.config.tail_risk_weight
                extra_weight += o["probability"] * (boost - 1.0)
                new_o["probability"] *= boost
            adjusted.append(new_o)

        # Renormalize probabilities
        total = sum(o["probability"] for o in adjusted)
        if total > 0:
            for o in adjusted:
                o["probability"] /= total

        return adjusted

    def certainty_equivalent(
        self, outcomes: list[dict[str, float]]
    ) -> float:
        """
        Compute the certainty equivalent: the guaranteed value that
        gives the same utility as the uncertain prospects.
        """
        expected_utility = self.compute_expected_utility(outcomes)
        return self._inverse_utility(expected_utility)

    def risk_premium(
        self, outcomes: list[dict[str, float]]
    ) -> float:
        """
        Compute the risk premium: how much value the agent is willing
        to give up to avoid uncertainty.
        """
        emv = sum(o["probability"] * o["payoff"] for o in outcomes)
        ce = self.certainty_equivalent(outcomes)
        return emv - ce

    # -------------------------------------------------------------------------
    # UTILITY FUNCTIONS
    # -------------------------------------------------------------------------

    def _linear(self, x: float) -> float:
        return x

    def _exponential(self, x: float) -> float:
        """CARA (Constant Absolute Risk Aversion) utility."""
        a = self.config.risk_aversion
        if a <= 0:
            return x
        return (1.0 - math.exp(-a * x)) / a

    def _power(self, x: float) -> float:
        """CRRA (Constant Relative Risk Aversion) utility."""
        rho = self.config.risk_aversion
        if rho == 1.0:
            return math.log(max(1e-10, x)) if x > 0 else -1e10
        if x > 0:
            return (x ** (1.0 - rho)) / (1.0 - rho)
        return -1e10

    def _logarithmic(self, x: float) -> float:
        """Log utility (special case of power with rho=1)."""
        if x > 0:
            return math.log(x)
        return -1e10

    def _prospect_theory(self, x: float) -> float:
        """
        Kahneman-Tversky Prospect Theory value function.
        Concave for gains, convex for losses, steeper for losses.
        """
        ref = self.config.reference_point
        alpha = 0.88  # Diminishing sensitivity parameter

        if x >= ref:
            gain = x - ref
            return gain ** alpha
        else:
            loss = ref - x
            return -self.config.loss_aversion * (loss ** alpha)

    def _weight_probability(self, p: float, payoff: float) -> float:
        """
        Probability weighting function (Prelec, 1998).
        People overweight small probabilities and underweight large ones.
        """
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0

        gamma = 0.65 if payoff >= self.config.reference_point else 0.60
        return math.exp(-((-math.log(p)) ** gamma))

    def _inverse_utility(self, u: float) -> float:
        """Approximate inverse of the utility function."""
        t = self.config.utility_type

        if t == UtilityType.LINEAR:
            return u
        elif t == UtilityType.EXPONENTIAL:
            a = self.config.risk_aversion
            if a <= 0:
                return u
            val = 1.0 - a * u
            if val <= 0:
                return 1e10
            return -math.log(val) / a
        elif t == UtilityType.LOGARITHMIC:
            return math.exp(u)
        elif t == UtilityType.PROSPECT_THEORY:
            # Approximate inverse for gains
            alpha = 0.88
            if u >= 0:
                return self.config.reference_point + u ** (1.0 / alpha)
            else:
                return self.config.reference_point - (
                    abs(u) / self.config.loss_aversion
                ) ** (1.0 / alpha)
        return u


# =============================================================================
# ADAPTIVE UTILITY FUNCTION (CARBON[6] Spec §5.2)
# =============================================================================

class AdaptiveUtilityFunction(UtilityFunction):
    """
    Context-aware utility function that adjusts risk aversion dynamically.

    Increases risk aversion when (CARBON[6] §5.2):
        - Stakes are high → risk_aversion *= 1.5
        - Resources are low → risk_aversion *= 1.3
        - Time pressure is high → risk_aversion *= 1.2

    Usage:
        adaptive = AdaptiveUtilityFunction()
        utility = adaptive.compute_adaptive_utility(value=100.0, context={
            "stakes": 0.9,
            "resource_availability": 0.2,
            "time_pressure": 0.8,
        })
    """

    def __init__(
        self,
        config: UtilityConfig | None = None,
        high_stakes_threshold: float = 0.7,
        low_resource_threshold: float = 0.3,
        high_time_pressure_threshold: float = 0.7,
    ):
        super().__init__(config)
        self.high_stakes_threshold = high_stakes_threshold
        self.low_resource_threshold = low_resource_threshold
        self.high_time_pressure_threshold = high_time_pressure_threshold

    def compute_adaptive_utility(
        self,
        value: float,
        context: dict[str, Any],
    ) -> float:
        """
        Compute utility with context-adjusted risk aversion.

        Args:
            value: Raw monetary/economic value
            context: Dict with keys: stakes, resource_availability, time_pressure

        Returns:
            Utility-adjusted value with dynamic risk aversion
        """
        adjusted_aversion = self.config.risk_aversion

        if context.get("stakes", 0) > self.high_stakes_threshold:
            adjusted_aversion *= 1.5

        if context.get("resource_availability", 1.0) < self.low_resource_threshold:
            adjusted_aversion *= 1.3

        if context.get("time_pressure", 0) > self.high_time_pressure_threshold:
            adjusted_aversion *= 1.2

        # Use power utility with adjusted risk aversion
        if value > 0:
            rho = adjusted_aversion
            if rho == 1.0:
                return math.log(max(1e-10, value))
            return (value ** (1.0 - rho)) / (1.0 - rho)
        return -abs(value) * adjusted_aversion

    def get_effective_risk_aversion(self, context: dict[str, Any]) -> float:
        """Return the effective risk aversion for the given context."""
        aversion = self.config.risk_aversion
        if context.get("stakes", 0) > self.high_stakes_threshold:
            aversion *= 1.5
        if context.get("resource_availability", 1.0) < self.low_resource_threshold:
            aversion *= 1.3
        if context.get("time_pressure", 0) > self.high_time_pressure_threshold:
            aversion *= 1.2
        return aversion
