"""
UTILITY FUNCTION MODULE
Adjusts EMV for preferences and risk attitudes
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Includes:
- Risk aversion modeling
- Loss penalties
- Tail-risk weighting
- Non-linear utility curves

Advanced agents do not optimize raw EMV.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import math


class RiskAttitude(Enum):
    """Risk attitude classifications."""
    RISK_SEEKING = "risk_seeking"
    RISK_NEUTRAL = "risk_neutral"
    RISK_AVERSE = "risk_averse"
    LOSS_AVERSE = "loss_averse"


class UtilityFunctionType(Enum):
    """Types of utility functions."""
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    POWER = "power"
    PROSPECT_THEORY = "prospect_theory"
    QUADRATIC = "quadratic"


@dataclass
class UtilityConfig:
    """Configuration for utility function."""
    # Risk aversion coefficient (higher = more risk averse)
    risk_aversion: float = 0.5  # 0 to 1

    # Loss aversion coefficient (how much losses hurt vs gains)
    loss_aversion: float = 2.25  # Kahneman-Tversky default

    # Reference point for gains/losses
    reference_point: float = 0.0

    # Tail risk parameters
    tail_risk_weight: float = 2.0  # Extra weight on extreme losses

    # Diminishing marginal utility
    diminishing_factor: float = 0.88  # Power function exponent

    # Satiation point (utility plateaus above this)
    satiation_point: Optional[float] = None

    # Subsistence point (minimum acceptable outcome)
    subsistence_point: Optional[float] = None


@dataclass
class UtilityResult:
    """Result of utility calculation."""
    raw_value: float
    utility: float
    certainty_equivalent: float
    risk_premium: float
    utility_function_type: UtilityFunctionType
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseUtilityFunction(ABC):
    """Abstract base class for utility functions."""

    def __init__(self, config: Optional[UtilityConfig] = None):
        self.config = config or UtilityConfig()

    @abstractmethod
    def calculate_utility(self, value: float) -> float:
        """Calculate utility of a monetary value."""
        pass

    @abstractmethod
    def inverse_utility(self, utility: float) -> float:
        """Calculate monetary value from utility (certainty equivalent)."""
        pass

    def calculate_expected_utility(
        self,
        values: List[float],
        probabilities: List[float]
    ) -> float:
        """Calculate expected utility of a probability distribution."""
        utilities = [self.calculate_utility(v) for v in values]
        return sum(u * p for u, p in zip(utilities, probabilities))

    def calculate_certainty_equivalent(
        self,
        values: List[float],
        probabilities: List[float]
    ) -> float:
        """Calculate certainty equivalent of a gamble."""
        expected_utility = self.calculate_expected_utility(values, probabilities)
        return self.inverse_utility(expected_utility)

    def calculate_risk_premium(
        self,
        values: List[float],
        probabilities: List[float]
    ) -> float:
        """
        Calculate risk premium.
        Risk premium = Expected Value - Certainty Equivalent
        """
        expected_value = sum(v * p for v, p in zip(values, probabilities))
        certainty_equivalent = self.calculate_certainty_equivalent(values, probabilities)
        return expected_value - certainty_equivalent

    def get_full_result(
        self,
        values: List[float],
        probabilities: List[float]
    ) -> UtilityResult:
        """Get comprehensive utility analysis."""
        expected_value = sum(v * p for v, p in zip(values, probabilities))
        expected_utility = self.calculate_expected_utility(values, probabilities)
        certainty_equivalent = self.calculate_certainty_equivalent(values, probabilities)
        risk_premium = expected_value - certainty_equivalent

        return UtilityResult(
            raw_value=expected_value,
            utility=expected_utility,
            certainty_equivalent=certainty_equivalent,
            risk_premium=risk_premium,
            utility_function_type=self.function_type,
            metadata={
                "risk_aversion": self.config.risk_aversion,
                "num_outcomes": len(values)
            }
        )

    @property
    @abstractmethod
    def function_type(self) -> UtilityFunctionType:
        """Return the utility function type."""
        pass


class LinearUtilityFunction(BaseUtilityFunction):
    """
    Linear utility function (risk neutral).
    U(x) = x
    """

    def calculate_utility(self, value: float) -> float:
        return value

    def inverse_utility(self, utility: float) -> float:
        return utility

    @property
    def function_type(self) -> UtilityFunctionType:
        return UtilityFunctionType.LINEAR


class LogarithmicUtilityFunction(BaseUtilityFunction):
    """
    Logarithmic utility function (CRRA with coefficient 1).
    U(x) = ln(x) for x > 0
    Exhibits constant relative risk aversion.
    """

    def __init__(self, config: Optional[UtilityConfig] = None):
        super().__init__(config)
        self.offset = 1.0  # Shift to handle non-positive values

    def calculate_utility(self, value: float) -> float:
        shifted = value + self.offset
        if shifted <= 0:
            return float('-inf')
        return math.log(shifted)

    def inverse_utility(self, utility: float) -> float:
        return math.exp(utility) - self.offset

    @property
    def function_type(self) -> UtilityFunctionType:
        return UtilityFunctionType.LOGARITHMIC


class ExponentialUtilityFunction(BaseUtilityFunction):
    """
    Exponential utility function (CARA).
    U(x) = -exp(-α × x) / α
    Exhibits constant absolute risk aversion.
    """

    def calculate_utility(self, value: float) -> float:
        alpha = self.config.risk_aversion
        if alpha == 0:
            return value
        return -math.exp(-alpha * value) / alpha

    def inverse_utility(self, utility: float) -> float:
        alpha = self.config.risk_aversion
        if alpha == 0:
            return utility
        return -math.log(-alpha * utility) / alpha

    @property
    def function_type(self) -> UtilityFunctionType:
        return UtilityFunctionType.EXPONENTIAL


class PowerUtilityFunction(BaseUtilityFunction):
    """
    Power utility function (CRRA).
    U(x) = x^(1-ρ) / (1-ρ) for ρ ≠ 1
    U(x) = ln(x) for ρ = 1
    Where ρ is the coefficient of relative risk aversion.
    """

    def calculate_utility(self, value: float) -> float:
        rho = self.config.risk_aversion
        gamma = self.config.diminishing_factor

        if value <= 0:
            # Handle non-positive values
            return self._handle_negative(value, rho)

        if abs(rho - 1) < 0.001:
            return math.log(value)

        return (value ** gamma) / gamma

    def inverse_utility(self, utility: float) -> float:
        gamma = self.config.diminishing_factor

        if utility <= 0:
            return 0

        return (utility * gamma) ** (1 / gamma)

    def _handle_negative(self, value: float, rho: float) -> float:
        """Handle negative values with loss aversion."""
        lambda_loss = self.config.loss_aversion
        gamma = self.config.diminishing_factor

        # Apply loss aversion: losses hurt more
        return -lambda_loss * ((-value) ** gamma) / gamma

    @property
    def function_type(self) -> UtilityFunctionType:
        return UtilityFunctionType.POWER


class ProspectTheoryUtilityFunction(BaseUtilityFunction):
    """
    Prospect Theory utility function (Kahneman & Tversky).
    Includes:
    - Reference dependence
    - Loss aversion
    - Diminishing sensitivity
    - Probability weighting
    """

    def __init__(self, config: Optional[UtilityConfig] = None):
        super().__init__(config)
        # Probability weighting parameter
        self.gamma_gain = 0.61
        self.gamma_loss = 0.69

    def calculate_utility(self, value: float) -> float:
        """
        Calculate prospect theory value function.
        v(x) = x^α for x ≥ 0 (gains)
        v(x) = -λ(-x)^β for x < 0 (losses)
        """
        reference = self.config.reference_point
        relative_value = value - reference

        alpha = self.config.diminishing_factor  # Typically 0.88
        beta = self.config.diminishing_factor
        lambda_loss = self.config.loss_aversion  # Typically 2.25

        if relative_value >= 0:
            # Gain domain
            return relative_value ** alpha
        else:
            # Loss domain (with loss aversion)
            return -lambda_loss * ((-relative_value) ** beta)

    def inverse_utility(self, utility: float) -> float:
        """Inverse of prospect theory value function."""
        alpha = self.config.diminishing_factor
        lambda_loss = self.config.loss_aversion

        if utility >= 0:
            return utility ** (1 / alpha) + self.config.reference_point
        else:
            return self.config.reference_point - ((-utility / lambda_loss) ** (1 / alpha))

    def weight_probability(self, p: float, is_gain: bool = True) -> float:
        """
        Apply probability weighting function.
        Overweights small probabilities, underweights large ones.
        """
        gamma = self.gamma_gain if is_gain else self.gamma_loss

        if p <= 0:
            return 0
        if p >= 1:
            return 1

        return (p ** gamma) / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))

    def calculate_weighted_expected_utility(
        self,
        values: List[float],
        probabilities: List[float]
    ) -> float:
        """
        Calculate expected utility with probability weighting.
        This is the full prospect theory model.
        """
        # Separate gains and losses
        gains = [(v, p) for v, p in zip(values, probabilities) if v >= self.config.reference_point]
        losses = [(v, p) for v, p in zip(values, probabilities) if v < self.config.reference_point]

        # Sort by absolute value for rank-dependent weighting
        gains.sort(key=lambda x: x[0], reverse=True)
        losses.sort(key=lambda x: x[0])

        total_utility = 0

        # Process gains (rank-dependent)
        cumulative = 0
        for value, prob in gains:
            weighted_prob = self.weight_probability(cumulative + prob, is_gain=True) - \
                          self.weight_probability(cumulative, is_gain=True)
            total_utility += self.calculate_utility(value) * weighted_prob
            cumulative += prob

        # Process losses (rank-dependent)
        cumulative = 0
        for value, prob in losses:
            weighted_prob = self.weight_probability(cumulative + prob, is_gain=False) - \
                          self.weight_probability(cumulative, is_gain=False)
            total_utility += self.calculate_utility(value) * weighted_prob
            cumulative += prob

        return total_utility

    @property
    def function_type(self) -> UtilityFunctionType:
        return UtilityFunctionType.PROSPECT_THEORY


class AdaptiveUtilityFunction(BaseUtilityFunction):
    """
    Adaptive utility function that adjusts based on context.
    Uses different sub-functions for different scenarios.
    """

    def __init__(self, config: Optional[UtilityConfig] = None):
        super().__init__(config)
        self._linear = LinearUtilityFunction(config)
        self._power = PowerUtilityFunction(config)
        self._prospect = ProspectTheoryUtilityFunction(config)

        # Context for adaptation
        self._current_wealth: float = 0
        self._risk_budget: float = 1.0
        self._recent_outcomes: List[float] = []

    def set_context(
        self,
        wealth: Optional[float] = None,
        risk_budget: Optional[float] = None
    ):
        """Set context for adaptive behavior."""
        if wealth is not None:
            self._current_wealth = wealth
        if risk_budget is not None:
            self._risk_budget = risk_budget

    def calculate_utility(self, value: float) -> float:
        """Calculate utility adaptively based on context."""
        # Small stakes relative to wealth: use linear
        if abs(value) < self._current_wealth * 0.01:
            return self._linear.calculate_utility(value)

        # Losses or low risk budget: use prospect theory
        if value < 0 or self._risk_budget < 0.3:
            return self._prospect.calculate_utility(value)

        # Large gains with high risk budget: use power
        return self._power.calculate_utility(value)

    def inverse_utility(self, utility: float) -> float:
        """Inverse utility (uses power function as default)."""
        return self._power.inverse_utility(utility)

    def record_outcome(self, outcome: float):
        """Record outcome for adaptation."""
        self._recent_outcomes.append(outcome)
        if len(self._recent_outcomes) > 50:
            self._recent_outcomes = self._recent_outcomes[-50:]

        # Adapt risk budget based on recent performance
        if len(self._recent_outcomes) >= 10:
            recent_mean = np.mean(self._recent_outcomes[-10:])
            if recent_mean < 0:
                self._risk_budget = max(0.1, self._risk_budget * 0.95)
            elif recent_mean > 0:
                self._risk_budget = min(1.0, self._risk_budget * 1.02)

    @property
    def function_type(self) -> UtilityFunctionType:
        return UtilityFunctionType.POWER  # Primary function


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_utility_function(
    function_type: UtilityFunctionType = UtilityFunctionType.PROSPECT_THEORY,
    config: Optional[UtilityConfig] = None
) -> BaseUtilityFunction:
    """Create a utility function of the specified type."""

    if function_type == UtilityFunctionType.LINEAR:
        return LinearUtilityFunction(config)
    elif function_type == UtilityFunctionType.LOGARITHMIC:
        return LogarithmicUtilityFunction(config)
    elif function_type == UtilityFunctionType.EXPONENTIAL:
        return ExponentialUtilityFunction(config)
    elif function_type == UtilityFunctionType.POWER:
        return PowerUtilityFunction(config)
    elif function_type == UtilityFunctionType.PROSPECT_THEORY:
        return ProspectTheoryUtilityFunction(config)
    else:
        return AdaptiveUtilityFunction(config)
