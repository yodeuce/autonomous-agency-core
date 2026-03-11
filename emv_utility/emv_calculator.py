"""
EMV CALCULATOR
Computes Expected Monetary/Economic Value for decisions
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Logic:
- Outcome enumeration
- Probability estimation
- Payoff calculation
- Discounting

This is the core decision metric engine.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import numpy as np


class OutcomeType(Enum):
    """Types of outcomes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NEUTRAL = "neutral"
    FAILURE = "failure"
    CRITICAL_FAILURE = "critical_failure"


@dataclass
class Outcome:
    """Represents a possible outcome of an action."""
    outcome_id: str
    outcome_type: OutcomeType
    description: str
    probability: float
    value: float  # Monetary or utility value
    risk_adjusted_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EMVResult:
    """Result of EMV calculation."""
    action: str
    emv: float
    outcomes: List[Outcome]
    probability_weighted_value: float
    best_case_value: float
    worst_case_value: float
    value_at_risk: float
    confidence: float
    calculation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EMVConfig:
    """Configuration for EMV calculator."""
    # Discounting
    discount_rate: float = 0.05  # Annual discount rate
    time_horizon_days: int = 365

    # Risk adjustment
    risk_free_rate: float = 0.02
    risk_premium: float = 0.05

    # Value at Risk
    var_confidence_level: float = 0.95

    # Probability calibration
    calibration_factor: float = 1.0
    pessimism_factor: float = 0.0  # 0 = neutral, 1 = pessimistic

    # Minimum values
    min_probability: float = 0.001
    min_value: float = -1000000
    max_value: float = 1000000


class EMVCalculator:
    """
    Calculator for Expected Monetary Value.
    Core decision metric engine for autonomous agents.
    """

    def __init__(self, config: Optional[EMVConfig] = None):
        self.config = config or EMVConfig()
        self._outcome_history: List[Tuple[str, Outcome, bool]] = []  # action, outcome, realized

    def calculate_emv(
        self,
        action: str,
        outcomes: List[Outcome],
        context: Optional[Dict[str, Any]] = None
    ) -> EMVResult:
        """
        Calculate Expected Monetary Value for an action.

        EMV = Σ P(outcome_i) × V(outcome_i)

        Args:
            action: Action being evaluated
            outcomes: List of possible outcomes with probabilities and values
            context: Additional context for calculation

        Returns:
            EMVResult with comprehensive analysis
        """
        context = context or {}

        # Validate and normalize probabilities
        outcomes = self._normalize_probabilities(outcomes)

        # Apply calibration and pessimism adjustments
        outcomes = self._apply_calibration(outcomes)

        # Calculate base EMV
        emv = sum(o.probability * o.value for o in outcomes)

        # Calculate probability-weighted value (same as EMV for basic case)
        pwv = emv

        # Calculate best and worst case
        values = [o.value for o in outcomes]
        best_case = max(values)
        worst_case = min(values)

        # Calculate Value at Risk (VaR)
        var = self._calculate_var(outcomes)

        # Apply time discounting if applicable
        time_horizon = context.get("time_horizon_days", self.config.time_horizon_days)
        if time_horizon > 0:
            emv = self._apply_discounting(emv, time_horizon)

        # Estimate confidence
        confidence = self._estimate_confidence(outcomes, context)

        return EMVResult(
            action=action,
            emv=round(emv, 2),
            outcomes=outcomes,
            probability_weighted_value=round(pwv, 2),
            best_case_value=round(best_case, 2),
            worst_case_value=round(worst_case, 2),
            value_at_risk=round(var, 2),
            confidence=round(confidence, 3),
            metadata={
                "num_outcomes": len(outcomes),
                "time_horizon_days": time_horizon,
                "discount_rate": self.config.discount_rate
            }
        )

    def calculate_incremental_emv(
        self,
        action: str,
        outcomes: List[Outcome],
        baseline_outcomes: List[Outcome]
    ) -> EMVResult:
        """
        Calculate incremental EMV compared to baseline.
        Useful for comparing against status quo.
        """
        action_result = self.calculate_emv(action, outcomes)
        baseline_result = self.calculate_emv("baseline", baseline_outcomes)

        incremental_emv = action_result.emv - baseline_result.emv

        return EMVResult(
            action=f"{action}_incremental",
            emv=round(incremental_emv, 2),
            outcomes=outcomes,
            probability_weighted_value=action_result.probability_weighted_value,
            best_case_value=action_result.best_case_value - baseline_result.worst_case_value,
            worst_case_value=action_result.worst_case_value - baseline_result.best_case_value,
            value_at_risk=action_result.value_at_risk,
            confidence=min(action_result.confidence, baseline_result.confidence),
            metadata={
                "baseline_emv": baseline_result.emv,
                "action_emv": action_result.emv
            }
        )

    def compare_actions(
        self,
        actions_outcomes: Dict[str, List[Outcome]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, EMVResult]:
        """
        Compare multiple actions and rank by EMV.

        Returns dict of action -> EMVResult, sorted by EMV descending.
        """
        results = {}
        for action, outcomes in actions_outcomes.items():
            results[action] = self.calculate_emv(action, outcomes, context)

        # Sort by EMV descending
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1].emv, reverse=True)
        )

        return sorted_results

    def calculate_portfolio_emv(
        self,
        actions_outcomes: Dict[str, List[Outcome]],
        weights: Dict[str, float]
    ) -> EMVResult:
        """
        Calculate portfolio EMV for multiple actions.
        Accounts for diversification effects.
        """
        total_emv = 0
        weighted_var = 0

        for action, outcomes in actions_outcomes.items():
            weight = weights.get(action, 0)
            result = self.calculate_emv(action, outcomes)
            total_emv += weight * result.emv
            weighted_var += weight * result.value_at_risk

        return EMVResult(
            action="portfolio",
            emv=round(total_emv, 2),
            outcomes=[],
            probability_weighted_value=round(total_emv, 2),
            best_case_value=0,
            worst_case_value=0,
            value_at_risk=round(weighted_var, 2),
            confidence=0.8,
            metadata={"weights": weights}
        )

    def _normalize_probabilities(self, outcomes: List[Outcome]) -> List[Outcome]:
        """Normalize probabilities to sum to 1."""
        total_prob = sum(o.probability for o in outcomes)

        if total_prob == 0:
            # Uniform distribution if all zero
            uniform = 1.0 / len(outcomes)
            for o in outcomes:
                o.probability = uniform
        elif abs(total_prob - 1.0) > 0.01:
            # Normalize
            for o in outcomes:
                o.probability = o.probability / total_prob

        # Enforce minimum probability
        for o in outcomes:
            o.probability = max(self.config.min_probability, o.probability)

        # Re-normalize after minimum enforcement
        total = sum(o.probability for o in outcomes)
        for o in outcomes:
            o.probability /= total

        return outcomes

    def _apply_calibration(self, outcomes: List[Outcome]) -> List[Outcome]:
        """Apply calibration and pessimism factors."""
        calibration = self.config.calibration_factor
        pessimism = self.config.pessimism_factor

        for o in outcomes:
            # Apply calibration
            o.probability *= calibration

            # Apply pessimism (increase probability of bad outcomes)
            if o.value < 0:
                o.probability *= (1 + pessimism)
            elif o.value > 0:
                o.probability *= (1 - pessimism * 0.5)

        # Re-normalize
        return self._normalize_probabilities(outcomes)

    def _apply_discounting(self, value: float, days: int) -> float:
        """Apply time value discounting."""
        years = days / 365
        discount_factor = (1 + self.config.discount_rate) ** years
        return value / discount_factor

    def _calculate_var(self, outcomes: List[Outcome]) -> float:
        """
        Calculate Value at Risk at configured confidence level.
        VaR represents worst-case loss at confidence level.
        """
        # Sort outcomes by value
        sorted_outcomes = sorted(outcomes, key=lambda o: o.value)

        cumulative_prob = 0
        target_prob = 1 - self.config.var_confidence_level

        for outcome in sorted_outcomes:
            cumulative_prob += outcome.probability
            if cumulative_prob >= target_prob:
                return -outcome.value if outcome.value < 0 else 0

        return 0

    def _estimate_confidence(
        self,
        outcomes: List[Outcome],
        context: Dict[str, Any]
    ) -> float:
        """Estimate confidence in EMV calculation."""
        confidence = 0.8  # Base confidence

        # Reduce confidence for many outcomes (more uncertainty)
        if len(outcomes) > 5:
            confidence -= 0.05 * (len(outcomes) - 5)

        # Reduce confidence for wide value range
        values = [o.value for o in outcomes]
        value_range = max(values) - min(values)
        if value_range > 10000:
            confidence -= 0.1

        # Reduce confidence for low probability outcomes with high impact
        for o in outcomes:
            if o.probability < 0.1 and abs(o.value) > 1000:
                confidence -= 0.05

        # Context-based adjustments
        if context.get("historical_data_available"):
            confidence += 0.1
        if context.get("expert_validated"):
            confidence += 0.1

        return max(0.1, min(1.0, confidence))

    def record_outcome(
        self,
        action: str,
        predicted_outcome: Outcome,
        realized: bool
    ):
        """Record actual outcome for calibration learning."""
        self._outcome_history.append((action, predicted_outcome, realized))

        # Keep limited history
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-1000:]

    def get_calibration_score(self) -> float:
        """Calculate calibration score from outcome history."""
        if not self._outcome_history:
            return 1.0

        # Compare predicted probabilities with actual outcomes
        predicted_probs = []
        realized = []

        for action, outcome, was_realized in self._outcome_history:
            predicted_probs.append(outcome.probability)
            realized.append(1 if was_realized else 0)

        # Brier score (lower is better)
        brier = sum(
            (p - r) ** 2 for p, r in zip(predicted_probs, realized)
        ) / len(predicted_probs)

        # Convert to score (1 = perfect, 0 = worst)
        return 1 - brier


class OutcomeEnumerator:
    """
    Helper class for enumerating possible outcomes.
    """

    @staticmethod
    def enumerate_binary(
        success_prob: float,
        success_value: float,
        failure_value: float
    ) -> List[Outcome]:
        """Enumerate simple success/failure outcomes."""
        return [
            Outcome(
                outcome_id="success",
                outcome_type=OutcomeType.SUCCESS,
                description="Action succeeds",
                probability=success_prob,
                value=success_value
            ),
            Outcome(
                outcome_id="failure",
                outcome_type=OutcomeType.FAILURE,
                description="Action fails",
                probability=1 - success_prob,
                value=failure_value
            )
        ]

    @staticmethod
    def enumerate_standard(
        success_prob: float = 0.6,
        partial_prob: float = 0.25,
        failure_prob: float = 0.14,
        critical_prob: float = 0.01,
        success_value: float = 100,
        partial_value: float = 50,
        failure_value: float = -50,
        critical_value: float = -500
    ) -> List[Outcome]:
        """Enumerate standard 4-outcome scenario."""
        return [
            Outcome(
                outcome_id="success",
                outcome_type=OutcomeType.SUCCESS,
                description="Complete success",
                probability=success_prob,
                value=success_value
            ),
            Outcome(
                outcome_id="partial",
                outcome_type=OutcomeType.PARTIAL_SUCCESS,
                description="Partial success",
                probability=partial_prob,
                value=partial_value
            ),
            Outcome(
                outcome_id="failure",
                outcome_type=OutcomeType.FAILURE,
                description="Failure",
                probability=failure_prob,
                value=failure_value
            ),
            Outcome(
                outcome_id="critical",
                outcome_type=OutcomeType.CRITICAL_FAILURE,
                description="Critical failure",
                probability=critical_prob,
                value=critical_value
            )
        ]

    @staticmethod
    def enumerate_from_distribution(
        values: List[float],
        probabilities: List[float]
    ) -> List[Outcome]:
        """Create outcomes from custom distribution."""
        outcomes = []
        for i, (value, prob) in enumerate(zip(values, probabilities)):
            outcome_type = (
                OutcomeType.SUCCESS if value > 0
                else OutcomeType.FAILURE if value < 0
                else OutcomeType.NEUTRAL
            )
            outcomes.append(Outcome(
                outcome_id=f"outcome_{i}",
                outcome_type=outcome_type,
                description=f"Outcome with value {value}",
                probability=prob,
                value=value
            ))
        return outcomes


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_emv_calculator(
    config: Optional[EMVConfig] = None
) -> EMVCalculator:
    """Create an EMV calculator."""
    return EMVCalculator(config)
