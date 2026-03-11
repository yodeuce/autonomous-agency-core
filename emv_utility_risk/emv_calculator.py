"""
FILE 14: emv_calculator.py
PURPOSE: Computes Expected Monetary/Economic Value
ROLE: The core decision metric engine
SPEC: CARBON[6] Technical Architecture Specification v1.0.0

Formal EMV Definition (CARBON[6] §5.1):
    EMV(a) = Σᵢ P(oᵢ | a) × V(oᵢ)

    Where:
        a       = action under consideration
        oᵢ      = possible outcome i
        P(oᵢ|a) = probability of outcome given action
        V(oᵢ)   = monetary value of outcome

Logic:
- Outcome enumeration via OutcomeEnumerator
- Probability estimation
- Payoff calculation
- Discounting
- Action comparison with ActionRanking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Outcome:
    """A possible outcome of an action."""
    name: str
    probability: float
    payoff: float
    description: str = ""
    risk_factors: list[str] = field(default_factory=list)

    def weighted_payoff(self) -> float:
        return self.probability * self.payoff


@dataclass
class ActionRanking:
    """Result of comparing multiple actions by EMV (CARBON[6] §5.1)."""
    best_action: str
    scores: dict[str, float]
    margin: float  # EMV difference between best and second-best
    ranked_results: list["EMVResult"] = field(default_factory=list)


@dataclass
class EMVResult:
    """Result of an EMV computation."""
    action: str
    emv: float
    outcomes: list[Outcome]
    confidence: float
    upside: float
    downside: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EMVCalculator:
    """
    Computes Expected Monetary/Economic Value for decision analysis.
    This is the core metric engine for the autonomous agent.
    """

    def __init__(self, discount_rate: float = 0.0, time_horizon: int = 1):
        self.discount_rate = discount_rate
        self.time_horizon = time_horizon
        self.computation_log: list[EMVResult] = []

    def compute_emv(
        self,
        action: str,
        outcomes: list[Outcome],
        discount_periods: int = 0,
    ) -> EMVResult:
        """
        Compute EMV for an action given its possible outcomes.

        Args:
            action: The action being evaluated
            outcomes: List of possible outcomes with probabilities and payoffs
            discount_periods: Number of periods to discount

        Returns:
            EMVResult with computed EMV and analysis
        """
        # Validate probabilities
        total_prob = sum(o.probability for o in outcomes)
        if abs(total_prob - 1.0) > 0.01:
            logger.warning(
                f"Outcome probabilities sum to {total_prob:.3f}, not 1.0. Normalizing."
            )
            for o in outcomes:
                o.probability /= total_prob

        # Compute raw EMV
        raw_emv = sum(o.weighted_payoff() for o in outcomes)

        # Apply time discounting
        if discount_periods > 0 and self.discount_rate > 0:
            discount_factor = 1.0 / ((1.0 + self.discount_rate) ** discount_periods)
            emv = raw_emv * discount_factor
        else:
            emv = raw_emv

        # Compute upside/downside
        positive = [o for o in outcomes if o.payoff > 0]
        negative = [o for o in outcomes if o.payoff < 0]

        upside = sum(o.weighted_payoff() for o in positive) if positive else 0.0
        downside = sum(o.weighted_payoff() for o in negative) if negative else 0.0

        # Confidence based on outcome count and probability spread
        confidence = self._estimate_confidence(outcomes)

        result = EMVResult(
            action=action,
            emv=emv,
            outcomes=outcomes,
            confidence=confidence,
            upside=upside,
            downside=downside,
            metadata={
                "raw_emv": raw_emv,
                "discount_periods": discount_periods,
                "discount_rate": self.discount_rate,
            },
        )

        self.computation_log.append(result)
        return result

    def compare_actions(
        self,
        action_outcomes: dict[str, list[Outcome]],
    ) -> ActionRanking:
        """
        Compare multiple actions by their EMV (CARBON[6] §5.1).

        Args:
            action_outcomes: {action_name: [outcomes]}

        Returns:
            ActionRanking with best action, scores, and margin
        """
        results = []
        for action, outcomes in action_outcomes.items():
            result = self.compute_emv(action, outcomes)
            results.append(result)

        results.sort(key=lambda r: r.emv, reverse=True)
        scores = {r.action: r.emv for r in results}

        margin = (
            results[0].emv - results[1].emv
            if len(results) > 1
            else float("inf")
        )

        return ActionRanking(
            best_action=results[0].action,
            scores=scores,
            margin=margin,
            ranked_results=results,
        )

    def compute_marginal_emv(
        self,
        base_emv: float,
        additional_outcomes: list[Outcome],
        additional_cost: float = 0.0,
    ) -> float:
        """
        Compute the marginal EMV of additional action/investment.

        Args:
            base_emv: EMV of the base case
            additional_outcomes: Outcomes of the additional action
            additional_cost: Cost of the additional action

        Returns:
            Marginal EMV (positive = worth doing)
        """
        additional_emv = sum(o.weighted_payoff() for o in additional_outcomes)
        return additional_emv - additional_cost - base_emv

    def sensitivity_analysis(
        self,
        action: str,
        outcomes: list[Outcome],
        vary_index: int,
        probability_range: tuple[float, float] = (0.0, 1.0),
        steps: int = 10,
    ) -> list[tuple[float, float]]:
        """
        Analyze how EMV changes as one outcome's probability varies.

        Returns:
            List of (probability, emv) pairs
        """
        results = []
        lo, hi = probability_range
        step_size = (hi - lo) / steps

        for i in range(steps + 1):
            p = lo + i * step_size
            test_outcomes = [Outcome(o.name, o.probability, o.payoff) for o in outcomes]
            original_p = test_outcomes[vary_index].probability
            test_outcomes[vary_index].probability = p

            # Redistribute remaining probability proportionally
            remaining = 1.0 - p
            other_total = sum(
                o.probability for j, o in enumerate(test_outcomes) if j != vary_index
            )
            if other_total > 0:
                for j, o in enumerate(test_outcomes):
                    if j != vary_index:
                        o.probability = o.probability / other_total * remaining

            emv = sum(o.weighted_payoff() for o in test_outcomes)
            results.append((p, emv))

        return results

    def _estimate_confidence(self, outcomes: list[Outcome]) -> float:
        """Estimate confidence in the EMV calculation."""
        if not outcomes:
            return 0.0

        # More outcomes generally means more thorough analysis
        count_factor = min(1.0, len(outcomes) / 5.0)

        # Well-spread probabilities indicate more careful analysis
        probs = [o.probability for o in outcomes]
        max_prob = max(probs)
        spread_factor = 1.0 - (max_prob - (1.0 / len(outcomes)))

        return min(1.0, (count_factor + spread_factor) / 2.0)


# =============================================================================
# OUTCOME ENUMERATOR (CARBON[6] Spec §5.1)
# =============================================================================

class OutcomeEnumerator:
    """
    Generates probability-weighted outcome sets from transition models.

    Process (CARBON[6] §5.1):
        1. Generate base outcomes from transition model
        2. Expand with uncertainty
        3. Normalize probabilities
    """

    def enumerate(
        self,
        state: dict[str, Any],
        action: str,
        transition_model: Any = None,
    ) -> list[Outcome]:
        """
        Enumerate possible outcomes for a state-action pair.

        Args:
            state: Current state
            action: Action under consideration
            transition_model: Model predicting outcomes (optional)

        Returns:
            List of Outcomes with normalized probabilities
        """
        if transition_model is not None and hasattr(transition_model, "predict_outcomes"):
            base_outcomes = transition_model.predict_outcomes(state, action)
        else:
            base_outcomes = self._default_outcomes(state, action)

        expanded = self.expand_uncertainty(base_outcomes)
        return self._normalize_probabilities(expanded)

    def expand_uncertainty(
        self,
        outcomes: list[Outcome],
        uncertainty_factor: float = 0.1,
    ) -> list[Outcome]:
        """
        Expand outcomes to account for uncertainty.
        Adds pessimistic and optimistic variants for each outcome.
        """
        if uncertainty_factor <= 0:
            return outcomes

        expanded = list(outcomes)
        for o in outcomes:
            if o.probability > uncertainty_factor * 2:
                shift = o.payoff * uncertainty_factor
                # Pessimistic variant
                expanded.append(Outcome(
                    name=f"{o.name}_pessimistic",
                    probability=o.probability * uncertainty_factor,
                    payoff=o.payoff - abs(shift),
                    description=f"Pessimistic variant of {o.name}",
                ))
                # Optimistic variant
                expanded.append(Outcome(
                    name=f"{o.name}_optimistic",
                    probability=o.probability * uncertainty_factor,
                    payoff=o.payoff + abs(shift),
                    description=f"Optimistic variant of {o.name}",
                ))
                # Reduce original probability
                o.probability *= (1.0 - 2.0 * uncertainty_factor)

        return expanded

    def _normalize_probabilities(self, outcomes: list[Outcome]) -> list[Outcome]:
        """Normalize probabilities to sum to 1.0."""
        total = sum(o.probability for o in outcomes)
        if total > 0:
            for o in outcomes:
                o.probability /= total
        return outcomes

    def _default_outcomes(
        self, state: dict[str, Any], action: str
    ) -> list[Outcome]:
        """Generate default outcomes when no transition model is available."""
        return [
            Outcome("success", 0.6, 10.0, "Action succeeds"),
            Outcome("partial", 0.25, 3.0, "Partial success"),
            Outcome("failure", 0.15, -5.0, "Action fails"),
        ]
