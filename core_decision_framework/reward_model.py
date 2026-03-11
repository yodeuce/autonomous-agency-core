"""
FILE 3: reward_model.py
PURPOSE: Encodes EMV into machine-usable rewards
ROLE: This is where EMV becomes executable
SPEC: CARBON[6] Technical Architecture Specification v1.0.0

Formal Reward Definitions:
    EMV Reward:
        R(s, a, s') = Σ_o P(o | s, a, s') * V(o)
        Where: o = outcome, P = probability, V = monetary value

    Risk-Adjusted Reward:
        R_adjusted(s, a) = U(EMV(s, a)) - λ * Risk(s, a)
        Where: U() = utility function, λ = risk aversion coefficient,
               Risk() = risk measure (VaR, CVaR, variance)

    Multi-Objective Reward:
        R_total = Σ_i w_i * R_i(s, a)
        Subject to: Σ w_i = 1, w_i ≥ 0

    Multi-Objective Weight Ranges (CARBON[6] §2.3):
        | Objective           | Weight Range | Description                      |
        |---------------------|-------------|----------------------------------|
        | Task Completion     | 0.3 - 0.5  | Primary task success             |
        | Resource Efficiency | 0.1 - 0.2  | Minimize resource consumption    |
        | Safety Margin       | 0.2 - 0.3  | Maintain safety buffer           |
        | User Satisfaction   | 0.1 - 0.2  | Inferred user preference alignment|

Contains:
- Immediate reward calculation
- Long-term reward aggregation
- Penalties for constraint violations
- Reward shaping logic
- Risk-adjusted reward model
- Multi-objective reward model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    TASK_COMPLETION = "task_completion"
    EFFICIENCY = "efficiency"
    RISK_AVOIDANCE = "risk_avoidance"
    INFORMATION_GAIN = "information_gain"
    ALIGNMENT = "alignment"
    CONSTRAINT_PENALTY = "constraint_penalty"


@dataclass
class RewardSignal:
    """A single reward signal with metadata."""
    component: RewardComponent
    value: float
    weight: float
    timestamp: str = ""
    explanation: str = ""


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    weights: dict[str, float] = field(default_factory=lambda: {
        "task_completion": 0.4,
        "efficiency": 0.2,
        "risk_avoidance": 0.2,
        "information_gain": 0.1,
        "alignment": 0.1,
    })
    constraint_penalty: float = -50.0
    reward_bounds: tuple[float, float] = (-100.0, 100.0)
    shaping_enabled: bool = True
    discount_factor: float = 0.95


class RewardModel:
    """
    Core reward model that converts agent actions and outcomes
    into scalar reward signals aligned with EMV objectives.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.reward_history: list[dict[str, Any]] = []
        self.cumulative_reward: float = 0.0
        self.step_count: int = 0

    def compute_immediate_reward(
        self,
        state: dict[str, Any],
        action: str,
        next_state: dict[str, Any],
        outcome: dict[str, Any],
    ) -> float:
        """
        Compute the immediate reward for a single state transition.

        Args:
            state: State before the action
            action: Action taken
            next_state: Resulting state
            outcome: Observed outcome metadata

        Returns:
            Scalar reward value (bounded)
        """
        signals: list[RewardSignal] = []

        # Task completion reward
        task_reward = self._compute_task_completion(state, action, next_state, outcome)
        signals.append(RewardSignal(
            component=RewardComponent.TASK_COMPLETION,
            value=task_reward,
            weight=self.config.weights["task_completion"],
            explanation=f"Task progress: {outcome.get('progress', 0):.1%}",
        ))

        # Efficiency reward
        efficiency_reward = self._compute_efficiency(state, action, outcome)
        signals.append(RewardSignal(
            component=RewardComponent.EFFICIENCY,
            value=efficiency_reward,
            weight=self.config.weights["efficiency"],
            explanation=f"Resource usage: {outcome.get('resource_cost', 0):.2f}",
        ))

        # Risk avoidance reward
        risk_reward = self._compute_risk_avoidance(state, next_state, outcome)
        signals.append(RewardSignal(
            component=RewardComponent.RISK_AVOIDANCE,
            value=risk_reward,
            weight=self.config.weights["risk_avoidance"],
            explanation=f"Risk delta: {outcome.get('risk_delta', 0):.3f}",
        ))

        # Information gain reward
        info_reward = self._compute_information_gain(state, next_state)
        signals.append(RewardSignal(
            component=RewardComponent.INFORMATION_GAIN,
            value=info_reward,
            weight=self.config.weights["information_gain"],
            explanation=f"Uncertainty reduction: {info_reward:.3f}",
        ))

        # Alignment reward
        alignment_reward = self._compute_alignment(action, outcome)
        signals.append(RewardSignal(
            component=RewardComponent.ALIGNMENT,
            value=alignment_reward,
            weight=self.config.weights["alignment"],
            explanation=f"Alignment score: {alignment_reward:.3f}",
        ))

        # Constraint violation penalties
        constraint_penalty = self._compute_constraint_penalty(action, outcome)
        if constraint_penalty < 0:
            signals.append(RewardSignal(
                component=RewardComponent.CONSTRAINT_PENALTY,
                value=constraint_penalty,
                weight=1.0,  # Penalties always at full weight
                explanation=f"Constraint violation detected",
            ))

        # Aggregate weighted reward
        total = sum(s.value * s.weight for s in signals)

        # Apply reward shaping if enabled
        if self.config.shaping_enabled:
            total += self._shape_reward(state, next_state)

        # Clamp to bounds
        total = max(self.config.reward_bounds[0], min(self.config.reward_bounds[1], total))

        # Record
        self._record_reward(state, action, total, signals)

        return total

    def compute_long_term_reward(
        self, reward_sequence: list[float]
    ) -> float:
        """
        Compute discounted cumulative reward over a sequence.

        Args:
            reward_sequence: List of immediate rewards in temporal order

        Returns:
            Discounted sum of rewards
        """
        gamma = self.config.discount_factor
        discounted = 0.0
        for t, r in enumerate(reward_sequence):
            discounted += (gamma ** t) * r
        return discounted

    # -------------------------------------------------------------------------
    # COMPONENT REWARD FUNCTIONS
    # -------------------------------------------------------------------------

    def _compute_task_completion(
        self,
        state: dict[str, Any],
        action: str,
        next_state: dict[str, Any],
        outcome: dict[str, Any],
    ) -> float:
        """Reward based on task progress and completion."""
        progress_before = state.get("task_progress", 0.0)
        progress_after = next_state.get("task_progress", 0.0)
        delta = progress_after - progress_before

        reward = delta * 100.0  # Scale progress to reward

        # Bonus for full completion
        if progress_after >= 1.0:
            reward += 20.0

        # Penalty for regression
        if delta < 0:
            reward *= 2.0  # Double the negative impact

        return reward

    def _compute_efficiency(
        self,
        state: dict[str, Any],
        action: str,
        outcome: dict[str, Any],
    ) -> float:
        """Reward for resource-efficient execution."""
        resource_cost = outcome.get("resource_cost", 0.0)
        resource_budget = state.get("resource_budget", 1.0)

        if resource_budget <= 0:
            return -10.0

        efficiency_ratio = 1.0 - (resource_cost / resource_budget)
        return max(-10.0, min(10.0, efficiency_ratio * 10.0))

    def _compute_risk_avoidance(
        self,
        state: dict[str, Any],
        next_state: dict[str, Any],
        outcome: dict[str, Any],
    ) -> float:
        """Reward for maintaining or reducing risk."""
        risk_before = state.get("risk_level", 0.0)
        risk_after = next_state.get("risk_level", 0.0)
        risk_delta = risk_after - risk_before

        # Reward for reducing risk, penalize for increasing
        reward = -risk_delta * 50.0

        # Extra penalty when risk exceeds threshold
        risk_threshold = state.get("risk_threshold", 0.7)
        if risk_after > risk_threshold:
            reward -= (risk_after - risk_threshold) * 100.0

        return max(-50.0, min(10.0, reward))

    def _compute_information_gain(
        self,
        state: dict[str, Any],
        next_state: dict[str, Any],
    ) -> float:
        """Reward for reducing uncertainty about the environment."""
        uncertainty_before = state.get("uncertainty", 1.0)
        uncertainty_after = next_state.get("uncertainty", 1.0)

        if uncertainty_before <= 0:
            return 0.0

        reduction = (uncertainty_before - uncertainty_after) / uncertainty_before
        return max(0.0, min(10.0, reduction * 10.0))

    def _compute_alignment(
        self,
        action: str,
        outcome: dict[str, Any],
    ) -> float:
        """Reward for adherence to the agent constitution."""
        alignment_score = outcome.get("alignment_score", 1.0)
        return (alignment_score - 0.5) * 20.0  # Scale [-10, 10]

    def _compute_constraint_penalty(
        self,
        action: str,
        outcome: dict[str, Any],
    ) -> float:
        """Hard penalties for constraint violations."""
        violations = outcome.get("constraint_violations", [])
        if not violations:
            return 0.0

        penalty = self.config.constraint_penalty * len(violations)
        logger.warning(
            f"Constraint violations detected: {violations}. Penalty: {penalty}"
        )
        return penalty

    # -------------------------------------------------------------------------
    # REWARD SHAPING
    # -------------------------------------------------------------------------

    def _shape_reward(
        self,
        state: dict[str, Any],
        next_state: dict[str, Any],
    ) -> float:
        """
        Potential-based reward shaping (preserves optimal policy).

        Uses the difference in potential function values:
            F(s, s') = gamma * Phi(s') - Phi(s)
        """
        phi_s = self._potential(state)
        phi_s_prime = self._potential(next_state)
        return self.config.discount_factor * phi_s_prime - phi_s

    def _potential(self, state: dict[str, Any]) -> float:
        """Potential function for reward shaping."""
        progress = state.get("task_progress", 0.0)
        risk = state.get("risk_level", 0.0)
        return progress * 10.0 - risk * 5.0

    # -------------------------------------------------------------------------
    # RECORDING
    # -------------------------------------------------------------------------

    def _record_reward(
        self,
        state: dict[str, Any],
        action: str,
        total_reward: float,
        signals: list[RewardSignal],
    ) -> None:
        """Record reward computation for audit."""
        self.step_count += 1
        self.cumulative_reward += total_reward

        self.reward_history.append({
            "step": self.step_count,
            "action": action,
            "total_reward": total_reward,
            "cumulative_reward": self.cumulative_reward,
            "components": {
                s.component.value: {"value": s.value, "weight": s.weight}
                for s in signals
            },
        })

    def get_reward_summary(self) -> dict[str, Any]:
        """Return summary statistics of reward history."""
        if not self.reward_history:
            return {"steps": 0, "cumulative": 0.0}

        rewards = [r["total_reward"] for r in self.reward_history]
        return {
            "steps": self.step_count,
            "cumulative": self.cumulative_reward,
            "mean": sum(rewards) / len(rewards),
            "min": min(rewards),
            "max": max(rewards),
        }


# =============================================================================
# RISK-ADJUSTED REWARD MODEL (CARBON[6] Spec §2.3)
# =============================================================================

class RiskAdjustedRewardModel(RewardModel):
    """
    Risk-adjusted reward model that incorporates utility and risk.

    Formal Definition:
        R_adjusted(s, a) = U(EMV(s, a)) - λ * Risk(s, a)

    Where:
        U()    = utility function (e.g., CRRA, exponential)
        λ      = risk aversion coefficient
        Risk() = risk measure (VaR, CVaR, variance)
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        utility_fn: Any = None,
        risk_model: Any = None,
        risk_aversion: float = 0.5,
    ):
        super().__init__(config)
        self.utility_fn = utility_fn
        self.risk_model = risk_model
        self.risk_aversion = risk_aversion

    def compute_risk_adjusted_reward(
        self,
        state: dict[str, Any],
        action: str,
        next_state: dict[str, Any],
        outcome: dict[str, Any],
    ) -> float:
        """
        Compute risk-adjusted reward: R_adjusted = U(EMV) - λ * Risk

        Args:
            state: State before the action
            action: Action taken
            next_state: Resulting state
            outcome: Observed outcome metadata

        Returns:
            Risk-adjusted scalar reward
        """
        # Base EMV reward
        emv_reward = self.compute_immediate_reward(state, action, next_state, outcome)

        # Apply utility transformation
        if self.utility_fn is not None:
            utility_reward = self.utility_fn.compute_utility(emv_reward)
        else:
            utility_reward = emv_reward

        # Apply risk penalty
        risk_penalty = 0.0
        if self.risk_model is not None:
            outcomes = outcome.get("outcome_distribution", [])
            if outcomes:
                metrics = self.risk_model.compute_risk_metrics(outcomes)
                risk_penalty = self.risk_aversion * abs(metrics.cvar)

        adjusted = utility_reward - risk_penalty

        # Clamp to bounds
        return max(
            self.config.reward_bounds[0],
            min(self.config.reward_bounds[1], adjusted),
        )


# =============================================================================
# MULTI-OBJECTIVE REWARD MODEL (CARBON[6] Spec §2.3)
# =============================================================================

@dataclass
class ObjectiveWeight:
    """A single objective with configurable weight range."""
    name: str
    weight: float
    weight_range: tuple[float, float]
    description: str = ""

    def validate(self) -> bool:
        lo, hi = self.weight_range
        return lo <= self.weight <= hi


class MultiObjectiveRewardModel:
    """
    Multi-objective reward with configurable weights per CARBON[6] spec.

    Formal Definition:
        R_total = Σ_i w_i * R_i(s, a)
        Subject to: Σ w_i = 1, w_i ≥ 0

    Default Objectives (CARBON[6] §2.3):
        | Objective           | Weight Range | Default |
        |---------------------|-------------|---------|
        | Task Completion     | 0.3 - 0.5  | 0.40    |
        | Resource Efficiency | 0.1 - 0.2  | 0.15    |
        | Safety Margin       | 0.2 - 0.3  | 0.25    |
        | User Satisfaction   | 0.1 - 0.2  | 0.20    |
    """

    DEFAULT_OBJECTIVES = [
        ObjectiveWeight("task_completion", 0.40, (0.3, 0.5), "Primary task success"),
        ObjectiveWeight("resource_efficiency", 0.15, (0.1, 0.2), "Minimize resource consumption"),
        ObjectiveWeight("safety_margin", 0.25, (0.2, 0.3), "Maintain safety buffer"),
        ObjectiveWeight("user_satisfaction", 0.20, (0.1, 0.2), "Inferred user preference alignment"),
    ]

    def __init__(self, objectives: list[ObjectiveWeight] | None = None):
        self.objectives = objectives or list(self.DEFAULT_OBJECTIVES)
        self._validate_weights()
        self.reward_functions: dict[str, Any] = {}

    def _validate_weights(self) -> None:
        total = sum(o.weight for o in self.objectives)
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Objective weights sum to {total:.3f}, normalizing to 1.0")
            for o in self.objectives:
                o.weight /= total

    def register_reward_function(self, objective_name: str, fn: Any) -> None:
        """Register a reward function for a specific objective."""
        self.reward_functions[objective_name] = fn

    def compute_multi_objective_reward(
        self,
        state: dict[str, Any],
        action: str,
        next_state: dict[str, Any],
        outcome: dict[str, Any],
    ) -> float:
        """
        Compute R_total = Σ_i w_i * R_i(s, a)

        Returns:
            Weighted sum of objective rewards
        """
        total = 0.0
        for obj in self.objectives:
            fn = self.reward_functions.get(obj.name)
            if fn is not None:
                r_i = fn(state, action, next_state, outcome)
            else:
                r_i = outcome.get(f"{obj.name}_reward", 0.0)
            total += obj.weight * r_i
        return total

    def set_weight(self, objective_name: str, weight: float) -> None:
        """Set the weight for an objective (must be within range)."""
        for obj in self.objectives:
            if obj.name == objective_name:
                lo, hi = obj.weight_range
                if weight < lo or weight > hi:
                    raise ValueError(
                        f"Weight {weight} for '{objective_name}' outside range [{lo}, {hi}]"
                    )
                obj.weight = weight
                self._validate_weights()
                return
        raise KeyError(f"Unknown objective: {objective_name}")
