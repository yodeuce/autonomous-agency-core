"""
REWARD MODEL MODULE
Encodes EMV (Expected Monetary Value) into machine-usable rewards
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

This is where EMV becomes executable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime


class RewardType(Enum):
    """Types of rewards."""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    SHAPED = "shaped"
    SPARSE = "sparse"


@dataclass
class RewardSignal:
    """A single reward signal."""
    value: float
    reward_type: RewardType
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    # Outcome rewards
    task_success: float = 100.0
    task_partial: float = 50.0
    task_failure: float = -50.0

    # Constraint penalties
    constraint_violation: float = -200.0
    hard_constraint_violation: float = -500.0

    # Escalation rewards
    escalation_appropriate: float = 20.0
    escalation_unnecessary: float = -10.0

    # Efficiency rewards
    time_bonus_multiplier: float = 1.1
    resource_efficiency_bonus: float = 15.0

    # Risk penalties
    risk_penalty_multiplier: float = 2.0

    # Shaping rewards
    confidence_increase_reward: float = 5.0
    memory_utilization_reward: float = 3.0
    progress_reward: float = 2.0


class BaseRewardModel(ABC):
    """Abstract base class for reward models."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.reward_history: List[RewardSignal] = []

    @abstractmethod
    def calculate_immediate_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> float:
        """Calculate immediate reward R(s, a, s')."""
        pass

    @abstractmethod
    def calculate_shaped_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> float:
        """Calculate reward shaping bonus."""
        pass

    def get_total_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> Tuple[float, RewardSignal]:
        """Get total reward including shaping."""
        immediate = self.calculate_immediate_reward(state, action, next_state)
        shaped = self.calculate_shaped_reward(state, action, next_state)

        total = immediate + shaped

        signal = RewardSignal(
            value=total,
            reward_type=RewardType.SHAPED,
            source="combined",
            metadata={
                "immediate": immediate,
                "shaped": shaped,
                "action": action
            }
        )

        self.reward_history.append(signal)
        return total, signal


class EMVRewardModel(BaseRewardModel):
    """
    Expected Monetary Value based reward model.
    Converts business outcomes to RL rewards.
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        outcome_values: Optional[Dict[str, float]] = None,
        outcome_probabilities: Optional[Dict[str, float]] = None
    ):
        super().__init__(config)

        # EMV components
        self.outcome_values = outcome_values or {
            "success": 1000.0,
            "partial_success": 500.0,
            "failure": -200.0,
            "critical_failure": -1000.0
        }

        self.outcome_probabilities = outcome_probabilities or {
            "success": 0.7,
            "partial_success": 0.2,
            "failure": 0.08,
            "critical_failure": 0.02
        }

    def calculate_emv(
        self,
        state: Dict[str, Any],
        action: str
    ) -> float:
        """
        Calculate Expected Monetary Value for state-action pair.
        EMV = Σ P(outcome) × V(outcome)
        """
        # Adjust probabilities based on state
        adjusted_probs = self._adjust_probabilities(state, action)

        emv = sum(
            adjusted_probs.get(outcome, 0) * value
            for outcome, value in self.outcome_values.items()
        )

        return emv

    def _adjust_probabilities(
        self,
        state: Dict[str, Any],
        action: str
    ) -> Dict[str, float]:
        """Adjust outcome probabilities based on state context."""
        probs = self.outcome_probabilities.copy()

        confidence = state.get("confidence_level", 0.5)
        risk = state.get("risk_exposure", 0.0)

        # Higher confidence increases success probability
        confidence_factor = confidence / 0.5  # Normalize around 0.5
        probs["success"] *= confidence_factor
        probs["partial_success"] *= (2 - confidence_factor)

        # Higher risk increases failure probability
        risk_factor = 1 + risk
        probs["failure"] *= risk_factor
        probs["critical_failure"] *= risk_factor

        # Normalize probabilities
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def calculate_immediate_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> float:
        """Calculate immediate reward based on state transition."""
        reward = 0.0

        # Task completion rewards
        if next_state.get("task_completed"):
            if next_state.get("task_success"):
                reward += self.config.task_success
            else:
                reward += self.config.task_partial

        if next_state.get("task_failed"):
            reward += self.config.task_failure

        # Constraint violation penalties
        violations = next_state.get("constraint_violations", 0)
        prev_violations = state.get("constraint_violations", 0)
        new_violations = violations - prev_violations

        if new_violations > 0:
            reward += self.config.constraint_violation * new_violations

        # Escalation rewards
        if action == "escalate":
            if state.get("risk_exposure", 0) > 0.7:
                reward += self.config.escalation_appropriate
            else:
                reward += self.config.escalation_unnecessary

        # Risk penalty
        risk = next_state.get("risk_exposure", 0)
        if risk > 0.5:
            reward -= (risk - 0.5) * self.config.risk_penalty_multiplier * 100

        return reward

    def calculate_shaped_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward shaping to guide learning.
        Uses potential-based shaping to preserve optimal policy.
        """
        shaped = 0.0

        # Confidence increase reward
        conf_delta = (
            next_state.get("confidence_level", 0) -
            state.get("confidence_level", 0)
        )
        if conf_delta > 0:
            shaped += conf_delta * self.config.confidence_increase_reward

        # Risk reduction reward
        risk_delta = (
            state.get("risk_exposure", 0) -
            next_state.get("risk_exposure", 0)
        )
        if risk_delta > 0:
            shaped += risk_delta * 10

        # Progress toward goal
        progress_delta = (
            next_state.get("progress", 0) -
            state.get("progress", 0)
        )
        if progress_delta > 0:
            shaped += progress_delta * self.config.progress_reward

        # Memory utilization
        if action == "retrieve_memory" and next_state.get("memory_hit"):
            shaped += self.config.memory_utilization_reward

        return shaped

    def calculate_discounted_return(
        self,
        rewards: List[float],
        gamma: float = 0.95
    ) -> float:
        """Calculate discounted cumulative return."""
        discounted = 0.0
        for t, r in enumerate(rewards):
            discounted += (gamma ** t) * r
        return discounted


class RiskAdjustedRewardModel(EMVRewardModel):
    """
    Risk-adjusted reward model that penalizes variance.
    Uses CVaR (Conditional Value at Risk) for tail risk.
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        risk_aversion: float = 0.5,
        cvar_alpha: float = 0.05
    ):
        super().__init__(config)
        self.risk_aversion = risk_aversion  # 0 = risk neutral, 1 = very risk averse
        self.cvar_alpha = cvar_alpha  # Bottom percentile for CVaR
        self.reward_samples: List[float] = []

    def calculate_risk_adjusted_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> float:
        """Calculate risk-adjusted reward using utility theory."""
        base_reward = self.calculate_immediate_reward(state, action, next_state)

        # Track for variance calculation
        self.reward_samples.append(base_reward)

        # Calculate variance penalty
        if len(self.reward_samples) > 10:
            variance = np.var(self.reward_samples[-100:])
            variance_penalty = self.risk_aversion * np.sqrt(variance)
            base_reward -= variance_penalty

        # Calculate CVaR penalty for tail risk
        if len(self.reward_samples) > 50:
            cvar = self._calculate_cvar()
            if cvar < 0:
                base_reward += cvar * self.risk_aversion * 0.1

        return base_reward

    def _calculate_cvar(self) -> float:
        """Calculate Conditional Value at Risk."""
        sorted_rewards = sorted(self.reward_samples)
        cutoff_idx = int(len(sorted_rewards) * self.cvar_alpha)
        if cutoff_idx == 0:
            cutoff_idx = 1

        tail_rewards = sorted_rewards[:cutoff_idx]
        return np.mean(tail_rewards)

    def get_total_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> Tuple[float, RewardSignal]:
        """Get risk-adjusted total reward."""
        risk_adjusted = self.calculate_risk_adjusted_reward(state, action, next_state)
        shaped = self.calculate_shaped_reward(state, action, next_state)

        total = risk_adjusted + shaped

        signal = RewardSignal(
            value=total,
            reward_type=RewardType.SHAPED,
            source="risk_adjusted",
            metadata={
                "risk_adjusted": risk_adjusted,
                "shaped": shaped,
                "action": action,
                "risk_aversion": self.risk_aversion
            }
        )

        self.reward_history.append(signal)
        return total, signal


class MultiObjectiveRewardModel(BaseRewardModel):
    """
    Multi-objective reward model for balancing competing goals.
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        objective_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(config)

        self.objective_weights = objective_weights or {
            "task_completion": 0.4,
            "efficiency": 0.2,
            "safety": 0.25,
            "user_satisfaction": 0.15
        }

        self.objective_rewards: Dict[str, List[float]] = {
            obj: [] for obj in self.objective_weights
        }

    def calculate_immediate_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> float:
        """Calculate weighted multi-objective reward."""
        objective_scores = {}

        # Task completion objective
        task_score = 0.0
        if next_state.get("task_completed"):
            task_score = 100.0 if next_state.get("task_success") else 50.0
        elif next_state.get("task_failed"):
            task_score = -50.0
        objective_scores["task_completion"] = task_score

        # Efficiency objective
        efficiency_score = 0.0
        steps = next_state.get("steps_taken", 1)
        if next_state.get("task_completed"):
            efficiency_score = max(0, 100 - steps * 5)
        objective_scores["efficiency"] = efficiency_score

        # Safety objective
        safety_score = 100.0
        violations = next_state.get("constraint_violations", 0)
        risk = next_state.get("risk_exposure", 0)
        safety_score -= violations * 50
        safety_score -= risk * 50
        objective_scores["safety"] = max(-100, safety_score)

        # User satisfaction objective
        satisfaction_score = next_state.get("user_satisfaction", 50)
        objective_scores["user_satisfaction"] = satisfaction_score

        # Store for tracking
        for obj, score in objective_scores.items():
            self.objective_rewards[obj].append(score)

        # Weighted combination
        total = sum(
            self.objective_weights.get(obj, 0) * score
            for obj, score in objective_scores.items()
        )

        return total

    def calculate_shaped_reward(
        self,
        state: Dict[str, Any],
        action: str,
        next_state: Dict[str, Any]
    ) -> float:
        """Minimal shaping for multi-objective."""
        return 0.0  # Keep objectives pure

    def get_pareto_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze Pareto efficiency across objectives."""
        analysis = {}
        for obj, rewards in self.objective_rewards.items():
            if rewards:
                analysis[obj] = {
                    "mean": np.mean(rewards),
                    "std": np.std(rewards),
                    "min": min(rewards),
                    "max": max(rewards)
                }
        return analysis


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_reward_model(
    model_type: str = "emv",
    config: Optional[RewardConfig] = None,
    **kwargs
) -> BaseRewardModel:
    """Create a reward model of the specified type."""

    if model_type == "emv":
        return EMVRewardModel(config, **kwargs)
    elif model_type == "risk_adjusted":
        return RiskAdjustedRewardModel(config, **kwargs)
    elif model_type == "multi_objective":
        return MultiObjectiveRewardModel(config, **kwargs)
    else:
        raise ValueError(f"Unknown reward model type: {model_type}")
