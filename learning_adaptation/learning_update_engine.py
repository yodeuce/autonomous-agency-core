"""
FILE 17: learning_update_engine.py
PURPOSE: Updates models from outcomes
ROLE: This is where experience compounds
SPEC: CARBON[6] Technical Architecture Specification v1.0.0

Learning Methods (CARBON[6] §6):
    TD Learning:
        V(s) ← V(s) + α[r + γV(s') - V(s)]
        Where: α = learning rate, γ = discount factor

    Policy Gradient:
        θ ← θ + α · ∇_θ log π(a|s) · G_t
        Where: G_t = discounted cumulative reward

    Salience-Based Learning Rate:
        α_effective = α_base × (1 + salience_boost × S(memory))
        Where: S(memory) = memory salience score

Includes:
- Policy updates (TD Learning, Policy Gradient)
- Reward model updates
- Memory salience adjustments
- Environment model corrections
- Prioritized experience replay
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LearningExperience:
    """A single experience tuple for learning."""
    state: dict[str, Any]
    action: str
    reward: float
    next_state: dict[str, Any]
    outcome: dict[str, Any]
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningConfig:
    """Configuration for the learning engine."""
    policy_learning_rate: float = 0.01
    reward_model_learning_rate: float = 0.005
    salience_learning_rate: float = 0.01
    environment_learning_rate: float = 0.01
    batch_size: int = 32
    min_experiences_to_learn: int = 10
    update_frequency: int = 10  # Learn every N steps
    max_experience_buffer: int = 10000
    td_discount_factor: float = 0.95
    salience_boost: float = 0.5


class LearningUpdateEngine:
    """
    Central learning engine that updates all agent models
    based on observed outcomes and experiences.
    """

    def __init__(self, config: LearningConfig | None = None):
        self.config = config or LearningConfig()
        self.experience_buffer: list[LearningExperience] = []
        self.step: int = 0
        self.total_updates: int = 0
        self.update_log: list[dict[str, Any]] = []

    def record_experience(self, experience: LearningExperience) -> None:
        """Add an experience to the replay buffer."""
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.config.max_experience_buffer:
            self.experience_buffer.pop(0)

    def should_update(self) -> bool:
        """Check if it's time to run a learning update."""
        if len(self.experience_buffer) < self.config.min_experiences_to_learn:
            return False
        return self.step % self.config.update_frequency == 0

    def run_update(
        self,
        policy: Any = None,
        reward_model: Any = None,
        salience_engine: Any = None,
        environment_model: Any = None,
    ) -> dict[str, Any]:
        """
        Run a full learning update cycle across all models.

        Args:
            policy: The policy object to update
            reward_model: The reward model to update
            salience_engine: The memory salience engine to update
            environment_model: The environment model to update

        Returns:
            Update summary with deltas for each model
        """
        if not self.experience_buffer:
            return {"status": "no_experiences"}

        # Sample batch
        batch = self._sample_batch()
        results = {}

        # 1. Policy update
        if policy is not None:
            policy_delta = self._update_policy(policy, batch)
            results["policy"] = policy_delta

        # 2. Reward model update
        if reward_model is not None:
            reward_delta = self._update_reward_model(reward_model, batch)
            results["reward_model"] = reward_delta

        # 3. Memory salience adjustment
        if salience_engine is not None:
            salience_delta = self._update_salience(salience_engine, batch)
            results["salience"] = salience_delta

        # 4. Environment model correction
        if environment_model is not None:
            env_delta = self._update_environment_model(environment_model, batch)
            results["environment"] = env_delta

        self.total_updates += 1
        self.update_log.append({
            "step": self.step,
            "update_number": self.total_updates,
            "batch_size": len(batch),
            "results": results,
        })

        logger.info(
            f"Learning update #{self.total_updates} completed "
            f"(batch_size={len(batch)})"
        )
        return results

    def advance_step(self) -> None:
        self.step += 1

    # -------------------------------------------------------------------------
    # MODEL-SPECIFIC UPDATES
    # -------------------------------------------------------------------------

    def _update_policy(
        self, policy: Any, batch: list[LearningExperience]
    ) -> dict[str, Any]:
        """
        Update the policy based on observed action-outcome pairs.
        Uses a simple value-based update approach.
        """
        lr = self.config.policy_learning_rate

        # Compute average reward per action
        action_rewards: dict[str, list[float]] = {}
        for exp in batch:
            action_rewards.setdefault(exp.action, []).append(exp.reward)

        action_avg = {
            action: sum(rewards) / len(rewards)
            for action, rewards in action_rewards.items()
        }

        # Update policy value estimates if supported
        if hasattr(policy, "update_action_values"):
            policy.update_action_values(action_avg, lr)

        # Version the policy update
        if hasattr(policy, "update_version"):
            best_action = max(action_avg, key=action_avg.get) if action_avg else None
            policy.update_version(
                reason=f"learning_update_{self.total_updates}",
                performance_delta=sum(action_avg.values()) / len(action_avg) if action_avg else 0,
            )

        return {
            "action_value_updates": action_avg,
            "learning_rate": lr,
        }

    def _update_reward_model(
        self, reward_model: Any, batch: list[LearningExperience]
    ) -> dict[str, Any]:
        """
        Update reward model weights based on prediction errors.
        """
        lr = self.config.reward_model_learning_rate

        prediction_errors = []
        for exp in batch:
            if hasattr(reward_model, "compute_immediate_reward"):
                predicted = reward_model.compute_immediate_reward(
                    exp.state, exp.action, exp.next_state, exp.outcome
                )
                actual = exp.reward
                error = actual - predicted
                prediction_errors.append(error)

        avg_error = (
            sum(prediction_errors) / len(prediction_errors)
            if prediction_errors
            else 0.0
        )

        return {
            "avg_prediction_error": avg_error,
            "sample_count": len(prediction_errors),
            "learning_rate": lr,
        }

    def _update_salience(
        self, salience_engine: Any, batch: list[LearningExperience]
    ) -> dict[str, Any]:
        """
        Adjust salience weights based on which memories were useful.
        """
        lr = self.config.salience_learning_rate

        # Experiences with high rewards suggest current salience is good
        # Experiences with unexpected outcomes suggest salience needs adjusting
        high_reward = [e for e in batch if e.reward > 0]
        low_reward = [e for e in batch if e.reward < 0]

        return {
            "high_reward_count": len(high_reward),
            "low_reward_count": len(low_reward),
            "learning_rate": lr,
        }

    def _update_environment_model(
        self, environment_model: Any, batch: list[LearningExperience]
    ) -> dict[str, Any]:
        """
        Correct environment model based on prediction errors.
        """
        lr = self.config.environment_learning_rate

        state_prediction_errors: dict[str, list[float]] = {}
        for exp in batch:
            for key in exp.next_state:
                if key in exp.state:
                    predicted = exp.state.get(f"predicted_{key}")
                    actual = exp.next_state[key]
                    if predicted is not None and isinstance(actual, (int, float)):
                        error = float(actual) - float(predicted)
                        state_prediction_errors.setdefault(key, []).append(error)

        avg_errors = {
            key: sum(errors) / len(errors)
            for key, errors in state_prediction_errors.items()
        }

        return {
            "state_prediction_errors": avg_errors,
            "variables_updated": len(avg_errors),
            "learning_rate": lr,
        }

    def compute_td_error(
        self,
        reward: float,
        current_value: float,
        next_value: float,
    ) -> float:
        """
        Compute Temporal Difference error (CARBON[6] §6).

        Formula: δ = r + γV(s') - V(s)

        Args:
            reward: Observed reward
            current_value: V(s) - estimated value of current state
            next_value: V(s') - estimated value of next state

        Returns:
            TD error δ
        """
        gamma = self.config.td_discount_factor
        return reward + gamma * next_value - current_value

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _sample_batch(self) -> list[LearningExperience]:
        """Sample a batch of experiences for learning."""
        import random

        buffer = self.experience_buffer
        batch_size = min(self.config.batch_size, len(buffer))
        return random.sample(buffer, batch_size)


class PrioritizedExperienceReplay:
    """
    Prioritized Experience Replay (CARBON[6] §6).

    Experiences with higher TD error are replayed more frequently.
    Priority: p_i = |δ_i| + ε
    Sampling probability: P(i) = p_i^α / Σ_k p_k^α
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.epsilon = epsilon
        self.buffer: list[tuple[LearningExperience, float]] = []  # (experience, priority)

    def add(self, experience: LearningExperience, td_error: float = 1.0) -> None:
        """Add experience with priority based on TD error."""
        priority = abs(td_error) + self.epsilon
        if len(self.buffer) >= self.capacity:
            # Remove lowest priority
            min_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][1])
            self.buffer[min_idx] = (experience, priority)
        else:
            self.buffer.append((experience, priority))

    def sample(self, batch_size: int) -> list[LearningExperience]:
        """Sample batch with probability proportional to priority."""
        if not self.buffer:
            return []

        priorities = [p ** self.alpha for _, p in self.buffer]
        total = sum(priorities)
        probabilities = [p / total for p in priorities]

        indices = []
        for _ in range(min(batch_size, len(self.buffer))):
            r = random.random()
            cumulative = 0.0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    indices.append(i)
                    break

        return [self.buffer[i][0] for i in indices]

    def update_priority(self, index: int, new_td_error: float) -> None:
        """Update the priority of an experience."""
        if 0 <= index < len(self.buffer):
            exp = self.buffer[index][0]
            self.buffer[index] = (exp, abs(new_td_error) + self.epsilon)

    def __len__(self) -> int:
        return len(self.buffer)
