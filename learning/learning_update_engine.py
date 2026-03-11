"""
LEARNING UPDATE ENGINE
Updates models from outcomes and experience
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Includes:
- Policy updates
- Reward model updates
- Memory salience adjustments
- Environment model corrections

This is where experience compounds.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict


class UpdateType(Enum):
    """Types of learning updates."""
    POLICY = "policy"
    REWARD_MODEL = "reward_model"
    MEMORY_SALIENCE = "memory_salience"
    ENVIRONMENT_MODEL = "environment_model"
    VALUE_FUNCTION = "value_function"
    CALIBRATION = "calibration"


class LearningSignal(Enum):
    """Types of learning signals."""
    REWARD = "reward"
    PENALTY = "penalty"
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTION = "correction"
    FEEDBACK = "feedback"


@dataclass
class Experience:
    """A single experience tuple for learning."""
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningUpdate:
    """Record of a learning update."""
    update_type: UpdateType
    signal: LearningSignal
    magnitude: float
    parameters_updated: List[str]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningConfig:
    """Configuration for learning engine."""
    # Learning rates
    policy_learning_rate: float = 0.01
    reward_learning_rate: float = 0.001
    salience_learning_rate: float = 0.05
    environment_learning_rate: float = 0.01

    # Discount factor
    gamma: float = 0.95

    # Experience replay
    replay_buffer_size: int = 10000
    batch_size: int = 32
    min_experiences_before_learning: int = 100

    # Update frequency
    policy_update_frequency: int = 10
    target_update_frequency: int = 100

    # Regularization
    l2_regularization: float = 0.01
    gradient_clip: float = 1.0

    # Exploration
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01


class BaseLearningEngine(ABC):
    """Abstract base class for learning engines."""

    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self._experience_buffer: List[Experience] = []
        self._update_history: List[LearningUpdate] = []
        self._step_count = 0

    @abstractmethod
    def learn_from_experience(
        self,
        experience: Experience
    ) -> Optional[LearningUpdate]:
        """Learn from a single experience."""
        pass

    @abstractmethod
    def learn_from_batch(
        self,
        experiences: List[Experience]
    ) -> List[LearningUpdate]:
        """Learn from a batch of experiences."""
        pass

    def add_experience(self, experience: Experience):
        """Add experience to replay buffer."""
        self._experience_buffer.append(experience)
        if len(self._experience_buffer) > self.config.replay_buffer_size:
            self._experience_buffer = self._experience_buffer[-self.config.replay_buffer_size:]
        self._step_count += 1

    def sample_batch(self, size: Optional[int] = None) -> List[Experience]:
        """Sample a batch of experiences for learning."""
        size = size or self.config.batch_size
        if len(self._experience_buffer) < size:
            return self._experience_buffer.copy()
        indices = np.random.choice(len(self._experience_buffer), size, replace=False)
        return [self._experience_buffer[i] for i in indices]

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_experiences": len(self._experience_buffer),
            "total_updates": len(self._update_history),
            "step_count": self._step_count,
            "exploration_rate": self.config.exploration_rate
        }


class TDLearningEngine(BaseLearningEngine):
    """
    Temporal Difference Learning Engine.
    Updates value estimates based on TD error.
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        super().__init__(config)
        self._value_function: Dict[str, float] = defaultdict(float)
        self._action_values: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._visit_counts: Dict[str, int] = defaultdict(int)

    def learn_from_experience(
        self,
        experience: Experience
    ) -> Optional[LearningUpdate]:
        """Learn from single experience using TD(0)."""
        self.add_experience(experience)

        # State representation (simplified - use state hash)
        state_key = self._state_to_key(experience.state)
        next_state_key = self._state_to_key(experience.next_state)
        action = experience.action

        # Current value estimate
        old_q = self._action_values[state_key][action]

        # TD Target
        if experience.done:
            target = experience.reward
        else:
            next_value = max(
                self._action_values[next_state_key].values()
            ) if self._action_values[next_state_key] else 0
            target = experience.reward + self.config.gamma * next_value

        # TD Error
        td_error = target - old_q

        # Learning rate with visit count decay
        self._visit_counts[state_key] += 1
        alpha = self.config.policy_learning_rate / np.sqrt(
            self._visit_counts[state_key]
        )

        # Update Q-value
        new_q = old_q + alpha * td_error
        self._action_values[state_key][action] = new_q

        # Update state value
        self._value_function[state_key] = max(
            self._action_values[state_key].values()
        )

        update = LearningUpdate(
            update_type=UpdateType.VALUE_FUNCTION,
            signal=LearningSignal.REWARD if td_error > 0 else LearningSignal.PENALTY,
            magnitude=abs(td_error),
            parameters_updated=[f"Q({state_key}, {action})"],
            old_values={"q_value": old_q},
            new_values={"q_value": new_q, "td_error": td_error}
        )

        self._update_history.append(update)
        return update

    def learn_from_batch(
        self,
        experiences: List[Experience]
    ) -> List[LearningUpdate]:
        """Learn from batch of experiences."""
        updates = []
        for exp in experiences:
            update = self.learn_from_experience(exp)
            if update:
                updates.append(update)
        return updates

    def get_action_value(self, state: Dict[str, Any], action: str) -> float:
        """Get Q-value for state-action pair."""
        state_key = self._state_to_key(state)
        return self._action_values[state_key][action]

    def get_state_value(self, state: Dict[str, Any]) -> float:
        """Get value of a state."""
        state_key = self._state_to_key(state)
        return self._value_function[state_key]

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dict to hashable key."""
        # Simplified state representation
        key_vars = ["task_context", "confidence_level", "risk_exposure"]
        key_parts = []
        for var in key_vars:
            val = state.get(var, "")
            if isinstance(val, float):
                val = round(val, 1)
            key_parts.append(f"{var}:{val}")
        return "|".join(key_parts)


class PolicyGradientEngine(BaseLearningEngine):
    """
    Policy Gradient Learning Engine.
    Directly optimizes policy parameters.
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        action_space: Optional[List[str]] = None
    ):
        super().__init__(config)
        self.action_space = action_space or []
        self._policy_params: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.action_space}
        )
        self._baseline: Dict[str, float] = defaultdict(float)
        self._episode_buffer: List[Experience] = []

    def learn_from_experience(
        self,
        experience: Experience
    ) -> Optional[LearningUpdate]:
        """Collect experience for end-of-episode update."""
        self.add_experience(experience)
        self._episode_buffer.append(experience)

        if experience.done:
            # End of episode - compute returns and update
            return self._update_policy()
        return None

    def learn_from_batch(
        self,
        experiences: List[Experience]
    ) -> List[LearningUpdate]:
        """Learn from batch of complete episodes."""
        updates = []
        for exp in experiences:
            update = self.learn_from_experience(exp)
            if update:
                updates.append(update)
        return updates

    def _update_policy(self) -> LearningUpdate:
        """Update policy at end of episode using REINFORCE."""
        if not self._episode_buffer:
            return None

        # Compute returns
        returns = []
        G = 0
        for exp in reversed(self._episode_buffer):
            G = exp.reward + self.config.gamma * G
            returns.insert(0, G)

        # Update for each step
        updated_params = []
        for exp, G in zip(self._episode_buffer, returns):
            state_key = self._state_to_key(exp.state)

            # Baseline (average return for this state)
            old_baseline = self._baseline[state_key]
            self._baseline[state_key] = old_baseline * 0.9 + G * 0.1

            # Advantage
            advantage = G - self._baseline[state_key]

            # Policy gradient update
            action_probs = self._softmax(self._policy_params[state_key])
            for action in self.action_space:
                if action == exp.action:
                    gradient = (1 - action_probs[action]) * advantage
                else:
                    gradient = -action_probs[action] * advantage

                # Apply gradient with clipping
                gradient = np.clip(
                    gradient,
                    -self.config.gradient_clip,
                    self.config.gradient_clip
                )

                self._policy_params[state_key][action] += (
                    self.config.policy_learning_rate * gradient
                )

            updated_params.append(state_key)

        # Clear episode buffer
        self._episode_buffer = []

        update = LearningUpdate(
            update_type=UpdateType.POLICY,
            signal=LearningSignal.REWARD if G > 0 else LearningSignal.PENALTY,
            magnitude=abs(G),
            parameters_updated=updated_params,
            old_values={},
            new_values={"episode_return": G}
        )

        self._update_history.append(update)
        return update

    def get_action_probabilities(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get action probabilities for a state."""
        state_key = self._state_to_key(state)
        return self._softmax(self._policy_params[state_key])

    def _softmax(self, values: Dict[str, float]) -> Dict[str, float]:
        """Compute softmax probabilities."""
        max_val = max(values.values())
        exp_vals = {k: np.exp(v - max_val) for k, v in values.items()}
        total = sum(exp_vals.values())
        return {k: v / total for k, v in exp_vals.items()}

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state to key."""
        key_vars = ["task_context", "confidence_level"]
        key_parts = []
        for var in key_vars:
            val = state.get(var, "")
            if isinstance(val, float):
                val = round(val, 2)
            key_parts.append(str(val))
        return "|".join(key_parts)


class SalienceUpdateEngine(BaseLearningEngine):
    """
    Engine for updating memory salience based on outcomes.
    Reinforces useful memories, decays irrelevant ones.
    """

    def __init__(self, config: Optional[LearningConfig] = None):
        super().__init__(config)
        self._memory_outcomes: Dict[str, List[float]] = defaultdict(list)
        self._salience_adjustments: Dict[str, float] = defaultdict(float)

    def learn_from_experience(
        self,
        experience: Experience
    ) -> Optional[LearningUpdate]:
        """Update memory salience based on experience outcome."""
        self.add_experience(experience)

        # Get memories used in this decision
        memories_used = experience.metadata.get("memories_retrieved", [])
        if not memories_used:
            return None

        # Record outcome for each memory
        for memory_id in memories_used:
            self._memory_outcomes[memory_id].append(experience.reward)

            # Keep limited history
            if len(self._memory_outcomes[memory_id]) > 100:
                self._memory_outcomes[memory_id] = \
                    self._memory_outcomes[memory_id][-100:]

        # Update salience adjustments
        updated = []
        for memory_id in memories_used:
            outcomes = self._memory_outcomes[memory_id]
            avg_outcome = np.mean(outcomes)

            # Positive outcomes reinforce, negative outcomes decay
            adjustment = avg_outcome * self.config.salience_learning_rate
            self._salience_adjustments[memory_id] += adjustment
            updated.append(memory_id)

        update = LearningUpdate(
            update_type=UpdateType.MEMORY_SALIENCE,
            signal=LearningSignal.REWARD if experience.reward > 0 else LearningSignal.PENALTY,
            magnitude=abs(experience.reward),
            parameters_updated=updated,
            old_values={},
            new_values={"memories_updated": len(updated)}
        )

        self._update_history.append(update)
        return update

    def learn_from_batch(
        self,
        experiences: List[Experience]
    ) -> List[LearningUpdate]:
        """Learn from batch."""
        updates = []
        for exp in experiences:
            update = self.learn_from_experience(exp)
            if update:
                updates.append(update)
        return updates

    def get_salience_adjustment(self, memory_id: str) -> float:
        """Get accumulated salience adjustment for a memory."""
        return self._salience_adjustments.get(memory_id, 0.0)


class CompositeLearningEngine(BaseLearningEngine):
    """
    Composite learning engine that coordinates multiple learning systems.
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        action_space: Optional[List[str]] = None
    ):
        super().__init__(config)

        self.td_engine = TDLearningEngine(config)
        self.policy_engine = PolicyGradientEngine(config, action_space)
        self.salience_engine = SalienceUpdateEngine(config)

    def learn_from_experience(
        self,
        experience: Experience
    ) -> Optional[LearningUpdate]:
        """Learn using all sub-engines."""
        self.add_experience(experience)

        updates = []

        # TD Learning for value function
        td_update = self.td_engine.learn_from_experience(experience)
        if td_update:
            updates.append(td_update)

        # Policy gradient (collects for episode end)
        policy_update = self.policy_engine.learn_from_experience(experience)
        if policy_update:
            updates.append(policy_update)

        # Salience updates
        salience_update = self.salience_engine.learn_from_experience(experience)
        if salience_update:
            updates.append(salience_update)

        # Decay exploration
        self.config.exploration_rate = max(
            self.config.min_exploration,
            self.config.exploration_rate * self.config.exploration_decay
        )

        return updates[0] if updates else None

    def learn_from_batch(
        self,
        experiences: List[Experience]
    ) -> List[LearningUpdate]:
        """Learn from batch using all engines."""
        all_updates = []
        all_updates.extend(self.td_engine.learn_from_batch(experiences))
        all_updates.extend(self.policy_engine.learn_from_batch(experiences))
        all_updates.extend(self.salience_engine.learn_from_batch(experiences))
        return all_updates


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_learning_engine(
    engine_type: str = "composite",
    config: Optional[LearningConfig] = None,
    action_space: Optional[List[str]] = None
) -> BaseLearningEngine:
    """Create a learning engine of the specified type."""

    if engine_type == "td":
        return TDLearningEngine(config)
    elif engine_type == "policy_gradient":
        return PolicyGradientEngine(config, action_space)
    elif engine_type == "salience":
        return SalienceUpdateEngine(config)
    elif engine_type == "composite":
        return CompositeLearningEngine(config, action_space)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
