"""
FILE 2: policy_definition.py
PURPOSE: Defines how actions are selected
ROLE: Without this, the agent has no coherent behavior

Supports:
- Deterministic or stochastic policy
- Exploration vs exploitation rules
- Constraints on actions
- Policy versioning
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB = "upper_confidence_bound"


class ExplorationStrategy(Enum):
    NONE = "none"
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class PolicyVersion:
    """Tracks policy versions for auditability."""
    version_id: str
    timestamp: str
    parent_version: str | None
    update_reason: str
    performance_delta: float | None = None
    checksum: str = ""

    def compute_checksum(self, policy_params: dict) -> str:
        raw = json.dumps(policy_params, sort_keys=True)
        self.checksum = hashlib.sha256(raw.encode()).hexdigest()
        return self.checksum


@dataclass
class ActionConstraint:
    """Defines constraints on action selection."""
    forbidden_actions: list[str] = field(default_factory=list)
    required_preconditions: dict[str, list[str]] = field(default_factory=dict)
    max_consecutive_same_action: int = 5
    cooldown_periods: dict[str, int] = field(default_factory=dict)


class BasePolicy(ABC):
    """Abstract base class for all policies."""

    def __init__(
        self,
        policy_type: PolicyType,
        action_space: list[str],
        constraints: ActionConstraint | None = None,
    ):
        self.policy_type = policy_type
        self.action_space = action_space
        self.constraints = constraints or ActionConstraint()
        self.version_history: list[PolicyVersion] = []
        self.current_version: str = "1.0.0"
        self.action_history: list[str] = []
        self._initialize_version()

    def _initialize_version(self) -> None:
        version = PolicyVersion(
            version_id=self.current_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parent_version=None,
            update_reason="initial_policy",
        )
        self.version_history.append(version)

    @abstractmethod
    def select_action(self, state: dict[str, Any]) -> str:
        """Select an action given the current state."""

    @abstractmethod
    def get_action_probabilities(self, state: dict[str, Any]) -> dict[str, float]:
        """Return probability distribution over actions."""

    def is_action_allowed(self, action: str, state: dict[str, Any]) -> bool:
        """Check if an action is permitted given constraints."""
        if action in self.constraints.forbidden_actions:
            logger.warning(f"Action '{action}' is forbidden by policy constraints.")
            return False

        if action in self.constraints.required_preconditions:
            for precondition in self.constraints.required_preconditions[action]:
                if not state.get(precondition, False):
                    logger.warning(
                        f"Precondition '{precondition}' not met for action '{action}'."
                    )
                    return False

        if action in self.constraints.cooldown_periods:
            cooldown = self.constraints.cooldown_periods[action]
            recent = self.action_history[-cooldown:]
            if action in recent:
                logger.warning(f"Action '{action}' is in cooldown period.")
                return False

        if len(self.action_history) >= self.constraints.max_consecutive_same_action:
            recent = self.action_history[
                -self.constraints.max_consecutive_same_action :
            ]
            if all(a == action for a in recent):
                logger.warning(
                    f"Action '{action}' exceeded max consecutive limit."
                )
                return False

        return True

    def get_allowed_actions(self, state: dict[str, Any]) -> list[str]:
        """Return list of currently allowed actions."""
        return [a for a in self.action_space if self.is_action_allowed(a, state)]

    def record_action(self, action: str) -> None:
        """Record an action for constraint tracking."""
        self.action_history.append(action)

    def update_version(self, reason: str, performance_delta: float | None = None) -> str:
        """Create a new policy version."""
        parts = self.current_version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        new_version = ".".join(parts)

        version = PolicyVersion(
            version_id=new_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parent_version=self.current_version,
            update_reason=reason,
            performance_delta=performance_delta,
        )
        self.version_history.append(version)
        self.current_version = new_version
        logger.info(f"Policy updated to version {new_version}: {reason}")
        return new_version


class DeterministicPolicy(BasePolicy):
    """Selects the highest-value action deterministically."""

    def __init__(
        self,
        action_space: list[str],
        value_function: Any = None,
        constraints: ActionConstraint | None = None,
    ):
        super().__init__(PolicyType.DETERMINISTIC, action_space, constraints)
        self.value_function = value_function

    def select_action(self, state: dict[str, Any]) -> str:
        allowed = self.get_allowed_actions(state)
        if not allowed:
            logger.error("No allowed actions available. Escalating.")
            return "escalate"

        if self.value_function is not None:
            values = {a: self.value_function(state, a) for a in allowed}
            action = max(values, key=values.get)
        else:
            action = allowed[0]

        self.record_action(action)
        return action

    def get_action_probabilities(self, state: dict[str, Any]) -> dict[str, float]:
        action = self.select_action(state)
        return {a: (1.0 if a == action else 0.0) for a in self.action_space}


class EpsilonGreedyPolicy(BasePolicy):
    """Epsilon-greedy exploration/exploitation policy."""

    def __init__(
        self,
        action_space: list[str],
        value_function: Any = None,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        constraints: ActionConstraint | None = None,
    ):
        super().__init__(PolicyType.EPSILON_GREEDY, action_space, constraints)
        self.value_function = value_function
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, state: dict[str, Any]) -> str:
        import random

        allowed = self.get_allowed_actions(state)
        if not allowed:
            return "escalate"

        if random.random() < self.epsilon:
            action = random.choice(allowed)
        elif self.value_function is not None:
            values = {a: self.value_function(state, a) for a in allowed}
            action = max(values, key=values.get)
        else:
            action = allowed[0]

        self.record_action(action)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def get_action_probabilities(self, state: dict[str, Any]) -> dict[str, float]:
        allowed = self.get_allowed_actions(state)
        n = len(allowed)
        if n == 0:
            return {a: 0.0 for a in self.action_space}

        probs = {a: 0.0 for a in self.action_space}
        explore_prob = self.epsilon / n

        if self.value_function is not None:
            values = {a: self.value_function({}, a) for a in allowed}
            best = max(values, key=values.get)
            for a in allowed:
                probs[a] = explore_prob
            probs[best] += 1.0 - self.epsilon
        else:
            for a in allowed:
                probs[a] = 1.0 / n

        return probs


class BoltzmannPolicy(BasePolicy):
    """Softmax/Boltzmann exploration policy."""

    def __init__(
        self,
        action_space: list[str],
        value_function: Any = None,
        temperature: float = 1.0,
        temperature_decay: float = 0.99,
        temperature_min: float = 0.1,
        constraints: ActionConstraint | None = None,
    ):
        super().__init__(PolicyType.BOLTZMANN, action_space, constraints)
        self.value_function = value_function
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min

    def select_action(self, state: dict[str, Any]) -> str:
        import random
        import math

        probs = self.get_action_probabilities(state)
        allowed = [a for a in probs if probs[a] > 0]
        if not allowed:
            return "escalate"

        weights = [probs[a] for a in allowed]
        action = random.choices(allowed, weights=weights, k=1)[0]
        self.record_action(action)
        self.temperature = max(
            self.temperature_min, self.temperature * self.temperature_decay
        )
        return action

    def get_action_probabilities(self, state: dict[str, Any]) -> dict[str, float]:
        import math

        allowed = self.get_allowed_actions(state)
        if not allowed:
            return {a: 0.0 for a in self.action_space}

        if self.value_function is not None:
            values = {a: self.value_function(state, a) for a in allowed}
        else:
            values = {a: 0.0 for a in allowed}

        max_val = max(values.values())
        exp_values = {
            a: math.exp((v - max_val) / self.temperature)
            for a, v in values.items()
        }
        total = sum(exp_values.values())

        probs = {a: 0.0 for a in self.action_space}
        for a in allowed:
            probs[a] = exp_values[a] / total
        return probs


# =============================================================================
# POLICY FACTORY
# =============================================================================

def create_policy(
    policy_type: str,
    action_space: list[str],
    value_function: Any = None,
    constraints: ActionConstraint | None = None,
    **kwargs,
) -> BasePolicy:
    """Factory function to create policies by type."""
    policies = {
        "deterministic": DeterministicPolicy,
        "epsilon_greedy": EpsilonGreedyPolicy,
        "boltzmann": BoltzmannPolicy,
    }

    if policy_type not in policies:
        raise ValueError(
            f"Unknown policy type '{policy_type}'. Available: {list(policies.keys())}"
        )

    return policies[policy_type](
        action_space=action_space,
        value_function=value_function,
        constraints=constraints,
        **kwargs,
    )
