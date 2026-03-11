"""
POLICY DEFINITION MODULE
Defines how actions are selected given states
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Without this, the agent has no coherent behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime
import json


class PolicyType(Enum):
    """Types of policies supported."""
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"
    UCB = "upper_confidence_bound"


class ExplorationStrategy(Enum):
    """Exploration vs exploitation strategies."""
    PURE_EXPLOITATION = "pure_exploitation"
    PURE_EXPLORATION = "pure_exploration"
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB1 = "ucb1"
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class PolicyConstraint:
    """Defines a constraint on policy actions."""
    name: str
    condition: Callable[[Dict], bool]
    forbidden_actions: List[str]
    priority: int = 0  # Higher = more important
    is_hard: bool = True  # Hard constraints cannot be violated


@dataclass
class PolicyVersion:
    """Tracks policy versioning."""
    version: str
    created_at: datetime
    description: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class BasePolicy(ABC):
    """Abstract base class for all policies."""

    def __init__(
        self,
        policy_id: str,
        policy_type: PolicyType,
        action_space: List[str],
        version: str = "1.0.0"
    ):
        self.policy_id = policy_id
        self.policy_type = policy_type
        self.action_space = action_space
        self.constraints: List[PolicyConstraint] = []
        self.version_history: List[PolicyVersion] = []
        self._current_version = version
        self._initialize_version(version)

    def _initialize_version(self, version: str):
        """Initialize version tracking."""
        self.version_history.append(PolicyVersion(
            version=version,
            created_at=datetime.now(),
            description="Initial policy",
            parameters=self.get_parameters()
        ))

    @abstractmethod
    def select_action(self, state: Dict[str, Any]) -> str:
        """Select an action given the current state."""
        pass

    @abstractmethod
    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Get probability distribution over actions."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current policy parameters."""
        pass

    def add_constraint(self, constraint: PolicyConstraint):
        """Add a constraint to the policy."""
        self.constraints.append(constraint)
        self.constraints.sort(key=lambda c: c.priority, reverse=True)

    def get_allowed_actions(self, state: Dict[str, Any]) -> List[str]:
        """Get actions allowed given current constraints."""
        allowed = set(self.action_space)

        for constraint in self.constraints:
            if constraint.condition(state):
                for action in constraint.forbidden_actions:
                    if constraint.is_hard:
                        allowed.discard(action)

        return list(allowed)

    def update_version(self, new_version: str, description: str):
        """Update policy version."""
        self._current_version = new_version
        self.version_history.append(PolicyVersion(
            version=new_version,
            created_at=datetime.now(),
            description=description,
            parameters=self.get_parameters()
        ))


class DeterministicPolicy(BasePolicy):
    """
    Deterministic policy: π(s) -> a
    Always selects the same action for a given state.
    """

    def __init__(
        self,
        policy_id: str,
        action_space: List[str],
        state_action_mapping: Dict[str, str],
        default_action: str
    ):
        super().__init__(policy_id, PolicyType.DETERMINISTIC, action_space)
        self.state_action_mapping = state_action_mapping
        self.default_action = default_action

    def select_action(self, state: Dict[str, Any]) -> str:
        """Select action deterministically based on state."""
        allowed_actions = self.get_allowed_actions(state)

        # Create state key for lookup
        state_key = self._state_to_key(state)

        # Get mapped action or default
        action = self.state_action_mapping.get(state_key, self.default_action)

        # Ensure action is allowed
        if action not in allowed_actions:
            # Fall back to first allowed action
            return allowed_actions[0] if allowed_actions else self.default_action

        return action

    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Deterministic policy has probability 1 for selected action."""
        action = self.select_action(state)
        return {a: 1.0 if a == action else 0.0 for a in self.action_space}

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "deterministic",
            "mapping_size": len(self.state_action_mapping),
            "default_action": self.default_action
        }

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dict to hashable key."""
        return json.dumps(state, sort_keys=True)


class StochasticPolicy(BasePolicy):
    """
    Stochastic policy: π(a|s) -> [0,1]
    Returns probability distribution over actions.
    """

    def __init__(
        self,
        policy_id: str,
        action_space: List[str],
        q_function: Optional[Callable[[Dict, str], float]] = None,
        temperature: float = 1.0
    ):
        super().__init__(policy_id, PolicyType.STOCHASTIC, action_space)
        self.q_function = q_function or self._default_q_function
        self.temperature = temperature

    def _default_q_function(self, state: Dict, action: str) -> float:
        """Default Q-function returns uniform values."""
        return 1.0 / len(self.action_space)

    def select_action(self, state: Dict[str, Any]) -> str:
        """Sample action from probability distribution."""
        probs = self.get_action_probabilities(state)
        actions = list(probs.keys())
        probabilities = list(probs.values())

        return np.random.choice(actions, p=probabilities)

    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Get softmax probabilities over actions."""
        allowed_actions = self.get_allowed_actions(state)

        # Compute Q-values for allowed actions
        q_values = {a: self.q_function(state, a) for a in allowed_actions}

        # Apply softmax with temperature
        max_q = max(q_values.values())
        exp_values = {a: np.exp((q - max_q) / self.temperature)
                      for a, q in q_values.items()}
        total = sum(exp_values.values())

        # Normalize
        probs = {a: exp_values.get(a, 0) / total for a in self.action_space}

        # Zero out forbidden actions
        for a in self.action_space:
            if a not in allowed_actions:
                probs[a] = 0.0

        return probs

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "stochastic",
            "temperature": self.temperature
        }


class EpsilonGreedyPolicy(BasePolicy):
    """
    Epsilon-greedy policy for exploration/exploitation balance.
    With probability epsilon, explore randomly.
    Otherwise, exploit best known action.
    """

    def __init__(
        self,
        policy_id: str,
        action_space: List[str],
        q_function: Callable[[Dict, str], float],
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        super().__init__(policy_id, PolicyType.EPSILON_GREEDY, action_space)
        self.q_function = q_function
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self._step_count = 0

    def select_action(self, state: Dict[str, Any]) -> str:
        """Select action using epsilon-greedy strategy."""
        allowed_actions = self.get_allowed_actions(state)

        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(allowed_actions)
        else:
            # Exploit: best Q-value action
            q_values = {a: self.q_function(state, a) for a in allowed_actions}
            return max(q_values, key=q_values.get)

    def get_action_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Get epsilon-greedy probability distribution."""
        allowed_actions = self.get_allowed_actions(state)
        n_allowed = len(allowed_actions)

        # Find greedy action
        q_values = {a: self.q_function(state, a) for a in allowed_actions}
        greedy_action = max(q_values, key=q_values.get)

        # Compute probabilities
        probs = {}
        for a in self.action_space:
            if a not in allowed_actions:
                probs[a] = 0.0
            elif a == greedy_action:
                probs[a] = 1 - self.epsilon + self.epsilon / n_allowed
            else:
                probs[a] = self.epsilon / n_allowed

        return probs

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._step_count += 1

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "epsilon_greedy",
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "step_count": self._step_count
        }


class PolicyManager:
    """
    Manages policy lifecycle, versioning, and switching.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.policies: Dict[str, BasePolicy] = {}
        self.active_policy_id: Optional[str] = None
        self.policy_performance: Dict[str, List[float]] = {}

    def register_policy(self, policy: BasePolicy, set_active: bool = False):
        """Register a new policy."""
        self.policies[policy.policy_id] = policy
        self.policy_performance[policy.policy_id] = []

        if set_active or self.active_policy_id is None:
            self.active_policy_id = policy.policy_id

    def get_active_policy(self) -> Optional[BasePolicy]:
        """Get the currently active policy."""
        if self.active_policy_id:
            return self.policies.get(self.active_policy_id)
        return None

    def switch_policy(self, policy_id: str) -> bool:
        """Switch to a different policy."""
        if policy_id in self.policies:
            self.active_policy_id = policy_id
            return True
        return False

    def record_performance(self, policy_id: str, reward: float):
        """Record policy performance for comparison."""
        if policy_id in self.policy_performance:
            self.policy_performance[policy_id].append(reward)

    def get_best_policy(self, window: int = 100) -> Optional[str]:
        """Get policy with best recent performance."""
        best_policy = None
        best_avg = float('-inf')

        for policy_id, rewards in self.policy_performance.items():
            if rewards:
                recent = rewards[-window:]
                avg = sum(recent) / len(recent)
                if avg > best_avg:
                    best_avg = avg
                    best_policy = policy_id

        return best_policy

    def select_action(self, state: Dict[str, Any]) -> Tuple[str, str]:
        """Select action using active policy."""
        policy = self.get_active_policy()
        if policy:
            action = policy.select_action(state)
            return action, policy.policy_id
        raise ValueError("No active policy")


# =============================================================================
# DEFAULT CONSTRAINTS (Immutable Safety Rules)
# =============================================================================

def create_default_constraints() -> List[PolicyConstraint]:
    """Create default safety constraints."""
    return [
        PolicyConstraint(
            name="high_risk_escalation",
            condition=lambda s: s.get("risk_exposure", 0) > 0.9,
            forbidden_actions=["execute_task", "delegate"],
            priority=100,
            is_hard=True
        ),
        PolicyConstraint(
            name="constraint_violation_block",
            condition=lambda s: s.get("constraint_violations", 0) > 0,
            forbidden_actions=["execute_task"],
            priority=99,
            is_hard=True
        ),
        PolicyConstraint(
            name="low_confidence_caution",
            condition=lambda s: s.get("confidence_level", 0) < 0.3,
            forbidden_actions=["execute_task", "delegate"],
            priority=50,
            is_hard=False
        ),
        PolicyConstraint(
            name="critical_urgency_focus",
            condition=lambda s: s.get("user_urgency") == "critical",
            forbidden_actions=["defer_action"],
            priority=80,
            is_hard=True
        ),
    ]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_default_policy(
    agent_id: str,
    action_space: List[str],
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
) -> PolicyManager:
    """Create a default policy manager with standard policies."""

    manager = PolicyManager(agent_id)

    # Default Q-function (uniform)
    def default_q(state: Dict, action: str) -> float:
        return 0.0

    # Create epsilon-greedy policy (default)
    epsilon_policy = EpsilonGreedyPolicy(
        policy_id=f"{agent_id}_epsilon_greedy",
        action_space=action_space,
        q_function=default_q,
        epsilon=0.1
    )

    # Add default constraints
    for constraint in create_default_constraints():
        epsilon_policy.add_constraint(constraint)

    manager.register_policy(epsilon_policy, set_active=True)

    # Create conservative policy for high-risk situations
    conservative_mapping = {
        "high_risk": "escalate",
        "low_confidence": "gather_information",
        "constraint_violation": "escalate"
    }

    conservative_policy = DeterministicPolicy(
        policy_id=f"{agent_id}_conservative",
        action_space=action_space,
        state_action_mapping=conservative_mapping,
        default_action="gather_information"
    )

    for constraint in create_default_constraints():
        conservative_policy.add_constraint(constraint)

    manager.register_policy(conservative_policy)

    return manager
