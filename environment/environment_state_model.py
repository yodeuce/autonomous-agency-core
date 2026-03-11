"""
ENVIRONMENT STATE MODEL
Canonical environment state definition
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Includes:
- Observable variables
- Latent variables
- Noise assumptions
- Update frequency

This is the external state vector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import deque


class VariableType(Enum):
    """Types of state variables."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"
    VECTOR = "vector"


class ObservabilityType(Enum):
    """Observability classification."""
    FULLY_OBSERVABLE = "fully_observable"
    PARTIALLY_OBSERVABLE = "partially_observable"
    LATENT = "latent"


class UpdateFrequency(Enum):
    """How often a variable is updated."""
    REAL_TIME = "real_time"
    HIGH_FREQUENCY = "high_frequency"
    MEDIUM_FREQUENCY = "medium_frequency"
    LOW_FREQUENCY = "low_frequency"
    ON_DEMAND = "on_demand"
    STATIC = "static"


@dataclass
class StateVariable:
    """Definition of a single state variable."""
    name: str
    variable_type: VariableType
    observability: ObservabilityType
    update_frequency: UpdateFrequency

    # Value constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None

    # Noise characteristics
    noise_std: float = 0.0
    noise_type: str = "gaussian"

    # Current value
    value: Any = None
    confidence: float = 1.0
    last_updated: Optional[datetime] = None

    # Metadata
    description: str = ""
    unit: str = ""
    source: str = ""

    def validate_value(self, value: Any) -> bool:
        """Validate a value against constraints."""
        if self.variable_type == VariableType.CONTINUOUS:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        elif self.variable_type == VariableType.CATEGORICAL:
            if self.allowed_values and value not in self.allowed_values:
                return False

        elif self.variable_type == VariableType.BINARY:
            if value not in [True, False, 0, 1]:
                return False

        return True


@dataclass
class StateSnapshot:
    """A snapshot of the environment state at a point in time."""
    timestamp: datetime
    variables: Dict[str, Any]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Configuration for environment state model."""
    # Update parameters
    default_update_interval_ms: int = 1000
    stale_threshold_ms: int = 5000

    # Noise handling
    noise_filtering_enabled: bool = True
    noise_filter_window: int = 5

    # History
    max_history_size: int = 1000
    snapshot_interval_ms: int = 60000

    # Partial observability
    inference_enabled: bool = True
    confidence_decay_rate: float = 0.95


class EnvironmentStateModel:
    """
    Canonical model of the environment state.
    Manages observable and latent variables.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()

        # Variable registry
        self._variables: Dict[str, StateVariable] = {}

        # State history
        self._history: deque = deque(maxlen=self.config.max_history_size)
        self._snapshots: List[StateSnapshot] = []

        # Noise filtering buffers
        self._noise_buffers: Dict[str, deque] = {}

        # Initialize default variables
        self._initialize_default_variables()

    def _initialize_default_variables(self):
        """Initialize standard environment variables."""
        default_variables = [
            StateVariable(
                name="task_context",
                variable_type=VariableType.CATEGORICAL,
                observability=ObservabilityType.FULLY_OBSERVABLE,
                update_frequency=UpdateFrequency.REAL_TIME,
                allowed_values=["idle", "planning", "executing", "reviewing", "escalating"],
                description="Current task execution context"
            ),
            StateVariable(
                name="user_urgency",
                variable_type=VariableType.ORDINAL,
                observability=ObservabilityType.FULLY_OBSERVABLE,
                update_frequency=UpdateFrequency.MEDIUM_FREQUENCY,
                allowed_values=["low", "normal", "high", "critical"],
                description="User-indicated urgency level"
            ),
            StateVariable(
                name="confidence_level",
                variable_type=VariableType.CONTINUOUS,
                observability=ObservabilityType.FULLY_OBSERVABLE,
                update_frequency=UpdateFrequency.REAL_TIME,
                min_value=0.0,
                max_value=1.0,
                noise_std=0.02,
                description="Agent confidence in current beliefs"
            ),
            StateVariable(
                name="risk_exposure",
                variable_type=VariableType.CONTINUOUS,
                observability=ObservabilityType.FULLY_OBSERVABLE,
                update_frequency=UpdateFrequency.HIGH_FREQUENCY,
                min_value=0.0,
                max_value=1.0,
                noise_std=0.05,
                description="Current risk level of pending actions"
            ),
            StateVariable(
                name="resource_availability",
                variable_type=VariableType.CONTINUOUS,
                observability=ObservabilityType.FULLY_OBSERVABLE,
                update_frequency=UpdateFrequency.MEDIUM_FREQUENCY,
                min_value=0.0,
                max_value=1.0,
                description="Available computational resources"
            ),
            StateVariable(
                name="constraint_violations",
                variable_type=VariableType.DISCRETE,
                observability=ObservabilityType.FULLY_OBSERVABLE,
                update_frequency=UpdateFrequency.REAL_TIME,
                min_value=0,
                description="Count of active constraint violations"
            ),
            StateVariable(
                name="user_intent",
                variable_type=VariableType.CATEGORICAL,
                observability=ObservabilityType.LATENT,
                update_frequency=UpdateFrequency.HIGH_FREQUENCY,
                allowed_values=["informational", "transactional", "analytical", "creative"],
                noise_std=0.1,
                description="Inferred user intent (latent)"
            ),
            StateVariable(
                name="environment_stability",
                variable_type=VariableType.CONTINUOUS,
                observability=ObservabilityType.PARTIALLY_OBSERVABLE,
                update_frequency=UpdateFrequency.LOW_FREQUENCY,
                min_value=0.0,
                max_value=1.0,
                noise_std=0.1,
                description="Stability of operating environment"
            ),
            StateVariable(
                name="session_duration",
                variable_type=VariableType.CONTINUOUS,
                observability=ObservabilityType.FULLY_OBSERVABLE,
                update_frequency=UpdateFrequency.REAL_TIME,
                min_value=0,
                unit="seconds",
                description="Time since session start"
            ),
            StateVariable(
                name="memory_utilization",
                variable_type=VariableType.CONTINUOUS,
                observability=ObservabilityType.FULLY_OBSERVABLE,
                update_frequency=UpdateFrequency.MEDIUM_FREQUENCY,
                min_value=0.0,
                max_value=1.0,
                description="Memory system utilization"
            )
        ]

        for var in default_variables:
            self.register_variable(var)

    def register_variable(self, variable: StateVariable):
        """Register a new state variable."""
        self._variables[variable.name] = variable

        # Initialize noise buffer for continuous variables
        if variable.variable_type == VariableType.CONTINUOUS:
            self._noise_buffers[variable.name] = deque(
                maxlen=self.config.noise_filter_window
            )

    def update_variable(
        self,
        name: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "observation"
    ) -> bool:
        """
        Update a state variable with a new value.
        Returns True if update was successful.
        """
        if name not in self._variables:
            return False

        variable = self._variables[name]

        # Validate value
        if not variable.validate_value(value):
            return False

        # Apply noise filtering for continuous variables
        if (
            self.config.noise_filtering_enabled and
            variable.variable_type == VariableType.CONTINUOUS and
            name in self._noise_buffers
        ):
            self._noise_buffers[name].append(value)
            if len(self._noise_buffers[name]) >= 3:
                # Use median filter
                value = float(np.median(list(self._noise_buffers[name])))

        # Update variable
        old_value = variable.value
        variable.value = value
        variable.confidence = confidence
        variable.last_updated = datetime.now()
        variable.source = source

        # Record in history
        self._history.append({
            "name": name,
            "old_value": old_value,
            "new_value": value,
            "confidence": confidence,
            "timestamp": datetime.now(),
            "source": source
        })

        return True

    def get_variable(self, name: str) -> Optional[StateVariable]:
        """Get a state variable by name."""
        return self._variables.get(name)

    def get_value(self, name: str, default: Any = None) -> Any:
        """Get the current value of a variable."""
        variable = self._variables.get(name)
        if variable is None:
            return default
        return variable.value if variable.value is not None else default

    def get_state_vector(self) -> Dict[str, Any]:
        """Get current state as a dictionary."""
        return {
            name: var.value
            for name, var in self._variables.items()
            if var.value is not None
        }

    def get_observable_state(self) -> Dict[str, Any]:
        """Get only observable variables."""
        return {
            name: var.value
            for name, var in self._variables.items()
            if var.observability == ObservabilityType.FULLY_OBSERVABLE
            and var.value is not None
        }

    def get_latent_state(self) -> Dict[str, Any]:
        """Get latent (inferred) variables."""
        return {
            name: var.value
            for name, var in self._variables.items()
            if var.observability == ObservabilityType.LATENT
            and var.value is not None
        }

    def get_confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores for all variables."""
        return {
            name: var.confidence
            for name, var in self._variables.items()
        }

    def is_stale(self, name: str) -> bool:
        """Check if a variable is stale (not recently updated)."""
        variable = self._variables.get(name)
        if variable is None or variable.last_updated is None:
            return True

        elapsed_ms = (datetime.now() - variable.last_updated).total_seconds() * 1000
        return elapsed_ms > self.config.stale_threshold_ms

    def get_stale_variables(self) -> List[str]:
        """Get list of stale variables."""
        return [name for name in self._variables if self.is_stale(name)]

    def take_snapshot(self) -> StateSnapshot:
        """Take a snapshot of current state."""
        snapshot = StateSnapshot(
            timestamp=datetime.now(),
            variables=self.get_state_vector(),
            confidence_scores=self.get_confidence_scores(),
            metadata={
                "stale_variables": self.get_stale_variables(),
                "total_variables": len(self._variables)
            }
        )
        self._snapshots.append(snapshot)
        return snapshot

    def get_recent_snapshots(self, count: int = 10) -> List[StateSnapshot]:
        """Get recent state snapshots."""
        return self._snapshots[-count:]

    def compute_state_delta(
        self,
        previous: Dict[str, Any],
        current: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Tuple[Any, Any]]:
        """Compute difference between two states."""
        current = current or self.get_state_vector()
        delta = {}

        all_keys = set(previous.keys()) | set(current.keys())
        for key in all_keys:
            prev_val = previous.get(key)
            curr_val = current.get(key)
            if prev_val != curr_val:
                delta[key] = (prev_val, curr_val)

        return delta

    def apply_confidence_decay(self):
        """Apply confidence decay to all variables based on staleness."""
        for name, variable in self._variables.items():
            if variable.last_updated is None:
                continue

            elapsed_ms = (datetime.now() - variable.last_updated).total_seconds() * 1000

            if elapsed_ms > self.config.stale_threshold_ms:
                # Apply exponential decay
                decay_factor = self.config.confidence_decay_rate ** (
                    elapsed_ms / self.config.stale_threshold_ms
                )
                variable.confidence *= decay_factor

    def reset(self):
        """Reset all variables to None."""
        for variable in self._variables.values():
            variable.value = None
            variable.confidence = 1.0
            variable.last_updated = None

        self._history.clear()
        for buffer in self._noise_buffers.values():
            buffer.clear()


class PartiallyObservableStateModel(EnvironmentStateModel):
    """
    Extension for partially observable environments.
    Includes belief state tracking.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__(config)
        self._belief_state: Dict[str, Dict[str, float]] = {}

    def update_belief(
        self,
        name: str,
        observation: Any,
        observation_probability: float = 1.0
    ):
        """
        Update belief state using Bayesian update.
        For latent/partially observable variables.
        """
        variable = self._variables.get(name)
        if variable is None:
            return

        if variable.allowed_values is None:
            return

        # Initialize belief if needed
        if name not in self._belief_state:
            # Uniform prior
            n_values = len(variable.allowed_values)
            self._belief_state[name] = {
                str(v): 1.0 / n_values
                for v in variable.allowed_values
            }

        # Simple likelihood update
        # P(hidden | obs) ∝ P(obs | hidden) * P(hidden)
        beliefs = self._belief_state[name]
        for value in variable.allowed_values:
            if value == observation:
                beliefs[str(value)] *= observation_probability
            else:
                beliefs[str(value)] *= (1 - observation_probability) / (len(variable.allowed_values) - 1)

        # Normalize
        total = sum(beliefs.values())
        if total > 0:
            for key in beliefs:
                beliefs[key] /= total

        # Update variable with MAP estimate
        map_value = max(beliefs, key=beliefs.get)
        variable.value = map_value
        variable.confidence = beliefs[map_value]

    def get_belief_state(self, name: str) -> Optional[Dict[str, float]]:
        """Get belief distribution for a variable."""
        return self._belief_state.get(name)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_environment_model(
    model_type: str = "standard",
    config: Optional[EnvironmentConfig] = None
) -> EnvironmentStateModel:
    """Create an environment state model."""

    if model_type == "standard":
        return EnvironmentStateModel(config)
    elif model_type == "pomdp":
        return PartiallyObservableStateModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
