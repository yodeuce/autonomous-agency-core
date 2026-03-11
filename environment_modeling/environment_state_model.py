"""
FILE 10: environment_state_model.py
PURPOSE: Canonical environment state definition
ROLE: The external state vector

Includes:
- Observable variables
- Latent variables
- Noise assumptions
- Update frequency
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class VariableType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BINARY = "binary"


class Observability(Enum):
    OBSERVABLE = "observable"
    LATENT = "latent"
    PARTIALLY_OBSERVABLE = "partially_observable"


@dataclass
class StateVariable:
    """Definition of a single environment state variable."""
    name: str
    variable_type: VariableType
    observability: Observability
    value: Any = None
    confidence: float = 1.0
    bounds: tuple[float | None, float | None] = (None, None)
    noise_model: str = "gaussian"
    noise_std: float = 0.0
    update_frequency: str = "per_step"
    last_updated: str = ""
    source: str = ""

    def update(self, new_value: Any, confidence: float = 1.0) -> None:
        self.value = new_value
        self.confidence = confidence
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def is_valid(self) -> bool:
        if self.value is None:
            return False
        if self.variable_type == VariableType.CONTINUOUS:
            lo, hi = self.bounds
            if lo is not None and self.value < lo:
                return False
            if hi is not None and self.value > hi:
                return False
        return True


class EnvironmentStateModel:
    """
    Canonical representation of the environment state.
    Separates observable, latent, and partially observable variables.
    """

    def __init__(self):
        self.variables: dict[str, StateVariable] = {}
        self.state_history: list[dict[str, Any]] = []
        self.step: int = 0

    def register_variable(self, var: StateVariable) -> None:
        """Register a new environment variable."""
        self.variables[var.name] = var
        logger.info(
            f"Registered env variable '{var.name}' "
            f"(type={var.variable_type.value}, obs={var.observability.value})"
        )

    def update_variable(
        self, name: str, value: Any, confidence: float = 1.0
    ) -> None:
        """Update a variable's value."""
        if name not in self.variables:
            raise KeyError(f"Unknown environment variable: {name}")
        self.variables[name].update(value, confidence)

    def get_observable_state(self) -> dict[str, Any]:
        """Return the current observable state vector."""
        return {
            name: {
                "value": var.value,
                "confidence": var.confidence,
                "type": var.variable_type.value,
            }
            for name, var in self.variables.items()
            if var.observability in (Observability.OBSERVABLE, Observability.PARTIALLY_OBSERVABLE)
        }

    def get_latent_state(self) -> dict[str, Any]:
        """Return estimated latent state variables."""
        return {
            name: {
                "value": var.value,
                "confidence": var.confidence,
                "type": var.variable_type.value,
            }
            for name, var in self.variables.items()
            if var.observability == Observability.LATENT
        }

    def get_full_state(self) -> dict[str, Any]:
        """Return complete state vector with all variables."""
        return {
            name: {
                "value": var.value,
                "confidence": var.confidence,
                "type": var.variable_type.value,
                "observability": var.observability.value,
                "valid": var.is_valid(),
            }
            for name, var in self.variables.items()
        }

    def snapshot(self) -> dict[str, Any]:
        """Take a timestamped snapshot of the current state."""
        snap = {
            "step": self.step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": self.get_full_state(),
        }
        self.state_history.append(snap)
        return snap

    def advance_step(self) -> None:
        """Advance to the next timestep."""
        self.step += 1

    def get_state_delta(self) -> dict[str, Any]:
        """Compute the change from the previous state snapshot."""
        if len(self.state_history) < 2:
            return {}

        prev = self.state_history[-2]["state"]
        curr = self.state_history[-1]["state"]
        delta = {}

        for name in curr:
            if name in prev:
                prev_val = prev[name]["value"]
                curr_val = curr[name]["value"]
                if prev_val != curr_val:
                    delta[name] = {
                        "previous": prev_val,
                        "current": curr_val,
                    }
                    if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                        delta[name]["change"] = curr_val - prev_val

        return delta

    def validate_state(self) -> list[str]:
        """Validate all variables and return list of issues."""
        issues = []
        for name, var in self.variables.items():
            if not var.is_valid():
                issues.append(f"Variable '{name}' has invalid value: {var.value}")
            if var.confidence < 0.3:
                issues.append(f"Variable '{name}' has low confidence: {var.confidence:.3f}")
        return issues
