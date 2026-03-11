"""
FILE 12: environment_observer.py
PURPOSE: Ingests environment signals
ROLE: Separates observation from belief

Handles:
- Sensors / APIs
- Data normalization
- Missing data handling
- Confidence tagging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class DataSource(Protocol):
    """Protocol for environment data sources (APIs, sensors, etc.)."""

    def fetch(self, variable_name: str) -> dict[str, Any]:
        """Fetch the latest value for a variable. Returns {value, timestamp, raw}."""
        ...

    def is_available(self) -> bool:
        """Check if the data source is currently available."""
        ...


@dataclass
class Observation:
    """A single observation from the environment."""
    variable_name: str
    raw_value: Any
    normalized_value: Any
    confidence: float
    timestamp: str
    source: str
    is_missing: bool = False
    noise_estimate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObserverConfig:
    """Configuration for the environment observer."""
    default_confidence: float = 0.8
    missing_data_strategy: str = "last_known"  # last_known | zero | nan | skip
    normalization_enabled: bool = True
    outlier_detection: bool = True
    outlier_std_threshold: float = 3.0
    max_staleness_steps: int = 50


class EnvironmentObserver:
    """
    Ingests raw environment signals from various sources,
    normalizes them, handles missing data, and tags with confidence.

    This is the boundary between the external world and the agent's
    internal state model.
    """

    def __init__(self, config: ObserverConfig | None = None):
        self.config = config or ObserverConfig()
        self.sources: dict[str, DataSource] = {}
        self.variable_registry: dict[str, dict[str, Any]] = {}
        self.last_observations: dict[str, Observation] = {}
        self.observation_history: list[Observation] = []
        self.step: int = 0
        # Running stats for normalization
        self._running_mean: dict[str, float] = {}
        self._running_var: dict[str, float] = {}
        self._observation_count: dict[str, int] = {}

    def register_source(self, name: str, source: DataSource) -> None:
        """Register a data source for environment observations."""
        self.sources[name] = source
        logger.info(f"Registered data source: {name}")

    def register_variable(
        self,
        name: str,
        source: str,
        var_type: str = "continuous",
        bounds: tuple[Any, Any] = (None, None),
        noise_std: float = 0.0,
    ) -> None:
        """Register a variable to be observed."""
        self.variable_registry[name] = {
            "source": source,
            "type": var_type,
            "bounds": bounds,
            "noise_std": noise_std,
        }

    def observe(self, variable_name: str) -> Observation:
        """
        Observe a single environment variable.

        Args:
            variable_name: Name of the variable to observe

        Returns:
            An Observation with normalized value and confidence
        """
        registry_entry = self.variable_registry.get(variable_name)
        if not registry_entry:
            return self._handle_unknown_variable(variable_name)

        source_name = registry_entry["source"]
        source = self.sources.get(source_name)

        # Handle unavailable source
        if source is None or not source.is_available():
            return self._handle_missing_data(variable_name, "source_unavailable")

        # Fetch raw data
        try:
            raw_data = source.fetch(variable_name)
        except Exception as e:
            logger.error(f"Error fetching '{variable_name}' from '{source_name}': {e}")
            return self._handle_missing_data(variable_name, "fetch_error")

        raw_value = raw_data.get("value")
        if raw_value is None:
            return self._handle_missing_data(variable_name, "null_value")

        # Normalize
        normalized = self._normalize(variable_name, raw_value, registry_entry)

        # Estimate confidence
        confidence = self._estimate_confidence(
            variable_name, normalized, registry_entry
        )

        # Detect outliers
        if self.config.outlier_detection:
            is_outlier = self._check_outlier(variable_name, normalized)
            if is_outlier:
                confidence *= 0.5
                logger.warning(f"Outlier detected for '{variable_name}': {normalized}")

        observation = Observation(
            variable_name=variable_name,
            raw_value=raw_value,
            normalized_value=normalized,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source_name,
            noise_estimate=registry_entry.get("noise_std", 0.0),
        )

        # Update tracking
        self.last_observations[variable_name] = observation
        self.observation_history.append(observation)
        self._update_running_stats(variable_name, normalized)

        return observation

    def observe_all(self) -> list[Observation]:
        """Observe all registered variables."""
        observations = []
        for var_name in self.variable_registry:
            obs = self.observe(var_name)
            observations.append(obs)
        self.step += 1
        return observations

    def get_observation_vector(self) -> dict[str, Any]:
        """Return the latest observations as a flat state dict."""
        return {
            name: {
                "value": obs.normalized_value,
                "confidence": obs.confidence,
                "is_missing": obs.is_missing,
            }
            for name, obs in self.last_observations.items()
        }

    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------

    def _normalize(
        self,
        variable_name: str,
        raw_value: Any,
        registry: dict[str, Any],
    ) -> Any:
        """Normalize a raw value based on variable type and bounds."""
        if not self.config.normalization_enabled:
            return raw_value

        var_type = registry.get("type", "continuous")

        if var_type in ("categorical", "binary"):
            return raw_value

        if var_type in ("continuous", "discrete"):
            bounds = registry.get("bounds", (None, None))
            lo, hi = bounds
            value = float(raw_value)

            # Clamp to bounds
            if lo is not None:
                value = max(lo, value)
            if hi is not None:
                value = min(hi, value)

            return value

        return raw_value

    # -------------------------------------------------------------------------
    # CONFIDENCE ESTIMATION
    # -------------------------------------------------------------------------

    def _estimate_confidence(
        self,
        variable_name: str,
        value: Any,
        registry: dict[str, Any],
    ) -> float:
        """Estimate confidence in an observation."""
        confidence = self.config.default_confidence

        # Higher noise = lower confidence
        noise_std = registry.get("noise_std", 0.0)
        if noise_std > 0:
            confidence -= min(0.3, noise_std * 0.5)

        # Check staleness of data
        if variable_name in self.last_observations:
            last = self.last_observations[variable_name]
            # Simple proxy: if the value hasn't changed, slight confidence reduction
            if last.normalized_value == value:
                confidence -= 0.05

        return max(0.1, min(1.0, confidence))

    # -------------------------------------------------------------------------
    # MISSING DATA HANDLING
    # -------------------------------------------------------------------------

    def _handle_missing_data(
        self, variable_name: str, reason: str
    ) -> Observation:
        """Handle missing data according to configured strategy."""
        strategy = self.config.missing_data_strategy

        if strategy == "last_known" and variable_name in self.last_observations:
            last = self.last_observations[variable_name]
            return Observation(
                variable_name=variable_name,
                raw_value=None,
                normalized_value=last.normalized_value,
                confidence=last.confidence * 0.5,  # Degraded confidence
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=f"last_known ({reason})",
                is_missing=True,
            )
        elif strategy == "zero":
            return Observation(
                variable_name=variable_name,
                raw_value=None,
                normalized_value=0.0,
                confidence=0.1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=f"default_zero ({reason})",
                is_missing=True,
            )
        else:
            return Observation(
                variable_name=variable_name,
                raw_value=None,
                normalized_value=None,
                confidence=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source=f"missing ({reason})",
                is_missing=True,
            )

    def _handle_unknown_variable(self, variable_name: str) -> Observation:
        logger.warning(f"Attempted to observe unregistered variable: {variable_name}")
        return Observation(
            variable_name=variable_name,
            raw_value=None,
            normalized_value=None,
            confidence=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="unknown",
            is_missing=True,
        )

    # -------------------------------------------------------------------------
    # OUTLIER DETECTION
    # -------------------------------------------------------------------------

    def _check_outlier(self, variable_name: str, value: Any) -> bool:
        """Check if a value is an outlier based on running statistics."""
        if not isinstance(value, (int, float)):
            return False

        count = self._observation_count.get(variable_name, 0)
        if count < 10:  # Not enough data
            return False

        mean = self._running_mean.get(variable_name, 0.0)
        var = self._running_var.get(variable_name, 1.0)
        std = var ** 0.5

        if std < 1e-10:
            return False

        z_score = abs(value - mean) / std
        return z_score > self.config.outlier_std_threshold

    def _update_running_stats(self, variable_name: str, value: Any) -> None:
        """Update running mean and variance using Welford's algorithm."""
        if not isinstance(value, (int, float)):
            return

        count = self._observation_count.get(variable_name, 0) + 1
        self._observation_count[variable_name] = count

        if count == 1:
            self._running_mean[variable_name] = float(value)
            self._running_var[variable_name] = 0.0
        else:
            mean = self._running_mean[variable_name]
            delta = float(value) - mean
            new_mean = mean + delta / count
            delta2 = float(value) - new_mean
            new_var = self._running_var[variable_name] + delta * delta2

            self._running_mean[variable_name] = new_mean
            self._running_var[variable_name] = new_var / count
