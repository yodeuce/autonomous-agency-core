"""
ENVIRONMENT OBSERVER MODULE
Ingests environment signals from various sources
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Handles:
- Sensors / APIs
- Data normalization
- Missing data handling
- Confidence tagging

Separates observation from belief.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json


class ObservationSource(Enum):
    """Sources of environment observations."""
    INTERNAL = "internal"        # Agent's own state
    USER_INPUT = "user_input"   # Direct user input
    API = "api"                  # External API calls
    SENSOR = "sensor"           # Simulated sensors
    INFERENCE = "inference"     # Derived observations
    SYSTEM = "system"           # System metrics


class DataQuality(Enum):
    """Quality classification of observed data."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"
    MISSING = "missing"


@dataclass
class RawObservation:
    """Raw observation before processing."""
    source: ObservationSource
    source_id: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedObservation:
    """Processed observation ready for state update."""
    variable_name: str
    value: Any
    confidence: float
    quality: DataQuality
    source: ObservationSource
    timestamp: datetime
    raw_value: Any = None
    normalization_applied: bool = False
    interpolated: bool = False


@dataclass
class ObserverConfig:
    """Configuration for environment observer."""
    # Normalization
    normalize_continuous: bool = True
    clip_outliers: bool = True
    outlier_std_threshold: float = 3.0

    # Missing data handling
    interpolate_missing: bool = True
    max_interpolation_gap_ms: int = 5000
    missing_value_strategy: str = "last_known"  # last_known, default, interpolate

    # Confidence estimation
    default_confidence: float = 0.8
    confidence_decay_per_second: float = 0.01
    min_confidence: float = 0.1

    # Filtering
    enable_spike_filter: bool = True
    spike_threshold_factor: float = 2.0

    # Batching
    batch_observations: bool = True
    batch_window_ms: int = 100


class BaseObserver(ABC):
    """Abstract base class for environment observers."""

    def __init__(
        self,
        config: Optional[ObserverConfig] = None,
        variable_registry: Optional[Dict] = None
    ):
        self.config = config or ObserverConfig()
        self.variable_registry = variable_registry or {}

        # Observation history for interpolation
        self._observation_history: Dict[str, List[ProcessedObservation]] = {}
        self._last_values: Dict[str, Tuple[Any, datetime]] = {}

        # Statistics for normalization
        self._running_stats: Dict[str, Dict[str, float]] = {}

    @abstractmethod
    def observe(self, source: ObservationSource, data: Any) -> List[ProcessedObservation]:
        """Process raw observations into structured observations."""
        pass

    @abstractmethod
    def get_variable_mapping(self, source: ObservationSource) -> Dict[str, str]:
        """Get mapping from source fields to state variables."""
        pass


class StandardEnvironmentObserver(BaseObserver):
    """
    Standard implementation of environment observer.
    Handles multiple observation sources and normalizes data.
    """

    def __init__(
        self,
        config: Optional[ObserverConfig] = None,
        variable_registry: Optional[Dict] = None
    ):
        super().__init__(config, variable_registry)

        # Source-specific handlers
        self._source_handlers: Dict[ObservationSource, Callable] = {
            ObservationSource.INTERNAL: self._handle_internal,
            ObservationSource.USER_INPUT: self._handle_user_input,
            ObservationSource.API: self._handle_api,
            ObservationSource.SENSOR: self._handle_sensor,
            ObservationSource.INFERENCE: self._handle_inference,
            ObservationSource.SYSTEM: self._handle_system
        }

        # Variable mappings per source
        self._variable_mappings: Dict[ObservationSource, Dict[str, str]] = {
            ObservationSource.INTERNAL: {
                "confidence": "confidence_level",
                "risk": "risk_exposure",
                "violations": "constraint_violations",
                "context": "task_context",
                "progress": "task_progress"
            },
            ObservationSource.USER_INPUT: {
                "urgency": "user_urgency",
                "intent": "user_intent",
                "satisfaction": "user_satisfaction"
            },
            ObservationSource.SYSTEM: {
                "cpu": "resource_availability",
                "memory": "memory_utilization",
                "latency": "response_latency",
                "session_time": "session_duration"
            }
        }

    def observe(
        self,
        source: ObservationSource,
        data: Any
    ) -> List[ProcessedObservation]:
        """
        Process raw observations into structured observations.
        Handles normalization, missing data, and confidence tagging.
        """
        handler = self._source_handlers.get(source)
        if handler is None:
            return []

        raw_obs = RawObservation(
            source=source,
            source_id=f"{source.value}_{datetime.now().timestamp()}",
            data=data
        )

        # Process through source-specific handler
        processed = handler(raw_obs)

        # Post-processing for all observations
        final_observations = []
        for obs in processed:
            # Apply normalization
            if self.config.normalize_continuous:
                obs = self._normalize_observation(obs)

            # Handle outliers
            if self.config.clip_outliers:
                obs = self._handle_outliers(obs)

            # Apply spike filter
            if self.config.enable_spike_filter:
                obs = self._apply_spike_filter(obs)

            # Update history
            self._update_history(obs)

            final_observations.append(obs)

        return final_observations

    def get_variable_mapping(
        self,
        source: ObservationSource
    ) -> Dict[str, str]:
        """Get mapping from source fields to state variables."""
        return self._variable_mappings.get(source, {})

    def handle_missing_data(
        self,
        variable_name: str
    ) -> Optional[ProcessedObservation]:
        """
        Handle missing data for a variable.
        Returns interpolated or default observation.
        """
        if not self.config.interpolate_missing:
            return None

        strategy = self.config.missing_value_strategy

        if strategy == "last_known":
            return self._get_last_known(variable_name)
        elif strategy == "interpolate":
            return self._interpolate_value(variable_name)
        elif strategy == "default":
            return self._get_default_observation(variable_name)

        return None

    def _handle_internal(
        self,
        raw: RawObservation
    ) -> List[ProcessedObservation]:
        """Handle internal state observations."""
        observations = []
        mapping = self._variable_mappings[ObservationSource.INTERNAL]

        if isinstance(raw.data, dict):
            for field, var_name in mapping.items():
                if field in raw.data:
                    obs = ProcessedObservation(
                        variable_name=var_name,
                        value=raw.data[field],
                        confidence=self.config.default_confidence,
                        quality=DataQuality.HIGH,
                        source=raw.source,
                        timestamp=raw.timestamp,
                        raw_value=raw.data[field]
                    )
                    observations.append(obs)

        return observations

    def _handle_user_input(
        self,
        raw: RawObservation
    ) -> List[ProcessedObservation]:
        """Handle user input observations."""
        observations = []
        mapping = self._variable_mappings[ObservationSource.USER_INPUT]

        if isinstance(raw.data, dict):
            for field, var_name in mapping.items():
                if field in raw.data:
                    # User input has high confidence
                    obs = ProcessedObservation(
                        variable_name=var_name,
                        value=raw.data[field],
                        confidence=0.95,
                        quality=DataQuality.HIGH,
                        source=raw.source,
                        timestamp=raw.timestamp,
                        raw_value=raw.data[field]
                    )
                    observations.append(obs)

        return observations

    def _handle_api(
        self,
        raw: RawObservation
    ) -> List[ProcessedObservation]:
        """Handle API response observations."""
        observations = []

        if isinstance(raw.data, dict):
            # API responses may have variable confidence
            api_confidence = raw.metadata.get("confidence", 0.8)

            for key, value in raw.data.items():
                obs = ProcessedObservation(
                    variable_name=key,
                    value=value,
                    confidence=api_confidence,
                    quality=DataQuality.MEDIUM,
                    source=raw.source,
                    timestamp=raw.timestamp,
                    raw_value=value
                )
                observations.append(obs)

        return observations

    def _handle_sensor(
        self,
        raw: RawObservation
    ) -> List[ProcessedObservation]:
        """Handle sensor observations with noise."""
        observations = []

        if isinstance(raw.data, dict):
            for key, value in raw.data.items():
                # Sensors have variable quality
                noise_level = raw.metadata.get("noise_std", 0.1)
                quality = (
                    DataQuality.HIGH if noise_level < 0.05
                    else DataQuality.MEDIUM if noise_level < 0.15
                    else DataQuality.LOW
                )

                obs = ProcessedObservation(
                    variable_name=key,
                    value=value,
                    confidence=1.0 - noise_level,
                    quality=quality,
                    source=raw.source,
                    timestamp=raw.timestamp,
                    raw_value=value
                )
                observations.append(obs)

        return observations

    def _handle_inference(
        self,
        raw: RawObservation
    ) -> List[ProcessedObservation]:
        """Handle inferred observations (lower confidence)."""
        observations = []

        if isinstance(raw.data, dict):
            inference_confidence = raw.metadata.get("confidence", 0.6)

            for key, value in raw.data.items():
                obs = ProcessedObservation(
                    variable_name=key,
                    value=value,
                    confidence=inference_confidence,
                    quality=DataQuality.UNCERTAIN,
                    source=raw.source,
                    timestamp=raw.timestamp,
                    raw_value=value
                )
                observations.append(obs)

        return observations

    def _handle_system(
        self,
        raw: RawObservation
    ) -> List[ProcessedObservation]:
        """Handle system metric observations."""
        observations = []
        mapping = self._variable_mappings[ObservationSource.SYSTEM]

        if isinstance(raw.data, dict):
            for field, var_name in mapping.items():
                if field in raw.data:
                    obs = ProcessedObservation(
                        variable_name=var_name,
                        value=raw.data[field],
                        confidence=0.99,  # System metrics are reliable
                        quality=DataQuality.HIGH,
                        source=raw.source,
                        timestamp=raw.timestamp,
                        raw_value=raw.data[field]
                    )
                    observations.append(obs)

        return observations

    def _normalize_observation(
        self,
        obs: ProcessedObservation
    ) -> ProcessedObservation:
        """Normalize continuous observations."""
        if not isinstance(obs.value, (int, float)):
            return obs

        var_name = obs.variable_name

        # Get or initialize running stats
        if var_name not in self._running_stats:
            self._running_stats[var_name] = {
                "count": 0,
                "mean": 0.0,
                "M2": 0.0,  # For Welford's algorithm
                "min": float('inf'),
                "max": float('-inf')
            }

        stats = self._running_stats[var_name]

        # Update running statistics (Welford's algorithm)
        stats["count"] += 1
        delta = obs.value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = obs.value - stats["mean"]
        stats["M2"] += delta * delta2
        stats["min"] = min(stats["min"], obs.value)
        stats["max"] = max(stats["max"], obs.value)

        obs.normalization_applied = True
        return obs

    def _handle_outliers(
        self,
        obs: ProcessedObservation
    ) -> ProcessedObservation:
        """Handle outliers in observations."""
        if not isinstance(obs.value, (int, float)):
            return obs

        var_name = obs.variable_name
        stats = self._running_stats.get(var_name)

        if stats is None or stats["count"] < 10:
            return obs

        # Calculate standard deviation
        variance = stats["M2"] / stats["count"]
        std = variance ** 0.5

        if std == 0:
            return obs

        # Check if outlier
        z_score = abs(obs.value - stats["mean"]) / std
        if z_score > self.config.outlier_std_threshold:
            # Clip to threshold
            if obs.value > stats["mean"]:
                obs.value = stats["mean"] + self.config.outlier_std_threshold * std
            else:
                obs.value = stats["mean"] - self.config.outlier_std_threshold * std

            # Reduce confidence for clipped values
            obs.confidence *= 0.7
            obs.quality = DataQuality.UNCERTAIN

        return obs

    def _apply_spike_filter(
        self,
        obs: ProcessedObservation
    ) -> ProcessedObservation:
        """Apply spike filter to detect sudden changes."""
        if not isinstance(obs.value, (int, float)):
            return obs

        var_name = obs.variable_name

        if var_name not in self._last_values:
            return obs

        last_value, last_time = self._last_values[var_name]

        if not isinstance(last_value, (int, float)):
            return obs

        # Check for spike
        change = abs(obs.value - last_value)
        time_delta = (obs.timestamp - last_time).total_seconds()

        if time_delta > 0:
            rate_of_change = change / time_delta
            stats = self._running_stats.get(var_name, {})
            typical_range = stats.get("max", 1) - stats.get("min", 0)

            if typical_range > 0:
                normalized_rate = rate_of_change / typical_range
                if normalized_rate > self.config.spike_threshold_factor:
                    # Potential spike - reduce confidence
                    obs.confidence *= 0.5
                    obs.quality = DataQuality.UNCERTAIN

        return obs

    def _update_history(self, obs: ProcessedObservation):
        """Update observation history."""
        var_name = obs.variable_name

        if var_name not in self._observation_history:
            self._observation_history[var_name] = []

        self._observation_history[var_name].append(obs)

        # Keep limited history
        if len(self._observation_history[var_name]) > 100:
            self._observation_history[var_name] = \
                self._observation_history[var_name][-100:]

        # Update last value
        self._last_values[var_name] = (obs.value, obs.timestamp)

    def _get_last_known(
        self,
        variable_name: str
    ) -> Optional[ProcessedObservation]:
        """Get last known observation for a variable."""
        history = self._observation_history.get(variable_name, [])
        if not history:
            return None

        last_obs = history[-1]

        # Apply confidence decay
        elapsed = (datetime.now() - last_obs.timestamp).total_seconds()
        decayed_confidence = max(
            self.config.min_confidence,
            last_obs.confidence - elapsed * self.config.confidence_decay_per_second
        )

        return ProcessedObservation(
            variable_name=variable_name,
            value=last_obs.value,
            confidence=decayed_confidence,
            quality=DataQuality.UNCERTAIN,
            source=ObservationSource.INFERENCE,
            timestamp=datetime.now(),
            interpolated=True
        )

    def _interpolate_value(
        self,
        variable_name: str
    ) -> Optional[ProcessedObservation]:
        """Interpolate missing value from history."""
        history = self._observation_history.get(variable_name, [])
        if len(history) < 2:
            return self._get_last_known(variable_name)

        # Simple linear interpolation from last two values
        recent = history[-2:]
        if not all(isinstance(o.value, (int, float)) for o in recent):
            return self._get_last_known(variable_name)

        v1, v2 = recent[0].value, recent[1].value
        t1, t2 = recent[0].timestamp, recent[1].timestamp

        time_diff = (t2 - t1).total_seconds()
        if time_diff == 0:
            return self._get_last_known(variable_name)

        # Extrapolate
        now = datetime.now()
        elapsed = (now - t2).total_seconds()
        rate = (v2 - v1) / time_diff
        interpolated_value = v2 + rate * elapsed

        return ProcessedObservation(
            variable_name=variable_name,
            value=interpolated_value,
            confidence=0.5,  # Lower confidence for interpolated
            quality=DataQuality.UNCERTAIN,
            source=ObservationSource.INFERENCE,
            timestamp=now,
            interpolated=True
        )

    def _get_default_observation(
        self,
        variable_name: str
    ) -> Optional[ProcessedObservation]:
        """Get default observation for a variable."""
        var_info = self.variable_registry.get(variable_name)
        if var_info is None:
            return None

        default_value = var_info.get("default")
        if default_value is None:
            return None

        return ProcessedObservation(
            variable_name=variable_name,
            value=default_value,
            confidence=0.3,
            quality=DataQuality.MISSING,
            source=ObservationSource.SYSTEM,
            timestamp=datetime.now()
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_environment_observer(
    config: Optional[ObserverConfig] = None,
    variable_registry: Optional[Dict] = None
) -> BaseObserver:
    """Create an environment observer."""
    return StandardEnvironmentObserver(config, variable_registry)
