"""
BELIEF STATE UPDATER
Converts observations into beliefs using Bayesian inference
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Includes:
- Bayesian updates
- State estimation
- Noise filtering
- Confidence propagation

Required when environment is partially observable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict


class BeliefUpdateMethod(Enum):
    """Methods for belief state updates."""
    BAYESIAN = "bayesian"
    KALMAN = "kalman"
    PARTICLE = "particle"
    EXPONENTIAL_MOVING = "exponential_moving"


@dataclass
class BeliefState:
    """Belief state for a single variable."""
    variable_name: str

    # For discrete/categorical variables: probability distribution
    distribution: Optional[Dict[str, float]] = None

    # For continuous variables: mean and variance
    mean: Optional[float] = None
    variance: Optional[float] = None

    # Common fields
    confidence: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0

    def get_map_estimate(self) -> Any:
        """Get Maximum A Posteriori estimate."""
        if self.distribution:
            return max(self.distribution, key=self.distribution.get)
        return self.mean

    def get_uncertainty(self) -> float:
        """Get uncertainty measure."""
        if self.distribution:
            # Entropy-based uncertainty
            probs = list(self.distribution.values())
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            max_entropy = np.log(len(probs))
            return entropy / max_entropy if max_entropy > 0 else 0
        elif self.variance is not None:
            return min(1.0, self.variance)
        return 1.0 - self.confidence


@dataclass
class Observation:
    """Single observation for belief update."""
    variable_name: str
    value: Any
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UpdaterConfig:
    """Configuration for belief state updater."""
    # Bayesian update parameters
    prior_strength: float = 1.0  # Strength of prior beliefs
    observation_weight: float = 1.0  # Weight of new observations

    # Kalman filter parameters
    process_noise: float = 0.01  # Q: Process noise variance
    measurement_noise: float = 0.1  # R: Measurement noise variance

    # Particle filter parameters
    num_particles: int = 100
    resampling_threshold: float = 0.5

    # Exponential moving average
    ema_alpha: float = 0.3  # Smoothing factor

    # General parameters
    confidence_decay_rate: float = 0.99
    min_confidence: float = 0.1
    max_history: int = 100


class BaseBeliefUpdater(ABC):
    """Abstract base class for belief state updaters."""

    def __init__(self, config: Optional[UpdaterConfig] = None):
        self.config = config or UpdaterConfig()
        self._belief_states: Dict[str, BeliefState] = {}
        self._observation_history: Dict[str, List[Observation]] = defaultdict(list)

    @abstractmethod
    def update(
        self,
        observation: Observation
    ) -> BeliefState:
        """Update belief state given new observation."""
        pass

    @abstractmethod
    def predict(
        self,
        variable_name: str,
        time_horizon: float = 0
    ) -> BeliefState:
        """Predict belief state at future time."""
        pass

    def get_belief(self, variable_name: str) -> Optional[BeliefState]:
        """Get current belief state for a variable."""
        return self._belief_states.get(variable_name)

    def get_all_beliefs(self) -> Dict[str, BeliefState]:
        """Get all belief states."""
        return self._belief_states.copy()

    def decay_confidence(self):
        """Apply confidence decay to all beliefs."""
        for belief in self._belief_states.values():
            elapsed = (datetime.now() - belief.last_updated).total_seconds()
            decay = self.config.confidence_decay_rate ** elapsed
            belief.confidence = max(
                self.config.min_confidence,
                belief.confidence * decay
            )


class BayesianBeliefUpdater(BaseBeliefUpdater):
    """
    Bayesian belief state updater for discrete/categorical variables.
    Updates probability distributions using Bayes' rule.
    """

    def __init__(
        self,
        config: Optional[UpdaterConfig] = None,
        variable_domains: Optional[Dict[str, List[Any]]] = None
    ):
        super().__init__(config)
        self.variable_domains = variable_domains or {}

    def update(self, observation: Observation) -> BeliefState:
        """
        Update belief using Bayesian update.
        P(H|E) = P(E|H) * P(H) / P(E)
        """
        var_name = observation.variable_name

        # Initialize belief if needed
        if var_name not in self._belief_states:
            self._initialize_belief(var_name)

        belief = self._belief_states[var_name]

        # Record observation
        self._observation_history[var_name].append(observation)
        if len(self._observation_history[var_name]) > self.config.max_history:
            self._observation_history[var_name] = \
                self._observation_history[var_name][-self.config.max_history:]

        # Perform Bayesian update
        if belief.distribution is not None:
            belief = self._bayesian_update_discrete(belief, observation)
        else:
            belief = self._bayesian_update_continuous(belief, observation)

        belief.last_updated = datetime.now()
        belief.update_count += 1
        self._belief_states[var_name] = belief

        return belief

    def predict(
        self,
        variable_name: str,
        time_horizon: float = 0
    ) -> BeliefState:
        """Predict belief (for Bayesian, just return current with decayed confidence)."""
        belief = self._belief_states.get(variable_name)
        if belief is None:
            self._initialize_belief(variable_name)
            belief = self._belief_states[variable_name]

        # Apply confidence decay for future prediction
        if time_horizon > 0:
            belief = BeliefState(
                variable_name=variable_name,
                distribution=belief.distribution.copy() if belief.distribution else None,
                mean=belief.mean,
                variance=belief.variance,
                confidence=belief.confidence * (self.config.confidence_decay_rate ** time_horizon)
            )

        return belief

    def _initialize_belief(self, variable_name: str):
        """Initialize belief with uniform prior."""
        domain = self.variable_domains.get(variable_name)

        if domain:
            # Discrete variable - uniform distribution
            n = len(domain)
            distribution = {str(v): 1.0 / n for v in domain}
            self._belief_states[variable_name] = BeliefState(
                variable_name=variable_name,
                distribution=distribution,
                confidence=0.5
            )
        else:
            # Continuous variable - uninformative prior
            self._belief_states[variable_name] = BeliefState(
                variable_name=variable_name,
                mean=0.5,
                variance=0.25,
                confidence=0.5
            )

    def _bayesian_update_discrete(
        self,
        belief: BeliefState,
        observation: Observation
    ) -> BeliefState:
        """Update discrete belief distribution."""
        obs_value = str(observation.value)
        obs_confidence = observation.confidence

        # Calculate likelihood P(E|H) for each hypothesis
        # Higher likelihood for observed value, lower for others
        distribution = belief.distribution.copy()
        total = 0.0

        for value in distribution:
            if value == obs_value:
                # Observed value gets high likelihood
                likelihood = obs_confidence
            else:
                # Other values share remaining probability
                n_other = len(distribution) - 1
                likelihood = (1 - obs_confidence) / max(1, n_other)

            # Bayes update: P(H|E) ∝ P(E|H) * P(H)
            prior = distribution[value]
            posterior = likelihood * prior
            distribution[value] = posterior
            total += posterior

        # Normalize
        if total > 0:
            for value in distribution:
                distribution[value] /= total

        # Update confidence based on observation confidence
        new_confidence = min(
            1.0,
            belief.confidence + obs_confidence * self.config.observation_weight * 0.1
        )

        return BeliefState(
            variable_name=belief.variable_name,
            distribution=distribution,
            confidence=new_confidence,
            last_updated=belief.last_updated,
            update_count=belief.update_count
        )

    def _bayesian_update_continuous(
        self,
        belief: BeliefState,
        observation: Observation
    ) -> BeliefState:
        """Update continuous belief (Gaussian conjugate prior)."""
        if not isinstance(observation.value, (int, float)):
            return belief

        # Treat as Gaussian with known variance (simplified)
        prior_mean = belief.mean or 0.5
        prior_var = belief.variance or 0.25
        obs_mean = observation.value
        obs_var = (1 - observation.confidence) ** 2 + 0.01  # Observation variance

        # Posterior parameters (Gaussian conjugate)
        posterior_var = 1 / (1/prior_var + 1/obs_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + obs_mean/obs_var)

        # Update confidence
        new_confidence = min(
            1.0,
            belief.confidence + observation.confidence * 0.1
        )

        return BeliefState(
            variable_name=belief.variable_name,
            mean=posterior_mean,
            variance=posterior_var,
            confidence=new_confidence,
            last_updated=belief.last_updated,
            update_count=belief.update_count
        )


class KalmanBeliefUpdater(BaseBeliefUpdater):
    """
    Kalman filter for continuous state estimation.
    Optimal for linear Gaussian systems.
    """

    def __init__(self, config: Optional[UpdaterConfig] = None):
        super().__init__(config)
        # Kalman state: mean and covariance
        self._kalman_states: Dict[str, Tuple[float, float]] = {}

    def update(self, observation: Observation) -> BeliefState:
        """Update belief using Kalman filter."""
        var_name = observation.variable_name

        if not isinstance(observation.value, (int, float)):
            # Fall back to simple update for non-numeric
            return self._simple_update(observation)

        # Initialize if needed
        if var_name not in self._kalman_states:
            self._kalman_states[var_name] = (observation.value, 1.0)

        # Get current state
        x, P = self._kalman_states[var_name]

        # Prediction step (assuming stationary process)
        Q = self.config.process_noise
        x_pred = x
        P_pred = P + Q

        # Update step
        z = observation.value
        R = self.config.measurement_noise / max(0.1, observation.confidence)

        # Kalman gain
        K = P_pred / (P_pred + R)

        # State update
        x_new = x_pred + K * (z - x_pred)
        P_new = (1 - K) * P_pred

        # Store updated state
        self._kalman_states[var_name] = (x_new, P_new)

        # Record observation
        self._observation_history[var_name].append(observation)

        # Create belief state
        confidence = 1 - min(1.0, P_new)  # Convert variance to confidence

        belief = BeliefState(
            variable_name=var_name,
            mean=x_new,
            variance=P_new,
            confidence=confidence,
            last_updated=datetime.now(),
            update_count=self._belief_states.get(var_name, BeliefState(var_name)).update_count + 1
        )

        self._belief_states[var_name] = belief
        return belief

    def predict(
        self,
        variable_name: str,
        time_horizon: float = 0
    ) -> BeliefState:
        """Predict future belief state."""
        if variable_name not in self._kalman_states:
            return BeliefState(
                variable_name=variable_name,
                mean=0.5,
                variance=1.0,
                confidence=0.1
            )

        x, P = self._kalman_states[variable_name]

        # Propagate uncertainty
        Q = self.config.process_noise
        P_pred = P + Q * time_horizon

        confidence = 1 - min(1.0, P_pred)

        return BeliefState(
            variable_name=variable_name,
            mean=x,
            variance=P_pred,
            confidence=confidence
        )

    def _simple_update(self, observation: Observation) -> BeliefState:
        """Simple update for non-numeric values."""
        var_name = observation.variable_name
        belief = self._belief_states.get(var_name) or BeliefState(var_name)

        belief.mean = None
        belief.variance = None
        belief.distribution = {str(observation.value): observation.confidence}
        belief.confidence = observation.confidence
        belief.last_updated = datetime.now()
        belief.update_count += 1

        self._belief_states[var_name] = belief
        return belief


class ExponentialMovingAverageUpdater(BaseBeliefUpdater):
    """
    Simple exponential moving average belief updater.
    Good for tracking slowly changing values.
    """

    def update(self, observation: Observation) -> BeliefState:
        """Update belief using exponential moving average."""
        var_name = observation.variable_name

        # Initialize if needed
        if var_name not in self._belief_states:
            self._belief_states[var_name] = BeliefState(
                variable_name=var_name,
                mean=observation.value if isinstance(observation.value, (int, float)) else None,
                confidence=observation.confidence
            )
            return self._belief_states[var_name]

        belief = self._belief_states[var_name]
        alpha = self.config.ema_alpha

        if isinstance(observation.value, (int, float)) and belief.mean is not None:
            # EMA update
            new_mean = alpha * observation.value + (1 - alpha) * belief.mean

            # Variance estimation
            if belief.variance is not None:
                diff = observation.value - belief.mean
                new_variance = (1 - alpha) * (belief.variance + alpha * diff ** 2)
            else:
                new_variance = 0.1

            belief.mean = new_mean
            belief.variance = new_variance
        else:
            # Non-numeric: just store latest
            belief.distribution = {str(observation.value): observation.confidence}

        # Update confidence (blend with observation confidence)
        belief.confidence = alpha * observation.confidence + (1 - alpha) * belief.confidence
        belief.last_updated = datetime.now()
        belief.update_count += 1

        self._belief_states[var_name] = belief
        return belief

    def predict(
        self,
        variable_name: str,
        time_horizon: float = 0
    ) -> BeliefState:
        """Predict future belief (return current with decayed confidence)."""
        belief = self._belief_states.get(variable_name)
        if belief is None:
            return BeliefState(variable_name=variable_name, confidence=0.1)

        # Decay confidence for prediction
        decay = self.config.confidence_decay_rate ** time_horizon
        return BeliefState(
            variable_name=variable_name,
            mean=belief.mean,
            variance=belief.variance,
            distribution=belief.distribution,
            confidence=belief.confidence * decay
        )


class CompositeBeliefUpdater(BaseBeliefUpdater):
    """
    Composite updater that uses different methods for different variable types.
    """

    def __init__(
        self,
        config: Optional[UpdaterConfig] = None,
        variable_domains: Optional[Dict[str, List[Any]]] = None
    ):
        super().__init__(config)

        # Create specialized updaters
        self.bayesian_updater = BayesianBeliefUpdater(config, variable_domains)
        self.kalman_updater = KalmanBeliefUpdater(config)
        self.ema_updater = ExponentialMovingAverageUpdater(config)

        # Variable type mapping
        self.variable_methods: Dict[str, BeliefUpdateMethod] = {}
        self.variable_domains = variable_domains or {}

    def set_update_method(self, variable_name: str, method: BeliefUpdateMethod):
        """Set update method for a specific variable."""
        self.variable_methods[variable_name] = method

    def update(self, observation: Observation) -> BeliefState:
        """Update using appropriate method for variable type."""
        var_name = observation.variable_name
        method = self.variable_methods.get(var_name)

        # Auto-detect method if not specified
        if method is None:
            if var_name in self.variable_domains:
                method = BeliefUpdateMethod.BAYESIAN
            elif isinstance(observation.value, (int, float)):
                method = BeliefUpdateMethod.KALMAN
            else:
                method = BeliefUpdateMethod.EXPONENTIAL_MOVING

            self.variable_methods[var_name] = method

        # Route to appropriate updater
        if method == BeliefUpdateMethod.BAYESIAN:
            belief = self.bayesian_updater.update(observation)
        elif method == BeliefUpdateMethod.KALMAN:
            belief = self.kalman_updater.update(observation)
        else:
            belief = self.ema_updater.update(observation)

        self._belief_states[var_name] = belief
        return belief

    def predict(
        self,
        variable_name: str,
        time_horizon: float = 0
    ) -> BeliefState:
        """Predict using appropriate method."""
        method = self.variable_methods.get(variable_name, BeliefUpdateMethod.EXPONENTIAL_MOVING)

        if method == BeliefUpdateMethod.BAYESIAN:
            return self.bayesian_updater.predict(variable_name, time_horizon)
        elif method == BeliefUpdateMethod.KALMAN:
            return self.kalman_updater.predict(variable_name, time_horizon)
        else:
            return self.ema_updater.predict(variable_name, time_horizon)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_belief_updater(
    method: BeliefUpdateMethod = BeliefUpdateMethod.BAYESIAN,
    config: Optional[UpdaterConfig] = None,
    variable_domains: Optional[Dict[str, List[Any]]] = None
) -> BaseBeliefUpdater:
    """Create a belief state updater."""

    if method == BeliefUpdateMethod.BAYESIAN:
        return BayesianBeliefUpdater(config, variable_domains)
    elif method == BeliefUpdateMethod.KALMAN:
        return KalmanBeliefUpdater(config)
    elif method == BeliefUpdateMethod.EXPONENTIAL_MOVING:
        return ExponentialMovingAverageUpdater(config)
    else:
        return CompositeBeliefUpdater(config, variable_domains)
