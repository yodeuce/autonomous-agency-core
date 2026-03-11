"""
FILE 13: belief_state_updater.py
PURPOSE: Converts observations into beliefs
ROLE: Required when environment is partially observable

Includes:
- Bayesian updates
- State estimation
- Noise filtering
- Confidence propagation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Belief:
    """A belief about a single state variable."""
    variable_name: str
    mean: float
    variance: float
    confidence: float
    last_observation: float | None = None
    update_count: int = 0

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable_name": self.variable_name,
            "mean": self.mean,
            "variance": self.variance,
            "confidence": self.confidence,
            "std": self.std,
            "update_count": self.update_count,
        }


@dataclass
class BeliefStateConfig:
    """Configuration for belief state updater."""
    prior_variance: float = 1.0
    observation_noise_default: float = 0.1
    confidence_decay_rate: float = 0.01
    min_confidence: float = 0.05
    max_variance: float = 100.0
    kalman_process_noise: float = 0.01


class BeliefStateUpdater:
    """
    Maintains and updates the agent's beliefs about the environment
    using Bayesian inference. Converts noisy, partial observations
    into calibrated belief distributions.
    """

    def __init__(self, config: BeliefStateConfig | None = None):
        self.config = config or BeliefStateConfig()
        self.beliefs: dict[str, Belief] = {}
        self.step: int = 0

    def initialize_belief(
        self,
        variable_name: str,
        prior_mean: float = 0.0,
        prior_variance: float | None = None,
    ) -> Belief:
        """Initialize a belief for a variable with a prior."""
        variance = prior_variance or self.config.prior_variance
        belief = Belief(
            variable_name=variable_name,
            mean=prior_mean,
            variance=variance,
            confidence=0.5,  # Start with moderate confidence
        )
        self.beliefs[variable_name] = belief
        return belief

    def update(
        self,
        variable_name: str,
        observed_value: float,
        observation_noise: float | None = None,
        observation_confidence: float = 1.0,
    ) -> Belief:
        """
        Update belief using Bayesian/Kalman update.

        Args:
            variable_name: Variable to update
            observed_value: The observed value
            observation_noise: Noise variance of the observation
            observation_confidence: Confidence in the observation [0, 1]

        Returns:
            Updated belief
        """
        if variable_name not in self.beliefs:
            self.initialize_belief(variable_name, prior_mean=observed_value)

        belief = self.beliefs[variable_name]
        noise = observation_noise or self.config.observation_noise_default

        # Scale noise inversely with observation confidence
        effective_noise = noise / max(0.01, observation_confidence)

        # Kalman-style Bayesian update
        # Prior: N(belief.mean, belief.variance)
        # Observation: N(observed_value, effective_noise)
        # Posterior: N(posterior_mean, posterior_variance)

        prior_precision = 1.0 / max(1e-10, belief.variance)
        obs_precision = 1.0 / max(1e-10, effective_noise)

        posterior_precision = prior_precision + obs_precision
        posterior_variance = 1.0 / posterior_precision
        posterior_mean = (
            prior_precision * belief.mean + obs_precision * observed_value
        ) / posterior_precision

        # Add process noise (environment changes over time)
        posterior_variance += self.config.kalman_process_noise

        # Update confidence
        innovation = abs(observed_value - belief.mean)
        expected_innovation = (belief.variance + effective_noise) ** 0.5
        if expected_innovation > 0:
            normalized_innovation = innovation / expected_innovation
            # High innovation relative to expected -> lower confidence
            confidence_update = math.exp(-0.5 * normalized_innovation ** 2)
        else:
            confidence_update = 1.0

        new_confidence = (
            belief.confidence * 0.7 + confidence_update * 0.3
        ) * observation_confidence

        # Clamp values
        posterior_variance = min(self.config.max_variance, max(1e-10, posterior_variance))
        new_confidence = max(self.config.min_confidence, min(1.0, new_confidence))

        # Update belief
        belief.mean = posterior_mean
        belief.variance = posterior_variance
        belief.confidence = new_confidence
        belief.last_observation = observed_value
        belief.update_count += 1

        return belief

    def update_batch(
        self, observations: list[dict[str, Any]]
    ) -> dict[str, Belief]:
        """
        Update beliefs for multiple observations at once.

        Args:
            observations: List of {variable_name, value, noise, confidence}

        Returns:
            Dict of updated beliefs
        """
        updated = {}
        for obs in observations:
            name = obs["variable_name"]
            value = obs["value"]
            noise = obs.get("noise")
            confidence = obs.get("confidence", 1.0)

            if value is not None:
                belief = self.update(name, value, noise, confidence)
                updated[name] = belief

        self.step += 1
        return updated

    def decay_confidence(self) -> None:
        """
        Decay confidence on all beliefs that weren't updated this step.
        Beliefs that are not refreshed become less trustworthy over time.
        """
        for belief in self.beliefs.values():
            belief.confidence = max(
                self.config.min_confidence,
                belief.confidence * (1.0 - self.config.confidence_decay_rate),
            )
            # Variance grows when not observed
            belief.variance = min(
                self.config.max_variance,
                belief.variance + self.config.kalman_process_noise,
            )

    def get_belief_state(self) -> dict[str, dict[str, Any]]:
        """Return the full belief state as a dictionary."""
        return {name: b.to_dict() for name, b in self.beliefs.items()}

    def get_uncertain_variables(
        self, threshold: float = 0.3
    ) -> list[str]:
        """Return variables where confidence is below threshold."""
        return [
            name
            for name, b in self.beliefs.items()
            if b.confidence < threshold
        ]

    def get_high_variance_variables(
        self, threshold: float | None = None
    ) -> list[str]:
        """Return variables with high uncertainty (variance)."""
        threshold = threshold or self.config.prior_variance * 2
        return [
            name
            for name, b in self.beliefs.items()
            if b.variance > threshold
        ]

    def compute_overall_uncertainty(self) -> float:
        """Compute aggregate uncertainty across all beliefs."""
        if not self.beliefs:
            return 1.0

        avg_confidence = sum(
            b.confidence for b in self.beliefs.values()
        ) / len(self.beliefs)

        return 1.0 - avg_confidence

    def reset_belief(self, variable_name: str) -> None:
        """Reset a belief to its uninformed prior."""
        if variable_name in self.beliefs:
            b = self.beliefs[variable_name]
            b.variance = self.config.prior_variance
            b.confidence = 0.5
            b.update_count = 0
