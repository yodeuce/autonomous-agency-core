"""
FILE 16: risk_model.py
PURPOSE: Explicit downside modeling
ROLE: Separates value from survivability

Includes:
- CVaR (Conditional Value at Risk)
- Worst-case loss
- Constraint violation probability
- Stress scenarios
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a decision."""
    var: float  # Value at Risk
    cvar: float  # Conditional Value at Risk (Expected Shortfall)
    worst_case_loss: float
    expected_loss: float
    loss_probability: float
    constraint_violation_probability: float
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown: float = 0.0
    risk_level: str = "low"  # low | medium | high | critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "var": self.var,
            "cvar": self.cvar,
            "worst_case_loss": self.worst_case_loss,
            "expected_loss": self.expected_loss,
            "loss_probability": self.loss_probability,
            "constraint_violation_probability": self.constraint_violation_probability,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "risk_level": self.risk_level,
        }


@dataclass
class StressScenario:
    """A stress test scenario."""
    name: str
    description: str
    probability: float
    payoff: float
    affected_variables: list[str] = field(default_factory=list)
    severity: str = "moderate"  # mild | moderate | severe | catastrophic


@dataclass
class RiskConfig:
    """Configuration for the risk model."""
    var_confidence: float = 0.95  # VaR confidence level
    cvar_confidence: float = 0.95
    max_acceptable_loss: float = -50.0
    max_constraint_violation_prob: float = 0.01
    risk_free_rate: float = 0.0
    risk_thresholds: dict[str, float] = field(default_factory=lambda: {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
        "critical": 1.0,
    })


class RiskModel:
    """
    Models and quantifies downside risk for agent decisions.
    Separates value optimization from survivability analysis.
    """

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self.stress_scenarios: list[StressScenario] = []
        self.risk_history: list[RiskMetrics] = []

    def compute_risk_metrics(
        self,
        outcomes: list[dict[str, float]],
        constraints: list[dict[str, Any]] | None = None,
    ) -> RiskMetrics:
        """
        Compute comprehensive risk metrics for a set of outcomes.

        Args:
            outcomes: List of {probability, payoff} dicts
            constraints: Optional list of constraint definitions

        Returns:
            RiskMetrics with full risk analysis
        """
        if not outcomes:
            return RiskMetrics(
                var=0, cvar=0, worst_case_loss=0, expected_loss=0,
                loss_probability=0, constraint_violation_probability=0,
                risk_level="low",
            )

        # Sort by payoff for quantile calculations
        sorted_outcomes = sorted(outcomes, key=lambda o: o["payoff"])
        payoffs = [o["payoff"] for o in sorted_outcomes]
        probs = [o["probability"] for o in sorted_outcomes]

        # VaR
        var = self._compute_var(sorted_outcomes)

        # CVaR
        cvar = self._compute_cvar(sorted_outcomes)

        # Worst-case loss
        worst_case = min(payoffs)

        # Expected loss (expected value of negative outcomes only)
        expected_loss = sum(
            o["probability"] * o["payoff"]
            for o in outcomes
            if o["payoff"] < 0
        )

        # Probability of any loss
        loss_prob = sum(o["probability"] for o in outcomes if o["payoff"] < 0)

        # Constraint violation probability
        cv_prob = self._compute_constraint_violation_prob(outcomes, constraints)

        # Risk-adjusted return ratios
        expected_return = sum(o["probability"] * o["payoff"] for o in outcomes)
        std_dev = self._compute_std(outcomes, expected_return)

        sharpe = None
        if std_dev > 0:
            sharpe = (expected_return - self.config.risk_free_rate) / std_dev

        sortino = None
        downside_std = self._compute_downside_std(outcomes, expected_return)
        if downside_std > 0:
            sortino = (expected_return - self.config.risk_free_rate) / downside_std

        # Classify risk level
        risk_score = self._compute_risk_score(loss_prob, cvar, cv_prob)
        risk_level = self._classify_risk(risk_score)

        metrics = RiskMetrics(
            var=var,
            cvar=cvar,
            worst_case_loss=worst_case,
            expected_loss=expected_loss,
            loss_probability=loss_prob,
            constraint_violation_probability=cv_prob,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            risk_level=risk_level,
        )

        self.risk_history.append(metrics)
        return metrics

    def add_stress_scenario(self, scenario: StressScenario) -> None:
        """Register a stress test scenario."""
        self.stress_scenarios.append(scenario)

    def run_stress_tests(
        self,
        base_outcomes: list[dict[str, float]],
    ) -> list[dict[str, Any]]:
        """
        Run all registered stress scenarios against base outcomes.

        Returns:
            List of stress test results
        """
        results = []
        for scenario in self.stress_scenarios:
            stressed = list(base_outcomes)
            stressed.append({
                "probability": scenario.probability,
                "payoff": scenario.payoff,
            })

            # Renormalize
            total = sum(o["probability"] for o in stressed)
            for o in stressed:
                o["probability"] /= total

            metrics = self.compute_risk_metrics(stressed)
            results.append({
                "scenario": scenario.name,
                "severity": scenario.severity,
                "metrics": metrics.to_dict(),
                "passes": metrics.cvar > self.config.max_acceptable_loss,
            })

        return results

    def is_acceptable(self, metrics: RiskMetrics) -> bool:
        """Check if risk metrics are within acceptable bounds."""
        if metrics.worst_case_loss < self.config.max_acceptable_loss:
            return False
        if metrics.constraint_violation_probability > self.config.max_constraint_violation_prob:
            return False
        if metrics.risk_level == "critical":
            return False
        return True

    # -------------------------------------------------------------------------
    # INTERNAL COMPUTATIONS
    # -------------------------------------------------------------------------

    def _compute_var(self, sorted_outcomes: list[dict[str, float]]) -> float:
        """Compute Value at Risk at configured confidence level."""
        alpha = 1.0 - self.config.var_confidence
        cumulative = 0.0

        for outcome in sorted_outcomes:
            cumulative += outcome["probability"]
            if cumulative >= alpha:
                return outcome["payoff"]

        return sorted_outcomes[0]["payoff"]

    def _compute_cvar(self, sorted_outcomes: list[dict[str, float]]) -> float:
        """Compute Conditional Value at Risk (Expected Shortfall)."""
        alpha = 1.0 - self.config.cvar_confidence
        cumulative = 0.0
        tail_sum = 0.0
        tail_prob = 0.0

        for outcome in sorted_outcomes:
            if cumulative + outcome["probability"] <= alpha:
                tail_sum += outcome["probability"] * outcome["payoff"]
                tail_prob += outcome["probability"]
                cumulative += outcome["probability"]
            elif cumulative < alpha:
                remaining = alpha - cumulative
                tail_sum += remaining * outcome["payoff"]
                tail_prob += remaining
                cumulative = alpha
                break
            else:
                break

        return tail_sum / tail_prob if tail_prob > 0 else sorted_outcomes[0]["payoff"]

    def _compute_constraint_violation_prob(
        self,
        outcomes: list[dict[str, float]],
        constraints: list[dict[str, Any]] | None,
    ) -> float:
        """Compute probability of violating any constraint."""
        if not constraints:
            return 0.0

        violation_prob = 0.0
        for outcome in outcomes:
            for constraint in constraints:
                threshold = constraint.get("threshold", self.config.max_acceptable_loss)
                if outcome["payoff"] < threshold:
                    violation_prob += outcome["probability"]
                    break  # Count once per outcome

        return min(1.0, violation_prob)

    def _compute_std(
        self, outcomes: list[dict[str, float]], mean: float
    ) -> float:
        variance = sum(
            o["probability"] * (o["payoff"] - mean) ** 2 for o in outcomes
        )
        return variance ** 0.5

    def _compute_downside_std(
        self, outcomes: list[dict[str, float]], target: float
    ) -> float:
        downside_var = sum(
            o["probability"] * min(0, o["payoff"] - target) ** 2
            for o in outcomes
        )
        return downside_var ** 0.5

    def _compute_risk_score(
        self, loss_prob: float, cvar: float, cv_prob: float
    ) -> float:
        score = loss_prob * 0.3
        if cvar < 0:
            score += min(0.4, abs(cvar) / abs(self.config.max_acceptable_loss) * 0.4)
        score += cv_prob * 0.3
        return min(1.0, score)

    def _classify_risk(self, score: float) -> str:
        thresholds = self.config.risk_thresholds
        if score < thresholds.get("low", 0.2):
            return "low"
        elif score < thresholds.get("medium", 0.5):
            return "medium"
        elif score < thresholds.get("high", 0.8):
            return "high"
        return "critical"
