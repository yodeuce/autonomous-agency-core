"""
RISK MODEL MODULE
Explicit downside modeling for autonomous agents
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Includes:
- CVaR (Conditional Value at Risk)
- Worst-case loss analysis
- Constraint violation probability
- Stress scenarios

Separates value from survivability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict


class RiskCategory(Enum):
    """Categories of risk."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    REPUTATIONAL = "reputational"
    COMPLIANCE = "compliance"
    STRATEGIC = "strategic"
    SAFETY = "safety"


class RiskLevel(Enum):
    """Risk level classifications."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional VaR (Expected Shortfall) at 95%
    cvar_99: float  # Conditional VaR at 99%
    worst_case_loss: float
    expected_loss: float
    probability_of_loss: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: Optional[float] = None


@dataclass
class RiskAssessment:
    """Complete risk assessment for an action."""
    action: str
    risk_level: RiskLevel
    risk_score: float  # 0-1 composite score
    metrics: RiskMetrics
    constraint_violation_prob: float
    stress_test_results: Dict[str, float]
    risk_factors: Dict[RiskCategory, float]
    mitigations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskConfig:
    """Configuration for risk model."""
    # VaR/CVaR parameters
    var_confidence_levels: List[float] = field(
        default_factory=lambda: [0.95, 0.99]
    )

    # Risk thresholds
    risk_tolerance: float = 0.3  # Maximum acceptable risk score
    max_acceptable_loss: float = -1000
    constraint_violation_limit: float = 0.05

    # Stress test parameters
    stress_multiplier: float = 2.0

    # Risk factor weights
    category_weights: Dict[RiskCategory, float] = field(
        default_factory=lambda: {
            RiskCategory.SAFETY: 0.25,
            RiskCategory.COMPLIANCE: 0.20,
            RiskCategory.FINANCIAL: 0.20,
            RiskCategory.OPERATIONAL: 0.15,
            RiskCategory.REPUTATIONAL: 0.10,
            RiskCategory.STRATEGIC: 0.10
        }
    )


class RiskModel:
    """
    Comprehensive risk modeling for autonomous agent decisions.
    Separates value optimization from survivability constraints.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self._risk_history: List[RiskAssessment] = []
        self._constraint_violations: List[Tuple[datetime, str]] = []

    def assess_risk(
        self,
        action: str,
        outcomes: List[Tuple[float, float]],  # (value, probability) pairs
        constraints: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment for an action.

        Args:
            action: Action being assessed
            outcomes: List of (value, probability) pairs
            constraints: Constraint thresholds
            context: Additional context

        Returns:
            Complete RiskAssessment
        """
        context = context or {}
        constraints = constraints or {}

        # Calculate core risk metrics
        metrics = self._calculate_risk_metrics(outcomes)

        # Calculate constraint violation probability
        constraint_prob = self._calculate_constraint_violation_prob(
            outcomes, constraints
        )

        # Run stress tests
        stress_results = self._run_stress_tests(outcomes, context)

        # Calculate risk factors by category
        risk_factors = self._assess_risk_factors(
            outcomes, metrics, context
        )

        # Calculate composite risk score
        risk_score = self._calculate_risk_score(
            metrics, constraint_prob, risk_factors
        )

        # Determine risk level
        risk_level = self._classify_risk_level(risk_score)

        # Generate mitigations
        mitigations = self._suggest_mitigations(
            risk_level, risk_factors, metrics
        )

        assessment = RiskAssessment(
            action=action,
            risk_level=risk_level,
            risk_score=risk_score,
            metrics=metrics,
            constraint_violation_prob=constraint_prob,
            stress_test_results=stress_results,
            risk_factors=risk_factors,
            mitigations=mitigations
        )

        self._risk_history.append(assessment)
        return assessment

    def _calculate_risk_metrics(
        self,
        outcomes: List[Tuple[float, float]]
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        values = [v for v, p in outcomes]
        probs = [p for v, p in outcomes]

        # Normalize probabilities
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]

        # Expected value
        expected = sum(v * p for v, p in zip(values, probs))

        # Sort for VaR calculations
        sorted_outcomes = sorted(zip(values, probs), key=lambda x: x[0])

        # Calculate VaR at different confidence levels
        var_95 = self._calculate_var(sorted_outcomes, 0.95)
        var_99 = self._calculate_var(sorted_outcomes, 0.99)

        # Calculate CVaR (Expected Shortfall)
        cvar_95 = self._calculate_cvar(sorted_outcomes, 0.95)
        cvar_99 = self._calculate_cvar(sorted_outcomes, 0.99)

        # Worst case and expected loss
        worst_case = min(values)
        losses = [v for v in values if v < 0]
        loss_probs = [p for v, p in zip(values, probs) if v < 0]
        expected_loss = sum(l * p for l, p in zip(losses, loss_probs)) if losses else 0

        # Probability of loss
        prob_loss = sum(p for v, p in zip(values, probs) if v < 0)

        # Volatility (standard deviation)
        variance = sum(p * (v - expected) ** 2 for v, p in zip(values, probs))
        volatility = np.sqrt(variance)

        # Max drawdown (simplified: max loss from peak)
        max_drawdown = abs(min(0, worst_case))

        # Sharpe ratio (if positive expected value)
        risk_free = 0.02
        sharpe = None
        if volatility > 0:
            sharpe = (expected - risk_free) / volatility

        return RiskMetrics(
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            cvar_95=round(cvar_95, 2),
            cvar_99=round(cvar_99, 2),
            worst_case_loss=round(worst_case, 2),
            expected_loss=round(expected_loss, 2),
            probability_of_loss=round(prob_loss, 4),
            max_drawdown=round(max_drawdown, 2),
            volatility=round(volatility, 2),
            sharpe_ratio=round(sharpe, 3) if sharpe else None
        )

    def _calculate_var(
        self,
        sorted_outcomes: List[Tuple[float, float]],
        confidence: float
    ) -> float:
        """
        Calculate Value at Risk at given confidence level.
        VaR is the maximum loss at the given confidence level.
        """
        target_prob = 1 - confidence
        cumulative = 0

        for value, prob in sorted_outcomes:
            cumulative += prob
            if cumulative >= target_prob:
                return -value if value < 0 else 0

        return 0

    def _calculate_cvar(
        self,
        sorted_outcomes: List[Tuple[float, float]],
        confidence: float
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        CVaR is the expected loss given that loss exceeds VaR.
        """
        target_prob = 1 - confidence
        cumulative = 0
        tail_values = []
        tail_probs = []

        for value, prob in sorted_outcomes:
            if cumulative < target_prob:
                remaining_prob = min(prob, target_prob - cumulative)
                tail_values.append(value)
                tail_probs.append(remaining_prob)
            cumulative += prob

        if not tail_values:
            return 0

        total_tail_prob = sum(tail_probs)
        if total_tail_prob == 0:
            return 0

        cvar = sum(v * p for v, p in zip(tail_values, tail_probs)) / total_tail_prob
        return -cvar if cvar < 0 else 0

    def _calculate_constraint_violation_prob(
        self,
        outcomes: List[Tuple[float, float]],
        constraints: Dict[str, float]
    ) -> float:
        """Calculate probability of constraint violation."""
        if not constraints:
            return 0.0

        violation_prob = 0

        # Check each constraint
        min_acceptable = constraints.get("min_value", float("-inf"))
        max_risk = constraints.get("max_risk", float("inf"))

        for value, prob in outcomes:
            if value < min_acceptable:
                violation_prob += prob
            if abs(value) > max_risk:
                violation_prob += prob * 0.5  # Partial violation

        return min(1.0, violation_prob)

    def _run_stress_tests(
        self,
        outcomes: List[Tuple[float, float]],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Run stress test scenarios."""
        results = {}
        multiplier = self.config.stress_multiplier

        # Scenario 1: Increased volatility
        stressed_outcomes = [
            (v * multiplier if v < 0 else v, p)
            for v, p in outcomes
        ]
        metrics = self._calculate_risk_metrics(stressed_outcomes)
        results["high_volatility"] = metrics.cvar_95

        # Scenario 2: Probability shift to negative outcomes
        shifted_outcomes = []
        for v, p in outcomes:
            if v < 0:
                shifted_outcomes.append((v, min(1.0, p * 1.5)))
            else:
                shifted_outcomes.append((v, p * 0.7))
        # Re-normalize
        total = sum(p for _, p in shifted_outcomes)
        shifted_outcomes = [(v, p / total) for v, p in shifted_outcomes]
        metrics = self._calculate_risk_metrics(shifted_outcomes)
        results["pessimistic"] = metrics.expected_loss

        # Scenario 3: Black swan (worst case with higher probability)
        worst_value = min(v for v, _ in outcomes)
        swan_outcomes = outcomes + [(worst_value * 2, 0.05)]
        total = sum(p for _, p in swan_outcomes)
        swan_outcomes = [(v, p / total) for v, p in swan_outcomes]
        metrics = self._calculate_risk_metrics(swan_outcomes)
        results["black_swan"] = metrics.worst_case_loss

        return results

    def _assess_risk_factors(
        self,
        outcomes: List[Tuple[float, float]],
        metrics: RiskMetrics,
        context: Dict[str, Any]
    ) -> Dict[RiskCategory, float]:
        """Assess risk by category."""
        factors = {}

        # Financial risk (based on loss metrics)
        financial_risk = min(1.0, (
            abs(metrics.cvar_95) / 1000 +
            metrics.probability_of_loss
        ) / 2)
        factors[RiskCategory.FINANCIAL] = financial_risk

        # Operational risk (from context)
        operational_risk = context.get("operational_risk", 0.3)
        factors[RiskCategory.OPERATIONAL] = operational_risk

        # Compliance risk (constraint violations)
        compliance_risk = context.get("compliance_risk", 0.2)
        factors[RiskCategory.COMPLIANCE] = compliance_risk

        # Reputational risk
        reputational_risk = context.get("reputational_risk", 0.2)
        if metrics.probability_of_loss > 0.3:
            reputational_risk += 0.2
        factors[RiskCategory.REPUTATIONAL] = min(1.0, reputational_risk)

        # Strategic risk
        strategic_risk = context.get("strategic_risk", 0.2)
        factors[RiskCategory.STRATEGIC] = strategic_risk

        # Safety risk
        safety_risk = context.get("safety_risk", 0.1)
        if metrics.worst_case_loss < self.config.max_acceptable_loss:
            safety_risk += 0.3
        factors[RiskCategory.SAFETY] = min(1.0, safety_risk)

        return factors

    def _calculate_risk_score(
        self,
        metrics: RiskMetrics,
        constraint_prob: float,
        risk_factors: Dict[RiskCategory, float]
    ) -> float:
        """Calculate composite risk score (0-1)."""
        # Weighted risk factors
        weighted_factors = sum(
            risk_factors.get(cat, 0) * weight
            for cat, weight in self.config.category_weights.items()
        )

        # Metrics-based component
        metrics_score = (
            min(1.0, abs(metrics.cvar_95) / 500) * 0.3 +
            metrics.probability_of_loss * 0.3 +
            min(1.0, metrics.volatility / 100) * 0.2 +
            constraint_prob * 0.2
        )

        # Combine
        risk_score = weighted_factors * 0.5 + metrics_score * 0.5

        return round(min(1.0, risk_score), 3)

    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk level from score."""
        if risk_score < 0.1:
            return RiskLevel.MINIMAL
        elif risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MODERATE
        elif risk_score < 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _suggest_mitigations(
        self,
        risk_level: RiskLevel,
        risk_factors: Dict[RiskCategory, float],
        metrics: RiskMetrics
    ) -> List[str]:
        """Suggest risk mitigations."""
        mitigations = []

        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            mitigations.append("ESCALATE: Risk level requires human oversight")

        # Category-specific mitigations
        for category, score in risk_factors.items():
            if score > 0.5:
                if category == RiskCategory.FINANCIAL:
                    mitigations.append("Consider smaller position sizing")
                    mitigations.append("Set stop-loss parameters")
                elif category == RiskCategory.COMPLIANCE:
                    mitigations.append("Review regulatory requirements")
                    mitigations.append("Document decision rationale")
                elif category == RiskCategory.SAFETY:
                    mitigations.append("Enable additional safety checks")
                    mitigations.append("Require confirmation for execution")
                elif category == RiskCategory.OPERATIONAL:
                    mitigations.append("Verify system health before proceeding")
                    mitigations.append("Have fallback plan ready")

        # Metrics-based mitigations
        if metrics.probability_of_loss > 0.5:
            mitigations.append("High loss probability - consider alternatives")

        if metrics.cvar_95 < -500:
            mitigations.append("Severe tail risk - implement hedging")

        return mitigations

    def is_within_tolerance(self, assessment: RiskAssessment) -> bool:
        """Check if risk is within tolerance."""
        return (
            assessment.risk_score <= self.config.risk_tolerance and
            assessment.constraint_violation_prob <= self.config.constraint_violation_limit and
            assessment.metrics.worst_case_loss >= self.config.max_acceptable_loss
        )

    def record_constraint_violation(self, constraint_name: str):
        """Record a constraint violation."""
        self._constraint_violations.append((datetime.now(), constraint_name))

    def get_risk_trend(self, lookback: int = 10) -> Dict[str, Any]:
        """Analyze recent risk trend."""
        recent = self._risk_history[-lookback:] if self._risk_history else []

        if not recent:
            return {"trend": "unknown", "average_score": 0}

        scores = [a.risk_score for a in recent]
        avg_score = np.mean(scores)

        if len(scores) >= 3:
            if scores[-1] > scores[0] * 1.2:
                trend = "increasing"
            elif scores[-1] < scores[0] * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "trend": trend,
            "average_score": round(avg_score, 3),
            "recent_scores": scores,
            "violations": len(self._constraint_violations)
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_risk_model(
    config: Optional[RiskConfig] = None
) -> RiskModel:
    """Create a risk model."""
    return RiskModel(config)
