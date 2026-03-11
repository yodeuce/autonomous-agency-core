"""
FILE 18: failure_analysis_module.py
PURPOSE: Post-decision diagnostics
ROLE: Prevents repeating mistakes

Captures:
- Expected vs realized EMV
- Root cause analysis
- Memory reinforcement
- Policy penalties
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    EMV_OVERESTIMATE = "emv_overestimate"
    EMV_UNDERESTIMATE = "emv_underestimate"
    RISK_UNDERESTIMATE = "risk_underestimate"
    CONSTRAINT_VIOLATION = "constraint_violation"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    MEMORY_FAILURE = "memory_failure"
    POLICY_FAILURE = "policy_failure"
    EXECUTION_ERROR = "execution_error"
    EXTERNAL_SHOCK = "external_shock"
    INFORMATION_FAILURE = "information_failure"
    ESTIMATION_ERROR = "estimation_error"
    GOAL_MISALIGNMENT = "goal_misalignment"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureAnalysis:
    """Complete analysis of a decision failure."""
    analysis_id: str
    step: int
    timestamp: str
    action: str
    expected_emv: float
    realized_emv: float
    emv_gap: float
    category: FailureCategory
    severity: Severity
    root_causes: list[str]
    contributing_factors: list[str]
    memory_reinforcements: list[dict[str, Any]]
    policy_adjustments: list[dict[str, Any]]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "step": self.step,
            "timestamp": self.timestamp,
            "action": self.action,
            "expected_emv": self.expected_emv,
            "realized_emv": self.realized_emv,
            "emv_gap": self.emv_gap,
            "category": self.category.value,
            "severity": self.severity.value,
            "root_causes": self.root_causes,
            "contributing_factors": self.contributing_factors,
            "recommendations": self.recommendations,
        }


@dataclass
class FailureAnalysisConfig:
    """Configuration for the failure analysis module."""
    emv_gap_threshold: float = 0.2  # Minimum gap to trigger analysis
    severity_thresholds: dict[str, float] = field(default_factory=lambda: {
        "low": 0.1,
        "medium": 0.3,
        "high": 0.6,
        "critical": 0.9,
    })
    max_root_causes: int = 5
    auto_penalize_policy: bool = True
    auto_reinforce_memory: bool = True


class FailureAnalysisModule:
    """
    Post-decision diagnostic engine.
    Analyzes failures, identifies root causes, and produces
    actionable corrections for the policy and memory systems.
    """

    def __init__(self, config: FailureAnalysisConfig | None = None):
        self.config = config or FailureAnalysisConfig()
        self.analysis_history: list[FailureAnalysis] = []
        self.failure_patterns: dict[str, int] = {}
        self.analysis_count: int = 0

    def analyze(
        self,
        step: int,
        action: str,
        expected_emv: float,
        realized_emv: float,
        state: dict[str, Any],
        next_state: dict[str, Any],
        outcome: dict[str, Any],
        retrieved_memories: list[dict[str, Any]] | None = None,
    ) -> FailureAnalysis | None:
        """
        Analyze a decision outcome for failures.

        Returns:
            FailureAnalysis if a failure is detected, else None
        """
        emv_gap = expected_emv - realized_emv

        # Check if this qualifies as a failure
        if abs(emv_gap) < self.config.emv_gap_threshold * max(abs(expected_emv), 1.0):
            return None  # Within acceptable range

        self.analysis_count += 1

        # Categorize
        category = self._categorize_failure(emv_gap, state, next_state, outcome)

        # Assess severity
        severity = self._assess_severity(emv_gap, expected_emv, outcome)

        # Root cause analysis
        root_causes = self._identify_root_causes(
            emv_gap, state, next_state, outcome, retrieved_memories
        )

        # Contributing factors
        contributing = self._identify_contributing_factors(
            state, next_state, outcome
        )

        # Generate memory reinforcements
        memory_reinforcements = self._generate_memory_reinforcements(
            category, root_causes, outcome
        )

        # Generate policy adjustments
        policy_adjustments = self._generate_policy_adjustments(
            action, category, severity, emv_gap
        )

        # Recommendations
        recommendations = self._generate_recommendations(
            category, severity, root_causes
        )

        analysis = FailureAnalysis(
            analysis_id=f"FA-{self.analysis_count:06d}",
            step=step,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            expected_emv=expected_emv,
            realized_emv=realized_emv,
            emv_gap=emv_gap,
            category=category,
            severity=severity,
            root_causes=root_causes,
            contributing_factors=contributing,
            memory_reinforcements=memory_reinforcements,
            policy_adjustments=policy_adjustments,
            recommendations=recommendations,
        )

        self.analysis_history.append(analysis)
        self._track_pattern(category)

        logger.warning(
            f"Failure detected: {category.value} (severity={severity.value}, "
            f"EMV gap={emv_gap:.2f})"
        )
        return analysis

    def get_recurring_failures(self, min_count: int = 3) -> dict[str, int]:
        """Identify failure patterns that keep recurring."""
        return {
            k: v for k, v in self.failure_patterns.items() if v >= min_count
        }

    def get_failure_summary(self) -> dict[str, Any]:
        """Summary statistics of all failures analyzed."""
        if not self.analysis_history:
            return {"total": 0}

        categories = {}
        severities = {}
        total_gap = 0.0

        for a in self.analysis_history:
            categories[a.category.value] = categories.get(a.category.value, 0) + 1
            severities[a.severity.value] = severities.get(a.severity.value, 0) + 1
            total_gap += abs(a.emv_gap)

        return {
            "total": len(self.analysis_history),
            "by_category": categories,
            "by_severity": severities,
            "avg_emv_gap": total_gap / len(self.analysis_history),
            "recurring_patterns": self.get_recurring_failures(),
        }

    def get_policy_response(self, category: FailureCategory) -> dict[str, Any]:
        """
        Get the recommended policy response for a failure category (CARBON[6] §6.2).

        Policy Responses:
            | Category             | Response                              |
            |---------------------|---------------------------------------|
            | Information Failure | Increase observation frequency         |
            | Estimation Error    | Adjust model parameters, increase data |
            | Execution Failure   | Retry with modified parameters         |
            | Goal Misalignment   | Escalate for human review             |
        """
        responses = {
            FailureCategory.INFORMATION_FAILURE: {
                "action": "increase_observation",
                "description": "Increase observation frequency and broaden data sources",
                "urgency": "medium",
            },
            FailureCategory.ESTIMATION_ERROR: {
                "action": "adjust_model",
                "description": "Adjust model parameters and increase training data",
                "urgency": "medium",
            },
            FailureCategory.EXECUTION_ERROR: {
                "action": "retry_modified",
                "description": "Retry with modified parameters or alternative approach",
                "urgency": "high",
            },
            FailureCategory.GOAL_MISALIGNMENT: {
                "action": "escalate",
                "description": "Escalate to human operator for objective review",
                "urgency": "critical",
            },
            FailureCategory.CONSTRAINT_VIOLATION: {
                "action": "halt_and_review",
                "description": "Halt action and review constraint enforcement",
                "urgency": "critical",
            },
        }
        return responses.get(category, {
            "action": "investigate",
            "description": "Investigate failure root cause",
            "urgency": "low",
        })

    # -------------------------------------------------------------------------
    # CATEGORIZATION
    # -------------------------------------------------------------------------

    def _categorize_failure(
        self,
        emv_gap: float,
        state: dict[str, Any],
        next_state: dict[str, Any],
        outcome: dict[str, Any],
    ) -> FailureCategory:
        if outcome.get("constraint_violations"):
            return FailureCategory.CONSTRAINT_VIOLATION
        if outcome.get("execution_error"):
            return FailureCategory.EXECUTION_ERROR
        if outcome.get("external_shock"):
            return FailureCategory.EXTERNAL_SHOCK

        risk_before = state.get("risk_level", 0)
        risk_after = next_state.get("risk_level", 0)
        if risk_after > risk_before * 1.5:
            return FailureCategory.RISK_UNDERESTIMATE

        if emv_gap > 0:
            return FailureCategory.EMV_OVERESTIMATE
        return FailureCategory.EMV_UNDERESTIMATE

    def _assess_severity(
        self, emv_gap: float, expected_emv: float, outcome: dict[str, Any]
    ) -> Severity:
        if outcome.get("constraint_violations"):
            return Severity.CRITICAL

        relative_gap = abs(emv_gap) / max(abs(expected_emv), 1.0)
        thresholds = self.config.severity_thresholds

        if relative_gap >= thresholds.get("critical", 0.9):
            return Severity.CRITICAL
        elif relative_gap >= thresholds.get("high", 0.6):
            return Severity.HIGH
        elif relative_gap >= thresholds.get("medium", 0.3):
            return Severity.MEDIUM
        return Severity.LOW

    # -------------------------------------------------------------------------
    # ROOT CAUSE ANALYSIS
    # -------------------------------------------------------------------------

    def _identify_root_causes(
        self,
        emv_gap: float,
        state: dict[str, Any],
        next_state: dict[str, Any],
        outcome: dict[str, Any],
        memories: list[dict[str, Any]] | None,
    ) -> list[str]:
        causes = []

        if outcome.get("environment_changed"):
            causes.append("Environment shifted between decision and outcome")

        if state.get("uncertainty", 0) > 0.5:
            causes.append("Decision made under high uncertainty")

        if memories and len(memories) == 0:
            causes.append("No relevant memories retrieved for decision")

        if outcome.get("execution_error"):
            causes.append(f"Execution error: {outcome['execution_error']}")

        if abs(emv_gap) > 0 and state.get("confidence", 1.0) > 0.8:
            causes.append("Overconfidence in predictions")

        if not causes:
            causes.append("Root cause undetermined - requires manual review")

        return causes[: self.config.max_root_causes]

    def _identify_contributing_factors(
        self,
        state: dict[str, Any],
        next_state: dict[str, Any],
        outcome: dict[str, Any],
    ) -> list[str]:
        factors = []
        if state.get("system_load", 0) > 0.8:
            factors.append("High system load at decision time")
        if state.get("time_pressure", False):
            factors.append("Decision made under time pressure")
        if outcome.get("data_quality", 1.0) < 0.5:
            factors.append("Low quality input data")
        return factors

    # -------------------------------------------------------------------------
    # CORRECTIONS
    # -------------------------------------------------------------------------

    def _generate_memory_reinforcements(
        self,
        category: FailureCategory,
        root_causes: list[str],
        outcome: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.config.auto_reinforce_memory:
            return []

        reinforcements = []
        reinforcements.append({
            "type": "create_memory",
            "memory_type": "episodic",
            "content": f"Failure ({category.value}): {'; '.join(root_causes)}",
            "salience_boost": 0.3,
            "tags": ["failure", category.value],
        })

        if category == FailureCategory.CONSTRAINT_VIOLATION:
            reinforcements.append({
                "type": "reinforce_existing",
                "memory_type": "constraint",
                "salience_boost": 0.5,
                "reason": "Constraint was violated - reinforce constraint memories",
            })

        return reinforcements

    def _generate_policy_adjustments(
        self,
        action: str,
        category: FailureCategory,
        severity: Severity,
        emv_gap: float,
    ) -> list[dict[str, Any]]:
        if not self.config.auto_penalize_policy:
            return []

        adjustments = []

        # Penalty proportional to severity
        penalty_map = {
            Severity.LOW: 0.01,
            Severity.MEDIUM: 0.05,
            Severity.HIGH: 0.1,
            Severity.CRITICAL: 0.2,
        }
        penalty = penalty_map.get(severity, 0.05)

        adjustments.append({
            "type": "action_penalty",
            "action": action,
            "penalty": penalty,
            "reason": f"{category.value} failure",
        })

        if category == FailureCategory.RISK_UNDERESTIMATE:
            adjustments.append({
                "type": "increase_exploration",
                "factor": 1.2,
                "reason": "Risk was underestimated - explore more",
            })

        return adjustments

    def _generate_recommendations(
        self,
        category: FailureCategory,
        severity: Severity,
        root_causes: list[str],
    ) -> list[str]:
        recs = []

        if category == FailureCategory.EMV_OVERESTIMATE:
            recs.append("Increase pessimism in outcome probability estimates")
        elif category == FailureCategory.RISK_UNDERESTIMATE:
            recs.append("Lower risk threshold or increase risk model sensitivity")
        elif category == FailureCategory.CONSTRAINT_VIOLATION:
            recs.append("Review and strengthen constraint enforcement")
        elif category == FailureCategory.ENVIRONMENT_MISMATCH:
            recs.append("Increase environment observation frequency")

        if severity in (Severity.HIGH, Severity.CRITICAL):
            recs.append("Consider human review of similar future decisions")

        return recs

    def _track_pattern(self, category: FailureCategory) -> None:
        key = category.value
        self.failure_patterns[key] = self.failure_patterns.get(key, 0) + 1
