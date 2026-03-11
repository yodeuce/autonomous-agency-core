"""
FAILURE ANALYSIS MODULE
Post-decision diagnostics and learning from failures
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Captures:
- Expected vs realized EMV
- Root cause analysis
- Memory reinforcement
- Policy penalties

Prevents repeating mistakes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
from collections import defaultdict
import numpy as np


class FailureType(Enum):
    """Types of failures."""
    OUTCOME_DEVIATION = "outcome_deviation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CASCADING_FAILURE = "cascading_failure"
    USER_REJECTION = "user_rejection"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


class RootCauseCategory(Enum):
    """Categories of root causes."""
    INCORRECT_STATE_ESTIMATE = "incorrect_state_estimate"
    INCORRECT_PROBABILITY = "incorrect_probability"
    INCOMPLETE_INFORMATION = "incomplete_information"
    POLICY_ERROR = "policy_error"
    EXECUTION_ERROR = "execution_error"
    EXTERNAL_FACTOR = "external_factor"
    CONSTRAINT_MISSED = "constraint_missed"
    MEMORY_GAP = "memory_gap"


@dataclass
class FailureEvent:
    """Record of a failure event."""
    failure_id: str
    failure_type: FailureType
    action: str
    expected_outcome: Dict[str, Any]
    actual_outcome: Dict[str, Any]
    state_at_failure: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RootCauseAnalysis:
    """Result of root cause analysis."""
    failure_event: FailureEvent
    root_causes: List[Tuple[RootCauseCategory, float]]  # (cause, confidence)
    contributing_factors: List[str]
    emv_deviation: float
    probability_calibration_error: float
    recommended_actions: List[str]
    affected_memories: List[str]
    policy_penalties: Dict[str, float]


@dataclass
class FailurePattern:
    """Pattern of recurring failures."""
    pattern_id: str
    failure_type: FailureType
    common_causes: List[RootCauseCategory]
    frequency: int
    average_impact: float
    trigger_conditions: Dict[str, Any]
    suggested_prevention: List[str]


@dataclass
class AnalysisConfig:
    """Configuration for failure analysis."""
    # EMV deviation thresholds
    significant_deviation_threshold: float = 0.2  # 20% deviation
    severe_deviation_threshold: float = 0.5  # 50% deviation

    # Pattern detection
    min_failures_for_pattern: int = 3
    pattern_similarity_threshold: float = 0.7

    # Penalty parameters
    base_penalty: float = 0.1
    repeat_penalty_multiplier: float = 1.5

    # Memory adjustment
    memory_reinforcement_factor: float = 0.3
    memory_decay_on_failure: float = 0.2

    # Retention
    max_failure_history: int = 1000
    pattern_retention_days: int = 30


class FailureAnalysisModule:
    """
    Module for analyzing failures and preventing repetition.
    Learns from mistakes to improve future decisions.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self._failure_history: List[FailureEvent] = []
        self._analyses: List[RootCauseAnalysis] = []
        self._patterns: Dict[str, FailurePattern] = {}
        self._action_penalties: Dict[str, float] = defaultdict(float)
        self._failure_counts: Dict[str, int] = defaultdict(int)

    def record_failure(
        self,
        action: str,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        state: Dict[str, Any],
        failure_type: Optional[FailureType] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> FailureEvent:
        """Record a failure event."""
        # Auto-detect failure type if not provided
        if failure_type is None:
            failure_type = self._detect_failure_type(expected, actual, context or {})

        failure = FailureEvent(
            failure_id=f"fail_{datetime.now().timestamp()}",
            failure_type=failure_type,
            action=action,
            expected_outcome=expected,
            actual_outcome=actual,
            state_at_failure=state,
            context=context or {}
        )

        self._failure_history.append(failure)
        self._failure_counts[action] += 1

        # Maintain history limit
        if len(self._failure_history) > self.config.max_failure_history:
            self._failure_history = self._failure_history[-self.config.max_failure_history:]

        return failure

    def analyze_failure(
        self,
        failure: FailureEvent
    ) -> RootCauseAnalysis:
        """Perform root cause analysis on a failure."""
        # Calculate EMV deviation
        emv_deviation = self._calculate_emv_deviation(
            failure.expected_outcome,
            failure.actual_outcome
        )

        # Calculate probability calibration error
        calibration_error = self._calculate_calibration_error(
            failure.expected_outcome,
            failure.actual_outcome
        )

        # Identify root causes
        root_causes = self._identify_root_causes(
            failure, emv_deviation, calibration_error
        )

        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(failure)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            failure, root_causes
        )

        # Identify affected memories
        affected_memories = self._identify_affected_memories(failure)

        # Calculate policy penalties
        penalties = self._calculate_penalties(failure, root_causes)
        self._apply_penalties(failure.action, penalties)

        analysis = RootCauseAnalysis(
            failure_event=failure,
            root_causes=root_causes,
            contributing_factors=contributing_factors,
            emv_deviation=emv_deviation,
            probability_calibration_error=calibration_error,
            recommended_actions=recommendations,
            affected_memories=affected_memories,
            policy_penalties=penalties
        )

        self._analyses.append(analysis)
        self._update_patterns(failure, analysis)

        return analysis

    def _detect_failure_type(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        context: Dict[str, Any]
    ) -> FailureType:
        """Auto-detect failure type from outcomes."""
        if context.get("constraint_violated"):
            return FailureType.CONSTRAINT_VIOLATION

        if context.get("timeout"):
            return FailureType.TIMEOUT

        if context.get("resource_exhausted"):
            return FailureType.RESOURCE_EXHAUSTION

        if context.get("user_rejected"):
            return FailureType.USER_REJECTION

        if context.get("system_error"):
            return FailureType.SYSTEM_ERROR

        # Check outcome deviation
        exp_value = expected.get("value", expected.get("emv", 0))
        act_value = actual.get("value", actual.get("realized_value", 0))

        if exp_value != 0:
            deviation = abs(act_value - exp_value) / abs(exp_value)
            if deviation > self.config.significant_deviation_threshold:
                return FailureType.OUTCOME_DEVIATION

        return FailureType.UNKNOWN

    def _calculate_emv_deviation(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> float:
        """Calculate deviation between expected and actual EMV."""
        exp_value = expected.get("emv", expected.get("value", 0))
        act_value = actual.get("realized_value", actual.get("value", 0))

        if exp_value == 0:
            return abs(act_value)

        return (act_value - exp_value) / abs(exp_value)

    def _calculate_calibration_error(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> float:
        """Calculate probability calibration error."""
        exp_prob = expected.get("probability", expected.get("confidence", 1.0))
        realized = 1.0 if actual.get("success", False) else 0.0

        return abs(exp_prob - realized)

    def _identify_root_causes(
        self,
        failure: FailureEvent,
        emv_deviation: float,
        calibration_error: float
    ) -> List[Tuple[RootCauseCategory, float]]:
        """Identify potential root causes with confidence scores."""
        causes = []

        # Check state estimation errors
        if failure.context.get("state_estimation_error"):
            causes.append((
                RootCauseCategory.INCORRECT_STATE_ESTIMATE,
                failure.context.get("state_estimation_error", 0.5)
            ))

        # Check probability errors
        if calibration_error > 0.3:
            causes.append((
                RootCauseCategory.INCORRECT_PROBABILITY,
                min(1.0, calibration_error)
            ))

        # Check for incomplete information
        if failure.context.get("missing_information"):
            causes.append((
                RootCauseCategory.INCOMPLETE_INFORMATION,
                0.7
            ))

        # Check constraint issues
        if failure.failure_type == FailureType.CONSTRAINT_VIOLATION:
            causes.append((
                RootCauseCategory.CONSTRAINT_MISSED,
                0.9
            ))

        # Check execution errors
        if failure.context.get("execution_error"):
            causes.append((
                RootCauseCategory.EXECUTION_ERROR,
                0.8
            ))

        # Check external factors
        if failure.context.get("external_event"):
            causes.append((
                RootCauseCategory.EXTERNAL_FACTOR,
                0.6
            ))

        # Check memory gaps
        if failure.context.get("relevant_memory_missing"):
            causes.append((
                RootCauseCategory.MEMORY_GAP,
                0.7
            ))

        # Default to policy error if no other cause found
        if not causes and abs(emv_deviation) > self.config.significant_deviation_threshold:
            causes.append((
                RootCauseCategory.POLICY_ERROR,
                0.5
            ))

        # Sort by confidence
        causes.sort(key=lambda x: x[1], reverse=True)
        return causes

    def _identify_contributing_factors(
        self,
        failure: FailureEvent
    ) -> List[str]:
        """Identify contributing factors to the failure."""
        factors = []

        state = failure.state_at_failure

        # Check confidence level
        if state.get("confidence_level", 1.0) < 0.5:
            factors.append("Low confidence at decision time")

        # Check risk level
        if state.get("risk_exposure", 0) > 0.7:
            factors.append("High risk exposure")

        # Check resource constraints
        if state.get("resource_availability", 1.0) < 0.3:
            factors.append("Limited resources available")

        # Check time pressure
        if failure.context.get("time_pressure"):
            factors.append("Time pressure present")

        # Check novelty
        if failure.context.get("novel_situation"):
            factors.append("Novel/unfamiliar situation")

        # Check information quality
        if failure.context.get("noisy_observations"):
            factors.append("Noisy or uncertain observations")

        return factors

    def _generate_recommendations(
        self,
        failure: FailureEvent,
        root_causes: List[Tuple[RootCauseCategory, float]]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        for cause, confidence in root_causes:
            if confidence < 0.4:
                continue

            if cause == RootCauseCategory.INCORRECT_STATE_ESTIMATE:
                recommendations.append("Improve state observation accuracy")
                recommendations.append("Increase observation frequency")

            elif cause == RootCauseCategory.INCORRECT_PROBABILITY:
                recommendations.append("Recalibrate probability estimates")
                recommendations.append("Use more conservative probabilities")

            elif cause == RootCauseCategory.INCOMPLETE_INFORMATION:
                recommendations.append("Gather more information before acting")
                recommendations.append("Request clarification when uncertain")

            elif cause == RootCauseCategory.POLICY_ERROR:
                recommendations.append("Review and update policy parameters")
                recommendations.append("Consider alternative actions")

            elif cause == RootCauseCategory.CONSTRAINT_MISSED:
                recommendations.append("Review constraint checking logic")
                recommendations.append("Add stricter constraint validation")

            elif cause == RootCauseCategory.MEMORY_GAP:
                recommendations.append("Store this failure as a memory")
                recommendations.append("Improve memory retrieval coverage")

        # Add generic recommendations
        if failure.failure_type == FailureType.OUTCOME_DEVIATION:
            recommendations.append("Add outcome monitoring and early detection")

        return list(set(recommendations))  # Deduplicate

    def _identify_affected_memories(
        self,
        failure: FailureEvent
    ) -> List[str]:
        """Identify memories that should be updated based on failure."""
        affected = []

        # Memories used in decision
        memories_used = failure.context.get("memories_retrieved", [])
        affected.extend(memories_used)

        # Related memories from state
        related = failure.state_at_failure.get("related_memories", [])
        affected.extend(related)

        return list(set(affected))

    def _calculate_penalties(
        self,
        failure: FailureEvent,
        root_causes: List[Tuple[RootCauseCategory, float]]
    ) -> Dict[str, float]:
        """Calculate policy penalties for the failure."""
        penalties = {}

        base = self.config.base_penalty

        # Increase penalty for repeat failures
        repeat_count = self._failure_counts.get(failure.action, 0)
        repeat_factor = self.config.repeat_penalty_multiplier ** (repeat_count - 1)

        # Action penalty
        penalties["action"] = base * repeat_factor

        # Additional penalties based on root causes
        for cause, confidence in root_causes:
            if cause == RootCauseCategory.CONSTRAINT_MISSED:
                penalties["constraint_checking"] = base * 2 * confidence
            if cause == RootCauseCategory.POLICY_ERROR:
                penalties["policy"] = base * 1.5 * confidence

        # Severity-based penalty
        if failure.failure_type in [
            FailureType.CONSTRAINT_VIOLATION,
            FailureType.CASCADING_FAILURE
        ]:
            penalties["severity"] = base * 2

        return penalties

    def _apply_penalties(self, action: str, penalties: Dict[str, float]):
        """Apply penalties to action."""
        total_penalty = sum(penalties.values())
        self._action_penalties[action] += total_penalty

    def _update_patterns(
        self,
        failure: FailureEvent,
        analysis: RootCauseAnalysis
    ):
        """Update failure patterns based on new analysis."""
        # Find similar failures
        similar = self._find_similar_failures(failure)

        if len(similar) >= self.config.min_failures_for_pattern - 1:
            # Pattern detected
            pattern_id = f"pattern_{failure.failure_type.value}_{failure.action}"

            if pattern_id in self._patterns:
                # Update existing pattern
                pattern = self._patterns[pattern_id]
                pattern.frequency += 1
                pattern.average_impact = (
                    pattern.average_impact * (pattern.frequency - 1) +
                    analysis.emv_deviation
                ) / pattern.frequency
            else:
                # Create new pattern
                common_causes = [c for c, _ in analysis.root_causes[:3]]
                self._patterns[pattern_id] = FailurePattern(
                    pattern_id=pattern_id,
                    failure_type=failure.failure_type,
                    common_causes=common_causes,
                    frequency=len(similar) + 1,
                    average_impact=analysis.emv_deviation,
                    trigger_conditions=failure.state_at_failure,
                    suggested_prevention=analysis.recommended_actions[:3]
                )

    def _find_similar_failures(
        self,
        failure: FailureEvent
    ) -> List[FailureEvent]:
        """Find failures similar to the given one."""
        similar = []

        for past in self._failure_history:
            if past.failure_id == failure.failure_id:
                continue

            # Same action and type
            if (past.action == failure.action and
                past.failure_type == failure.failure_type):
                similar.append(past)

        return similar

    def get_action_penalty(self, action: str) -> float:
        """Get accumulated penalty for an action."""
        return self._action_penalties.get(action, 0.0)

    def get_patterns(self) -> List[FailurePattern]:
        """Get detected failure patterns."""
        return list(self._patterns.values())

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics."""
        total = len(self._failure_history)
        by_type = defaultdict(int)
        by_action = defaultdict(int)

        for failure in self._failure_history:
            by_type[failure.failure_type.value] += 1
            by_action[failure.action] += 1

        return {
            "total_failures": total,
            "by_type": dict(by_type),
            "by_action": dict(by_action),
            "patterns_detected": len(self._patterns),
            "most_common_type": max(by_type, key=by_type.get) if by_type else None,
            "most_problematic_action": max(by_action, key=by_action.get) if by_action else None
        }

    def should_block_action(self, action: str) -> Tuple[bool, str]:
        """Determine if an action should be blocked based on failure history."""
        penalty = self._action_penalties.get(action, 0)
        failure_count = self._failure_counts.get(action, 0)

        # Block if penalty exceeds threshold
        if penalty > 1.0:
            return True, f"Accumulated penalty ({penalty:.2f}) exceeds threshold"

        # Block if pattern detected with high frequency
        for pattern in self._patterns.values():
            if (pattern.failure_type in [FailureType.CONSTRAINT_VIOLATION,
                                         FailureType.CASCADING_FAILURE] and
                pattern.frequency >= 5):
                return True, f"Repeated critical failure pattern detected"

        return False, ""


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_failure_analyzer(
    config: Optional[AnalysisConfig] = None
) -> FailureAnalysisModule:
    """Create a failure analysis module."""
    return FailureAnalysisModule(config)
