"""
CONSTRAINT ENFORCEMENT ENGINE
Hard stop mechanism for autonomous agents
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Ensures:
- No forbidden actions
- No policy overrides
- Deterministic enforcement

Constraints are not suggestions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from datetime import datetime
from enum import Enum
import re


class ConstraintType(Enum):
    """Types of constraints."""
    HARD = "hard"       # Cannot be violated under any circumstances
    SOFT = "soft"       # Can be violated with override approval
    ADVISORY = "advisory"  # Warning only


class ConstraintCategory(Enum):
    """Categories of constraints."""
    SAFETY = "safety"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    ETHICAL = "ethical"
    RESOURCE = "resource"
    TEMPORAL = "temporal"


class ViolationSeverity(Enum):
    """Severity of constraint violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EnforcementAction(Enum):
    """Actions taken on constraint violation."""
    BLOCK = "block"
    ESCALATE = "escalate"
    WARN = "warn"
    LOG = "log"
    MODIFY = "modify"


@dataclass
class Constraint:
    """Definition of a constraint."""
    constraint_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    category: ConstraintCategory
    condition: Callable[[Dict[str, Any], str], bool]  # Returns True if violated
    severity: ViolationSeverity
    enforcement_action: EnforcementAction
    message: str
    enabled: bool = True
    priority: int = 0  # Higher = checked first
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    violation_id: str
    constraint: Constraint
    state: Dict[str, Any]
    action: str
    timestamp: datetime
    enforcement_applied: EnforcementAction
    override_allowed: bool
    override_applied: bool = False
    override_by: Optional[str] = None


@dataclass
class EnforcementResult:
    """Result of constraint enforcement check."""
    action: str
    allowed: bool
    violations: List[ConstraintViolation]
    warnings: List[str]
    modified_action: Optional[str] = None
    escalation_required: bool = False
    audit_log_entry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnforcementConfig:
    """Configuration for constraint enforcement."""
    # Enforcement behavior
    fail_fast: bool = True  # Stop on first hard constraint violation
    log_all_checks: bool = True
    require_audit_trail: bool = True

    # Override settings
    allow_soft_overrides: bool = True
    override_requires_reason: bool = True
    max_override_chain: int = 3

    # Escalation settings
    auto_escalate_critical: bool = True
    escalation_cooldown_seconds: int = 60

    # Rate limiting
    max_violations_per_minute: int = 10
    violation_rate_action: EnforcementAction = EnforcementAction.BLOCK


class ConstraintEnforcementEngine:
    """
    Engine for enforcing constraints on agent actions.
    Provides deterministic, auditable constraint checking.
    """

    def __init__(self, config: Optional[EnforcementConfig] = None):
        self.config = config or EnforcementConfig()
        self._constraints: Dict[str, Constraint] = {}
        self._violation_history: List[ConstraintViolation] = []
        self._override_chain: List[str] = []
        self._forbidden_actions: Set[str] = set()

        # Initialize default constraints
        self._initialize_default_constraints()

    def _initialize_default_constraints(self):
        """Initialize default safety constraints."""
        default_constraints = [
            Constraint(
                constraint_id="c_safety_001",
                name="High Risk Block",
                description="Block actions when risk exposure exceeds threshold",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.SAFETY,
                condition=lambda state, action: state.get("risk_exposure", 0) > 0.9,
                severity=ViolationSeverity.CRITICAL,
                enforcement_action=EnforcementAction.BLOCK,
                message="Risk exposure too high for autonomous action",
                priority=100
            ),
            Constraint(
                constraint_id="c_safety_002",
                name="Constraint Violation Block",
                description="Block when existing constraint violations present",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.SAFETY,
                condition=lambda state, action: state.get("constraint_violations", 0) > 0 and action == "execute_task",
                severity=ViolationSeverity.HIGH,
                enforcement_action=EnforcementAction.BLOCK,
                message="Cannot execute with existing constraint violations",
                priority=95
            ),
            Constraint(
                constraint_id="c_confidence_001",
                name="Low Confidence Warning",
                description="Warn when confidence is below threshold",
                constraint_type=ConstraintType.SOFT,
                category=ConstraintCategory.OPERATIONAL,
                condition=lambda state, action: state.get("confidence_level", 1.0) < 0.3 and action in ["execute_task", "delegate"],
                severity=ViolationSeverity.MEDIUM,
                enforcement_action=EnforcementAction.WARN,
                message="Low confidence - consider gathering more information",
                priority=50
            ),
            Constraint(
                constraint_id="c_resource_001",
                name="Resource Exhaustion Block",
                description="Block when resources are critically low",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.RESOURCE,
                condition=lambda state, action: state.get("resource_availability", 1.0) < 0.1,
                severity=ViolationSeverity.HIGH,
                enforcement_action=EnforcementAction.ESCALATE,
                message="Insufficient resources for operation",
                priority=80
            ),
            Constraint(
                constraint_id="c_forbidden_001",
                name="Forbidden Action Block",
                description="Block explicitly forbidden actions",
                constraint_type=ConstraintType.HARD,
                category=ConstraintCategory.COMPLIANCE,
                condition=lambda state, action: action in self._forbidden_actions,
                severity=ViolationSeverity.CRITICAL,
                enforcement_action=EnforcementAction.BLOCK,
                message="Action is explicitly forbidden",
                priority=110
            ),
        ]

        for constraint in default_constraints:
            self.register_constraint(constraint)

    def register_constraint(self, constraint: Constraint):
        """Register a new constraint."""
        self._constraints[constraint.constraint_id] = constraint

    def unregister_constraint(self, constraint_id: str) -> bool:
        """Unregister a constraint (soft constraints only)."""
        constraint = self._constraints.get(constraint_id)
        if constraint and constraint.constraint_type != ConstraintType.HARD:
            del self._constraints[constraint_id]
            return True
        return False

    def add_forbidden_action(self, action: str):
        """Add an action to the forbidden list."""
        self._forbidden_actions.add(action)

    def remove_forbidden_action(self, action: str):
        """Remove an action from the forbidden list."""
        self._forbidden_actions.discard(action)

    def check_constraints(
        self,
        state: Dict[str, Any],
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EnforcementResult:
        """
        Check all constraints for a proposed action.
        Returns enforcement result with violations and recommendations.
        """
        context = context or {}
        violations = []
        warnings = []
        allowed = True
        escalation_required = False
        modified_action = None

        # Get constraints sorted by priority
        sorted_constraints = sorted(
            self._constraints.values(),
            key=lambda c: c.priority,
            reverse=True
        )

        for constraint in sorted_constraints:
            if not constraint.enabled:
                continue

            # Check if constraint is violated
            try:
                is_violated = constraint.condition(state, action)
            except Exception as e:
                # Constraint check failed - treat as violation for safety
                is_violated = True
                warnings.append(f"Constraint check error: {constraint.name} - {str(e)}")

            if is_violated:
                # Create violation record
                violation = ConstraintViolation(
                    violation_id=f"v_{datetime.now().timestamp()}",
                    constraint=constraint,
                    state=state.copy(),
                    action=action,
                    timestamp=datetime.now(),
                    enforcement_applied=constraint.enforcement_action,
                    override_allowed=(
                        constraint.constraint_type != ConstraintType.HARD and
                        self.config.allow_soft_overrides
                    )
                )

                violations.append(violation)
                self._violation_history.append(violation)

                # Apply enforcement action
                if constraint.enforcement_action == EnforcementAction.BLOCK:
                    if constraint.constraint_type == ConstraintType.HARD:
                        allowed = False
                        if self.config.fail_fast:
                            break
                    else:
                        warnings.append(f"SOFT BLOCK: {constraint.message}")

                elif constraint.enforcement_action == EnforcementAction.ESCALATE:
                    escalation_required = True
                    if constraint.constraint_type == ConstraintType.HARD:
                        allowed = False

                elif constraint.enforcement_action == EnforcementAction.WARN:
                    warnings.append(f"WARNING: {constraint.message}")

                elif constraint.enforcement_action == EnforcementAction.MODIFY:
                    modified_action = self._suggest_modified_action(
                        action, constraint, state
                    )
                    warnings.append(f"Action modified: {action} -> {modified_action}")

        # Check violation rate limit
        if self._check_violation_rate_exceeded():
            allowed = False
            warnings.append("Violation rate limit exceeded")

        # Build audit log entry
        audit_entry = self._build_audit_entry(
            state, action, violations, allowed, context
        )

        return EnforcementResult(
            action=action,
            allowed=allowed,
            violations=violations,
            warnings=warnings,
            modified_action=modified_action,
            escalation_required=escalation_required,
            audit_log_entry=audit_entry
        )

    def apply_override(
        self,
        violation: ConstraintViolation,
        override_by: str,
        reason: str
    ) -> bool:
        """
        Apply override to a soft constraint violation.
        Returns True if override was successful.
        """
        if not violation.override_allowed:
            return False

        if self.config.override_requires_reason and not reason:
            return False

        if len(self._override_chain) >= self.config.max_override_chain:
            return False

        violation.override_applied = True
        violation.override_by = override_by

        self._override_chain.append(violation.violation_id)

        return True

    def _suggest_modified_action(
        self,
        action: str,
        constraint: Constraint,
        state: Dict[str, Any]
    ) -> str:
        """Suggest a modified action that satisfies the constraint."""
        # Default modifications based on constraint category
        modifications = {
            ConstraintCategory.SAFETY: "escalate",
            ConstraintCategory.RESOURCE: "defer_action",
            ConstraintCategory.OPERATIONAL: "gather_information",
            ConstraintCategory.COMPLIANCE: "request_clarification"
        }

        return modifications.get(constraint.category, "defer_action")

    def _check_violation_rate_exceeded(self) -> bool:
        """Check if violation rate exceeds limit."""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(minutes=1)
        recent_violations = [
            v for v in self._violation_history
            if v.timestamp > cutoff
        ]

        return len(recent_violations) > self.config.max_violations_per_minute

    def _build_audit_entry(
        self,
        state: Dict[str, Any],
        action: str,
        violations: List[ConstraintViolation],
        allowed: bool,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build audit log entry for the constraint check."""
        return {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "allowed": allowed,
            "state_snapshot": {
                k: v for k, v in state.items()
                if k in ["confidence_level", "risk_exposure", "constraint_violations"]
            },
            "violations": [
                {
                    "constraint_id": v.constraint.constraint_id,
                    "constraint_name": v.constraint.name,
                    "severity": v.constraint.severity.value,
                    "enforcement": v.enforcement_applied.value
                }
                for v in violations
            ],
            "context": {
                k: v for k, v in context.items()
                if k in ["session_id", "user_id", "task_id"]
            }
        }

    def get_active_constraints(self) -> List[Constraint]:
        """Get all active constraints."""
        return [c for c in self._constraints.values() if c.enabled]

    def get_violation_history(
        self,
        limit: int = 100,
        constraint_id: Optional[str] = None
    ) -> List[ConstraintViolation]:
        """Get violation history, optionally filtered."""
        history = self._violation_history

        if constraint_id:
            history = [
                v for v in history
                if v.constraint.constraint_id == constraint_id
            ]

        return history[-limit:]

    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get violation statistics."""
        from collections import defaultdict

        by_constraint = defaultdict(int)
        by_severity = defaultdict(int)
        by_action = defaultdict(int)

        for v in self._violation_history:
            by_constraint[v.constraint.constraint_id] += 1
            by_severity[v.constraint.severity.value] += 1
            by_action[v.enforcement_applied.value] += 1

        return {
            "total_violations": len(self._violation_history),
            "by_constraint": dict(by_constraint),
            "by_severity": dict(by_severity),
            "by_enforcement_action": dict(by_action),
            "overrides_applied": sum(
                1 for v in self._violation_history if v.override_applied
            ),
            "forbidden_actions": list(self._forbidden_actions)
        }


class ConstraintBuilder:
    """Helper class for building constraints."""

    def __init__(self):
        self._constraint_id = None
        self._name = None
        self._description = ""
        self._type = ConstraintType.SOFT
        self._category = ConstraintCategory.OPERATIONAL
        self._condition = None
        self._severity = ViolationSeverity.MEDIUM
        self._action = EnforcementAction.WARN
        self._message = ""
        self._priority = 50

    def with_id(self, constraint_id: str) -> 'ConstraintBuilder':
        self._constraint_id = constraint_id
        return self

    def with_name(self, name: str) -> 'ConstraintBuilder':
        self._name = name
        return self

    def with_description(self, description: str) -> 'ConstraintBuilder':
        self._description = description
        return self

    def as_hard_constraint(self) -> 'ConstraintBuilder':
        self._type = ConstraintType.HARD
        return self

    def as_soft_constraint(self) -> 'ConstraintBuilder':
        self._type = ConstraintType.SOFT
        return self

    def in_category(self, category: ConstraintCategory) -> 'ConstraintBuilder':
        self._category = category
        return self

    def with_condition(
        self,
        condition: Callable[[Dict[str, Any], str], bool]
    ) -> 'ConstraintBuilder':
        self._condition = condition
        return self

    def with_severity(self, severity: ViolationSeverity) -> 'ConstraintBuilder':
        self._severity = severity
        return self

    def with_enforcement(self, action: EnforcementAction) -> 'ConstraintBuilder':
        self._action = action
        return self

    def with_message(self, message: str) -> 'ConstraintBuilder':
        self._message = message
        return self

    def with_priority(self, priority: int) -> 'ConstraintBuilder':
        self._priority = priority
        return self

    def build(self) -> Constraint:
        if not self._constraint_id or not self._name or not self._condition:
            raise ValueError("Constraint ID, name, and condition are required")

        return Constraint(
            constraint_id=self._constraint_id,
            name=self._name,
            description=self._description,
            constraint_type=self._type,
            category=self._category,
            condition=self._condition,
            severity=self._severity,
            enforcement_action=self._action,
            message=self._message or f"Constraint violated: {self._name}",
            priority=self._priority
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_constraint_engine(
    config: Optional[EnforcementConfig] = None
) -> ConstraintEnforcementEngine:
    """Create a constraint enforcement engine."""
    return ConstraintEnforcementEngine(config)
