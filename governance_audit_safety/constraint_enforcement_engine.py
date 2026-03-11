"""
FILE 20: constraint_enforcement_engine.py
PURPOSE: Hard stop mechanism
ROLE: Constraints are not suggestions

Ensures:
- No forbidden actions
- No policy overrides
- Deterministic enforcement
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EnforcementAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"
    ESCALATE = "escalate"
    HALT = "halt"


class ConstraintType(Enum):
    HARD = "hard"  # Must never be violated - blocks action
    SOFT = "soft"  # Should not be violated - warns and logs
    ADVISORY = "advisory"  # Logged but not enforced


@dataclass
class Constraint:
    """A single enforceable constraint."""
    constraint_id: str
    description: str
    constraint_type: ConstraintType
    condition: str  # Python expression or rule name
    enforcement_action: EnforcementAction
    priority: int = 0  # Higher = checked first
    enabled: bool = True
    violation_count: int = 0
    last_violation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "description": self.description,
            "type": self.constraint_type.value,
            "enforcement": self.enforcement_action.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "violation_count": self.violation_count,
        }


@dataclass
class EnforcementResult:
    """Result of a constraint enforcement check."""
    allowed: bool
    action: str
    enforcement_action: EnforcementAction
    violations: list[dict[str, Any]]
    warnings: list[str]
    timestamp: str
    modified_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "enforcement_action": self.enforcement_action.value,
            "violations": self.violations,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
            "modified_action": self.modified_action,
        }


class ConstraintEnforcementEngine:
    """
    Deterministic constraint enforcement engine.
    This is the hard stop mechanism that ensures no forbidden actions
    are taken and no policy overrides occur.

    Constraints are NOT suggestions. They are laws.
    """

    def __init__(self):
        self.constraints: dict[str, Constraint] = {}
        self.enforcement_log: list[EnforcementResult] = []
        self.total_checks: int = 0
        self.total_violations: int = 0
        self.halted: bool = False
        self.halt_reason: str = ""

        # Built-in constraints
        self._register_builtin_constraints()

    def register_constraint(self, constraint: Constraint) -> None:
        """Register a new constraint."""
        self.constraints[constraint.constraint_id] = constraint
        logger.info(
            f"Registered constraint '{constraint.constraint_id}': "
            f"{constraint.description}"
        )

    def check(
        self,
        action: str,
        state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> EnforcementResult:
        """
        Check an action against all constraints.
        This is the primary enforcement entry point.

        Args:
            action: The proposed action
            state: Current environment/agent state
            context: Additional context (EMV, risk metrics, etc.)

        Returns:
            EnforcementResult indicating whether the action is allowed
        """
        if self.halted:
            return EnforcementResult(
                allowed=False,
                action=action,
                enforcement_action=EnforcementAction.HALT,
                violations=[{"reason": f"System halted: {self.halt_reason}"}],
                warnings=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        self.total_checks += 1
        violations: list[dict[str, Any]] = []
        warnings: list[str] = []
        enforcement = EnforcementAction.ALLOW
        modified_action = None

        context = context or {}

        # Check constraints in priority order
        sorted_constraints = sorted(
            self.constraints.values(),
            key=lambda c: c.priority,
            reverse=True,
        )

        for constraint in sorted_constraints:
            if not constraint.enabled:
                continue

            violated = self._evaluate_constraint(constraint, action, state, context)

            if violated:
                constraint.violation_count += 1
                constraint.last_violation = datetime.now(timezone.utc).isoformat()
                self.total_violations += 1

                violation_record = {
                    "constraint_id": constraint.constraint_id,
                    "description": constraint.description,
                    "type": constraint.constraint_type.value,
                    "action": action,
                }

                if constraint.constraint_type == ConstraintType.HARD:
                    violations.append(violation_record)
                    if constraint.enforcement_action == EnforcementAction.HALT:
                        enforcement = EnforcementAction.HALT
                        self.halted = True
                        self.halt_reason = (
                            f"Hard constraint '{constraint.constraint_id}' violated"
                        )
                        logger.critical(f"SYSTEM HALT: {self.halt_reason}")
                    elif constraint.enforcement_action == EnforcementAction.BLOCK:
                        if enforcement != EnforcementAction.HALT:
                            enforcement = EnforcementAction.BLOCK
                    elif constraint.enforcement_action == EnforcementAction.ESCALATE:
                        if enforcement not in (
                            EnforcementAction.HALT,
                            EnforcementAction.BLOCK,
                        ):
                            enforcement = EnforcementAction.ESCALATE

                elif constraint.constraint_type == ConstraintType.SOFT:
                    warnings.append(
                        f"Soft constraint '{constraint.constraint_id}' violated: "
                        f"{constraint.description}"
                    )
                    logger.warning(warnings[-1])

                elif constraint.constraint_type == ConstraintType.ADVISORY:
                    warnings.append(
                        f"Advisory '{constraint.constraint_id}': {constraint.description}"
                    )

        allowed = enforcement == EnforcementAction.ALLOW

        result = EnforcementResult(
            allowed=allowed,
            action=action,
            enforcement_action=enforcement,
            violations=violations,
            warnings=warnings,
            timestamp=datetime.now(timezone.utc).isoformat(),
            modified_action=modified_action,
        )

        self.enforcement_log.append(result)

        if not allowed:
            logger.warning(
                f"Action '{action}' BLOCKED by constraint enforcement. "
                f"Violations: {len(violations)}"
            )

        return result

    def reset_halt(self, authorization: str) -> bool:
        """Reset a halted system (requires authorization)."""
        if not authorization:
            logger.error("Cannot reset halt without authorization.")
            return False

        logger.warning(f"System halt reset by: {authorization}")
        self.halted = False
        self.halt_reason = ""
        return True

    def get_constraint_status(self) -> dict[str, Any]:
        """Return status of all constraints."""
        return {
            "total_constraints": len(self.constraints),
            "total_checks": self.total_checks,
            "total_violations": self.total_violations,
            "halted": self.halted,
            "constraints": {
                cid: c.to_dict() for cid, c in self.constraints.items()
            },
        }

    # -------------------------------------------------------------------------
    # CONSTRAINT EVALUATION
    # -------------------------------------------------------------------------

    def _evaluate_constraint(
        self,
        constraint: Constraint,
        action: str,
        state: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """
        Evaluate whether a constraint is violated.
        Returns True if violated.
        """
        condition = constraint.condition

        # Built-in rule evaluation
        if condition == "no_forbidden_actions":
            forbidden = state.get("forbidden_actions", [])
            return action in forbidden

        elif condition == "budget_limit":
            cost = context.get("action_cost", 0)
            budget = state.get("resource_budget_remaining", float("inf"))
            return cost > budget

        elif condition == "risk_threshold":
            risk = context.get("risk_level", 0)
            threshold = state.get("max_risk_threshold", 0.9)
            return risk > threshold

        elif condition == "human_override_required":
            requires_approval = context.get("requires_human_approval", False)
            has_approval = context.get("human_approved", False)
            return requires_approval and not has_approval

        elif condition == "max_consecutive_actions":
            recent = state.get("recent_actions", [])
            limit = 10
            return len(recent) >= limit and len(set(recent[-limit:])) == 1

        elif condition == "audit_trail_required":
            return not context.get("trace_logged", True)

        # Default: not violated
        return False

    # -------------------------------------------------------------------------
    # BUILT-IN CONSTRAINTS
    # -------------------------------------------------------------------------

    def _register_builtin_constraints(self) -> None:
        """Register the built-in safety constraints."""
        builtins = [
            Constraint(
                constraint_id="BUILTIN-001",
                description="No forbidden actions may be executed",
                constraint_type=ConstraintType.HARD,
                condition="no_forbidden_actions",
                enforcement_action=EnforcementAction.BLOCK,
                priority=100,
            ),
            Constraint(
                constraint_id="BUILTIN-002",
                description="Actions must not exceed budget",
                constraint_type=ConstraintType.HARD,
                condition="budget_limit",
                enforcement_action=EnforcementAction.BLOCK,
                priority=90,
            ),
            Constraint(
                constraint_id="BUILTIN-003",
                description="Risk must not exceed maximum threshold",
                constraint_type=ConstraintType.HARD,
                condition="risk_threshold",
                enforcement_action=EnforcementAction.ESCALATE,
                priority=95,
            ),
            Constraint(
                constraint_id="BUILTIN-004",
                description="Human override must be available when required",
                constraint_type=ConstraintType.HARD,
                condition="human_override_required",
                enforcement_action=EnforcementAction.BLOCK,
                priority=100,
            ),
            Constraint(
                constraint_id="BUILTIN-005",
                description="Prevent infinite action loops",
                constraint_type=ConstraintType.SOFT,
                condition="max_consecutive_actions",
                enforcement_action=EnforcementAction.BLOCK,
                priority=80,
            ),
            Constraint(
                constraint_id="BUILTIN-006",
                description="All decisions must have audit trail",
                constraint_type=ConstraintType.HARD,
                condition="audit_trail_required",
                enforcement_action=EnforcementAction.BLOCK,
                priority=85,
            ),
        ]

        for c in builtins:
            self.register_constraint(c)
