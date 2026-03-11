"""
FILE 22: multi_agent_interface.py
PURPOSE: Multi-agent coordination
ROLE: Shared memory, negotiation, coordination between agents

Supports:
- Agent discovery and registration
- Shared memory protocol
- Task delegation and negotiation
- Conflict resolution
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    PRIMARY = "primary"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"


class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    MEMORY_SHARE = "memory_share"
    NEGOTIATION = "negotiation"
    STATUS_UPDATE = "status_update"
    CONFLICT_REPORT = "conflict_report"
    COORDINATION = "coordination"


@dataclass
class AgentRegistration:
    """Registration record for an agent in the multi-agent system."""
    agent_id: str
    role: AgentRole
    capabilities: list[str]
    status: str = "active"
    registered_at: str = ""
    last_heartbeat: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.registered_at:
            self.registered_at = datetime.now(timezone.utc).isoformat()
        self.last_heartbeat = self.registered_at


@dataclass
class AgentMessage:
    """A message between agents."""
    message_id: str
    sender_id: str
    recipient_id: str | None  # None = broadcast
    message_type: MessageType
    payload: dict[str, Any]
    timestamp: str = ""
    requires_response: bool = False
    response_to: str | None = None
    priority: int = 0

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class NegotiationProposal:
    """A proposal in a multi-agent negotiation."""
    proposal_id: str
    proposer_id: str
    task: str
    terms: dict[str, Any]
    status: str = "pending"  # pending | accepted | rejected | counter
    counter_proposal: dict[str, Any] | None = None


class MultiAgentInterface:
    """
    Interface for multi-agent coordination, communication,
    shared memory access, and task negotiation.
    """

    def __init__(self, agent_id: str, role: AgentRole = AgentRole.PRIMARY):
        self.agent_id = agent_id
        self.role = role
        self.registry: dict[str, AgentRegistration] = {}
        self.inbox: list[AgentMessage] = []
        self.outbox: list[AgentMessage] = []
        self.shared_memory: dict[str, Any] = {}
        self.active_negotiations: dict[str, NegotiationProposal] = {}
        self.task_delegations: dict[str, dict[str, Any]] = {}

        # Register self
        self.register_agent(AgentRegistration(
            agent_id=agent_id,
            role=role,
            capabilities=[],
        ))

    def register_agent(self, registration: AgentRegistration) -> None:
        """Register an agent in the multi-agent system."""
        self.registry[registration.agent_id] = registration
        logger.info(
            f"Agent '{registration.agent_id}' registered "
            f"(role={registration.role.value})"
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        if agent_id in self.registry:
            del self.registry[agent_id]
            logger.info(f"Agent '{agent_id}' unregistered")

    def get_active_agents(self) -> list[AgentRegistration]:
        """Return all active registered agents."""
        return [a for a in self.registry.values() if a.status == "active"]

    def find_agents_by_capability(self, capability: str) -> list[AgentRegistration]:
        """Find agents that have a specific capability."""
        return [
            a for a in self.registry.values()
            if capability in a.capabilities and a.status == "active"
        ]

    # -------------------------------------------------------------------------
    # MESSAGING
    # -------------------------------------------------------------------------

    def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent (or broadcast)."""
        message.sender_id = self.agent_id
        self.outbox.append(message)
        logger.debug(
            f"Message sent: {message.message_type.value} -> "
            f"{message.recipient_id or 'broadcast'}"
        )

    def receive_messages(self) -> list[AgentMessage]:
        """Retrieve and clear the inbox."""
        messages = list(self.inbox)
        self.inbox.clear()
        return messages

    def deliver_message(self, message: AgentMessage) -> None:
        """Deliver a message to this agent's inbox."""
        self.inbox.append(message)

    # -------------------------------------------------------------------------
    # SHARED MEMORY
    # -------------------------------------------------------------------------

    def share_memory(self, key: str, value: Any, scope: str = "global") -> None:
        """Share a memory item with other agents."""
        self.shared_memory[key] = {
            "value": value,
            "shared_by": self.agent_id,
            "scope": scope,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.send_message(AgentMessage(
            message_id="",
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.MEMORY_SHARE,
            payload={"key": key, "scope": scope},
        ))

    def get_shared_memory(self, key: str) -> Any | None:
        """Retrieve a shared memory item."""
        entry = self.shared_memory.get(key)
        return entry["value"] if entry else None

    # -------------------------------------------------------------------------
    # TASK DELEGATION
    # -------------------------------------------------------------------------

    def delegate_task(
        self,
        task_id: str,
        task_description: str,
        target_agent_id: str,
        requirements: dict[str, Any] | None = None,
    ) -> str:
        """Delegate a task to another agent."""
        delegation = {
            "task_id": task_id,
            "description": task_description,
            "delegated_to": target_agent_id,
            "delegated_by": self.agent_id,
            "requirements": requirements or {},
            "status": "pending",
            "delegated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.task_delegations[task_id] = delegation

        self.send_message(AgentMessage(
            message_id="",
            sender_id=self.agent_id,
            recipient_id=target_agent_id,
            message_type=MessageType.TASK_REQUEST,
            payload=delegation,
            requires_response=True,
        ))

        return task_id

    def respond_to_task(
        self,
        task_id: str,
        accepted: bool,
        result: Any = None,
    ) -> None:
        """Respond to a delegated task."""
        self.send_message(AgentMessage(
            message_id="",
            sender_id=self.agent_id,
            recipient_id=None,
            message_type=MessageType.TASK_RESPONSE,
            payload={
                "task_id": task_id,
                "accepted": accepted,
                "result": result,
                "status": "completed" if result else ("accepted" if accepted else "rejected"),
            },
        ))

    # -------------------------------------------------------------------------
    # NEGOTIATION
    # -------------------------------------------------------------------------

    def propose(
        self,
        task: str,
        terms: dict[str, Any],
        recipient_id: str,
    ) -> str:
        """Make a negotiation proposal to another agent."""
        proposal = NegotiationProposal(
            proposal_id=str(uuid.uuid4()),
            proposer_id=self.agent_id,
            task=task,
            terms=terms,
        )
        self.active_negotiations[proposal.proposal_id] = proposal

        self.send_message(AgentMessage(
            message_id="",
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=MessageType.NEGOTIATION,
            payload={
                "proposal_id": proposal.proposal_id,
                "action": "propose",
                "task": task,
                "terms": terms,
            },
            requires_response=True,
        ))

        return proposal.proposal_id

    def respond_to_proposal(
        self,
        proposal_id: str,
        accept: bool,
        counter_terms: dict[str, Any] | None = None,
    ) -> None:
        """Accept, reject, or counter a negotiation proposal."""
        if proposal_id in self.active_negotiations:
            proposal = self.active_negotiations[proposal_id]
            if accept:
                proposal.status = "accepted"
            elif counter_terms:
                proposal.status = "counter"
                proposal.counter_proposal = counter_terms
            else:
                proposal.status = "rejected"

    # -------------------------------------------------------------------------
    # CONFLICT RESOLUTION
    # -------------------------------------------------------------------------

    def report_conflict(
        self,
        conflicting_agent_id: str,
        conflict_type: str,
        details: dict[str, Any],
    ) -> None:
        """Report a conflict with another agent for resolution."""
        self.send_message(AgentMessage(
            message_id="",
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast to coordinators
            message_type=MessageType.CONFLICT_REPORT,
            payload={
                "conflicting_agent": conflicting_agent_id,
                "conflict_type": conflict_type,
                "details": details,
            },
            priority=10,
        ))
