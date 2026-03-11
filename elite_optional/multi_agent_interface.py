"""
MULTI-AGENT INTERFACE
Shared memory, negotiation, and coordination between agents
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY (ELITE)

Features:
- Shared memory access
- Inter-agent negotiation
- Task coordination
- Conflict resolution
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from collections import defaultdict
import uuid


class AgentRole(Enum):
    """Roles in multi-agent coordination."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    OBSERVER = "observer"


class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    NEGOTIATION_OFFER = "negotiation_offer"
    NEGOTIATION_ACCEPT = "negotiation_accept"
    NEGOTIATION_REJECT = "negotiation_reject"
    STATUS_UPDATE = "status_update"
    COORDINATION = "coordination"
    CONFLICT_REPORT = "conflict_report"
    MEMORY_SHARE = "memory_share"


class CoordinationProtocol(Enum):
    """Coordination protocols."""
    HIERARCHICAL = "hierarchical"
    DEMOCRATIC = "democratic"
    AUCTION = "auction"
    CONTRACT_NET = "contract_net"
    BLACKBOARD = "blackboard"


@dataclass
class AgentCapability:
    """Describes an agent's capabilities."""
    capability_id: str
    name: str
    domain: str
    proficiency: float  # 0-1
    cost: float  # Resource cost to use
    availability: float  # 0-1


@dataclass
class AgentProfile:
    """Profile of an agent in the multi-agent system."""
    agent_id: str
    name: str
    role: AgentRole
    capabilities: List[AgentCapability]
    status: str = "active"
    load: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """Inter-agent message."""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: int = 300


@dataclass
class SharedMemoryEntry:
    """Entry in shared memory."""
    key: str
    value: Any
    owner_id: str
    created: datetime
    expires: Optional[datetime] = None
    access_list: Set[str] = field(default_factory=set)  # Empty = all agents
    version: int = 1


@dataclass
class NegotiationState:
    """State of a negotiation."""
    negotiation_id: str
    initiator_id: str
    participants: Set[str]
    topic: str
    offers: List[Tuple[str, Dict[str, Any]]]  # (agent_id, offer)
    status: str = "active"
    deadline: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


class SharedMemory:
    """
    Shared memory system for multi-agent coordination.
    Provides thread-safe access to shared state.
    """

    def __init__(self):
        self._entries: Dict[str, SharedMemoryEntry] = {}
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # key -> agent_ids
        self._lock = asyncio.Lock()

    async def put(
        self,
        key: str,
        value: Any,
        owner_id: str,
        ttl_seconds: Optional[int] = None,
        access_list: Optional[Set[str]] = None
    ) -> bool:
        """Put a value in shared memory."""
        async with self._lock:
            expires = None
            if ttl_seconds:
                expires = datetime.now() + timedelta(seconds=ttl_seconds)

            existing = self._entries.get(key)
            version = (existing.version + 1) if existing else 1

            self._entries[key] = SharedMemoryEntry(
                key=key,
                value=value,
                owner_id=owner_id,
                created=datetime.now(),
                expires=expires,
                access_list=access_list or set(),
                version=version
            )

            return True

    async def get(
        self,
        key: str,
        requester_id: str
    ) -> Optional[Any]:
        """Get a value from shared memory."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            # Check expiration
            if entry.expires and datetime.now() > entry.expires:
                del self._entries[key]
                return None

            # Check access control
            if entry.access_list and requester_id not in entry.access_list:
                return None

            return entry.value

    async def delete(self, key: str, requester_id: str) -> bool:
        """Delete a value from shared memory."""
        async with self._lock:
            entry = self._entries.get(key)
            if entry and entry.owner_id == requester_id:
                del self._entries[key]
                return True
            return False

    async def subscribe(self, key: str, agent_id: str):
        """Subscribe to changes on a key."""
        self._subscriptions[key].add(agent_id)

    async def unsubscribe(self, key: str, agent_id: str):
        """Unsubscribe from changes on a key."""
        self._subscriptions[key].discard(agent_id)

    def get_subscribers(self, key: str) -> Set[str]:
        """Get subscribers for a key."""
        return self._subscriptions.get(key, set())


class MessageBus:
    """
    Message bus for inter-agent communication.
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)

    def register_agent(self, agent_id: str):
        """Register an agent on the message bus."""
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the message bus."""
        if agent_id in self._queues:
            del self._queues[agent_id]

    async def send(self, message: Message):
        """Send a message to an agent."""
        queue = self._queues.get(message.recipient_id)
        if queue:
            await queue.put(message)
            return True
        return False

    async def broadcast(self, message: Message, exclude: Optional[Set[str]] = None):
        """Broadcast a message to all agents."""
        exclude = exclude or set()
        for agent_id, queue in self._queues.items():
            if agent_id not in exclude and agent_id != message.sender_id:
                msg_copy = Message(
                    message_id=f"{message.message_id}_{agent_id}",
                    message_type=message.message_type,
                    sender_id=message.sender_id,
                    recipient_id=agent_id,
                    content=message.content,
                    correlation_id=message.correlation_id
                )
                await queue.put(msg_copy)

    async def receive(
        self,
        agent_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Receive a message for an agent."""
        queue = self._queues.get(agent_id)
        if not queue:
            return None

        try:
            if timeout:
                return await asyncio.wait_for(queue.get(), timeout)
            return await queue.get()
        except asyncio.TimeoutError:
            return None

    def register_handler(
        self,
        agent_id: str,
        handler: Callable[[Message], None]
    ):
        """Register a message handler for an agent."""
        self._handlers[agent_id].append(handler)


class NegotiationProtocol:
    """
    Protocol for inter-agent negotiation.
    """

    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self._negotiations: Dict[str, NegotiationState] = {}

    async def initiate_negotiation(
        self,
        initiator_id: str,
        participants: Set[str],
        topic: str,
        initial_offer: Dict[str, Any],
        deadline_seconds: int = 60
    ) -> str:
        """Initiate a negotiation."""
        neg_id = str(uuid.uuid4())

        state = NegotiationState(
            negotiation_id=neg_id,
            initiator_id=initiator_id,
            participants=participants,
            topic=topic,
            offers=[(initiator_id, initial_offer)],
            deadline=datetime.now() + timedelta(seconds=deadline_seconds)
        )

        self._negotiations[neg_id] = state

        # Send negotiation offers to participants
        for participant in participants:
            if participant != initiator_id:
                await self.message_bus.send(Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.NEGOTIATION_OFFER,
                    sender_id=initiator_id,
                    recipient_id=participant,
                    content={
                        "negotiation_id": neg_id,
                        "topic": topic,
                        "offer": initial_offer
                    },
                    correlation_id=neg_id
                ))

        return neg_id

    async def submit_offer(
        self,
        negotiation_id: str,
        agent_id: str,
        offer: Dict[str, Any]
    ) -> bool:
        """Submit an offer in a negotiation."""
        state = self._negotiations.get(negotiation_id)
        if not state or state.status != "active":
            return False

        if agent_id not in state.participants:
            return False

        if state.deadline and datetime.now() > state.deadline:
            state.status = "expired"
            return False

        state.offers.append((agent_id, offer))

        # Notify other participants
        for participant in state.participants:
            if participant != agent_id:
                await self.message_bus.send(Message(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.NEGOTIATION_OFFER,
                    sender_id=agent_id,
                    recipient_id=participant,
                    content={
                        "negotiation_id": negotiation_id,
                        "offer": offer
                    },
                    correlation_id=negotiation_id
                ))

        return True

    async def accept_offer(
        self,
        negotiation_id: str,
        agent_id: str,
        accepted_offer_index: int
    ) -> bool:
        """Accept an offer in a negotiation."""
        state = self._negotiations.get(negotiation_id)
        if not state or state.status != "active":
            return False

        if accepted_offer_index >= len(state.offers):
            return False

        accepted = state.offers[accepted_offer_index]
        state.result = {
            "accepted_by": agent_id,
            "offer": accepted[1],
            "offered_by": accepted[0]
        }
        state.status = "accepted"

        # Notify all participants
        for participant in state.participants:
            await self.message_bus.send(Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.NEGOTIATION_ACCEPT,
                sender_id=agent_id,
                recipient_id=participant,
                content={
                    "negotiation_id": negotiation_id,
                    "result": state.result
                },
                correlation_id=negotiation_id
            ))

        return True

    def get_negotiation_state(
        self,
        negotiation_id: str
    ) -> Optional[NegotiationState]:
        """Get current negotiation state."""
        return self._negotiations.get(negotiation_id)


class MultiAgentCoordinator:
    """
    Coordinator for multi-agent systems.
    Manages agents, shared memory, and coordination.
    """

    def __init__(
        self,
        protocol: CoordinationProtocol = CoordinationProtocol.HIERARCHICAL
    ):
        self.protocol = protocol
        self.shared_memory = SharedMemory()
        self.message_bus = MessageBus()
        self.negotiation = NegotiationProtocol(self.message_bus)

        self._agents: Dict[str, AgentProfile] = {}
        self._task_assignments: Dict[str, str] = {}  # task_id -> agent_id

    def register_agent(self, profile: AgentProfile):
        """Register an agent with the coordinator."""
        self._agents[profile.agent_id] = profile
        self.message_bus.register_agent(profile.agent_id)

    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            self.message_bus.unregister_agent(agent_id)

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent's profile."""
        return self._agents.get(agent_id)

    def get_agents_by_capability(
        self,
        capability_name: str,
        min_proficiency: float = 0.0
    ) -> List[AgentProfile]:
        """Find agents with a specific capability."""
        matching = []
        for agent in self._agents.values():
            for cap in agent.capabilities:
                if (cap.name == capability_name and
                    cap.proficiency >= min_proficiency):
                    matching.append(agent)
                    break
        return matching

    async def assign_task(
        self,
        task_id: str,
        task_description: str,
        required_capabilities: List[str],
        requester_id: str
    ) -> Optional[str]:
        """
        Assign a task to an appropriate agent.
        Uses the configured coordination protocol.
        """
        if self.protocol == CoordinationProtocol.HIERARCHICAL:
            return await self._assign_hierarchical(
                task_id, required_capabilities
            )
        elif self.protocol == CoordinationProtocol.AUCTION:
            return await self._assign_auction(
                task_id, task_description, required_capabilities, requester_id
            )
        elif self.protocol == CoordinationProtocol.CONTRACT_NET:
            return await self._assign_contract_net(
                task_id, task_description, required_capabilities, requester_id
            )
        else:
            return await self._assign_hierarchical(
                task_id, required_capabilities
            )

    async def _assign_hierarchical(
        self,
        task_id: str,
        required_capabilities: List[str]
    ) -> Optional[str]:
        """Assign task using hierarchical protocol (coordinator decides)."""
        best_agent = None
        best_score = -1

        for agent in self._agents.values():
            if agent.status != "active":
                continue

            # Calculate capability match score
            score = 0
            for req_cap in required_capabilities:
                for cap in agent.capabilities:
                    if cap.name == req_cap:
                        score += cap.proficiency * cap.availability
                        break

            # Penalize high load
            score *= (1 - agent.load)

            if score > best_score:
                best_score = score
                best_agent = agent

        if best_agent:
            self._task_assignments[task_id] = best_agent.agent_id
            return best_agent.agent_id

        return None

    async def _assign_auction(
        self,
        task_id: str,
        task_description: str,
        required_capabilities: List[str],
        requester_id: str
    ) -> Optional[str]:
        """Assign task using auction protocol (agents bid)."""
        # Broadcast task announcement
        await self.message_bus.broadcast(Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_REQUEST,
            sender_id=requester_id,
            recipient_id="broadcast",
            content={
                "task_id": task_id,
                "description": task_description,
                "required_capabilities": required_capabilities
            },
            correlation_id=task_id
        ))

        # Collect bids (simplified - in practice would wait for responses)
        # For now, use hierarchical as fallback
        return await self._assign_hierarchical(task_id, required_capabilities)

    async def _assign_contract_net(
        self,
        task_id: str,
        task_description: str,
        required_capabilities: List[str],
        requester_id: str
    ) -> Optional[str]:
        """Assign task using contract net protocol."""
        # Similar to auction but with proposal evaluation
        return await self._assign_hierarchical(task_id, required_capabilities)

    async def report_conflict(
        self,
        reporter_id: str,
        conflict_type: str,
        involved_agents: Set[str],
        details: Dict[str, Any]
    ):
        """Report a conflict for resolution."""
        # Find coordinator
        coordinators = [
            a for a in self._agents.values()
            if a.role == AgentRole.COORDINATOR
        ]

        if coordinators:
            coordinator = coordinators[0]
            await self.message_bus.send(Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.CONFLICT_REPORT,
                sender_id=reporter_id,
                recipient_id=coordinator.agent_id,
                content={
                    "conflict_type": conflict_type,
                    "involved_agents": list(involved_agents),
                    "details": details
                }
            ))

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        active_agents = sum(
            1 for a in self._agents.values() if a.status == "active"
        )
        avg_load = sum(
            a.load for a in self._agents.values()
        ) / len(self._agents) if self._agents else 0

        return {
            "total_agents": len(self._agents),
            "active_agents": active_agents,
            "average_load": avg_load,
            "active_tasks": len(self._task_assignments),
            "coordination_protocol": self.protocol.value
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_multi_agent_coordinator(
    protocol: CoordinationProtocol = CoordinationProtocol.HIERARCHICAL
) -> MultiAgentCoordinator:
    """Create a multi-agent coordinator."""
    return MultiAgentCoordinator(protocol)
