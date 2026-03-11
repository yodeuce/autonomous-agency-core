"""
FILE 7: memory_retrieval_policy.py
PURPOSE: Controls when and how memory is accessed
ROLE: Prevents hallucination and over-retrieval

Triggers include:
- Decision uncertainty
- Risk threshold breaches
- Policy conflicts
- Novel environment states
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RetrievalTrigger(Enum):
    DECISION_UNCERTAINTY = "decision_uncertainty"
    RISK_THRESHOLD_BREACH = "risk_threshold_breach"
    POLICY_CONFLICT = "policy_conflict"
    NOVEL_STATE = "novel_state"
    EXPLICIT_QUERY = "explicit_query"
    PERIODIC_REFRESH = "periodic_refresh"
    CONSTRAINT_CHECK = "constraint_check"


class RetrievalMode(Enum):
    SIMILARITY = "similarity"
    RECENCY = "recency"
    SALIENCE = "salience"
    TAG_MATCH = "tag_match"
    TYPE_FILTER = "type_filter"
    COMPOSITE = "composite"


@dataclass
class RetrievalRequest:
    """A structured request to retrieve memories."""
    trigger: RetrievalTrigger
    query_context: dict[str, Any]
    mode: RetrievalMode = RetrievalMode.COMPOSITE
    max_results: int = 10
    min_salience: float = 0.1
    min_confidence: float = 0.3
    memory_types: list[str] | None = None
    required_tags: list[str] | None = None
    exclude_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    memories: list[dict[str, Any]]
    trigger: RetrievalTrigger
    retrieval_count: int
    filtered_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval policy."""
    uncertainty_threshold: float = 0.6
    risk_threshold: float = 0.7
    novelty_threshold: float = 0.8
    max_retrievals_per_step: int = 3
    cooldown_steps: int = 5
    over_retrieval_limit: int = 50
    periodic_refresh_interval: int = 100


class MemoryRetrievalPolicy:
    """
    Governs when and how the agent accesses its memory store.
    Prevents both hallucination (using wrong memories) and
    over-retrieval (excessive memory access degrading performance).
    """

    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.retrieval_count_this_step: int = 0
        self.total_retrievals: int = 0
        self.last_retrieval_step: int = -999
        self.current_step: int = 0

    def set_step(self, step: int) -> None:
        """Advance to a new timestep, resetting per-step counters."""
        if step != self.current_step:
            self.current_step = step
            self.retrieval_count_this_step = 0

    def should_retrieve(
        self,
        state: dict[str, Any],
        trigger: RetrievalTrigger | None = None,
    ) -> RetrievalRequest | None:
        """
        Determine if memory retrieval should be triggered.

        Args:
            state: Current agent state
            trigger: Optional explicit trigger

        Returns:
            A RetrievalRequest if retrieval is warranted, else None
        """
        # Rate limiting
        if self.retrieval_count_this_step >= self.config.max_retrievals_per_step:
            logger.debug("Retrieval rate limit reached for this step.")
            return None

        # Cooldown check
        steps_since_last = self.current_step - self.last_retrieval_step
        if steps_since_last < self.config.cooldown_steps and trigger != RetrievalTrigger.CONSTRAINT_CHECK:
            logger.debug("Retrieval in cooldown period.")
            return None

        # Explicit trigger always allowed
        if trigger == RetrievalTrigger.EXPLICIT_QUERY:
            return self._build_explicit_request(state)

        # Constraint checks always allowed
        if trigger == RetrievalTrigger.CONSTRAINT_CHECK:
            return self._build_constraint_request(state)

        # Auto-detect triggers if none specified
        if trigger is None:
            trigger = self._detect_trigger(state)
            if trigger is None:
                return None

        return self._build_request(trigger, state)

    def execute_retrieval(
        self,
        request: RetrievalRequest,
        memory_store: list[dict[str, Any]],
    ) -> RetrievalResult:
        """
        Execute a memory retrieval against the memory store.

        Args:
            request: The retrieval request
            memory_store: Full list of memories

        Returns:
            Filtered and ranked retrieval results
        """
        # Filter by basic criteria
        candidates = self._filter_candidates(request, memory_store)
        initial_count = len(candidates)

        # Rank by retrieval mode
        ranked = self._rank_candidates(request, candidates)

        # Apply result limit
        results = ranked[: request.max_results]

        # Update access metadata
        for memory in results:
            memory["access_count"] = memory.get("access_count", 0) + 1
            memory["last_accessed"] = str(self.current_step)

        # Track retrieval stats
        self.retrieval_count_this_step += 1
        self.total_retrievals += 1
        self.last_retrieval_step = self.current_step

        logger.info(
            f"Retrieved {len(results)}/{initial_count} memories "
            f"(trigger={request.trigger.value})"
        )

        return RetrievalResult(
            memories=results,
            trigger=request.trigger,
            retrieval_count=len(results),
            filtered_count=initial_count,
            metadata={
                "mode": request.mode.value,
                "step": self.current_step,
            },
        )

    # -------------------------------------------------------------------------
    # TRIGGER DETECTION
    # -------------------------------------------------------------------------

    def _detect_trigger(self, state: dict[str, Any]) -> RetrievalTrigger | None:
        """Auto-detect which trigger condition is met."""
        # Decision uncertainty
        uncertainty = state.get("uncertainty", 0.0)
        if uncertainty > self.config.uncertainty_threshold:
            return RetrievalTrigger.DECISION_UNCERTAINTY

        # Risk threshold breach
        risk_level = state.get("risk_level", 0.0)
        if risk_level > self.config.risk_threshold:
            return RetrievalTrigger.RISK_THRESHOLD_BREACH

        # Policy conflict
        if state.get("policy_conflict", False):
            return RetrievalTrigger.POLICY_CONFLICT

        # Novel state
        novelty = state.get("novelty_score", 0.0)
        if novelty > self.config.novelty_threshold:
            return RetrievalTrigger.NOVEL_STATE

        # Periodic refresh
        if (
            self.current_step > 0
            and self.current_step % self.config.periodic_refresh_interval == 0
        ):
            return RetrievalTrigger.PERIODIC_REFRESH

        return None

    # -------------------------------------------------------------------------
    # REQUEST BUILDING
    # -------------------------------------------------------------------------

    def _build_request(
        self,
        trigger: RetrievalTrigger,
        state: dict[str, Any],
    ) -> RetrievalRequest:
        """Build a retrieval request based on trigger type."""
        # Customize based on trigger
        if trigger == RetrievalTrigger.DECISION_UNCERTAINTY:
            return RetrievalRequest(
                trigger=trigger,
                query_context=state,
                mode=RetrievalMode.COMPOSITE,
                max_results=10,
                min_salience=0.2,
                memory_types=["episodic", "semantic", "strategic"],
            )
        elif trigger == RetrievalTrigger.RISK_THRESHOLD_BREACH:
            return RetrievalRequest(
                trigger=trigger,
                query_context=state,
                mode=RetrievalMode.SALIENCE,
                max_results=15,
                min_salience=0.1,
                memory_types=["constraint", "episodic"],
                required_tags=["risk", "constraint", "boundary"],
            )
        elif trigger == RetrievalTrigger.NOVEL_STATE:
            return RetrievalRequest(
                trigger=trigger,
                query_context=state,
                mode=RetrievalMode.SIMILARITY,
                max_results=10,
                min_salience=0.1,
            )
        else:
            return RetrievalRequest(
                trigger=trigger,
                query_context=state,
                mode=RetrievalMode.COMPOSITE,
                max_results=10,
            )

    def _build_explicit_request(self, state: dict[str, Any]) -> RetrievalRequest:
        return RetrievalRequest(
            trigger=RetrievalTrigger.EXPLICIT_QUERY,
            query_context=state,
            mode=RetrievalMode.COMPOSITE,
            max_results=20,
            min_salience=0.0,
            min_confidence=0.1,
        )

    def _build_constraint_request(self, state: dict[str, Any]) -> RetrievalRequest:
        return RetrievalRequest(
            trigger=RetrievalTrigger.CONSTRAINT_CHECK,
            query_context=state,
            mode=RetrievalMode.TYPE_FILTER,
            max_results=50,
            min_salience=0.0,
            memory_types=["constraint"],
        )

    # -------------------------------------------------------------------------
    # FILTERING & RANKING
    # -------------------------------------------------------------------------

    def _filter_candidates(
        self,
        request: RetrievalRequest,
        memory_store: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter memories by request criteria."""
        candidates = []
        for memory in memory_store:
            # Exclude specific IDs
            if memory["memory_id"] in request.exclude_ids:
                continue

            # Superseded memories are skipped
            if memory.get("superseded_by") is not None:
                continue

            # Salience filter
            if memory.get("salience_score", 0) < request.min_salience:
                continue

            # Confidence filter
            if memory.get("confidence_score", 0) < request.min_confidence:
                continue

            # Type filter
            if request.memory_types:
                if memory.get("memory_type") not in request.memory_types:
                    continue

            # Tag filter
            if request.required_tags:
                memory_tags = set(memory.get("tags", []))
                if not memory_tags & set(request.required_tags):
                    continue

            candidates.append(memory)

        return candidates

    def _rank_candidates(
        self,
        request: RetrievalRequest,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank candidates based on retrieval mode."""
        if request.mode == RetrievalMode.SALIENCE:
            candidates.sort(
                key=lambda m: m.get("salience_score", 0), reverse=True
            )
        elif request.mode == RetrievalMode.RECENCY:
            candidates.sort(
                key=lambda m: m.get("timestamp", ""), reverse=True
            )
        elif request.mode == RetrievalMode.COMPOSITE:
            candidates.sort(
                key=lambda m: (
                    m.get("salience_score", 0) * 0.5
                    + m.get("confidence_score", 0) * 0.3
                    + (m.get("access_count", 0) > 0) * 0.2
                ),
                reverse=True,
            )
        # SIMILARITY mode would use embeddings - placeholder for now
        return candidates
