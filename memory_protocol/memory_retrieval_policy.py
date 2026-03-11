"""
MEMORY RETRIEVAL POLICY
Controls when and how memory is accessed
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Triggers include:
- Decision uncertainty
- Risk threshold breaches
- Policy conflicts
- Novel environment states

Prevents hallucination and over-retrieval.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import hashlib


class RetrievalTrigger(Enum):
    """Events that trigger memory retrieval."""
    DECISION_UNCERTAINTY = "decision_uncertainty"
    RISK_THRESHOLD_BREACH = "risk_threshold_breach"
    POLICY_CONFLICT = "policy_conflict"
    NOVEL_STATE = "novel_state"
    EXPLICIT_REQUEST = "explicit_request"
    PERIODIC_REFRESH = "periodic_refresh"
    CONTEXT_SWITCH = "context_switch"
    ERROR_RECOVERY = "error_recovery"


class RetrievalMode(Enum):
    """Modes of memory retrieval."""
    SEMANTIC = "semantic"       # Content similarity
    TEMPORAL = "temporal"       # Time-based
    ASSOCIATIVE = "associative" # Related memories
    CONSTRAINT = "constraint"   # Rules/boundaries only
    COMPREHENSIVE = "comprehensive"  # All modes


@dataclass
class RetrievalRequest:
    """A request to retrieve memories."""
    query: str
    trigger: RetrievalTrigger
    mode: RetrievalMode
    context: Dict[str, Any]
    max_results: int = 10
    min_salience: float = 0.3
    memory_types: Optional[List[str]] = None
    time_window: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    memories: List[Dict[str, Any]]
    request: RetrievalRequest
    total_candidates: int
    filtered_count: int
    retrieval_time_ms: float
    cache_hit: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval policy."""
    # Trigger thresholds
    uncertainty_threshold: float = 0.4  # Below this, trigger retrieval
    risk_threshold: float = 0.7  # Above this, trigger retrieval
    novelty_threshold: float = 0.8  # State novelty score

    # Retrieval limits
    max_results_per_query: int = 20
    max_retrievals_per_minute: int = 30
    min_interval_between_same_query_ms: int = 1000

    # Anti-hallucination
    require_source_verification: bool = True
    max_inference_chain_length: int = 3

    # Cache settings
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000

    # Resource limits
    max_memory_scan_size: int = 10000
    timeout_ms: int = 5000


class BaseRetrievalPolicy(ABC):
    """Abstract base class for retrieval policies."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._retrieval_log: List[Tuple[datetime, RetrievalRequest]] = []
        self._cache: Dict[str, Tuple[datetime, RetrievalResult]] = {}

    @abstractmethod
    def should_retrieve(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[RetrievalTrigger]]:
        """Determine if memory retrieval should be triggered."""
        pass

    @abstractmethod
    def build_retrieval_request(
        self,
        trigger: RetrievalTrigger,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RetrievalRequest:
        """Build a retrieval request based on trigger and context."""
        pass

    @abstractmethod
    def filter_results(
        self,
        candidates: List[Dict[str, Any]],
        request: RetrievalRequest
    ) -> List[Dict[str, Any]]:
        """Filter and validate retrieval results."""
        pass

    def check_rate_limit(self) -> bool:
        """Check if retrieval is within rate limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        recent = [
            log for log in self._retrieval_log
            if log[0] > minute_ago
        ]

        return len(recent) < self.config.max_retrievals_per_minute

    def log_retrieval(self, request: RetrievalRequest):
        """Log a retrieval for rate limiting."""
        self._retrieval_log.append((datetime.now(), request))

        # Prune old entries
        cutoff = datetime.now() - timedelta(minutes=5)
        self._retrieval_log = [
            log for log in self._retrieval_log
            if log[0] > cutoff
        ]

    def get_cached(self, query_hash: str) -> Optional[RetrievalResult]:
        """Get cached retrieval result."""
        if query_hash in self._cache:
            timestamp, result = self._cache[query_hash]
            if (datetime.now() - timestamp).seconds < self.config.cache_ttl_seconds:
                result.cache_hit = True
                return result
            else:
                del self._cache[query_hash]
        return None

    def cache_result(self, query_hash: str, result: RetrievalResult):
        """Cache a retrieval result."""
        # Enforce cache size limit
        if len(self._cache) >= self.config.cache_max_entries:
            # Remove oldest entries
            sorted_cache = sorted(
                self._cache.items(),
                key=lambda x: x[1][0]
            )
            for key, _ in sorted_cache[:len(sorted_cache)//4]:
                del self._cache[key]

        self._cache[query_hash] = (datetime.now(), result)


class StandardRetrievalPolicy(BaseRetrievalPolicy):
    """
    Standard implementation of memory retrieval policy.
    Balances retrieval utility against resource cost.
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        salience_engine: Optional[Any] = None
    ):
        super().__init__(config)
        self.salience_engine = salience_engine
        self._recent_states: List[Dict[str, Any]] = []
        self._state_history_size = 100

    def should_retrieve(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[RetrievalTrigger]]:
        """
        Determine if memory retrieval should be triggered.
        Evaluates multiple conditions and returns the most relevant trigger.
        """
        # Check rate limits first
        if not self.check_rate_limit():
            return False, None

        # Priority 1: Explicit request
        if context.get("explicit_memory_request"):
            return True, RetrievalTrigger.EXPLICIT_REQUEST

        # Priority 2: Error recovery
        if state.get("error_state") or context.get("recovering_from_error"):
            return True, RetrievalTrigger.ERROR_RECOVERY

        # Priority 3: Risk threshold breach
        risk_exposure = state.get("risk_exposure", 0)
        if risk_exposure > self.config.risk_threshold:
            return True, RetrievalTrigger.RISK_THRESHOLD_BREACH

        # Priority 4: Policy conflict
        if state.get("constraint_violations", 0) > 0 or context.get("policy_conflict"):
            return True, RetrievalTrigger.POLICY_CONFLICT

        # Priority 5: Decision uncertainty
        confidence = state.get("confidence_level", 1.0)
        if confidence < self.config.uncertainty_threshold:
            return True, RetrievalTrigger.DECISION_UNCERTAINTY

        # Priority 6: Novel environment state
        novelty = self._compute_state_novelty(state)
        if novelty > self.config.novelty_threshold:
            return True, RetrievalTrigger.NOVEL_STATE

        # Priority 7: Context switch
        if context.get("context_switched"):
            return True, RetrievalTrigger.CONTEXT_SWITCH

        # No trigger conditions met
        return False, None

    def build_retrieval_request(
        self,
        trigger: RetrievalTrigger,
        state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RetrievalRequest:
        """Build optimized retrieval request based on trigger type."""

        # Base query from context
        query = context.get("query", context.get("task", ""))

        # Determine retrieval mode and parameters based on trigger
        if trigger == RetrievalTrigger.RISK_THRESHOLD_BREACH:
            mode = RetrievalMode.CONSTRAINT
            memory_types = ["constraint", "strategic"]
            min_salience = 0.5
            query = f"risk mitigation {query}"

        elif trigger == RetrievalTrigger.POLICY_CONFLICT:
            mode = RetrievalMode.CONSTRAINT
            memory_types = ["constraint"]
            min_salience = 0.4
            query = f"policy rule constraint {query}"

        elif trigger == RetrievalTrigger.DECISION_UNCERTAINTY:
            mode = RetrievalMode.COMPREHENSIVE
            memory_types = None  # All types
            min_salience = 0.3
            # Enrich query with state context
            query = self._enrich_query_for_uncertainty(query, state)

        elif trigger == RetrievalTrigger.NOVEL_STATE:
            mode = RetrievalMode.SEMANTIC
            memory_types = ["episodic", "semantic"]
            min_salience = 0.25
            query = self._build_novelty_query(state)

        elif trigger == RetrievalTrigger.ERROR_RECOVERY:
            mode = RetrievalMode.COMPREHENSIVE
            memory_types = ["episodic", "procedural"]
            min_salience = 0.3
            error_type = state.get("error_type", "error")
            query = f"error recovery {error_type} {query}"

        elif trigger == RetrievalTrigger.CONTEXT_SWITCH:
            mode = RetrievalMode.ASSOCIATIVE
            memory_types = None
            min_salience = 0.35
            new_context = context.get("new_context", {})
            query = f"{query} {new_context.get('task_type', '')}"

        else:  # EXPLICIT_REQUEST or default
            mode = RetrievalMode.SEMANTIC
            memory_types = None
            min_salience = 0.3

        return RetrievalRequest(
            query=query,
            trigger=trigger,
            mode=mode,
            context=context,
            max_results=self.config.max_results_per_query,
            min_salience=min_salience,
            memory_types=memory_types,
            metadata={
                "state_confidence": state.get("confidence_level"),
                "state_risk": state.get("risk_exposure"),
                "trigger_time": datetime.now().isoformat()
            }
        )

    def filter_results(
        self,
        candidates: List[Dict[str, Any]],
        request: RetrievalRequest
    ) -> List[Dict[str, Any]]:
        """
        Filter and validate retrieval results.
        Prevents hallucination and ensures quality.
        """
        filtered = []
        warnings = []

        for memory in candidates:
            # Check memory type filter
            if request.memory_types:
                mem_type = memory.get("memory_type", "")
                if mem_type not in request.memory_types:
                    continue

            # Check time window filter
            if request.time_window:
                created = memory.get("timestamps", {}).get("created")
                if created:
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    cutoff = datetime.now() - request.time_window
                    if created.replace(tzinfo=None) < cutoff:
                        continue

            # Check source verification if required
            if self.config.require_source_verification:
                provenance = memory.get("provenance", {})
                status = provenance.get("verification_status", "unverified")
                if status == "deprecated":
                    continue
                if status == "unverified":
                    # Add warning but include
                    warnings.append(f"Unverified memory: {memory.get('memory_id')}")

            # Check confidence threshold
            confidence = memory.get("scores", {}).get("confidence", 0)
            if confidence < 0.3:
                warnings.append(f"Low confidence memory: {memory.get('memory_id')}")

            # Validate inference chain length
            provenance = memory.get("provenance", {})
            if provenance.get("source_type") == "inference":
                chain = provenance.get("chain_of_custody", [])
                inference_steps = sum(
                    1 for c in chain if c.get("action") == "inferred"
                )
                if inference_steps > self.config.max_inference_chain_length:
                    warnings.append(
                        f"Long inference chain ({inference_steps}): "
                        f"{memory.get('memory_id')}"
                    )
                    continue

            filtered.append(memory)

        # Apply salience-based ranking if engine available
        if self.salience_engine and filtered:
            ranked = self.salience_engine.rank_memories(
                filtered,
                request.context,
                top_k=request.max_results
            )
            filtered = [m for m, s in ranked if s.total_score >= request.min_salience]

        return filtered[:request.max_results]

    def _compute_state_novelty(self, state: Dict[str, Any]) -> float:
        """
        Compute how novel the current state is compared to recent history.
        Returns value between 0 (seen before) and 1 (completely new).
        """
        if not self._recent_states:
            self._recent_states.append(state.copy())
            return 1.0  # First state is maximally novel

        # Compute state hash for comparison
        state_features = {
            k: v for k, v in state.items()
            if k in ["task_context", "user_urgency", "task_type"]
        }
        state_hash = hashlib.md5(
            str(sorted(state_features.items())).encode()
        ).hexdigest()

        # Check similarity to recent states
        similarities = []
        for past_state in self._recent_states[-20:]:
            past_features = {
                k: v for k, v in past_state.items()
                if k in ["task_context", "user_urgency", "task_type"]
            }
            past_hash = hashlib.md5(
                str(sorted(past_features.items())).encode()
            ).hexdigest()

            # Simple binary similarity
            if state_hash == past_hash:
                similarities.append(1.0)
            else:
                # Compute feature overlap
                overlap = sum(
                    1 for k, v in state_features.items()
                    if past_features.get(k) == v
                )
                similarities.append(overlap / max(len(state_features), 1))

        # Update state history
        self._recent_states.append(state.copy())
        if len(self._recent_states) > self._state_history_size:
            self._recent_states = self._recent_states[-self._state_history_size:]

        # Novelty is inverse of max similarity
        max_similarity = max(similarities) if similarities else 0
        return 1.0 - max_similarity

    def _enrich_query_for_uncertainty(
        self,
        query: str,
        state: Dict[str, Any]
    ) -> str:
        """Enrich query with state context for uncertainty resolution."""
        enrichments = []

        task_context = state.get("task_context", "")
        if task_context:
            enrichments.append(task_context)

        task_type = state.get("task_type", "")
        if task_type:
            enrichments.append(f"task:{task_type}")

        if enrichments:
            return f"{query} {' '.join(enrichments)}"
        return query

    def _build_novelty_query(self, state: Dict[str, Any]) -> str:
        """Build query for novel state exploration."""
        components = []

        task_context = state.get("task_context", "")
        if task_context:
            components.append(f"similar to {task_context}")

        entities = state.get("entities", [])
        if entities:
            components.append(" ".join(entities[:5]))

        return " ".join(components) or "similar situations"


class ConservativeRetrievalPolicy(StandardRetrievalPolicy):
    """
    Conservative retrieval policy that minimizes retrieval frequency.
    Use when resource conservation is priority.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        conservative_config = config or RetrievalConfig()
        # Tighten thresholds
        conservative_config.uncertainty_threshold = 0.25
        conservative_config.risk_threshold = 0.85
        conservative_config.novelty_threshold = 0.9
        conservative_config.max_retrievals_per_minute = 15

        super().__init__(conservative_config)


class AggressiveRetrievalPolicy(StandardRetrievalPolicy):
    """
    Aggressive retrieval policy that retrieves more frequently.
    Use when memory utilization is priority.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        aggressive_config = config or RetrievalConfig()
        # Loosen thresholds
        aggressive_config.uncertainty_threshold = 0.6
        aggressive_config.risk_threshold = 0.5
        aggressive_config.novelty_threshold = 0.6
        aggressive_config.max_retrievals_per_minute = 60

        super().__init__(aggressive_config)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_retrieval_policy(
    policy_type: str = "standard",
    config: Optional[RetrievalConfig] = None,
    salience_engine: Optional[Any] = None
) -> BaseRetrievalPolicy:
    """Create a retrieval policy of the specified type."""

    if policy_type == "standard":
        return StandardRetrievalPolicy(config, salience_engine)
    elif policy_type == "conservative":
        return ConservativeRetrievalPolicy(config)
    elif policy_type == "aggressive":
        return AggressiveRetrievalPolicy(config)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
