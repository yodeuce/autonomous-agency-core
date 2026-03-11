"""
MEMORY SALIENCE ENGINE
Computes priority of memories for retrieval
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Logic includes:
- EMV impact weighting
- Risk amplification
- Recency decay
- Frequency reinforcement

This ensures the agent recalls what matters most.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import math


class SalienceFactorType(Enum):
    """Types of factors affecting salience."""
    EMV_IMPACT = "emv_impact"
    RISK_AMPLIFICATION = "risk_amplification"
    RECENCY = "recency"
    FREQUENCY = "frequency"
    RELEVANCE = "relevance"
    CONFIDENCE = "confidence"


@dataclass
class SalienceFactor:
    """A single factor contributing to salience."""
    factor_type: SalienceFactorType
    weight: float
    value: float
    contribution: float = 0.0


@dataclass
class SalienceScore:
    """Complete salience score with breakdown."""
    total_score: float
    factors: List[SalienceFactor]
    computed_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SalienceConfig:
    """Configuration for salience computation."""
    # Factor weights (must sum to 1.0)
    emv_weight: float = 0.25
    risk_weight: float = 0.20
    recency_weight: float = 0.20
    frequency_weight: float = 0.15
    relevance_weight: float = 0.15
    confidence_weight: float = 0.05

    # Decay parameters
    recency_half_life_hours: float = 72.0  # 3 days
    frequency_decay_rate: float = 0.95

    # Amplification parameters
    risk_amplification_factor: float = 2.0
    emv_amplification_threshold: float = 0.7

    # Minimum thresholds
    minimum_salience: float = 0.01
    retrieval_threshold: float = 0.3


class MemorySalienceEngine:
    """
    Engine for computing and managing memory salience.
    Determines which memories are most important to recall.
    """

    def __init__(self, config: Optional[SalienceConfig] = None):
        self.config = config or SalienceConfig()
        self._validate_weights()

        # Cache for recent computations
        self._salience_cache: Dict[str, SalienceScore] = {}
        self._cache_ttl = timedelta(minutes=5)

    def _validate_weights(self):
        """Validate that weights sum to approximately 1.0."""
        total = (
            self.config.emv_weight +
            self.config.risk_weight +
            self.config.recency_weight +
            self.config.frequency_weight +
            self.config.relevance_weight +
            self.config.confidence_weight
        )
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Salience weights must sum to 1.0, got {total}")

    def compute_salience(
        self,
        memory: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SalienceScore:
        """
        Compute comprehensive salience score for a memory.

        Args:
            memory: The memory object to score
            context: Current context for relevance computation

        Returns:
            SalienceScore with total and factor breakdown
        """
        memory_id = memory.get("memory_id", "")

        # Check cache
        if memory_id in self._salience_cache:
            cached = self._salience_cache[memory_id]
            if datetime.now() - cached.computed_at < self._cache_ttl:
                return cached

        factors = []

        # Compute each factor
        emv_factor = self._compute_emv_impact(memory, context)
        factors.append(emv_factor)

        risk_factor = self._compute_risk_amplification(memory, context)
        factors.append(risk_factor)

        recency_factor = self._compute_recency_decay(memory)
        factors.append(recency_factor)

        frequency_factor = self._compute_frequency_reinforcement(memory)
        factors.append(frequency_factor)

        relevance_factor = self._compute_relevance(memory, context)
        factors.append(relevance_factor)

        confidence_factor = self._compute_confidence_boost(memory)
        factors.append(confidence_factor)

        # Compute weighted total
        total = sum(f.contribution for f in factors)
        total = max(self.config.minimum_salience, min(1.0, total))

        score = SalienceScore(
            total_score=round(total, 4),
            factors=factors,
            context={"query": context.get("query", "")[:100]}
        )

        # Cache result
        self._salience_cache[memory_id] = score

        return score

    def _compute_emv_impact(
        self,
        memory: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SalienceFactor:
        """
        Compute salience factor based on EMV (Expected Monetary Value) impact.
        Memories related to high-value outcomes get higher salience.
        """
        scores = memory.get("scores", {})
        utility_relevance = scores.get("utility_relevance", 0.5)

        # Check if memory relates to high-value decisions
        content = memory.get("content", {}).get("raw", "").lower()
        high_value_indicators = [
            "revenue", "cost", "profit", "loss", "investment",
            "decision", "critical", "strategic", "milestone"
        ]
        indicator_count = sum(1 for i in high_value_indicators if i in content)
        indicator_boost = min(0.3, indicator_count * 0.05)

        value = utility_relevance + indicator_boost

        # Amplify if above threshold
        if value > self.config.emv_amplification_threshold:
            value = min(1.0, value * 1.2)

        contribution = value * self.config.emv_weight

        return SalienceFactor(
            factor_type=SalienceFactorType.EMV_IMPACT,
            weight=self.config.emv_weight,
            value=round(value, 4),
            contribution=round(contribution, 4)
        )

    def _compute_risk_amplification(
        self,
        memory: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SalienceFactor:
        """
        Compute salience factor with risk amplification.
        Memories related to risks get amplified salience.
        """
        scores = memory.get("scores", {})
        risk_impact = abs(scores.get("risk_impact", 0.0))

        # Check for risk-related content
        content = memory.get("content", {}).get("raw", "").lower()
        risk_indicators = [
            "risk", "danger", "warning", "error", "failure",
            "violation", "breach", "threat", "vulnerability"
        ]
        risk_count = sum(1 for r in risk_indicators if r in content)

        # Apply amplification
        if risk_count > 0 or risk_impact > 0.3:
            value = min(1.0, (risk_impact + risk_count * 0.1) *
                       self.config.risk_amplification_factor)
        else:
            value = risk_impact

        # Context-aware: amplify if current context is risky
        current_risk = context.get("risk_exposure", 0)
        if current_risk > 0.5:
            value = min(1.0, value * 1.3)

        contribution = value * self.config.risk_weight

        return SalienceFactor(
            factor_type=SalienceFactorType.RISK_AMPLIFICATION,
            weight=self.config.risk_weight,
            value=round(value, 4),
            contribution=round(contribution, 4)
        )

    def _compute_recency_decay(self, memory: Dict[str, Any]) -> SalienceFactor:
        """
        Compute salience factor based on memory recency.
        More recent memories have higher salience.
        """
        timestamps = memory.get("timestamps", {})

        # Use last accessed time, falling back to created time
        last_accessed = timestamps.get("last_accessed")
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
        elif last_accessed is None:
            created = timestamps.get("created")
            if isinstance(created, str):
                last_accessed = datetime.fromisoformat(created.replace("Z", "+00:00"))
            else:
                last_accessed = created or datetime.now()

        # Handle timezone-naive comparison
        now = datetime.now()
        if last_accessed.tzinfo is not None:
            last_accessed = last_accessed.replace(tzinfo=None)

        # Calculate hours since last access
        hours_elapsed = (now - last_accessed).total_seconds() / 3600

        # Exponential decay
        half_life = self.config.recency_half_life_hours
        decay_factor = math.pow(0.5, hours_elapsed / half_life)

        # Check if memory is immutable (no decay)
        decay_profile = memory.get("decay", {})
        if decay_profile.get("is_immutable"):
            decay_factor = 1.0

        value = decay_factor
        contribution = value * self.config.recency_weight

        return SalienceFactor(
            factor_type=SalienceFactorType.RECENCY,
            weight=self.config.recency_weight,
            value=round(value, 4),
            contribution=round(contribution, 4)
        )

    def _compute_frequency_reinforcement(
        self,
        memory: Dict[str, Any]
    ) -> SalienceFactor:
        """
        Compute salience factor based on access frequency.
        Frequently accessed memories get reinforced.
        """
        scores = memory.get("scores", {})
        access_frequency = scores.get("access_frequency", 0)
        reinforcement_count = scores.get("reinforcement_count", 0)

        # Logarithmic scaling to prevent runaway reinforcement
        if access_frequency > 0:
            frequency_value = math.log1p(access_frequency) / 10
        else:
            frequency_value = 0

        # Add reinforcement bonus
        reinforcement_bonus = min(0.3, reinforcement_count * 0.02)

        value = min(1.0, frequency_value + reinforcement_bonus)
        contribution = value * self.config.frequency_weight

        return SalienceFactor(
            factor_type=SalienceFactorType.FREQUENCY,
            weight=self.config.frequency_weight,
            value=round(value, 4),
            contribution=round(contribution, 4)
        )

    def _compute_relevance(
        self,
        memory: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SalienceFactor:
        """
        Compute salience factor based on contextual relevance.
        How relevant is this memory to the current query/task?
        """
        # Get memory content and context
        content = memory.get("content", {})
        memory_text = content.get("raw", "").lower()
        memory_entities = set(content.get("key_entities", []))
        memory_tags = set(memory.get("associations", {}).get("tags", []))

        # Get current context
        query = context.get("query", "").lower()
        context_entities = set(context.get("entities", []))
        context_tags = set(context.get("tags", []))
        current_task = context.get("task_type", "")

        relevance = 0.0

        # Text overlap (simple word matching)
        query_words = set(query.split())
        memory_words = set(memory_text.split())
        if query_words and memory_words:
            overlap = len(query_words & memory_words)
            relevance += min(0.4, overlap / len(query_words) * 0.4)

        # Entity overlap
        if context_entities and memory_entities:
            entity_overlap = len(context_entities & memory_entities)
            relevance += min(0.3, entity_overlap / len(context_entities) * 0.3)

        # Tag overlap
        if context_tags and memory_tags:
            tag_overlap = len(context_tags & memory_tags)
            relevance += min(0.2, tag_overlap / len(context_tags) * 0.2)

        # Task type matching
        memory_type = memory.get("memory_type", "")
        task_type_relevance = {
            ("strategic", "planning"): 0.2,
            ("procedural", "execution"): 0.2,
            ("constraint", "validation"): 0.25,
            ("semantic", "research"): 0.15
        }
        for (mem_type, task), boost in task_type_relevance.items():
            if memory_type == mem_type and current_task == task:
                relevance += boost
                break

        value = min(1.0, relevance)
        contribution = value * self.config.relevance_weight

        return SalienceFactor(
            factor_type=SalienceFactorType.RELEVANCE,
            weight=self.config.relevance_weight,
            value=round(value, 4),
            contribution=round(contribution, 4)
        )

    def _compute_confidence_boost(self, memory: Dict[str, Any]) -> SalienceFactor:
        """
        Compute salience factor based on memory confidence.
        Higher confidence memories get slight boost.
        """
        scores = memory.get("scores", {})
        confidence = scores.get("confidence", 0.5)

        # Only boost high-confidence memories
        if confidence > 0.8:
            value = confidence
        else:
            value = confidence * 0.8  # Slight penalty for low confidence

        contribution = value * self.config.confidence_weight

        return SalienceFactor(
            factor_type=SalienceFactorType.CONFIDENCE,
            weight=self.config.confidence_weight,
            value=round(value, 4),
            contribution=round(contribution, 4)
        )

    def rank_memories(
        self,
        memories: List[Dict[str, Any]],
        context: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        Rank memories by salience score.

        Returns:
            List of (memory, score) tuples, sorted by score descending
        """
        scored = []
        for memory in memories:
            score = self.compute_salience(memory, context)
            scored.append((memory, score))

        # Sort by total score descending
        scored.sort(key=lambda x: x[1].total_score, reverse=True)

        # Filter by threshold
        scored = [
            (m, s) for m, s in scored
            if s.total_score >= self.config.retrieval_threshold
        ]

        if top_k:
            scored = scored[:top_k]

        return scored

    def update_salience_on_access(
        self,
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update memory salience scores after access.
        Reinforces frequently accessed memories.
        """
        scores = memory.get("scores", {})
        scores["access_frequency"] = scores.get("access_frequency", 0) + 1
        scores["reinforcement_count"] = scores.get("reinforcement_count", 0) + 1

        timestamps = memory.get("timestamps", {})
        timestamps["last_accessed"] = datetime.now().isoformat()

        memory["scores"] = scores
        memory["timestamps"] = timestamps

        # Invalidate cache
        memory_id = memory.get("memory_id", "")
        if memory_id in self._salience_cache:
            del self._salience_cache[memory_id]

        return memory

    def clear_cache(self):
        """Clear the salience computation cache."""
        self._salience_cache.clear()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_salience_engine(
    config: Optional[SalienceConfig] = None
) -> MemorySalienceEngine:
    """Create a memory salience engine with the given config."""
    return MemorySalienceEngine(config)
