"""
FILE 5: memory_encoder.py
PURPOSE: Converts raw experience into structured memory
ROLE: Event -> memory transformation pipeline

Handles:
- Event to memory transformation
- Metadata attachment
- Confidence estimation
- Initial salience scoring
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RawEvent:
    """A raw experience event before encoding into memory."""
    event_type: str
    data: dict[str, Any]
    source_type: str = "direct_experience"
    source_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class EncodedMemory:
    """A fully encoded memory conforming to memory_schema.json."""
    memory_id: str
    memory_type: str
    content: dict[str, Any]
    timestamp: str
    confidence_score: float
    salience_score: float
    utility_relevance: float
    risk_impact: float
    source_provenance: dict[str, Any]
    decay_profile: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    associations: list[dict[str, Any]] = field(default_factory=list)
    is_immutable: bool = False
    access_count: int = 0
    last_accessed: str = ""
    superseded_by: str | None = None
    compressed: bool = False
    original_memory_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed or self.timestamp,
            "access_count": self.access_count,
            "confidence_score": self.confidence_score,
            "salience_score": self.salience_score,
            "utility_relevance": self.utility_relevance,
            "risk_impact": self.risk_impact,
            "source_provenance": self.source_provenance,
            "decay_profile": self.decay_profile,
            "tags": self.tags,
            "associations": self.associations,
            "is_immutable": self.is_immutable,
            "superseded_by": self.superseded_by,
            "compressed": self.compressed,
            "original_memory_ids": self.original_memory_ids,
        }


class MemoryEncoder:
    """
    Transforms raw events into structured memories.
    Acts as the gateway between experience and the memory store.
    """

    # Memory type classification rules
    MEMORY_TYPE_MAP = {
        "action_outcome": "episodic",
        "observation": "episodic",
        "learned_fact": "semantic",
        "pattern": "semantic",
        "plan": "strategic",
        "goal_update": "strategic",
        "rule": "constraint",
        "boundary": "constraint",
        "skill": "procedural",
        "procedure": "procedural",
    }

    # Default decay profiles by memory type
    DECAY_PROFILES = {
        "episodic": {
            "function": "exponential",
            "half_life_steps": 500,
            "minimum_salience": 0.05,
            "immune": False,
        },
        "semantic": {
            "function": "linear",
            "half_life_steps": 2000,
            "minimum_salience": 0.1,
            "immune": False,
        },
        "strategic": {
            "function": "exponential",
            "half_life_steps": 1000,
            "minimum_salience": 0.1,
            "immune": False,
        },
        "constraint": {
            "function": "none",
            "half_life_steps": 0,
            "minimum_salience": 1.0,
            "immune": True,
        },
        "procedural": {
            "function": "linear",
            "half_life_steps": 5000,
            "minimum_salience": 0.2,
            "immune": False,
        },
    }

    # Source reliability defaults
    SOURCE_RELIABILITY = {
        "direct_experience": 0.9,
        "observation": 0.7,
        "inference": 0.5,
        "external_input": 0.6,
        "human_instruction": 0.95,
    }

    def __init__(self):
        self.encoding_count: int = 0

    def encode(self, event: RawEvent) -> EncodedMemory:
        """
        Encode a raw event into a structured memory.

        Args:
            event: The raw experience event

        Returns:
            A fully structured EncodedMemory
        """
        memory_type = self._classify_memory_type(event)
        confidence = self._estimate_confidence(event)
        salience = self._compute_initial_salience(event, confidence)
        utility_relevance = self._assess_utility_relevance(event)
        risk_impact = self._assess_risk_impact(event)
        tags = self._extract_tags(event)

        memory = EncodedMemory(
            memory_id=str(uuid.uuid4()),
            memory_type=memory_type,
            content={
                "summary": self._generate_summary(event),
                "data": event.data,
            },
            timestamp=event.timestamp,
            confidence_score=confidence,
            salience_score=salience,
            utility_relevance=utility_relevance,
            risk_impact=risk_impact,
            source_provenance={
                "source_type": event.source_type,
                "source_id": event.source_id or self._generate_source_id(event),
                "reliability": self.SOURCE_RELIABILITY.get(event.source_type, 0.5),
            },
            decay_profile=self.DECAY_PROFILES.get(
                memory_type, self.DECAY_PROFILES["episodic"]
            ),
            tags=tags,
            is_immutable=(memory_type == "constraint"),
        )

        self.encoding_count += 1
        logger.info(
            f"Encoded memory {memory.memory_id} "
            f"(type={memory_type}, salience={salience:.3f})"
        )
        return memory

    def encode_batch(self, events: list[RawEvent]) -> list[EncodedMemory]:
        """Encode multiple events into memories."""
        return [self.encode(event) for event in events]

    # -------------------------------------------------------------------------
    # CLASSIFICATION
    # -------------------------------------------------------------------------

    def _classify_memory_type(self, event: RawEvent) -> str:
        """Determine the memory type from the event type."""
        memory_type = self.MEMORY_TYPE_MAP.get(event.event_type)
        if memory_type:
            return memory_type

        # Fallback heuristics
        data = event.data
        if "constraint" in str(data).lower() or "rule" in str(data).lower():
            return "constraint"
        if "plan" in str(data).lower() or "strategy" in str(data).lower():
            return "strategic"
        return "episodic"

    # -------------------------------------------------------------------------
    # CONFIDENCE ESTIMATION
    # -------------------------------------------------------------------------

    def _estimate_confidence(self, event: RawEvent) -> float:
        """
        Estimate confidence in the memory based on source and data quality.
        """
        base_confidence = self.SOURCE_RELIABILITY.get(event.source_type, 0.5)

        # Adjust for data completeness
        data = event.data
        if not data:
            return base_confidence * 0.5

        # More data fields generally means higher confidence
        field_count = len(data)
        completeness_bonus = min(0.1, field_count * 0.01)

        # Check for explicit confidence in the data
        if "confidence" in data:
            explicit = float(data["confidence"])
            return min(1.0, (base_confidence + explicit) / 2.0)

        return min(1.0, base_confidence + completeness_bonus)

    # -------------------------------------------------------------------------
    # SALIENCE SCORING
    # -------------------------------------------------------------------------

    def _compute_initial_salience(self, event: RawEvent, confidence: float) -> float:
        """
        Compute initial salience (importance) score for a new memory.

        Higher salience for:
        - High-confidence memories
        - Risk-related events
        - Constraint/rule memories
        - Unexpected outcomes
        """
        salience = confidence * 0.5  # Base from confidence

        data = event.data

        # Risk events are inherently more salient
        if data.get("risk_level", 0) > 0.5:
            salience += 0.2

        # Constraint memories are always maximally salient
        if event.event_type in ("rule", "boundary"):
            return 1.0

        # Unexpected outcomes are more salient
        if "expected" in data and "actual" in data:
            deviation = abs(data["expected"] - data["actual"])
            salience += min(0.3, deviation)

        # Reward signals boost salience
        if abs(data.get("reward", 0)) > 10.0:
            salience += 0.15

        return min(1.0, max(0.0, salience))

    # -------------------------------------------------------------------------
    # UTILITY & RISK ASSESSMENT
    # -------------------------------------------------------------------------

    def _assess_utility_relevance(self, event: RawEvent) -> float:
        """Assess how relevant this memory is to EMV objectives."""
        data = event.data
        if "emv_impact" in data:
            return max(-1.0, min(1.0, float(data["emv_impact"])))
        if "reward" in data:
            return max(-1.0, min(1.0, float(data["reward"]) / 100.0))
        return 0.0

    def _assess_risk_impact(self, event: RawEvent) -> float:
        """Assess how much this memory relates to risk."""
        data = event.data
        if "risk_level" in data:
            return max(0.0, min(1.0, float(data["risk_level"])))
        if "risk_impact" in data:
            return max(0.0, min(1.0, float(data["risk_impact"])))
        if event.event_type in ("rule", "boundary", "constraint"):
            return 0.8
        return 0.0

    # -------------------------------------------------------------------------
    # METADATA GENERATION
    # -------------------------------------------------------------------------

    def _generate_summary(self, event: RawEvent) -> str:
        """Generate a human-readable summary of the event."""
        data = event.data
        summary = data.get("summary", data.get("description", ""))
        if summary:
            return str(summary)
        return f"{event.event_type} event from {event.source_type}"

    def _extract_tags(self, event: RawEvent) -> list[str]:
        """Extract categorical tags from the event."""
        tags = [event.event_type, event.source_type]
        if "tags" in event.data:
            tags.extend(event.data["tags"])
        return list(set(tags))

    def _generate_source_id(self, event: RawEvent) -> str:
        """Generate a deterministic source ID from event content."""
        raw = f"{event.event_type}:{event.timestamp}:{hash(str(event.data))}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
