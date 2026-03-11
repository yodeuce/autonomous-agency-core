"""
MEMORY DECAY AND COMPRESSION MODULE
Intentional forgetting and summarization
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Includes:
- Temporal decay functions
- Utility-based pruning
- Memory compression rules
- Supersession logic

This keeps memory lean and strategic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import math
import hashlib


class DecayFunction(Enum):
    """Types of memory decay functions."""
    EXPONENTIAL = "exponential"
    POWER = "power"
    LINEAR = "linear"
    STEP = "step"
    NONE = "none"


class CompressionStrategy(Enum):
    """Strategies for memory compression."""
    SUMMARIZE = "summarize"
    MERGE = "merge"
    ABSTRACT = "abstract"
    TRUNCATE = "truncate"


@dataclass
class DecayConfig:
    """Configuration for memory decay."""
    # Default decay parameters
    default_half_life_hours: float = 168.0  # 1 week
    minimum_retention: float = 0.1

    # Decay modifiers
    access_reinforcement_factor: float = 1.2
    importance_retention_boost: float = 0.3

    # Pruning thresholds
    prune_threshold: float = 0.05
    prune_check_interval_hours: float = 24.0

    # Compression thresholds
    compression_age_hours: float = 720.0  # 30 days
    compression_salience_threshold: float = 0.3
    max_uncompressed_memories: int = 10000


@dataclass
class CompressionConfig:
    """Configuration for memory compression."""
    target_compression_ratio: float = 0.3
    min_content_length: int = 50
    max_compressed_length: int = 500
    preserve_key_entities: bool = True
    preserve_key_relations: bool = True


@dataclass
class DecayResult:
    """Result of applying decay to a memory."""
    memory_id: str
    original_salience: float
    new_salience: float
    decay_factor: float
    should_prune: bool
    should_compress: bool


class MemoryDecayEngine:
    """
    Engine for computing and applying memory decay.
    Implements multiple decay functions.
    """

    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()

    def compute_decay_factor(
        self,
        memory: Dict[str, Any],
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Compute the decay factor for a memory.
        Returns value between 0 (fully decayed) and 1 (no decay).
        """
        current_time = current_time or datetime.now()

        decay_profile = memory.get("decay", {})
        decay_function = DecayFunction(
            decay_profile.get("function", "exponential")
        )

        # Immutable memories don't decay
        if decay_profile.get("is_immutable"):
            return 1.0

        # Get time since last access
        timestamps = memory.get("timestamps", {})
        last_accessed = timestamps.get("last_accessed", timestamps.get("created"))

        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(
                last_accessed.replace("Z", "+00:00")
            ).replace(tzinfo=None)

        hours_elapsed = (current_time - last_accessed).total_seconds() / 3600

        # Get half-life
        half_life = decay_profile.get(
            "half_life_hours",
            self.config.default_half_life_hours
        )

        # Get minimum retention
        min_retention = decay_profile.get(
            "minimum_retention",
            self.config.minimum_retention
        )

        # Compute base decay
        if decay_function == DecayFunction.NONE:
            decay_factor = 1.0

        elif decay_function == DecayFunction.EXPONENTIAL:
            decay_factor = math.pow(0.5, hours_elapsed / half_life)

        elif decay_function == DecayFunction.POWER:
            # Power law decay (forgetting curve)
            decay_factor = 1.0 / (1.0 + (hours_elapsed / half_life))

        elif decay_function == DecayFunction.LINEAR:
            decay_factor = max(0, 1.0 - (hours_elapsed / (half_life * 2)))

        elif decay_function == DecayFunction.STEP:
            # Step function: full retention until half_life, then drop
            decay_factor = 1.0 if hours_elapsed < half_life else 0.5

        else:
            decay_factor = 1.0

        # Apply access reinforcement
        scores = memory.get("scores", {})
        access_freq = scores.get("access_frequency", 0)
        if access_freq > 0:
            reinforcement = math.log1p(access_freq) * 0.05
            decay_factor = min(1.0, decay_factor + reinforcement)

        # Apply importance boost
        salience = scores.get("salience", 0.5)
        if salience > 0.7:
            decay_factor = min(
                1.0,
                decay_factor + self.config.importance_retention_boost * (salience - 0.7)
            )

        # Enforce minimum retention
        decay_factor = max(min_retention, decay_factor)

        return round(decay_factor, 4)

    def apply_decay(
        self,
        memory: Dict[str, Any],
        current_time: Optional[datetime] = None
    ) -> DecayResult:
        """Apply decay to a memory and determine if it should be pruned/compressed."""
        decay_factor = self.compute_decay_factor(memory, current_time)

        scores = memory.get("scores", {})
        original_salience = scores.get("salience", 0.5)
        new_salience = original_salience * decay_factor

        # Determine if should prune
        should_prune = (
            new_salience < self.config.prune_threshold and
            not memory.get("decay", {}).get("is_immutable")
        )

        # Determine if should compress
        timestamps = memory.get("timestamps", {})
        created = timestamps.get("created")
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00")).replace(tzinfo=None)

        current_time = current_time or datetime.now()
        age_hours = (current_time - created).total_seconds() / 3600

        should_compress = (
            age_hours > self.config.compression_age_hours and
            new_salience < self.config.compression_salience_threshold and
            not memory.get("metadata", {}).get("is_compressed") and
            not memory.get("decay", {}).get("is_immutable")
        )

        return DecayResult(
            memory_id=memory.get("memory_id", ""),
            original_salience=original_salience,
            new_salience=round(new_salience, 4),
            decay_factor=decay_factor,
            should_prune=should_prune,
            should_compress=should_compress
        )

    def batch_apply_decay(
        self,
        memories: List[Dict[str, Any]],
        current_time: Optional[datetime] = None
    ) -> List[DecayResult]:
        """Apply decay to a batch of memories."""
        return [self.apply_decay(m, current_time) for m in memories]


class MemoryCompressor:
    """
    Compresses memories to reduce storage while preserving essential information.
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        summarizer: Optional[Callable[[str], str]] = None
    ):
        self.config = config or CompressionConfig()
        self.summarizer = summarizer or self._default_summarizer

    def _default_summarizer(self, text: str) -> str:
        """Default summarization: take first and last sentences."""
        sentences = text.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 2:
            return text

        # Take first sentence and last sentence
        summary = f"{sentences[0]}. [...] {sentences[-1]}."
        return summary[:self.config.max_compressed_length]

    def compress(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a single memory."""
        compressed = memory.copy()
        content = memory.get("content", {})

        # Get original content
        raw = content.get("raw", "")

        if len(raw) < self.config.min_content_length:
            return memory  # Too short to compress

        # Generate summary
        summary = self.summarizer(raw)

        # Build compressed content
        compressed_content = {
            "raw": summary,
            "summary": summary,
            "original_length": len(raw),
            "compressed_length": len(summary)
        }

        # Preserve key entities if configured
        if self.config.preserve_key_entities:
            compressed_content["key_entities"] = content.get("key_entities", [])[:10]

        # Preserve key relations if configured
        if self.config.preserve_key_relations:
            compressed_content["key_relations"] = content.get("key_relations", [])[:5]

        compressed["content"] = compressed_content

        # Update metadata
        metadata = compressed.get("metadata", {})
        metadata["is_compressed"] = True
        metadata["compression_ratio"] = round(len(summary) / len(raw), 3)
        metadata["compressed_at"] = datetime.now().isoformat()
        compressed["metadata"] = metadata

        return compressed

    def batch_compress(
        self,
        memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compress a batch of memories."""
        return [self.compress(m) for m in memories]


class SupersessionManager:
    """
    Manages memory supersession - when new memories replace old ones.
    """

    def __init__(self):
        self._supersession_index: Dict[str, str] = {}  # old_id -> new_id

    def should_supersede(
        self,
        new_memory: Dict[str, Any],
        existing_memory: Dict[str, Any]
    ) -> bool:
        """Determine if new memory should supersede existing one."""
        # Check if same topic/entity
        new_entities = set(
            new_memory.get("content", {}).get("key_entities", [])
        )
        existing_entities = set(
            existing_memory.get("content", {}).get("key_entities", [])
        )

        # Require significant entity overlap
        if not new_entities or not existing_entities:
            return False

        overlap = len(new_entities & existing_entities) / len(existing_entities)
        if overlap < 0.5:
            return False

        # Check if new memory is more recent
        new_created = new_memory.get("timestamps", {}).get("created")
        existing_created = existing_memory.get("timestamps", {}).get("created")

        if isinstance(new_created, str):
            new_created = datetime.fromisoformat(new_created.replace("Z", "+00:00"))
        if isinstance(existing_created, str):
            existing_created = datetime.fromisoformat(existing_created.replace("Z", "+00:00"))

        if new_created <= existing_created:
            return False

        # Check if new memory has higher confidence
        new_conf = new_memory.get("scores", {}).get("confidence", 0)
        existing_conf = existing_memory.get("scores", {}).get("confidence", 0)

        if new_conf < existing_conf:
            return False

        # Check memory types compatibility
        new_type = new_memory.get("memory_type")
        existing_type = existing_memory.get("memory_type")

        if new_type != existing_type:
            return False

        return True

    def mark_superseded(
        self,
        old_memory: Dict[str, Any],
        new_memory: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Mark memories with supersession relationship."""
        old_id = old_memory.get("memory_id")
        new_id = new_memory.get("memory_id")

        # Update old memory
        old_metadata = old_memory.get("metadata", {})
        old_metadata["superseded_by"] = new_id
        old_memory["metadata"] = old_metadata

        # Update new memory
        new_metadata = new_memory.get("metadata", {})
        new_metadata["supersedes"] = old_id
        new_memory["metadata"] = new_metadata

        # Track in index
        self._supersession_index[old_id] = new_id

        return old_memory, new_memory

    def get_current_version(self, memory_id: str) -> str:
        """Get the current (non-superseded) version of a memory."""
        current = memory_id
        while current in self._supersession_index:
            current = self._supersession_index[current]
        return current

    def is_superseded(self, memory_id: str) -> bool:
        """Check if a memory has been superseded."""
        return memory_id in self._supersession_index


class MemoryMaintenanceScheduler:
    """
    Schedules and coordinates memory maintenance operations.
    """

    def __init__(
        self,
        decay_engine: MemoryDecayEngine,
        compressor: MemoryCompressor,
        supersession_manager: SupersessionManager
    ):
        self.decay_engine = decay_engine
        self.compressor = compressor
        self.supersession_manager = supersession_manager
        self._last_maintenance: Optional[datetime] = None

    def run_maintenance(
        self,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run full maintenance cycle on memory store.
        Returns statistics about operations performed.
        """
        stats = {
            "total_memories": len(memories),
            "decayed": 0,
            "pruned": 0,
            "compressed": 0,
            "superseded": 0,
            "maintenance_time": datetime.now().isoformat()
        }

        to_prune = []
        to_compress = []
        updated_memories = []

        # Phase 1: Apply decay and identify candidates
        for memory in memories:
            decay_result = self.decay_engine.apply_decay(memory)

            if decay_result.should_prune:
                to_prune.append(memory.get("memory_id"))
                stats["pruned"] += 1
            elif decay_result.should_compress:
                to_compress.append(memory)
            else:
                # Update salience with decay
                memory["scores"]["salience"] = decay_result.new_salience
                updated_memories.append(memory)
                stats["decayed"] += 1

        # Phase 2: Compress eligible memories
        for memory in to_compress:
            compressed = self.compressor.compress(memory)
            updated_memories.append(compressed)
            stats["compressed"] += 1

        # Phase 3: Check for supersession opportunities
        # Group by key entities for supersession check
        entity_groups: Dict[str, List[Dict]] = {}
        for memory in updated_memories:
            entities = tuple(sorted(
                memory.get("content", {}).get("key_entities", [])[:3]
            ))
            if entities:
                if entities not in entity_groups:
                    entity_groups[entities] = []
                entity_groups[entities].append(memory)

        # Check supersession within groups
        for group in entity_groups.values():
            if len(group) > 1:
                # Sort by created time
                group.sort(
                    key=lambda m: m.get("timestamps", {}).get("created", ""),
                    reverse=True
                )
                newest = group[0]
                for older in group[1:]:
                    if self.supersession_manager.should_supersede(newest, older):
                        self.supersession_manager.mark_superseded(older, newest)
                        stats["superseded"] += 1

        stats["remaining_memories"] = len(updated_memories) - stats["pruned"]
        self._last_maintenance = datetime.now()

        return stats


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_decay_engine(
    config: Optional[DecayConfig] = None
) -> MemoryDecayEngine:
    """Create a memory decay engine."""
    return MemoryDecayEngine(config)


def create_compressor(
    config: Optional[CompressionConfig] = None,
    summarizer: Optional[Callable[[str], str]] = None
) -> MemoryCompressor:
    """Create a memory compressor."""
    return MemoryCompressor(config, summarizer)


def create_maintenance_scheduler(
    decay_config: Optional[DecayConfig] = None,
    compression_config: Optional[CompressionConfig] = None
) -> MemoryMaintenanceScheduler:
    """Create a full maintenance scheduler."""
    return MemoryMaintenanceScheduler(
        decay_engine=create_decay_engine(decay_config),
        compressor=create_compressor(compression_config),
        supersession_manager=SupersessionManager()
    )
