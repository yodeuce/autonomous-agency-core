"""
FILE 8: memory_decay_and_compression.py
PURPOSE: Intentional forgetting and summarization
ROLE: Keeps memory lean and strategic
SPEC: CARBON[6] Technical Architecture Specification v1.0.0

Decay Functions (CARBON[6] §3.5):
    Exponential:  S(t) = S₀ · e^(-λt)        [Default for episodic]
    Power Law:    S(t) = S₀ · (1 + t)^(-α)   [For semantic memories]
    Linear:       S(t) = max(S_min, S₀ - kt)  [For time-bounded tasks]
    Step:         S(t) = S₀ if t < T else 0    [For deadline-based events]

Compression Strategy (CARBON[6] §3.5):
    1. Identify Candidates — Memories with salience below compression threshold
    2. Group by Type — Cluster similar low-salience memories
    3. Generate Summary — Create compressed representation preserving key insights
    4. Supersede Originals — Replace individual memories with summary
    5. Update References — Maintain provenance chain to originals

Includes:
- Temporal decay functions (exponential, power law, linear, step)
- Utility-based pruning
- Memory compression rules
- Supersession logic
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DecayConfig:
    """Configuration for memory decay and compression."""
    prune_threshold: float = 0.05
    compression_threshold: float = 0.15
    max_memory_count: int = 10000
    compression_batch_size: int = 5
    min_similarity_for_merge: float = 0.7
    prune_interval_steps: int = 100


class MemoryDecayEngine:
    """
    Manages intentional forgetting and memory compression.
    Ensures the memory store stays lean and strategically relevant.
    """

    def __init__(self, config: DecayConfig | None = None):
        self.config = config or DecayConfig()
        self.current_step: int = 0
        self.pruned_count: int = 0
        self.compressed_count: int = 0

    def set_step(self, step: int) -> None:
        self.current_step = step

    def apply_decay(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Apply decay functions to all memories, updating salience scores.

        Args:
            memories: Full memory store

        Returns:
            Updated memories with decayed salience values
        """
        for memory in memories:
            if memory.get("is_immutable", False):
                continue

            decay_profile = memory.get("decay_profile", {})
            if decay_profile.get("immune", False):
                continue

            created_step = memory.get("created_step", 0)
            age = max(0, self.current_step - created_step)
            function = decay_profile.get("function", "exponential")
            half_life = decay_profile.get("half_life_steps", 500)
            minimum = decay_profile.get("minimum_salience", 0.05)

            original = memory.get("salience_score", 0.5)
            decayed = self._apply_decay_function(original, age, function, half_life)

            # Frequency reinforcement: accessed memories decay slower
            access_count = memory.get("access_count", 0)
            if access_count > 0:
                reinforcement = min(0.3, 0.05 * math.log(1 + access_count))
                decayed += reinforcement

            memory["salience_score"] = max(minimum, min(1.0, decayed))

        return memories

    def prune(self, memories: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Remove memories below the prune threshold.

        Returns:
            (remaining_memories, pruned_memories)
        """
        remaining = []
        pruned = []

        for memory in memories:
            if memory.get("is_immutable", False):
                remaining.append(memory)
                continue

            salience = memory.get("salience_score", 0.0)
            minimum = memory.get("decay_profile", {}).get("minimum_salience", 0.05)

            if salience <= self.config.prune_threshold and salience <= minimum:
                pruned.append(memory)
            else:
                remaining.append(memory)

        self.pruned_count += len(pruned)
        if pruned:
            logger.info(f"Pruned {len(pruned)} memories below threshold.")

        return remaining, pruned

    def compress(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Compress similar low-salience memories into summary memories.
        Merges groups of related memories into single compressed memories.

        Returns:
            Memory store with compressed entries
        """
        # Separate compressible from non-compressible
        compressible = []
        keep = []

        for memory in memories:
            if memory.get("is_immutable", False):
                keep.append(memory)
            elif memory.get("salience_score", 1.0) < self.config.compression_threshold:
                compressible.append(memory)
            else:
                keep.append(memory)

        if len(compressible) < self.config.compression_batch_size:
            return memories  # Not enough to compress

        # Group by memory type and tags
        groups = self._group_for_compression(compressible)

        for group in groups:
            if len(group) < 2:
                keep.extend(group)
                continue

            compressed = self._merge_memories(group)
            keep.append(compressed)
            self.compressed_count += len(group)
            logger.info(
                f"Compressed {len(group)} memories into {compressed['memory_id']}"
            )

        return keep

    def supersede(
        self,
        old_memory: dict[str, Any],
        new_memory: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Mark an old memory as superseded by a new one.

        Returns:
            (updated_old, updated_new) with supersession links
        """
        old_memory["superseded_by"] = new_memory["memory_id"]
        old_memory["salience_score"] = 0.0  # Will be pruned next cycle

        if "associations" not in new_memory:
            new_memory["associations"] = []
        new_memory["associations"].append({
            "memory_id": old_memory["memory_id"],
            "relationship": "supersedes",
            "strength": 1.0,
        })

        logger.info(
            f"Memory {old_memory['memory_id']} superseded by {new_memory['memory_id']}"
        )
        return old_memory, new_memory

    def enforce_capacity(
        self, memories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Enforce maximum memory capacity.
        If over limit, prune lowest-salience non-immutable memories.
        """
        if len(memories) <= self.config.max_memory_count:
            return memories

        # Sort non-immutable by salience
        immutable = [m for m in memories if m.get("is_immutable", False)]
        mutable = [m for m in memories if not m.get("is_immutable", False)]
        mutable.sort(key=lambda m: m.get("salience_score", 0.0), reverse=True)

        # Keep top memories up to limit
        keep_count = self.config.max_memory_count - len(immutable)
        kept_mutable = mutable[:keep_count]
        pruned_count = len(mutable) - keep_count

        logger.warning(
            f"Memory capacity enforced: pruned {pruned_count} lowest-salience memories."
        )
        return immutable + kept_mutable

    def run_maintenance(
        self, memories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Full maintenance cycle: decay -> compress -> prune -> enforce capacity.
        """
        memories = self.apply_decay(memories)
        memories = self.compress(memories)
        memories, _ = self.prune(memories)
        memories = self.enforce_capacity(memories)
        return memories

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------

    def _apply_decay_function(
        self,
        value: float,
        age: int,
        function: str,
        half_life: int,
    ) -> float:
        """
        Apply a specific decay function.

        Formal Definitions (CARBON[6] §3.5):
            Exponential:  S(t) = S₀ · e^(-λt)        [Default for episodic]
            Power Law:    S(t) = S₀ · (1 + t)^(-α)   [For semantic memories]
            Linear:       S(t) = max(S_min, S₀ - kt)  [For time-bounded tasks]
            Step:         S(t) = S₀ if t < T else 0    [For deadline-based events]
        """
        if half_life <= 0 or function == "none":
            return value

        if function == "exponential":
            # S(t) = S₀ · e^(-λt), where λ = ln(2)/half_life
            return value * math.exp(-0.693 * age / half_life)
        elif function == "power_law":
            # S(t) = S₀ · (1 + t)^(-α), where α derived from half_life
            alpha = 0.693 / math.log(1 + half_life) if half_life > 0 else 1.0
            return value * (1 + age) ** (-alpha)
        elif function == "linear":
            # S(t) = max(S_min, S₀ - kt)
            factor = max(0.0, 1.0 - (age / (2.0 * half_life)))
            return value * factor
        elif function == "step":
            # S(t) = S₀ if t < T else 0
            return value if age < half_life else 0.0
        return value

    def _group_for_compression(
        self, memories: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Group memories by type and overlapping tags for compression."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for memory in memories:
            key = memory.get("memory_type", "unknown")
            groups.setdefault(key, []).append(memory)

        result = []
        for group_memories in groups.values():
            # Split into batches
            for i in range(0, len(group_memories), self.config.compression_batch_size):
                batch = group_memories[i : i + self.config.compression_batch_size]
                if len(batch) >= 2:
                    result.append(batch)
                else:
                    result.append(batch)
        return result

    def _merge_memories(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge a group of memories into a single compressed memory."""
        summaries = [
            m.get("content", {}).get("summary", "") for m in group
        ]
        combined_summary = " | ".join(s for s in summaries if s)

        # Aggregate metrics
        avg_confidence = sum(
            m.get("confidence_score", 0.5) for m in group
        ) / len(group)
        max_salience = max(m.get("salience_score", 0.0) for m in group)
        max_risk = max(m.get("risk_impact", 0.0) for m in group)

        # Merge tags
        all_tags: set[str] = set()
        for m in group:
            all_tags.update(m.get("tags", []))

        original_ids = [m["memory_id"] for m in group]

        return {
            "memory_id": str(uuid.uuid4()),
            "memory_type": group[0].get("memory_type", "episodic"),
            "content": {
                "summary": f"[Compressed from {len(group)} memories] {combined_summary}",
                "data": {"source_count": len(group)},
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_score": avg_confidence,
            "salience_score": max_salience * 1.1,  # Slight boost for compressed
            "utility_relevance": max(
                m.get("utility_relevance", 0.0) for m in group
            ),
            "risk_impact": max_risk,
            "source_provenance": {
                "source_type": "inference",
                "source_id": "compression_engine",
                "reliability": avg_confidence,
            },
            "decay_profile": group[0].get("decay_profile", {
                "function": "exponential",
                "half_life_steps": 1000,
                "minimum_salience": 0.1,
                "immune": False,
            }),
            "tags": list(all_tags),
            "associations": [],
            "is_immutable": False,
            "compressed": True,
            "original_memory_ids": original_ids,
            "access_count": 0,
        }
