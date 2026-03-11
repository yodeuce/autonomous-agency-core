"""
FILE 6: memory_salience_engine.py
PURPOSE: Computes priority of memories
ROLE: Ensures the agent recalls what matters most
SPEC: CARBON[6] Technical Architecture Specification v1.0.0

Formal Salience Formula (CARBON[6] §3.3):
    S(m) = w₁·EMV_impact(m) + w₂·risk_factor(m) + w₃·recency(m) + w₄·frequency(m)

    Where:
        EMV_impact(m)  = normalized absolute impact on expected value
        risk_factor(m) = amplification for risk-related memories
        recency(m)     = exponential decay based on time since creation
        frequency(m)   = log(access_count + 1) / log(max_access + 1)

    Default Weights (CARBON[6] §3.3):
        | Factor     | Weight | Rationale                           |
        |-----------|--------|-------------------------------------|
        | EMV Impact | 0.35  | Primary driver of decision relevance |
        | Risk Factor| 0.25  | Safety-critical memories amplified   |
        | Recency    | 0.25  | Recent experiences more relevant     |
        | Frequency  | 0.15  | Frequently accessed = frequently useful|

Logic includes:
- EMV impact weighting
- Risk amplification
- Recency decay
- Frequency reinforcement
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SalienceWeights:
    """Configurable weights for salience computation (CARBON[6] §3.3)."""
    emv_impact: float = 0.35
    risk_amplification: float = 0.25
    recency: float = 0.25
    frequency: float = 0.15


class MemorySalienceEngine:
    """
    Computes and updates the salience (priority) of memories.
    Determines what the agent should recall when making decisions.
    """

    def __init__(self, weights: SalienceWeights | None = None, max_access: int = 100):
        self.weights = weights or SalienceWeights()
        self.current_step: int = 0
        self.max_access: int = max_access  # For frequency normalization

    def set_step(self, step: int) -> None:
        """Update the current timestep for recency calculations."""
        self.current_step = step

    def compute_salience(
        self,
        memory: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> float:
        """
        Compute the current salience score for a memory.

        Args:
            memory: Memory dict conforming to memory_schema.json
            context: Optional current decision context for relevance scoring

        Returns:
            Salience score in [0.0, 1.0]
        """
        if memory.get("is_immutable", False):
            return 1.0

        decay = memory.get("decay_profile", {})
        if decay.get("immune", False):
            return 1.0

        # Compute components
        emv_score = self._emv_impact_score(memory, context)
        risk_score = self._risk_amplification_score(memory, context)
        recency_score = self._recency_score(memory)
        frequency_score = self._frequency_score(memory)

        # Weighted combination: S(m) = w₁·EMV + w₂·risk + w₃·recency + w₄·frequency
        salience = (
            self.weights.emv_impact * emv_score
            + self.weights.risk_amplification * risk_score
            + self.weights.recency * recency_score
            + self.weights.frequency * frequency_score
        )

        # Apply decay function
        salience = self._apply_decay(salience, memory)

        return max(0.0, min(1.0, salience))

    def rank_memories(
        self,
        memories: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank memories by salience and return top-k.

        Returns:
            List of (memory_id, salience_score) sorted descending
        """
        scored = [
            (m["memory_id"], self.compute_salience(m, context))
            for m in memories
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored = scored[:top_k]

        return scored

    def batch_update_salience(
        self,
        memories: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recompute salience for all memories and update in place.

        Returns:
            Updated memories with new salience_score values
        """
        for memory in memories:
            memory["salience_score"] = self.compute_salience(memory, context)
        return memories

    # -------------------------------------------------------------------------
    # COMPONENT SCORES
    # -------------------------------------------------------------------------

    def _emv_impact_score(
        self,
        memory: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> float:
        """Score based on how much this memory affects EMV decisions."""
        utility = memory.get("utility_relevance", 0.0)
        # Convert from [-1, 1] to [0, 1], with higher weight for large impacts
        score = (abs(utility) + utility) / 2.0  # Positive impact
        negative_penalty = max(0, -utility) * 0.5  # Negative impact still relevant

        base = score + negative_penalty

        # Context relevance boost
        if context:
            context_tags = set(context.get("relevant_tags", []))
            memory_tags = set(memory.get("tags", []))
            overlap = len(context_tags & memory_tags)
            if overlap > 0:
                base *= 1.0 + (overlap * 0.2)

        return min(1.0, base)

    def _risk_amplification_score(
        self,
        memory: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> float:
        """
        Risk-related memories get amplified salience.
        Critical for survival - the agent must remember dangers.
        """
        risk_impact = memory.get("risk_impact", 0.0)

        # Non-linear amplification: high-risk memories are disproportionately salient
        amplified = risk_impact ** 0.5 if risk_impact > 0 else 0.0

        # Further amplification if current context is risky
        if context and context.get("risk_level", 0) > 0.5:
            amplified *= 1.5

        return min(1.0, amplified)

    def _recency_score(self, memory: dict[str, Any]) -> float:
        """
        More recent memories are more salient.
        Uses exponential decay from creation time.
        """
        if self.current_step == 0:
            return 0.5  # Neutral if step not set

        # Approximate age from access_count as proxy for steps since creation
        created_step = memory.get("created_step", 0)
        age = max(0, self.current_step - created_step)

        if age == 0:
            return 1.0

        # Exponential decay with configurable half-life
        half_life = memory.get("decay_profile", {}).get("half_life_steps", 500)
        if half_life <= 0:
            return 1.0

        return math.exp(-0.693 * age / half_life)  # 0.693 = ln(2)

    def _frequency_score(self, memory: dict[str, Any]) -> float:
        """
        Frequently accessed memories are reinforced.

        Formula (CARBON[6] §3.3):
            frequency(m) = log(access_count + 1) / log(max_access + 1)
        """
        access_count = memory.get("access_count", 0)
        if access_count <= 0:
            return 0.0

        max_access = max(self.max_access, access_count)
        return math.log(access_count + 1) / math.log(max_access + 1)

    # -------------------------------------------------------------------------
    # DECAY APPLICATION
    # -------------------------------------------------------------------------

    def _apply_decay(self, salience: float, memory: dict[str, Any]) -> float:
        """Apply the memory's decay profile to the salience score."""
        decay = memory.get("decay_profile", {})
        function = decay.get("function", "exponential")
        minimum = decay.get("minimum_salience", 0.05)

        if function == "none" or decay.get("immune", False):
            return salience

        if self.current_step == 0:
            return salience

        created_step = memory.get("created_step", 0)
        age = max(0, self.current_step - created_step)
        half_life = decay.get("half_life_steps", 500)

        if half_life <= 0:
            return salience

        if function == "exponential":
            decay_factor = math.exp(-0.693 * age / half_life)
        elif function == "linear":
            decay_factor = max(0.0, 1.0 - (age / (2.0 * half_life)))
        elif function == "step":
            decay_factor = 1.0 if age < half_life else 0.0
        else:
            decay_factor = 1.0

        decayed = salience * decay_factor
        return max(minimum, decayed)
