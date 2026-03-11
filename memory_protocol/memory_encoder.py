"""
MEMORY ENCODER MODULE
Converts raw experience into structured memory
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY

Handles:
- Event -> memory transformation
- Metadata attachment
- Confidence estimation
- Initial salience scoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import json
import re


class MemoryType(Enum):
    """Types of memories that can be encoded."""
    EPISODIC = "episodic"       # Specific events/experiences
    SEMANTIC = "semantic"       # Facts and knowledge
    STRATEGIC = "strategic"     # Plans and strategies
    CONSTRAINT = "constraint"   # Rules and boundaries
    PROCEDURAL = "procedural"   # How to do things


class SourceType(Enum):
    """Sources of memory content."""
    OBSERVATION = "observation"
    INFERENCE = "inference"
    USER_INPUT = "user_input"
    SYSTEM = "system"
    AGENT = "agent"
    EXTERNAL_API = "external_api"


@dataclass
class RawExperience:
    """Raw experience to be encoded into memory."""
    content: str
    source_type: SourceType
    source_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedMemory:
    """Fully encoded memory ready for storage."""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    timestamps: Dict[str, datetime]
    scores: Dict[str, float]
    decay: Dict[str, Any]
    provenance: Dict[str, Any]
    embedding: Optional[List[float]]
    associations: Dict[str, Any]
    metadata: Dict[str, Any]


class BaseMemoryEncoder(ABC):
    """Abstract base class for memory encoders."""

    def __init__(self, agent_id: str, namespace: str = "default"):
        self.agent_id = agent_id
        self.namespace = namespace

    @abstractmethod
    def encode(self, experience: RawExperience) -> EncodedMemory:
        """Encode raw experience into structured memory."""
        pass

    @abstractmethod
    def classify_memory_type(self, content: str, context: Dict) -> MemoryType:
        """Classify the type of memory based on content."""
        pass

    @abstractmethod
    def estimate_confidence(self, experience: RawExperience) -> float:
        """Estimate confidence score for the memory."""
        pass

    @abstractmethod
    def calculate_initial_salience(self, experience: RawExperience) -> float:
        """Calculate initial salience score."""
        pass

    def generate_memory_id(self, content: str, timestamp: datetime) -> str:
        """Generate unique memory ID."""
        hash_input = f"{content}{timestamp.isoformat()}{self.agent_id}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:24]
        return f"mem_{hash_value}"


class StandardMemoryEncoder(BaseMemoryEncoder):
    """
    Standard implementation of memory encoder.
    Transforms raw experiences into structured memories.
    """

    def __init__(
        self,
        agent_id: str,
        namespace: str = "default",
        embedding_model: Optional[Any] = None
    ):
        super().__init__(agent_id, namespace)
        self.embedding_model = embedding_model

        # Patterns for classification
        self.constraint_patterns = [
            r"must not", r"never", r"always", r"required",
            r"forbidden", r"mandatory", r"prohibited"
        ]
        self.strategic_patterns = [
            r"plan", r"strategy", r"goal", r"objective",
            r"approach", r"next steps", r"roadmap"
        ]
        self.procedural_patterns = [
            r"how to", r"steps to", r"process for",
            r"procedure", r"workflow", r"instructions"
        ]

    def encode(self, experience: RawExperience) -> EncodedMemory:
        """Encode raw experience into structured memory."""
        memory_id = self.generate_memory_id(
            experience.content,
            experience.timestamp
        )

        memory_type = self.classify_memory_type(
            experience.content,
            experience.context
        )

        content = self._extract_content(experience)
        confidence = self.estimate_confidence(experience)
        salience = self.calculate_initial_salience(experience)

        # Generate embedding if model available
        embedding = None
        if self.embedding_model:
            embedding = self._generate_embedding(experience.content)

        # Build decay profile based on memory type
        decay = self._build_decay_profile(memory_type)

        # Build provenance chain
        provenance = self._build_provenance(experience)

        # Initialize associations
        associations = self._extract_associations(experience)

        return EncodedMemory(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            timestamps={
                "created": experience.timestamp,
                "last_accessed": experience.timestamp,
                "last_modified": experience.timestamp
            },
            scores={
                "confidence": confidence,
                "salience": salience,
                "utility_relevance": 0.5,  # Default, updated by salience engine
                "risk_impact": 0.0,
                "access_frequency": 0.0,
                "reinforcement_count": 0
            },
            decay=decay,
            provenance=provenance,
            embedding=embedding,
            associations=associations,
            metadata={
                "agent_id": self.agent_id,
                "namespace": self.namespace,
                "session_id": experience.metadata.get("session_id"),
                "is_compressed": False
            }
        )

    def classify_memory_type(self, content: str, context: Dict) -> MemoryType:
        """Classify memory type based on content patterns."""
        content_lower = content.lower()

        # Check for constraint patterns
        if any(re.search(p, content_lower) for p in self.constraint_patterns):
            return MemoryType.CONSTRAINT

        # Check for strategic patterns
        if any(re.search(p, content_lower) for p in self.strategic_patterns):
            return MemoryType.STRATEGIC

        # Check for procedural patterns
        if any(re.search(p, content_lower) for p in self.procedural_patterns):
            return MemoryType.PROCEDURAL

        # Check context hints
        if context.get("type") == "fact":
            return MemoryType.SEMANTIC

        # Default to episodic (specific event/experience)
        return MemoryType.EPISODIC

    def estimate_confidence(self, experience: RawExperience) -> float:
        """Estimate confidence based on source reliability."""
        base_confidence = {
            SourceType.SYSTEM: 0.95,
            SourceType.USER_INPUT: 0.85,
            SourceType.OBSERVATION: 0.75,
            SourceType.AGENT: 0.70,
            SourceType.EXTERNAL_API: 0.65,
            SourceType.INFERENCE: 0.55
        }

        confidence = base_confidence.get(experience.source_type, 0.5)

        # Adjust based on context
        if experience.context.get("verified"):
            confidence = min(1.0, confidence + 0.15)

        if experience.context.get("contradicts_existing"):
            confidence = max(0.1, confidence - 0.2)

        if experience.metadata.get("high_quality_source"):
            confidence = min(1.0, confidence + 0.1)

        return round(confidence, 3)

    def calculate_initial_salience(self, experience: RawExperience) -> float:
        """Calculate initial salience based on content characteristics."""
        salience = 0.5  # Base salience

        content = experience.content.lower()

        # Increase salience for important keywords
        important_keywords = [
            "important", "critical", "urgent", "error", "warning",
            "success", "failure", "decision", "deadline", "priority"
        ]
        keyword_matches = sum(1 for kw in important_keywords if kw in content)
        salience += min(0.3, keyword_matches * 0.05)

        # Increase for user-initiated content
        if experience.source_type == SourceType.USER_INPUT:
            salience += 0.15

        # Increase for constraint-related content
        if any(re.search(p, content) for p in self.constraint_patterns):
            salience += 0.2

        # Adjust based on context urgency
        urgency = experience.context.get("urgency", "normal")
        urgency_boost = {"low": -0.1, "normal": 0, "high": 0.15, "critical": 0.3}
        salience += urgency_boost.get(urgency, 0)

        return round(min(1.0, max(0.0, salience)), 3)

    def _extract_content(self, experience: RawExperience) -> Dict[str, Any]:
        """Extract and structure content from experience."""
        raw = experience.content

        # Generate summary (simple truncation for now)
        summary = raw[:200] + "..." if len(raw) > 200 else raw

        # Extract entities (simple pattern matching)
        entities = self._extract_entities(raw)

        # Extract key-value pairs if structured
        structured = experience.context.get("structured_data", {})

        return {
            "raw": raw,
            "summary": summary,
            "structured": structured,
            "key_entities": entities,
            "key_relations": []  # Populated by relation extraction
        }

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text."""
        entities = []

        # Extract capitalized phrases (proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(proper_nouns[:10])

        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        entities.extend(quoted[:5])

        # Extract numbers with context
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|\s*\w+)?\b', text)
        entities.extend(numbers[:5])

        return list(set(entities))

    def _build_decay_profile(self, memory_type: MemoryType) -> Dict[str, Any]:
        """Build decay profile based on memory type."""
        decay_profiles = {
            MemoryType.EPISODIC: {
                "function": "exponential",
                "half_life_hours": 168,  # 1 week
                "minimum_retention": 0.1,
                "is_immutable": False
            },
            MemoryType.SEMANTIC: {
                "function": "power",
                "half_life_hours": 720,  # 1 month
                "minimum_retention": 0.2,
                "is_immutable": False
            },
            MemoryType.STRATEGIC: {
                "function": "linear",
                "half_life_hours": 336,  # 2 weeks
                "minimum_retention": 0.15,
                "is_immutable": False
            },
            MemoryType.CONSTRAINT: {
                "function": "none",
                "half_life_hours": None,
                "minimum_retention": 1.0,
                "is_immutable": True
            },
            MemoryType.PROCEDURAL: {
                "function": "power",
                "half_life_hours": 2160,  # 3 months
                "minimum_retention": 0.3,
                "is_immutable": False
            }
        }

        return decay_profiles.get(memory_type, decay_profiles[MemoryType.EPISODIC])

    def _build_provenance(self, experience: RawExperience) -> Dict[str, Any]:
        """Build provenance information."""
        return {
            "source_type": experience.source_type.value,
            "source_id": experience.source_id,
            "timestamp": experience.timestamp.isoformat(),
            "verification_status": "unverified",
            "chain_of_custody": [
                {
                    "agent_id": self.agent_id,
                    "action": "encoded",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }

    def _extract_associations(self, experience: RawExperience) -> Dict[str, Any]:
        """Extract initial associations from experience."""
        tags = experience.metadata.get("tags", [])

        # Add auto-tags based on memory type
        content_lower = experience.content.lower()
        if "error" in content_lower or "fail" in content_lower:
            tags.append("error")
        if "success" in content_lower or "complete" in content_lower:
            tags.append("success")

        context_keys = list(experience.context.keys())

        return {
            "related_memories": [],
            "tags": list(set(tags)),
            "context_keys": context_keys
        }

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text."""
        if self.embedding_model is None:
            return None

        try:
            # Placeholder for actual embedding generation
            # In production, use: self.embedding_model.encode(text)
            return None
        except Exception:
            return None


class BatchMemoryEncoder(StandardMemoryEncoder):
    """Encoder optimized for batch processing."""

    def encode_batch(
        self,
        experiences: List[RawExperience]
    ) -> List[EncodedMemory]:
        """Encode multiple experiences efficiently."""
        memories = []

        # Batch embedding generation if model available
        if self.embedding_model:
            texts = [e.content for e in experiences]
            # embeddings = self.embedding_model.encode_batch(texts)
            embeddings = [None] * len(texts)  # Placeholder
        else:
            embeddings = [None] * len(experiences)

        for experience, embedding in zip(experiences, embeddings):
            memory = self.encode(experience)
            memory.embedding = embedding
            memories.append(memory)

        return memories


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_memory_encoder(
    agent_id: str,
    namespace: str = "default",
    encoder_type: str = "standard",
    embedding_model: Optional[Any] = None
) -> BaseMemoryEncoder:
    """Create a memory encoder of the specified type."""

    if encoder_type == "standard":
        return StandardMemoryEncoder(agent_id, namespace, embedding_model)
    elif encoder_type == "batch":
        return BatchMemoryEncoder(agent_id, namespace, embedding_model)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
