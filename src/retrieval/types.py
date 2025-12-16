"""Type definitions and enums for retrieval module."""

from enum import Enum


class QueryIntent(str, Enum):
    """Query intent classification."""
    PROCEDURAL = "procedural"
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    TROUBLESHOOTING = "troubleshooting"
    DECISION = "decision"


class QueryComplexity(str, Enum):
    """Query complexity classification."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ProcessingCost(str, Enum):
    """Estimated processing cost level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConnectionLogic(str, Enum):
    """How sub-queries relate to each other."""
    AND = "AND"
    OR = "OR"
    SEQUENTIAL = "SEQUENTIAL"


class ExecutionStrategy(str, Enum):
    """How to execute sub-queries."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"


class AugmentationType(str, Enum):
    """Type of query augmentation performed."""
    DOMAIN_CONTEXT = "domain_context"
    PRONOUN_RESOLUTION = "pronoun_resolution"
    ACTION_COMPLETION = "action_completion"
