"""Query analysis for detecting decomposition and augmentation needs."""

import re
from dataclasses import dataclass
from typing import List, Tuple

from .types import QueryComplexity, ProcessingCost


@dataclass
class QueryAnalysis:
    """Analysis results for query complexity and processing needs."""

    needs_decomposition: bool
    decomposition_confidence: float
    decomposition_reasons: List[str]

    needs_augmentation: bool
    augmentation_confidence: float
    augmentation_reasons: List[str]

    query_complexity: QueryComplexity
    estimated_processing_cost: ProcessingCost


class QueryAnalyzer:
    """Analyze queries to determine if decomposition or augmentation is needed.

    Uses heuristic pattern matching to detect multi-part queries and vague queries.
    Fast (~5ms) with no LLM calls required.
    """

    def __init__(self, config: dict = None):
        """Initialize query analyzer.

        Args:
            config: Optional configuration dictionary with thresholds
        """
        self.config = config or {}
        self.min_decompose_confidence = self.config.get("min_decompose_confidence", 0.6)
        self.min_augment_confidence = self.config.get("min_augment_confidence", 0.5)

        self._init_patterns()

    def _init_patterns(self):
        """Initialize detection patterns for query analysis."""
        self.sequential_patterns = [
            "and then", "after that", "followed by", "next", "finally",
            "first", "second", "third", "then"
        ]

        self.enumeration_patterns = [
            r"\b(first|second|third|fourth|fifth)\b",
            r"\b\d+\.",  # "1.", "2.", etc.
            r"\b[a-c]\)"  # "a)", "b)", etc.
        ]

        self.vague_pronouns = ["this", "that", "it", "these", "those"]
        self.generic_terms = [
            "issue", "problem", "thing", "stuff", "feature",
            "module", "system", "tool", "function"
        ]
        self.incomplete_actions = [
            "export", "import", "create", "delete", "update",
            "set up", "configure", "install", "remove"
        ]

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze query and determine processing needs.

        Args:
            query: User query string to analyze

        Returns:
            QueryAnalysis object with detection results
        """
        decomp_score, decomp_reasons = self._check_decomposition_need(query)
        needs_decomp = decomp_score >= self.min_decompose_confidence

        aug_score, aug_reasons = self._check_augmentation_need(query)
        needs_aug = aug_score >= self.min_augment_confidence

        complexity = self._classify_complexity(decomp_score, aug_score, query)
        cost = self._estimate_cost(needs_decomp, needs_aug)

        return QueryAnalysis(
            needs_decomposition=needs_decomp,
            decomposition_confidence=decomp_score,
            decomposition_reasons=decomp_reasons,
            needs_augmentation=needs_aug,
            augmentation_confidence=aug_score,
            augmentation_reasons=aug_reasons,
            query_complexity=complexity,
            estimated_processing_cost=cost
        )

    def _check_decomposition_need(self, query: str) -> Tuple[float, List[str]]:
        """Check if query needs decomposition using heuristics.

        Args:
            query: User query string

        Returns:
            Tuple of (confidence_score, reasons_list)
        """
        score = 0.0
        reasons = []
        query_lower = query.lower()

        # Sequential indicators (strong signal)
        for pattern in self.sequential_patterns:
            if pattern in query_lower:
                score += 0.4
                reasons.append(f"Sequential indicator: '{pattern}'")
                break

        # Conjunctions (moderate signal)
        if " and " in query_lower or ", and " in query_lower:
            score += 0.3
            reasons.append("Multiple parts connected by 'and'")

        # Multiple questions
        question_marks = query.count("?")
        if question_marks > 1:
            score += 0.4
            reasons.append(f"Multiple questions ({question_marks})")

        # Enumeration
        for pattern in self.enumeration_patterns:
            if re.search(pattern, query_lower):
                score += 0.3
                reasons.append("Enumerated list detected")
                break

        # High clause count
        clause_indicators = [",", " and ", " or ", " then "]
        clause_count = sum(query_lower.count(indicator) for indicator in clause_indicators)
        if clause_count >= 3:
            score += 0.2
            reasons.append(f"High clause count ({clause_count})")

        return min(score, 1.0), reasons

    def _check_augmentation_need(self, query: str) -> Tuple[float, List[str]]:
        """Check if query needs augmentation with domain context.

        Args:
            query: User query string

        Returns:
            Tuple of (confidence_score, reasons_list)
        """
        score = 0.0
        reasons = []
        query_lower = query.lower()
        words = query_lower.split()

        # Vague pronouns (strong signal)
        for pronoun in self.vague_pronouns:
            if f" {pronoun} " in f" {query_lower} ":
                score += 0.4
                reasons.append(f"Vague pronoun: '{pronoun}'")
                break

        # Generic terms (moderate signal)
        for term in self.generic_terms:
            if term in words:
                score += 0.2
                reasons.append(f"Generic term: '{term}'")
                break

        # Very short queries
        if len(words) <= 3:
            score += 0.3
            reasons.append(f"Short query ({len(words)} words)")

        # Incomplete actions
        for action in self.incomplete_actions:
            if action in query_lower and len(words) <= 4:
                score += 0.3
                reasons.append(f"Incomplete action: '{action}'")
                break

        return min(score, 1.0), reasons

    def _classify_complexity(
        self,
        decomp_score: float,
        aug_score: float,
        query: str
    ) -> QueryComplexity:
        """Classify overall query complexity.

        Args:
            decomp_score: Decomposition confidence score
            aug_score: Augmentation confidence score
            query: Original query string

        Returns:
            QueryComplexity enum value
        """
        if decomp_score >= 0.6 or aug_score >= 0.7:
            return QueryComplexity.COMPLEX
        elif decomp_score >= 0.3 or aug_score >= 0.4:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def _estimate_cost(self, needs_decomp: bool, needs_aug: bool) -> ProcessingCost:
        """Estimate processing cost based on required operations.

        Args:
            needs_decomp: Whether decomposition is needed
            needs_aug: Whether augmentation is needed

        Returns:
            ProcessingCost enum value
        """
        if needs_decomp and needs_aug:
            return ProcessingCost.HIGH
        elif needs_decomp or needs_aug:
            return ProcessingCost.MEDIUM
        else:
            return ProcessingCost.LOW
