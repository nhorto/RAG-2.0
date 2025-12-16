"""Query augmentation module for enhancing vague queries with domain context.

This module uses LLM-based augmentation to:
1. Resolve vague pronouns to likely domain entities
2. Complete incomplete actions with likely objects
3. Add domain-specific context to underspecified queries

Author: Atlas (Principal Software Engineer)
Created: 2025-12-15
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

from ..utils.llm_client import LLMClient
from .query_analyzer import QueryAnalysis
from .types import AugmentationType

logger = logging.getLogger(__name__)


@dataclass
class AugmentedQuery:
    """Result of query augmentation.

    Attributes:
        original_query: The original user query
        augmented_variants: List of augmented query variants
        augmentation_type: Type of augmentation performed
        confidence: Confidence score from analysis
    """
    original_query: str
    augmented_variants: List[str]
    augmentation_type: AugmentationType
    confidence: float


class QueryAugmenter:
    """Augment vague queries with domain-specific context.

    This class uses GPT-3.5-turbo (cheaper model) to enhance underspecified
    queries by adding likely PowerFab domain context and specific entities.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_variants: int = 5,
        domain_vocab: Optional[Dict] = None
    ):
        """Initialize query augmenter.

        Args:
            llm_client: LLMClient instance for LLM calls
            max_variants: Maximum number of variants to generate
            domain_vocab: Optional domain vocabulary dictionary
        """
        self.llm_client = llm_client
        self.max_variants = max_variants
        self.domain_vocab = domain_vocab or {}

    def augment(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict] = None
    ) -> AugmentedQuery:
        """Augment vague query with domain context.

        This method generates 2-5 specific variants of the original query
        by adding PowerFab domain context, module names, and likely entities.

        Args:
            query: Original user query
            analysis: Query analysis results from QueryAnalyzer
            context: Optional conversation context for pronoun resolution

        Returns:
            AugmentedQuery with domain-specific variants

        Raises:
            Exception: If LLM call fails (falls back to rule-based augmentation)
        """
        # Determine augmentation type
        aug_type = self._determine_augmentation_type(analysis.augmentation_reasons)

        # Build augmentation prompt
        prompt = self._build_augmentation_prompt(query, aug_type, context)

        try:
            # Call LLM with retry logic
            response = self.llm_client.call_with_retry(
                system_prompt=self._get_system_prompt(),
                user_prompt=prompt,
                response_format={"type": "json_object"}
            )

            # Parse LLM response
            result = json.loads(response)
            variants = result.get("augmented_queries", [query])

            # Ensure we have 2-5 variants (not too many)
            variants = variants[:self.max_variants]
            if len(variants) < 2:
                variants.append(query)  # Always include original as fallback

            return AugmentedQuery(
                original_query=query,
                augmented_variants=variants,
                augmentation_type=aug_type,
                confidence=analysis.augmentation_confidence
            )

        except Exception as e:
            # Fallback: Use rule-based augmentation
            logger.warning(f"Query augmentation failed: {e}")
            return self._fallback_augmentation(query, aug_type)

    def _determine_augmentation_type(self, reasons: List[str]) -> AugmentationType:
        """Determine what type of augmentation is needed.

        Args:
            reasons: List of augmentation reasons from analysis

        Returns:
            AugmentationType enum value
        """
        reasons_str = " ".join(reasons).lower()

        if "pronoun" in reasons_str:
            return AugmentationType.PRONOUN_RESOLUTION
        elif "incomplete action" in reasons_str:
            return AugmentationType.ACTION_COMPLETION
        elif "generic term" in reasons_str or "short query" in reasons_str:
            return AugmentationType.DOMAIN_CONTEXT
        else:
            return AugmentationType.DOMAIN_CONTEXT  # Default

    def _get_system_prompt(self) -> str:
        """Get system prompt for query augmentation.

        Returns:
            System prompt string optimized for augmentation task
        """
        return """You are a query augmentation expert for a Tekla PowerFab steel fabrication consulting RAG system.

Your task: Enhance vague or underspecified queries with domain-specific context.

Domain context:
- PowerFab modules: Estimating, Production Control, Purchasing, Inventory, Shipping
- Common entities: BOM (Bill of Materials), Work Order (WO), Materials, Reports, Clients, Projects
- Common workflows: BOM creation, Work Order management, Material tracking, Report generation, Data import/export
- Common issues: Data import errors, permissions, configuration, integrations, troubleshooting

Rules for augmentation:
1. Preserve the user's core intent and question type
2. Add likely domain context (module names, workflows, specific entities)
3. Generate 2-5 specific variants (diverse but all relevant)
4. Don't change the question structure - only add context
5. Make variants diverse to cover different interpretations
6. Use specific PowerFab terminology

Guidelines by augmentation type:
- Pronoun resolution: Replace "this", "that", "it" with likely PowerFab entities (BOM, WO, reports, etc.)
- Action completion: Add likely objects to incomplete verbs (export → export BOM/WO/reports)
- Domain context: Add module names and workflow context to generic queries

Return JSON with this exact structure:
{
  "augmented_queries": [
    "How to export BOM reports from PowerFab Estimating?",
    "How to export Work Orders from PowerFab Production Control?",
    "How to export material lists from PowerFab Purchasing?"
  ]
}"""

    def _build_augmentation_prompt(
        self,
        query: str,
        aug_type: AugmentationType,
        context: Optional[Dict] = None
    ) -> str:
        """Build user prompt for augmentation.

        Args:
            query: Original user query
            aug_type: Type of augmentation needed
            context: Optional conversation context

        Returns:
            Formatted prompt string
        """
        context_info = ""
        if context and "last_query" in context:
            context_info = f"\nRecent conversation context: {context['last_query']}"

        # Type-specific instructions
        if aug_type == AugmentationType.PRONOUN_RESOLUTION:
            instruction = "Resolve pronouns (this/that/it) to likely PowerFab entities (BOM, Work Order, reports, materials, etc.)."
        elif aug_type == AugmentationType.ACTION_COMPLETION:
            instruction = "Complete the action with likely PowerFab objects. Consider: BOM, Work Order, reports, materials, data imports/exports."
        else:  # domain_context
            instruction = "Add PowerFab module context and common workflow details. Consider all modules and typical consulting scenarios."

        return f"""Augment this vague query with domain-specific context:

Query: "{query}"

Task: {instruction}{context_info}

Generate 2-5 specific variants that:
- Add PowerFab module names or entity types
- Include relevant workflow context
- Preserve the original question intent
- Are diverse but all plausible interpretations

Generate variants from different perspectives (different modules, different entities, different workflows)."""

    def _fallback_augmentation(self, query: str, aug_type: AugmentationType) -> AugmentedQuery:
        """Rule-based fallback augmentation if LLM fails.

        Args:
            query: Original user query
            aug_type: Type of augmentation needed

        Returns:
            AugmentedQuery with rule-based variants
        """
        variants = [query]  # Always include original

        query_lower = query.lower()

        # Type-specific fallback rules
        if aug_type == AugmentationType.ACTION_COMPLETION:
            # Map common actions to likely completions
            action_completions = {
                "export": [
                    "export BOM reports from PowerFab Estimating",
                    "export Work Orders from PowerFab Production Control",
                    "export material lists from PowerFab"
                ],
                "create": [
                    "create a Bill of Materials in PowerFab Estimating",
                    "create a Work Order in PowerFab Production Control",
                    "create materials in PowerFab Purchasing"
                ],
                "import": [
                    "import data into PowerFab",
                    "import materials into PowerFab",
                    "import BOMs into PowerFab Estimating"
                ],
                "configure": [
                    "configure PowerFab Estimating module",
                    "configure PowerFab Production Control",
                    "configure user permissions in PowerFab"
                ],
            }

            for action, completions in action_completions.items():
                if action in query_lower:
                    variants.extend([f"How to {c}?" for c in completions[:3]])
                    break

        elif aug_type == AugmentationType.DOMAIN_CONTEXT:
            # Add module names to generic queries
            modules = ["Estimating", "Production Control", "Purchasing", "Inventory"]
            for module in modules[:3]:
                variants.append(f"{query} in PowerFab {module}")

        # Limit variants
        variants = variants[:self.max_variants]

        # Ensure at least 2 variants
        if len(variants) < 2:
            variants.append(f"{query} in Tekla PowerFab")

        return AugmentedQuery(
            original_query=query,
            augmented_variants=list(set(variants)),  # Deduplicate
            augmentation_type=aug_type,
            confidence=0.5  # Lower confidence for fallback
        )
