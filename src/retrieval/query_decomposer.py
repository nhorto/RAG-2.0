"""Query decomposition for breaking complex multi-part queries into sub-queries."""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from ..utils.llm_client import LLMClient
from .query_analyzer import QueryAnalysis
from .types import QueryIntent, ConnectionLogic, ExecutionStrategy

logger = logging.getLogger(__name__)


@dataclass
class SubQuery:
    """Represents a decomposed sub-query."""

    query_text: str
    order: int
    dependency: Optional[str]
    intent: QueryIntent


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""

    original_query: str
    sub_queries: List[SubQuery]
    connection_logic: ConnectionLogic
    execution_strategy: ExecutionStrategy


class QueryDecomposer:
    """Decompose complex queries into atomic sub-queries using LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_sub_queries: int = 5
    ):
        """Initialize query decomposer.

        Args:
            llm_client: LLMClient instance for LLM calls
            max_sub_queries: Maximum number of sub-queries to generate
        """
        self.llm_client = llm_client
        self.max_sub_queries = max_sub_queries

    def decompose(self, query: str, analysis: QueryAnalysis) -> DecomposedQuery:
        """Decompose complex query into sub-queries.

        Args:
            query: Original user query
            analysis: Query analysis results from QueryAnalyzer

        Returns:
            DecomposedQuery with sub-queries and execution strategy
        """
        prompt = self._build_decomposition_prompt(query, analysis)

        try:
            response = self.llm_client.call_with_retry(
                system_prompt=self._get_system_prompt(),
                user_prompt=prompt,
                response_format={"type": "json_object"}
            )

            result = json.loads(response)

            sub_queries = [
                SubQuery(
                    query_text=sq["query_text"],
                    order=sq["order"],
                    dependency=sq.get("dependency"),
                    intent=QueryIntent(sq.get("intent", "factual"))
                )
                for sq in result["sub_queries"]
            ]

            if len(sub_queries) > self.max_sub_queries:
                sub_queries = sub_queries[:self.max_sub_queries]

            return DecomposedQuery(
                original_query=query,
                sub_queries=sub_queries,
                connection_logic=ConnectionLogic(result.get("connection_logic", "SEQUENTIAL")),
                execution_strategy=ExecutionStrategy(result.get("execution_strategy", "sequential"))
            )

        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return self._create_fallback_result(query)

    def _get_system_prompt(self) -> str:
        """Get system prompt for query decomposition."""
        return """You are a query decomposition expert for a RAG system focused on steel fabrication consulting and Tekla PowerFab software.

Your task: Break complex multi-part queries into simple, independently answerable sub-queries.

Domain context:
- Tekla PowerFab is steel fabrication software with modules: Estimating, Production Control, Purchasing, Inventory, Shipping
- Common workflows: BOM (Bill of Materials) creation, Work Order management, Material tracking, Report generation
- Common topics: Troubleshooting, configuration, data import/export, permissions, integrations

Rules for decomposition:
1. Each sub-query should be atomic (one clear information need)
2. Preserve domain context (mention "PowerFab", module names when relevant)
3. Maintain chronological order for sequential steps
4. Use clear, specific language
5. Avoid pronouns - use explicit nouns (e.g., "BOM" not "it")
6. Keep sub-queries independent (each should be answerable alone)

Connection logic:
- "SEQUENTIAL": Steps that follow a process (A, then B, then C)
- "AND": Multiple independent topics that are all needed
- "OR": Alternative solutions or parallel options

Execution strategy:
- "sequential": Execute in order (for procedural workflows)
- "parallel": Execute concurrently (for independent topics)
- "conditional": Evaluate based on results (for multi-hop, future)

Return JSON with this exact structure:
{
  "sub_queries": [
    {"query_text": "How to create a Bill of Materials in PowerFab Estimating?", "order": 0, "dependency": null, "intent": "procedural"},
    {"query_text": "How to assign a BOM to a Work Order in PowerFab?", "order": 1, "dependency": null, "intent": "procedural"}
  ],
  "connection_logic": "SEQUENTIAL",
  "execution_strategy": "sequential"
}"""

    def _build_decomposition_prompt(self, query: str, analysis: QueryAnalysis) -> str:
        """Build user prompt for decomposition."""
        reasons = ", ".join(analysis.decomposition_reasons)

        return f"""Decompose this query about Tekla PowerFab consulting:

Query: "{query}"

Detected complexity indicators: {reasons}

Break this into simple sub-queries that can be answered independently. Each sub-query should:
- Be a complete, specific question
- Include domain context (e.g., "in PowerFab Estimating module")
- Avoid pronouns (use explicit nouns like "BOM", "Work Order")
- Be independently answerable

If the query describes sequential steps (A, then B, then C), use "SEQUENTIAL" logic and "sequential" strategy.
If it asks multiple independent questions, use "AND" logic and "parallel" strategy.
If it asks for alternatives, use "OR" logic and "parallel" strategy.

Limit to {self.max_sub_queries} sub-queries maximum."""

    def _create_fallback_result(self, query: str) -> DecomposedQuery:
        """Create fallback result when decomposition fails."""
        return DecomposedQuery(
            original_query=query,
            sub_queries=[
                SubQuery(
                    query_text=query,
                    order=0,
                    dependency=None,
                    intent=QueryIntent.FACTUAL
                )
            ],
            connection_logic=ConnectionLogic.AND,
            execution_strategy=ExecutionStrategy.SEQUENTIAL
        )
