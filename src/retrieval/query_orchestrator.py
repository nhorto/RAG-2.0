"""Query orchestration module for multi-query retrieval and result merging.

This module coordinates:
1. Execution of decomposed sub-queries (parallel or sequential)
2. Execution of augmented query variants
3. Result merging and deduplication

Author: Atlas (Principal Software Engineer)
Created: 2025-12-15
"""

import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..database.qdrant_client import SearchResult
from .query_decomposer import DecomposedQuery
from .query_augmenter import AugmentedQuery
from .types import ExecutionStrategy, ConnectionLogic

logger = logging.getLogger(__name__)


@dataclass
class OrchestratedResults:
    """Results from orchestrated multi-query retrieval.

    Attributes:
        results_by_subquery: Dict mapping sub-query order to results
        merged_results: Final merged and ranked results
        execution_time_ms: Total execution time in milliseconds
    """
    results_by_subquery: Dict[int, List[SearchResult]]
    merged_results: List[SearchResult]
    execution_time_ms: float


class QueryOrchestrator:
    """Orchestrate retrieval for decomposed and augmented queries.

    This class manages the execution of multiple queries and merges their
    results with deduplication. It supports both parallel and sequential
    execution strategies.
    """

    def __init__(
        self,
        hybrid_searcher,  # Type: HybridSearcher (avoid circular import)
        parallel_workers: int = 5,
        enable_deduplication: bool = True
    ):
        """Initialize query orchestrator.

        Args:
            hybrid_searcher: HybridSearcher instance for executing searches
            parallel_workers: Number of parallel workers for concurrent execution
            enable_deduplication: Whether to deduplicate results by chunk_id
        """
        self.searcher = hybrid_searcher
        self.parallel_workers = parallel_workers
        self.enable_deduplication = enable_deduplication

    def execute_decomposed(
        self,
        decomposed: DecomposedQuery,
        filters: Optional[Dict] = None,
        top_k_per_query: int = 10,
        final_top_k: int = 20
    ) -> OrchestratedResults:
        """Execute retrieval for decomposed query.

        This method handles both sequential and parallel execution based on
        the decomposed query's execution strategy.

        Args:
            decomposed: Decomposed query with sub-queries
            filters: Optional metadata filters to apply to all sub-queries
            top_k_per_query: Number of results to retrieve per sub-query
            final_top_k: Number of final merged results

        Returns:
            OrchestratedResults with per-query and merged results
        """
        start_time = time.time()

        # Execute sub-queries based on strategy
        if decomposed.execution_strategy == ExecutionStrategy.PARALLEL:
            results_map = self._execute_parallel(
                decomposed.sub_queries,
                filters,
                top_k_per_query
            )
        else:  # sequential
            results_map = self._execute_sequential(
                decomposed.sub_queries,
                filters,
                top_k_per_query
            )

        # Merge results based on connection logic
        if decomposed.connection_logic == ConnectionLogic.OR:
            merged = self._merge_or_logic(results_map, final_top_k)
        else:  # AND or SEQUENTIAL
            merged = self._merge_and_logic(results_map, final_top_k)

        execution_time = (time.time() - start_time) * 1000

        return OrchestratedResults(
            results_by_subquery=results_map,
            merged_results=merged,
            execution_time_ms=execution_time
        )

    def execute_augmented(
        self,
        augmented: AugmentedQuery,
        filters: Optional[Dict] = None,
        top_k_per_variant: int = 5,
        final_top_k: int = 10
    ) -> List[SearchResult]:
        """Execute retrieval for augmented query variants.

        This method searches with all augmented variants in parallel and
        merges the results with deduplication.

        Args:
            augmented: Augmented query with variants
            filters: Optional metadata filters
            top_k_per_variant: Results to retrieve per variant
            final_top_k: Number of final merged results

        Returns:
            Merged and deduplicated results sorted by score
        """
        all_results = []
        seen_ids = set()

        # Execute searches for all variants in parallel
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all variant searches
            future_to_variant = {
                executor.submit(
                    self.searcher.search,
                    variant,
                    filters,
                    top_k_per_variant
                ): variant
                for variant in augmented.augmented_variants
            }

            # Collect results as they complete
            for future in as_completed(future_to_variant):
                variant = future_to_variant[future]
                try:
                    results = future.result()

                    # Deduplicate by chunk_id
                    for result in results:
                        chunk_id = self._get_chunk_id(result)
                        if chunk_id not in seen_ids:
                            all_results.append(result)
                            seen_ids.add(chunk_id)

                except Exception as e:
                    logger.warning(f"Variant search failed for '{variant}': {e}")
                    continue

        # Sort by score descending
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:final_top_k]

    def _execute_parallel(
        self,
        sub_queries: List,  # List[SubQuery]
        filters: Optional[Dict],
        top_k: int
    ) -> Dict[int, List[SearchResult]]:
        """Execute sub-queries in parallel using ThreadPoolExecutor.

        Args:
            sub_queries: List of SubQuery objects
            filters: Optional metadata filters
            top_k: Results per sub-query

        Returns:
            Dict mapping sub-query order to results
        """
        results_map = {}

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            # Submit all sub-query searches
            future_to_order = {
                executor.submit(
                    self.searcher.search,
                    sq.query_text,
                    filters,
                    top_k
                ): sq.order
                for sq in sub_queries
            }

            # Collect results as they complete
            for future in as_completed(future_to_order):
                order = future_to_order[future]
                try:
                    results = future.result()
                    results_map[order] = results
                except Exception as e:
                    logger.warning(f"Sub-query {order} failed: {e}")
                    results_map[order] = []

        return results_map

    def _execute_sequential(
        self,
        sub_queries: List,  # List[SubQuery]
        filters: Optional[Dict],
        top_k: int
    ) -> Dict[int, List[SearchResult]]:
        """Execute sub-queries sequentially in order.

        Args:
            sub_queries: List of SubQuery objects
            filters: Optional metadata filters
            top_k: Results per sub-query

        Returns:
            Dict mapping sub-query order to results
        """
        results_map = {}

        # Execute in order
        for sq in sorted(sub_queries, key=lambda x: x.order):
            try:
                results = self.searcher.search(
                    sq.query_text,
                    filters,
                    top_k
                )
                results_map[sq.order] = results
            except Exception as e:
                logger.warning(f"Sub-query {sq.order} failed: {e}")
                results_map[sq.order] = []

        return results_map

    def _merge_or_logic(
        self,
        results_map: Dict[int, List[SearchResult]],
        top_k: int
    ) -> List[SearchResult]:
        """Merge results with OR logic (union, ranked by score).

        For OR queries, we want the union of all results, prioritizing
        the highest scoring chunks across all sub-queries.

        Args:
            results_map: Dict mapping sub-query order to results
            top_k: Number of final results

        Returns:
            Merged results sorted by score
        """
        all_results = []
        seen_ids = set()

        # Collect all unique results
        for results in results_map.values():
            for result in results:
                chunk_id = self._get_chunk_id(result)
                if chunk_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(chunk_id)

        # Sort by score descending
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:top_k]

    def _merge_and_logic(
        self,
        results_map: Dict[int, List[SearchResult]],
        top_k: int
    ) -> List[SearchResult]:
        """Merge results with AND logic (interleave, preserve order).

        For AND/SEQUENTIAL queries, we interleave results to ensure
        representation from all sub-queries while preserving relevance.

        Args:
            results_map: Dict mapping sub-query order to results
            top_k: Number of final results

        Returns:
            Merged results with round-robin interleaving
        """
        merged = []
        seen_ids = set()

        # Interleave results from sub-queries in round-robin fashion
        max_len = max(len(results) for results in results_map.values()) if results_map else 0

        for i in range(max_len):
            for order in sorted(results_map.keys()):
                results = results_map[order]
                if i < len(results):
                    result = results[i]
                    chunk_id = self._get_chunk_id(result)

                    # Avoid duplicates
                    if self.enable_deduplication and chunk_id in seen_ids:
                        continue

                    merged.append(result)
                    seen_ids.add(chunk_id)

                    # Stop if we have enough results
                    if len(merged) >= top_k:
                        return merged

        return merged

    def _get_chunk_id(self, result: SearchResult) -> str:
        """Extract chunk ID from search result for deduplication.

        Args:
            result: SearchResult object

        Returns:
            Unique chunk identifier
        """
        return result.chunk_id
