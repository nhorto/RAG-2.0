"""Hybrid search combining dense vector search with sparse BM25."""

import logging
from typing import List, Dict, Optional, Literal
from collections import defaultdict
from qdrant_client import models
from qdrant_client.models import SparseVector

from ..database.qdrant_client import QdrantManager, SearchResult, get_qdrant_manager
from ..ingestion.embedding_generator import EmbeddingGenerator, HybridEmbeddingGenerator
from ..utils.config_loader import get_config

try:
    from .reranker import get_reranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Hybrid search combining dense and sparse retrieval."""

    def __init__(
        self,
        qdrant_manager: QdrantManager = None,
        embedding_generator: EmbeddingGenerator = None,
        fusion_method: Literal["rrf", "weighted_sum"] = None,
        dense_weight: float = None,
        sparse_weight: float = None,
        rrf_k: int = None,
    ):
        """Initialize hybrid searcher.

        Args:
            qdrant_manager: Qdrant manager instance
            embedding_generator: Embedding generator instance
            fusion_method: Fusion strategy ("rrf" or "weighted_sum")
            dense_weight: Weight for dense scores in weighted fusion
            sparse_weight: Weight for sparse scores in weighted fusion
            rrf_k: Constant for RRF (default: 60)
        """
        config = get_config()
        retrieval_config = config.retrieval

        # Initialize components
        self.qdrant = qdrant_manager or get_qdrant_manager()
        self.embedder = embedding_generator or EmbeddingGenerator()

        # Fusion configuration
        fusion_config = retrieval_config.get("fusion", {})
        self.fusion_method = fusion_method or fusion_config.get("method", "rrf")
        self.dense_weight = dense_weight or fusion_config.get("dense_weight", 0.7)
        self.sparse_weight = sparse_weight or fusion_config.get("sparse_weight", 0.3)
        self.rrf_k = rrf_k or fusion_config.get("rrf_k", 60)

        # Search configuration
        dense_config = retrieval_config.get("dense_search", {})
        sparse_config = retrieval_config.get("sparse_search", {})

        self.dense_top_k = dense_config.get("top_k", 20)
        self.sparse_top_k = sparse_config.get("top_k", 20)
        self.final_top_k = retrieval_config.get("final_top_k", 10)

        # Reranking configuration
        rerank_config = retrieval_config.get("reranking", {})
        self.rerank_enabled = rerank_config.get("enabled", False)

        if self.rerank_enabled and RERANKER_AVAILABLE:
            rerank_provider = rerank_config.get("provider", "cohere")
            try:
                self.reranker = get_reranker(provider=rerank_provider)
                logger.info(f"Reranking enabled with {rerank_provider}")
            except Exception as e:
                logger.warning(f"Could not initialize reranker: {e}")
                self.rerank_enabled = False
                self.reranker = None
        else:
            self.reranker = None

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = None,
        dense_only: bool = False,
        sparse_only: bool = False,
    ) -> List[SearchResult]:
        """Perform hybrid search.

        Args:
            query: Search query text
            filters: Qdrant filter conditions
            top_k: Number of final results (default from config)
            dense_only: Use only dense search (no hybrid)
            sparse_only: Use only sparse search (no hybrid)

        Returns:
            List of SearchResult objects
        """
        top_k = top_k or self.final_top_k

        # If using prefetch method and hybrid mode, delegate to optimized version
        if self.fusion_method == "prefetch" and not dense_only and not sparse_only:
            results = self.search_with_prefetch(query, filters, top_k=20 if self.rerank_enabled else top_k)
        else:

            # Otherwise, use manual fusion (existing code)
            # Generate hybrid query embeddings (dense + sparse)
            # Check if embedder is HybridEmbeddingGenerator
            if isinstance(self.embedder, HybridEmbeddingGenerator):
                query_embeddings = self.embedder.generate_query_embeddings(query)
                query_dense = query_embeddings["dense"]
                query_sparse = query_embeddings["sparse"]
            else:
                # Fall back to dense-only if using old EmbeddingGenerator
                query_dense = self.embedder.generate_embedding(query)
                query_sparse = None

            # Dense vector search
            if not sparse_only:
                dense_results = self._dense_search(
                    query_dense, filters, self.dense_top_k
                )
            else:
                dense_results = []

            # Sparse BM25 search (using real indexed sparse vectors)
            if not dense_only and query_sparse is not None:
                sparse_results = self._sparse_search(
                    query_sparse, filters, self.sparse_top_k
                )
            else:
                sparse_results = []

            # If only one type of search, return directly (with reranking if enabled)
            if dense_only:
                results = dense_results[:20 if self.rerank_enabled else top_k]
            elif sparse_only:
                results = sparse_results[:20 if self.rerank_enabled else top_k]
            else:
                # Hybrid fusion
                if self.fusion_method == "rrf":
                    fused_results = self._reciprocal_rank_fusion(
                        dense_results, sparse_results, self.rrf_k
                    )
                else:
                    fused_results = self._weighted_fusion(
                        dense_results, sparse_results
                    )
                results = fused_results[:20 if self.rerank_enabled else top_k]

        # Apply reranking if enabled
        if self.rerank_enabled and self.reranker:
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_n=top_k
            )
        else:
            results = results[:top_k]

        return results

    def _dense_search(
        self,
        query_embedding: List[float],
        filters: Optional[any],
        top_k: int,
    ) -> List[SearchResult]:
        """Perform dense vector search.

        Args:
            query_embedding: Query embedding vector
            filters: Metadata filters
            top_k: Number of results

        Returns:
            List of SearchResult objects
        """
        return self.qdrant.search(
            query_vector=query_embedding,
            filters=filters,
            limit=top_k,
        )

    def _sparse_search(
        self,
        query_sparse: Dict,  # Sparse embedding from FastEmbed
        filters: Optional[Dict],
        top_k: int,
    ) -> List[SearchResult]:
        """Perform sparse BM25 search using native Qdrant sparse vectors.

        Args:
            query_sparse: Sparse query embedding (from FastEmbed)
            filters: Metadata filters
            top_k: Number of results

        Returns:
            List of SearchResult objects
        """
        try:
            # Convert sparse dict to SparseVector model
            sparse_vector = SparseVector(
                indices=query_sparse["indices"],
                values=query_sparse["values"]
            )
            # Use Qdrant's native sparse vector search with query_points
            search_results = self.qdrant.client.query_points(
                collection_name=self.qdrant.collection_name,
                query=sparse_vector,
                using="sparse",
                query_filter=filters,
                limit=top_k,
                with_payload=True,
            )

            # Convert to SearchResult objects
            results = []
            for hit in search_results.points:
                results.append(
                    SearchResult(
                        chunk_id=str(hit.id),
                        text=hit.payload.get("text", ""),
                        score=hit.score,
                        metadata=hit.payload,
                    )
                )

            return results

        except Exception as e:
            logger.warning(f"Sparse search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion.

        RRF formula: score(d) = Σ 1 / (k + rank_i(d))

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            k: RRF constant (default: 60)

        Returns:
            Fused and sorted list of SearchResult objects
        """
        rrf_scores = defaultdict(float)
        result_map = {}

        # Score dense results
        for rank, result in enumerate(dense_results, start=1):
            rrf_scores[result.chunk_id] += 1.0 / (k + rank)
            result_map[result.chunk_id] = result

        # Score sparse results
        for rank, result in enumerate(sparse_results, start=1):
            rrf_scores[result.chunk_id] += 1.0 / (k + rank)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Create new SearchResult objects with RRF scores
        fused_results = []
        for chunk_id, score in sorted_ids:
            result = result_map[chunk_id]
            fused_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=score,
                    metadata=result.metadata,
                )
            )

        return fused_results

    def _weighted_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
    ) -> List[SearchResult]:
        """Combine results using weighted score fusion.

        Formula: score = α * dense_score + (1-α) * sparse_score

        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search

        Returns:
            Fused and sorted list of SearchResult objects
        """
        weighted_scores = defaultdict(float)
        result_map = {}

        # Normalize and weight dense scores
        if dense_results:
            max_dense = max(r.score for r in dense_results)
            for result in dense_results:
                normalized_score = result.score / max_dense if max_dense > 0 else 0
                weighted_scores[result.chunk_id] += self.dense_weight * normalized_score
                result_map[result.chunk_id] = result

        # Normalize and weight sparse scores
        if sparse_results:
            max_sparse = max(r.score for r in sparse_results)
            for result in sparse_results:
                normalized_score = result.score / max_sparse if max_sparse > 0 else 0
                weighted_scores[result.chunk_id] += self.sparse_weight * normalized_score
                if result.chunk_id not in result_map:
                    result_map[result.chunk_id] = result

        # Sort by weighted score
        sorted_ids = sorted(
            weighted_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Create new SearchResult objects
        fused_results = []
        for chunk_id, score in sorted_ids:
            result = result_map[chunk_id]
            fused_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=score,
                    metadata=result.metadata,
                )
            )

        return fused_results

    def search_with_prefetch(
        self,
        query: str,
        filters: Optional[any] = None,
        top_k: int = None,
    ) -> List[SearchResult]:
        """Perform hybrid search using Qdrant's native prefetch fusion.

        This is more efficient than manual RRF as fusion happens server-side.

        Args:
            query: Search query text
            filters: Qdrant filter conditions
            top_k: Number of final results

        Returns:
            List of SearchResult objects
        """
        from qdrant_client.models import Prefetch

        top_k = top_k or self.final_top_k

        # Generate hybrid query embeddings
        if isinstance(self.embedder, HybridEmbeddingGenerator):
            query_embeddings = self.embedder.generate_query_embeddings(query)
        else:
            # Fallback to dense-only
            query_dense = self.embedder.generate_embedding(query)
            return self._dense_search(query_dense, filters, top_k)

        try:
            # Convert sparse dict to SparseVector model
            sparse_query = SparseVector(
                indices=query_embeddings["sparse"]["indices"],
                values=query_embeddings["sparse"]["values"]
            )
            # Use Qdrant's prefetch for hybrid search
            search_results = self.qdrant.client.query_points(
                collection_name=self.qdrant.collection_name,
                prefetch=[
                    # Dense semantic search
                    Prefetch(
                        using="dense",
                        query=query_embeddings["dense"],
                        limit=self.dense_top_k,
                        filter=filters,
                    ),
                    # Sparse keyword search
                    Prefetch(
                        using="sparse",
                        query=sparse_query,
                        limit=self.sparse_top_k,
                        filter=filters,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

            # Convert to SearchResult objects
            results = []
            for hit in search_results.points:
                results.append(
                    SearchResult(
                        chunk_id=str(hit.id),
                        text=hit.payload.get("text", ""),
                        score=hit.score,
                        metadata=hit.payload,
                    )
                )

            return results

        except Exception as e:
            logger.warning(f"Prefetch search failed, falling back to manual RRF: {e}")
            # Fallback to manual RRF if prefetch fails
            return self.search(query, filters, top_k)
