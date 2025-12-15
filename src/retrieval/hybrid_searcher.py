"""Hybrid search combining dense vector search with sparse BM25."""

from typing import List, Dict, Optional, Literal
from collections import defaultdict

from ..database.qdrant_client import QdrantManager, SearchResult, get_qdrant_manager
from ..ingestion.embedding_generator import EmbeddingGenerator
from ..utils.config_loader import get_config


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

    def search(
        self,
        query: str,
        filters: Optional[any] = None,
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

        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)

        # Dense vector search
        if not sparse_only:
            dense_results = self._dense_search(
                query_embedding, filters, self.dense_top_k
            )
        else:
            dense_results = []

        # Sparse BM25 search (simulated - Qdrant native sparse vectors require v1.7+)
        if not dense_only:
            sparse_results = self._sparse_search(query, filters, self.sparse_top_k)
        else:
            sparse_results = []

        # If only one type of search, return directly
        if dense_only:
            return dense_results[:top_k]
        if sparse_only:
            return sparse_results[:top_k]

        # Hybrid fusion
        if self.fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(
                dense_results, sparse_results, self.rrf_k
            )
        else:
            fused_results = self._weighted_fusion(
                dense_results, sparse_results
            )

        return fused_results[:top_k]

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
        query: str,
        filters: Optional[any],
        top_k: int,
    ) -> List[SearchResult]:
        """Perform sparse BM25 search.

        Note: This is a simplified implementation using keyword matching.
        For true BM25, Qdrant v1.7+ with sparse vectors is recommended.

        Args:
            query: Query text
            filters: Metadata filters
            top_k: Number of results

        Returns:
            List of SearchResult objects (scored by keyword overlap)
        """
        # Tokenize query
        query_tokens = set(query.lower().split())

        # For now, do a broad search and score by keyword overlap
        # In production with Qdrant v1.7+, use native sparse vectors

        # Get candidate chunks (this is a simplified approach)
        # In real implementation, Qdrant would handle BM25 scoring natively
        try:
            # Scroll through collection to score by keywords
            # This is inefficient and just for demonstration
            # Real implementation would use Qdrant's sparse vector support

            candidates = []
            offset = None
            max_candidates = 1000  # Limit for performance

            while len(candidates) < max_candidates:
                points, offset = self.qdrant.scroll_points(
                    limit=100, offset=offset, with_vectors=False
                )

                if not points:
                    break

                for point in points:
                    text = point["payload"].get("text", "").lower()
                    text_tokens = set(text.split())

                    # Calculate simple overlap score
                    overlap = len(query_tokens & text_tokens)

                    if overlap > 0:
                        candidates.append((overlap, point))

                if offset is None:
                    break

            # Sort by overlap score and convert to SearchResult
            candidates.sort(key=lambda x: x[0], reverse=True)

            results = []
            for score, point in candidates[:top_k]:
                results.append(
                    SearchResult(
                        chunk_id=str(point["id"]),
                        text=point["payload"].get("text", ""),
                        score=float(score) / len(query_tokens),  # Normalize
                        metadata=point["payload"],
                    )
                )

            return results

        except Exception as e:
            print(f"Warning: Sparse search failed: {e}")
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


# For testing
if __name__ == "__main__":
    # This would require a running Qdrant instance with data
    # Example usage:

    searcher = HybridSearcher()

    query = "How do I create a BOM in Estimating?"

    try:
        results = searcher.search(query, top_k=5)

        print(f"Found {len(results)} results for: {query}\n")

        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.4f}")
            print(f"   Text: {result.text[:100]}...")
            print()

    except Exception as e:
        print(f"Search failed (Qdrant may not be running): {e}")
