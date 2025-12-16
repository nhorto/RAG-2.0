"""Reranking module for refining search results."""

import logging
from typing import List, Optional
import os

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from ..database.qdrant_client import SearchResult
from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


class CohereReranker:
    """Rerank search results using Cohere Rerank API."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        top_n: int = None,
    ):
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Rerank model name
            top_n: Number of results to return after reranking
        """
        if not COHERE_AVAILABLE:
            raise ImportError(
                "Cohere library not installed. Run: pip install cohere"
            )

        config = get_config()
        rerank_config = config.retrieval.get("reranking", {})

        # Get API key
        if api_key is None:
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError(
                    "COHERE_API_KEY not found in environment. "
                    "Please set it in .env file."
                )

        self.client = cohere.Client(api_key=api_key)
        self.model = model or rerank_config.get("model", "rerank-english-v3.0")
        self.top_n = top_n or rerank_config.get("top_k", 5)

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int = None,
    ) -> List[SearchResult]:
        """Rerank search results.

        Args:
            query: Original search query
            results: Search results from hybrid search
            top_n: Number of results to return (default from config)

        Returns:
            Reranked list of SearchResult objects
        """
        top_n = top_n or self.top_n

        if not results:
            return []

        # Prepare documents for reranking
        documents = [result.text for result in results]

        try:
            # Call Cohere Rerank API
            rerank_response = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False,  # We already have the documents
            )

            # Map reranked results back to SearchResult objects
            reranked_results = []
            for hit in rerank_response.results:
                # Get original result
                original_result = results[hit.index]

                # Create new SearchResult with rerank score
                reranked_results.append(
                    SearchResult(
                        chunk_id=original_result.chunk_id,
                        text=original_result.text,
                        score=hit.relevance_score,  # Cohere rerank score
                        metadata={
                            **original_result.metadata,
                            "original_score": original_result.score,
                            "rerank_score": hit.relevance_score,
                        },
                    )
                )

            return reranked_results

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            # Fallback to original results
            return results[:top_n]


class LocalReranker:
    """Rerank using local ColBERT model (FastEmbed)."""

    def __init__(self, model: str = "colbert-ir/colbertv2.0"):
        """Initialize local reranker.

        Args:
            model: ColBERT model name for FastEmbed
        """
        try:
            from fastembed import LateInteractionTextEmbedding
        except ImportError:
            raise ImportError(
                "FastEmbed with late interaction not installed. "
                "Run: pip install fastembed"
            )

        self.model = LateInteractionTextEmbedding(model_name=model)

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int = 5,
    ) -> List[SearchResult]:
        """Rerank using local ColBERT model.

        Args:
            query: Original search query
            results: Search results from hybrid search
            top_n: Number of results to return

        Returns:
            Reranked list of SearchResult objects
        """
        if not results:
            return []

        # Get query embedding
        query_embedding = list(self.model.embed([query]))[0]

        # Get document embeddings
        documents = [result.text for result in results]
        doc_embeddings = list(self.model.embed(documents))

        # Calculate relevance scores (dot product for ColBERT)
        import numpy as np

        scored_results = []
        for i, (result, doc_emb) in enumerate(zip(results, doc_embeddings)):
            # ColBERT uses MaxSim scoring
            # Simplified: average dot product
            score = float(np.mean(np.dot(query_embedding, doc_emb.T)))

            scored_results.append((score, result))

        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Create reranked results
        reranked_results = []
        for score, result in scored_results[:top_n]:
            reranked_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=score,
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                        "rerank_score": score,
                    },
                )
            )

        return reranked_results


# Factory function
def get_reranker(provider: str = "cohere", **kwargs):
    """Get reranker instance.

    Args:
        provider: "cohere" or "local"
        **kwargs: Additional arguments for reranker

    Returns:
        Reranker instance
    """
    if provider == "cohere":
        return CohereReranker(**kwargs)
    elif provider == "local":
        return LocalReranker(**kwargs)
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
