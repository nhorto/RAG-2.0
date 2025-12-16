#!/usr/bin/env python3
"""Test reranking quality."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_searcher import HybridSearcher


def test_with_and_without_reranking(query: str):
    """Compare results with and without reranking."""

    print(f"\nQuery: {query}")
    print("=" * 80)

    # Without reranking
    searcher_no_rerank = HybridSearcher()
    searcher_no_rerank.rerank_enabled = False
    results_no_rerank = searcher_no_rerank.search(query, top_k=5)

    print("\nWithout Reranking:")
    for i, result in enumerate(results_no_rerank, 1):
        print(f"{i}. Score: {result.score:.4f}")
        print(f"   {result.text[:100]}...\n")

    # With reranking
    searcher_rerank = HybridSearcher()
    # Will use reranking if enabled in config
    results_rerank = searcher_rerank.search(query, top_k=5)

    print("\nWith Reranking:")
    for i, result in enumerate(results_rerank, 1):
        original_score = result.metadata.get("original_score", "N/A")
        rerank_score = result.metadata.get("rerank_score", result.score)
        if isinstance(original_score, float):
            print(f"{i}. Original: {original_score:.4f} → Rerank: {rerank_score:.4f}")
        else:
            print(f"{i}. Score: {rerank_score:.4f}")
        print(f"   {result.text[:100]}...\n")


if __name__ == "__main__":
    test_queries = [
        "How to create BOM in Estimating?",
        "Work Order production issues",
        "Customer portal setup",
    ]

    for query in test_queries:
        test_with_and_without_reranking(query)
        print("\n" + "=" * 80 + "\n")
