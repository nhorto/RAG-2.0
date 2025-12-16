#!/usr/bin/env python3
"""Test complete hybrid search implementation."""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.embedding_generator import HybridEmbeddingGenerator
from src.retrieval.hybrid_searcher import HybridSearcher


def test_hybrid_embeddings():
    """Test hybrid embedding generation."""
    print("=" * 60)
    print("TEST 1: Hybrid Embeddings")
    print("=" * 60)

    try:
        embedder = HybridEmbeddingGenerator()
        query = "How to create BOM in Estimating?"
        embeddings = embedder.generate_query_embeddings(query)

        print(f"Query: {query}")
        print(f"✅ Dense embedding shape: {len(embeddings['dense'])}")
        print(f"✅ Sparse embedding type: {type(embeddings['sparse'])}")
        print("✅ Hybrid embeddings generated successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def test_search_modes():
    """Test different search modes."""
    print("=" * 60)
    print("TEST 2: Search Modes")
    print("=" * 60)

    query = "BOM creation"
    print(f"Query: {query}\n")

    try:
        searcher = HybridSearcher()

        # Test dense only
        print("Testing Dense Search...")
        start = time.time()
        dense_results = searcher.search(query, dense_only=True, top_k=3)
        dense_time = (time.time() - start) * 1000
        print(f"✅ Dense search: {len(dense_results)} results in {dense_time:.1f}ms")

        # Test sparse only (if hybrid embedder is available)
        if hasattr(searcher.embedder, 'generate_query_embeddings'):
            print("\nTesting Sparse Search...")
            start = time.time()
            sparse_results = searcher.search(query, sparse_only=True, top_k=3)
            sparse_time = (time.time() - start) * 1000
            print(f"✅ Sparse search: {len(sparse_results)} results in {sparse_time:.1f}ms")
        else:
            print("\n⚠️  Sparse search requires HybridEmbeddingGenerator")

        # Test hybrid
        print("\nTesting Hybrid Search...")
        start = time.time()
        hybrid_results = searcher.search(query, top_k=3)
        hybrid_time = (time.time() - start) * 1000
        print(f"✅ Hybrid search: {len(hybrid_results)} results in {hybrid_time:.1f}ms")

        print("\n✅ All search modes working!\n")
        return True

    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def test_fusion_methods():
    """Test different fusion methods."""
    print("=" * 60)
    print("TEST 3: Fusion Methods")
    print("=" * 60)

    query = "Work Order issues"
    print(f"Query: {query}\n")

    fusion_methods = ["prefetch", "rrf", "weighted_sum"]
    results = {}

    for method in fusion_methods:
        try:
            print(f"Testing {method.upper()} fusion...")
            searcher = HybridSearcher(fusion_method=method)
            start = time.time()
            search_results = searcher.search(query, top_k=3)
            latency = (time.time() - start) * 1000

            results[method] = {
                "count": len(search_results),
                "latency": latency,
                "success": True
            }
            print(f"✅ {method}: {len(search_results)} results in {latency:.1f}ms")

        except Exception as e:
            print(f"❌ {method}: Error - {e}")
            results[method] = {"success": False}

    print("\n✅ Fusion methods tested!\n")
    return all(r.get("success", False) for r in results.values())


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HYBRID SEARCH COMPLETE TEST SUITE")
    print("=" * 60 + "\n")

    results = []

    results.append(("Hybrid Embeddings", test_hybrid_embeddings()))
    results.append(("Search Modes", test_search_modes()))
    results.append(("Fusion Methods", test_fusion_methods()))

    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
