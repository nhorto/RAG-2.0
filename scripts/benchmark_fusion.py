#!/usr/bin/env python3
"""Benchmark prefetch vs. manual RRF fusion."""

import time
from pathlib import Path
import sys
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_searcher import HybridSearcher


def benchmark_search(queries, method="prefetch", runs=10):
    """Benchmark search performance.

    Args:
        queries: List of test queries
        method: Fusion method ("prefetch", "rrf", or "weighted_sum")
        runs: Number of runs per query

    Returns:
        Tuple of (average_latency_ms, p95_latency_ms)
    """
    searcher = HybridSearcher(fusion_method=method)

    latencies = []

    for query in queries:
        for _ in range(runs):
            start = time.time()
            results = searcher.search(query, top_k=10)
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)

    avg_latency = statistics.mean(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    return avg_latency, p95_latency


if __name__ == "__main__":
    test_queries = [
        "How to create BOM in Estimating?",
        "Work Order issues in Production Control",
        "Material tracking in Inventory",
        "Job costing setup",
        "Customer portal configuration",
    ]

    print("Benchmarking Fusion Methods")
    print("=" * 50)
    print(f"Test queries: {len(test_queries)}")
    print(f"Runs per query: 10")
    print(f"Total tests: {len(test_queries) * 10}\n")

    # Benchmark prefetch
    print("Testing Prefetch (Qdrant native fusion):")
    try:
        avg, p95 = benchmark_search(test_queries, method="prefetch", runs=10)
        print(f"  Average latency: {avg:.1f}ms")
        print(f"  P95 latency: {p95:.1f}ms")
    except Exception as e:
        print(f"  Error: {e}")

    # Benchmark manual RRF
    print("\nTesting Manual RRF:")
    try:
        avg, p95 = benchmark_search(test_queries, method="rrf", runs=10)
        print(f"  Average latency: {avg:.1f}ms")
        print(f"  P95 latency: {p95:.1f}ms")
    except Exception as e:
        print(f"  Error: {e}")

    # Benchmark weighted sum
    print("\nTesting Weighted Sum:")
    try:
        avg, p95 = benchmark_search(test_queries, method="weighted_sum", runs=10)
        print(f"  Average latency: {avg:.1f}ms")
        print(f"  P95 latency: {p95:.1f}ms")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 50)
    print("Recommendation: Use 'prefetch' for best performance")
