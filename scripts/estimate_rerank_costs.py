#!/usr/bin/env python3
"""Estimate reranking costs."""


def estimate_monthly_cost(
    queries_per_day: int,
    candidates_per_query: int = 20,
):
    """Estimate monthly Cohere Rerank costs.

    Args:
        queries_per_day: Average queries per day
        candidates_per_query: Number of candidates to rerank

    Returns:
        Monthly cost estimate
    """
    queries_per_month = queries_per_day * 30
    cost_per_1000 = 1.00  # USD for Cohere rerank-english-v3.0

    monthly_cost = (queries_per_month / 1000) * cost_per_1000

    print(f"Rerank Cost Estimate")
    print(f"=" * 40)
    print(f"Queries per day: {queries_per_day}")
    print(f"Queries per month: {queries_per_month}")
    print(f"Candidates per query: {candidates_per_query}")
    print(f"Monthly cost: ${monthly_cost:.2f}")
    print(f"Cost per query: ${monthly_cost/queries_per_month:.4f}")

    return monthly_cost


if __name__ == "__main__":
    scenarios = [
        ("Low usage", 10),
        ("Medium usage", 100),
        ("High usage", 1000),
        ("Enterprise usage", 10000),
    ]

    print("Cohere Rerank Cost Estimates")
    print("=" * 60)
    print("Pricing: $1.00 per 1,000 searches (rerank-english-v3.0)\n")

    for name, queries in scenarios:
        print(f"\n{name} ({queries} queries/day):")
        estimate_monthly_cost(queries)

    print("\n" + "=" * 60)
    print("\nNote: Actual costs may vary based on usage patterns.")
    print("Consider local ColBERT reranking for cost-sensitive applications.")
