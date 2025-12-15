#!/usr/bin/env python3
"""Evaluate RAG system performance."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.query_processor import QueryProcessor
from src.retrieval.hybrid_searcher import HybridSearcher
from src.generation.llm_interface import LLMInterface
from src.evaluation.metrics import MetricsCalculator, EvaluationResult
from src.utils.config_loader import get_config


def load_test_dataset(dataset_path: str) -> list:
    """Load test queries from JSON file.

    Args:
        dataset_path: Path to test dataset

    Returns:
        List of test query dictionaries
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)

    return data.get("queries", [])


def evaluate_query(
    query_data: dict,
    query_processor: QueryProcessor,
    searcher: HybridSearcher,
    llm: LLMInterface,
    metrics_calc: MetricsCalculator,
) -> EvaluationResult:
    """Evaluate a single query.

    Args:
        query_data: Test query data
        query_processor: Query processor instance
        searcher: Hybrid searcher instance
        llm: LLM interface instance
        metrics_calc: Metrics calculator instance

    Returns:
        EvaluationResult object
    """
    query_text = query_data["query_text"]
    ground_truth_chunks = query_data.get("ground_truth_chunks", [])
    expected_answer_points = query_data.get("expected_answer_points", [])

    # Process and search
    processed = query_processor.process(query_text)
    results = searcher.search(
        query=query_text,
        filters=processed.filter,
        top_k=10,
    )

    # Extract retrieved chunk IDs
    retrieved_chunk_ids = [r.chunk_id for r in results]

    # Compute retrieval metrics
    retrieval_metrics = metrics_calc.compute_retrieval_metrics(
        retrieved_chunk_ids, ground_truth_chunks
    )

    # Generate response
    response = llm.generate_response(query_text, results)

    # Compute generation metrics
    retrieved_contexts = [r.text for r in results]
    generation_metrics = metrics_calc.compute_generation_metrics(
        query=query_text,
        retrieved_contexts=retrieved_contexts,
        generated_answer=response.answer,
        expected_answer_points=expected_answer_points,
    )

    return EvaluationResult(
        query_id=query_data["query_id"],
        query_text=query_text,
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
    )


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system performance"
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for evaluation results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to evaluation config YAML",
    )

    args = parser.parse_args()

    # Load test dataset
    print(f"Loading test dataset: {args.test_dataset}")
    test_queries = load_test_dataset(args.test_dataset)
    print(f"Loaded {len(test_queries)} test queries")

    # Initialize components
    print("\nInitializing RAG components...")
    query_processor = QueryProcessor()
    searcher = HybridSearcher()
    llm = LLMInterface()
    metrics_calc = MetricsCalculator()

    # Evaluate each query
    print("\nEvaluating queries...")
    results = []

    for i, query_data in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] {query_data['query_text'][:50]}...")

        try:
            result = evaluate_query(
                query_data,
                query_processor,
                searcher,
                llm,
                metrics_calc,
            )
            results.append(result)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Aggregate metrics
    print("\nComputing aggregate metrics...")

    avg_precision_5 = sum(
        r.retrieval_metrics.precision_at_5 for r in results if r.retrieval_metrics
    ) / len(results)

    avg_precision_10 = sum(
        r.retrieval_metrics.precision_at_10 for r in results if r.retrieval_metrics
    ) / len(results)

    avg_recall_10 = sum(
        r.retrieval_metrics.recall_at_10 for r in results if r.retrieval_metrics
    ) / len(results)

    avg_mrr = sum(
        r.retrieval_metrics.mrr for r in results if r.retrieval_metrics
    ) / len(results)

    avg_faithfulness = sum(
        r.generation_metrics.faithfulness for r in results if r.generation_metrics
    ) / len(results)

    avg_relevancy = sum(
        r.generation_metrics.answer_relevancy for r in results if r.generation_metrics
    ) / len(results)

    avg_completeness = sum(
        r.generation_metrics.answer_completeness for r in results if r.generation_metrics
    ) / len(results)

    # Prepare output
    output_data = {
        "run_id": f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "test_dataset": args.test_dataset,
        "num_queries": len(test_queries),
        "num_evaluated": len(results),
        "summary": {
            "retrieval": {
                "precision@5": round(avg_precision_5, 3),
                "precision@10": round(avg_precision_10, 3),
                "recall@10": round(avg_recall_10, 3),
                "mrr": round(avg_mrr, 3),
            },
            "generation": {
                "faithfulness": round(avg_faithfulness, 3),
                "answer_relevancy": round(avg_relevancy, 3),
                "answer_completeness": round(avg_completeness, 3),
            },
        },
        "per_query_results": [r.to_dict() for r in results],
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Queries evaluated: {len(results)}/{len(test_queries)}")
    print(f"\nRetrieval Metrics:")
    print(f"  Precision@5:  {avg_precision_5:.3f}")
    print(f"  Precision@10: {avg_precision_10:.3f}")
    print(f"  Recall@10:    {avg_recall_10:.3f}")
    print(f"  MRR:          {avg_mrr:.3f}")
    print(f"\nGeneration Metrics:")
    print(f"  Faithfulness:  {avg_faithfulness:.3f}")
    print(f"  Relevancy:     {avg_relevancy:.3f}")
    print(f"  Completeness:  {avg_completeness:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
