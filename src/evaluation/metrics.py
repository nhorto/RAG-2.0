"""Metrics calculation for RAG evaluation."""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from ..utils.config_loader import get_config


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""

    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    mrr: float  # Mean Reciprocal Rank

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GenerationMetrics:
    """Generation quality metrics."""

    faithfulness: float
    answer_relevancy: float
    answer_completeness: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCalculator:
    """Calculate evaluation metrics for RAG system."""

    def __init__(self, llm_client=None):
        """Initialize metrics calculator.

        Args:
            llm_client: LLM client for RAGAS evaluation
        """
        self.llm_client = llm_client
        self.config = get_config()

    def compute_retrieval_metrics(
        self,
        retrieved_chunk_ids: List[str],
        ground_truth_chunk_ids: List[str],
    ) -> RetrievalMetrics:
        """Compute retrieval metrics.

        Args:
            retrieved_chunk_ids: IDs of retrieved chunks
            ground_truth_chunk_ids: IDs of ground truth chunks

        Returns:
            RetrievalMetrics object
        """
        # Precision at K
        relevant_at_5 = sum(
            1 for cid in retrieved_chunk_ids[:5] if cid in ground_truth_chunk_ids
        )
        relevant_at_10 = sum(
            1 for cid in retrieved_chunk_ids[:10] if cid in ground_truth_chunk_ids
        )

        precision_at_5 = relevant_at_5 / 5 if len(retrieved_chunk_ids) >= 5 else 0
        precision_at_10 = relevant_at_10 / 10 if len(retrieved_chunk_ids) >= 10 else 0

        # Recall at 10
        total_relevant = len(ground_truth_chunk_ids)
        recall_at_10 = (
            relevant_at_10 / total_relevant if total_relevant > 0 else 0
        )

        # Mean Reciprocal Rank
        first_relevant_rank = None
        for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
            if chunk_id in ground_truth_chunk_ids:
                first_relevant_rank = rank
                break

        mrr = 1 / first_relevant_rank if first_relevant_rank else 0

        return RetrievalMetrics(
            precision_at_5=precision_at_5,
            precision_at_10=precision_at_10,
            recall_at_10=recall_at_10,
            mrr=mrr,
        )

    def compute_generation_metrics(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        expected_answer_points: List[str] = None,
    ) -> GenerationMetrics:
        """Compute generation metrics using RAGAS.

        Args:
            query: User query
            retrieved_contexts: Retrieved context chunks
            generated_answer: Generated answer
            expected_answer_points: Expected answer key points (for completeness)

        Returns:
            GenerationMetrics object
        """
        if not RAGAS_AVAILABLE:
            print(
                "Warning: RAGAS not available. Install with: pip install ragas datasets"
            )
            return GenerationMetrics(
                faithfulness=0.0,
                answer_relevancy=0.0,
                answer_completeness=0.0,
            )

        # Prepare data for RAGAS
        data = {
            "question": [query],
            "contexts": [retrieved_contexts],
            "answer": [generated_answer],
        }

        dataset = Dataset.from_dict(data)

        try:
            # Evaluate using RAGAS
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy],
            )

            faithfulness_score = result["faithfulness"]
            relevancy_score = result["answer_relevancy"]

        except Exception as e:
            print(f"Warning: RAGAS evaluation failed: {e}")
            faithfulness_score = 0.0
            relevancy_score = 0.0

        # Compute answer completeness (custom metric)
        completeness = self._compute_answer_completeness(
            generated_answer, expected_answer_points
        )

        return GenerationMetrics(
            faithfulness=faithfulness_score,
            answer_relevancy=relevancy_score,
            answer_completeness=completeness,
        )

    def _compute_answer_completeness(
        self, answer: str, expected_points: List[str] = None
    ) -> float:
        """Compute answer completeness based on expected key points.

        Args:
            answer: Generated answer
            expected_points: Expected key points

        Returns:
            Completeness score (0-1)
        """
        if not expected_points:
            return 1.0  # No ground truth, assume complete

        answer_lower = answer.lower()
        covered = 0

        for point in expected_points:
            # Simple keyword matching - could be enhanced with semantic similarity
            point_lower = point.lower()
            if point_lower in answer_lower:
                covered += 1

        return covered / len(expected_points)


class EvaluationResult:
    """Stores complete evaluation results."""

    def __init__(
        self,
        query_id: str,
        query_text: str,
        retrieval_metrics: RetrievalMetrics = None,
        generation_metrics: GenerationMetrics = None,
    ):
        """Initialize evaluation result.

        Args:
            query_id: Query identifier
            query_text: Query text
            retrieval_metrics: Retrieval metrics
            generation_metrics: Generation metrics
        """
        self.query_id = query_id
        self.query_text = query_text
        self.retrieval_metrics = retrieval_metrics
        self.generation_metrics = generation_metrics

    def to_dict(self) -> Dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "retrieval_metrics": (
                self.retrieval_metrics.to_dict() if self.retrieval_metrics else None
            ),
            "generation_metrics": (
                self.generation_metrics.to_dict()
                if self.generation_metrics
                else None
            ),
        }

    def to_json(self, filepath: str):
        """Save to JSON file.

        Args:
            filepath: Output file path
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# For testing
if __name__ == "__main__":
    calculator = MetricsCalculator()

    # Test retrieval metrics
    retrieved = ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"]
    ground_truth = ["chunk-2", "chunk-5", "chunk-7"]

    retrieval_metrics = calculator.compute_retrieval_metrics(retrieved, ground_truth)

    print("Retrieval Metrics:")
    print(f"  Precision@5: {retrieval_metrics.precision_at_5:.3f}")
    print(f"  Precision@10: {retrieval_metrics.precision_at_10:.3f}")
    print(f"  Recall@10: {retrieval_metrics.recall_at_10:.3f}")
    print(f"  MRR: {retrieval_metrics.mrr:.3f}")

    # Test answer completeness
    answer = "To create a BOM, navigate to Estimating and click New BOM."
    expected = ["Navigate to Estimating", "Click New BOM", "Add items"]

    completeness = calculator._compute_answer_completeness(answer, expected)
    print(f"\nAnswer Completeness: {completeness:.3f}")
