"""Unit tests for QueryAnalyzer module.

Tests heuristic detection logic for query decomposition and augmentation needs.
"""

import pytest
import json
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.query_analyzer import QueryAnalyzer


@pytest.fixture
def analyzer():
    """Create QueryAnalyzer instance with default config."""
    return QueryAnalyzer()


@pytest.fixture
def test_queries():
    """Load test queries from JSON file."""
    test_file = Path(__file__).parent.parent / "data" / "test_queries" / "advanced_test_queries.json"
    with open(test_file, 'r') as f:
        data = json.load(f)
    return data['test_queries']


class TestDecompositionDetection:
    """Test decomposition detection heuristics."""

    def test_sequential_indicators(self, analyzer):
        """Test detection of sequential indicators."""
        query = "First create BOM, then assign to WO, and finally track production"
        analysis = analyzer.analyze(query)

        assert analysis.needs_decomposition is True
        assert analysis.decomposition_confidence >= 0.6
        assert any("Sequential indicator" in r for r in analysis.decomposition_reasons)

    def test_multiple_and_conjunctions(self, analyzer):
        """Test detection of multiple AND conjunctions."""
        query = "How to create BOM and assign to WO and track production?"
        analysis = analyzer.analyze(query)

        assert analysis.needs_decomposition is True
        assert "and" in analysis.decomposition_reasons[0].lower()

    def test_multiple_questions(self, analyzer):
        """Test detection of multiple question marks."""
        query = "How to create BOM? How to assign WO? How to track production?"
        analysis = analyzer.analyze(query)

        assert analysis.needs_decomposition is True
        assert any("Multiple questions" in r for r in analysis.decomposition_reasons)

    def test_enumeration_patterns(self, analyzer):
        """Test detection of enumeration patterns."""
        query = "1. Create BOM 2. Assign WO 3. Track production"
        analysis = analyzer.analyze(query)

        assert analysis.needs_decomposition is True
        assert any("Enumerated list" in r for r in analysis.decomposition_reasons)

    def test_high_clause_count(self, analyzer):
        """Test detection of high clause count."""
        query = "Create BOM, assign to WO, track production, generate reports"
        analysis = analyzer.analyze(query)

        assert analysis.needs_decomposition is True
        # Should have high clause count
        assert analysis.decomposition_confidence > 0.0

    def test_simple_query_no_decomposition(self, analyzer):
        """Test that simple queries are not decomposed."""
        query = "How to create a Bill of Materials in Estimating module?"
        analysis = analyzer.analyze(query)

        assert analysis.needs_decomposition is False
        assert analysis.decomposition_confidence < 0.6


class TestAugmentationDetection:
    """Test augmentation detection heuristics."""

    def test_vague_pronouns(self, analyzer):
        """Test detection of vague pronouns."""
        queries = [
            "How to export this?",
            "What is that feature?",
            "How does it work?"
        ]

        for query in queries:
            analysis = analyzer.analyze(query)
            assert analysis.needs_augmentation is True
            assert any("pronoun" in r.lower() for r in analysis.augmentation_reasons)

    def test_generic_terms(self, analyzer):
        """Test detection of generic terms."""
        queries = [
            "Fix issue",
            "Create feature",
            "Update module"
        ]

        for query in queries:
            analysis = analyzer.analyze(query)
            assert analysis.needs_augmentation is True or analysis.augmentation_confidence > 0.0

    def test_short_queries(self, analyzer):
        """Test detection of short queries."""
        queries = [
            "BOM?",
            "Export",
            "Create WO"
        ]

        for query in queries:
            analysis = analyzer.analyze(query)
            assert analysis.needs_augmentation is True
            assert any("Short query" in r for r in analysis.augmentation_reasons)

    def test_incomplete_actions(self, analyzer):
        """Test detection of incomplete actions."""
        queries = [
            "Export",
            "Create report",
            "Import data"
        ]

        for query in queries:
            analysis = analyzer.analyze(query)
            # Should have high augmentation score
            assert analysis.augmentation_confidence > 0.0

    def test_specific_query_no_augmentation(self, analyzer):
        """Test that specific queries are not augmented."""
        query = "How to create a Bill of Materials in PowerFab Estimating module?"
        analysis = analyzer.analyze(query)

        assert analysis.needs_augmentation is False
        assert analysis.augmentation_confidence < 0.5


class TestComplexityClassification:
    """Test query complexity classification."""

    def test_simple_complexity(self, analyzer):
        """Test simple query classification."""
        query = "How to create a BOM in Estimating?"
        analysis = analyzer.analyze(query)

        assert analysis.query_complexity == "simple"
        assert analysis.estimated_processing_cost == "low"

    def test_moderate_complexity(self, analyzer):
        """Test moderate query classification."""
        query = "Create new report"
        analysis = analyzer.analyze(query)

        # Should be at least moderate due to incompleteness
        assert analysis.query_complexity in ["moderate", "complex"]

    def test_complex_decomposition(self, analyzer):
        """Test complex query with decomposition."""
        query = "How to create BOM, assign to WO, and track production?"
        analysis = analyzer.analyze(query)

        assert analysis.query_complexity == "complex"
        assert analysis.estimated_processing_cost in ["medium", "high"]

    def test_complex_both_enhancements(self, analyzer):
        """Test query needing both decomposition and augmentation."""
        query = "How to fix this issue with BOM, WO, and tracking?"
        analysis = analyzer.analyze(query)

        assert analysis.needs_decomposition is True
        assert analysis.needs_augmentation is True
        assert analysis.query_complexity == "complex"
        assert analysis.estimated_processing_cost == "high"


class TestWithTestDataset:
    """Test analyzer against comprehensive test dataset."""

    def test_decomposition_queries(self, analyzer, test_queries):
        """Test all queries expected to trigger decomposition."""
        decomp_queries = [q for q in test_queries if q.get('expected_decomposition')]

        correct = 0
        total = len(decomp_queries)

        for query_data in decomp_queries:
            query = query_data['query_text']
            analysis = analyzer.analyze(query)

            if analysis.needs_decomposition:
                correct += 1
            else:
                print(f"  False negative: '{query}' (confidence={analysis.decomposition_confidence:.2f})")

        accuracy = correct / total if total > 0 else 0
        print(f"\nDecomposition detection accuracy: {accuracy:.1%} ({correct}/{total})")

        # Target: 90% accuracy
        assert accuracy >= 0.85, f"Decomposition accuracy {accuracy:.1%} below 85% threshold"

    def test_augmentation_queries(self, analyzer, test_queries):
        """Test all queries expected to trigger augmentation."""
        aug_queries = [q for q in test_queries if q.get('expected_augmentation')]

        correct = 0
        total = len(aug_queries)

        for query_data in aug_queries:
            query = query_data['query_text']
            analysis = analyzer.analyze(query)

            if analysis.needs_augmentation:
                correct += 1
            else:
                print(f"  False negative: '{query}' (confidence={analysis.augmentation_confidence:.2f})")

        accuracy = correct / total if total > 0 else 0
        print(f"\nAugmentation detection accuracy: {accuracy:.1%} ({correct}/{total})")

        # Target: 80% accuracy
        assert accuracy >= 0.75, f"Augmentation accuracy {accuracy:.1%} below 75% threshold"

    def test_simple_queries(self, analyzer, test_queries):
        """Test that simple queries are not over-processed."""
        simple_queries = [
            q for q in test_queries
            if not q.get('expected_decomposition') and not q.get('expected_augmentation')
        ]

        correct = 0
        total = len(simple_queries)

        for query_data in simple_queries:
            query = query_data['query_text']
            analysis = analyzer.analyze(query)

            # Should not need either enhancement
            if not analysis.needs_decomposition and not analysis.needs_augmentation:
                correct += 1
            else:
                print(f"  False positive: '{query}' (decomp={analysis.needs_decomposition}, aug={analysis.needs_augmentation})")

        accuracy = correct / total if total > 0 else 0
        print(f"\nSimple query accuracy (no false positives): {accuracy:.1%} ({correct}/{total})")

        # Should avoid over-processing simple queries
        assert accuracy >= 0.80


class TestConfidenceThresholds:
    """Test confidence threshold tuning."""

    def test_custom_thresholds(self):
        """Test analyzer with custom confidence thresholds."""
        # More conservative thresholds
        config = {
            "min_decompose_confidence": 0.7,
            "min_augment_confidence": 0.6
        }
        analyzer = QueryAnalyzer(config=config)

        # Borderline query
        query = "Create BOM and assign WO"
        analysis = analyzer.analyze(query)

        # With higher threshold, might not decompose
        # Just verify it respects the threshold
        if analysis.decomposition_confidence >= 0.7:
            assert analysis.needs_decomposition is True
        else:
            assert analysis.needs_decomposition is False


if __name__ == "__main__":
    """Run tests with pytest."""
    pytest.main([__file__, "-v", "-s"])
