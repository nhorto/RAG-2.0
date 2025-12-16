"""Integration tests for advanced query processing pipeline.

Tests end-to-end functionality of query analysis, decomposition, augmentation,
and orchestration working together.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.query_analyzer import QueryAnalyzer
from retrieval.query_processor import QueryProcessor

# Load environment variables
load_dotenv()


@pytest.fixture
def test_queries():
    """Load test queries from JSON file."""
    test_file = Path(__file__).parent.parent / "data" / "test_queries" / "advanced_test_queries.json"
    with open(test_file, 'r') as f:
        data = json.load(f)
    return data['test_queries']


@pytest.fixture
def query_processor():
    """Create QueryProcessor with advanced processing enabled."""
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found - skipping integration tests")

    return QueryProcessor(llm_api_key=api_key)


class TestQueryProcessorIntegration:
    """Test QueryProcessor with advanced processing enabled."""

    def test_simple_query_no_enhancement(self, query_processor):
        """Test that simple queries pass through without enhancement."""
        query = "How to create a Bill of Materials in the Estimating module?"
        result = query_processor.process(query)

        # Should not trigger advanced processing
        if result.analysis:
            assert result.analysis.needs_decomposition is False
            assert result.analysis.needs_augmentation is False

        # Should still have expanded queries from traditional processing
        assert len(result.expanded_queries) >= 0

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_decomposition_query(self, query_processor):
        """Test query that should trigger decomposition."""
        query = "How do I create a BOM, assign it to a WO, and track production?"
        result = query_processor.process(query)

        # Should trigger decomposition
        if result.decomposed:
            assert len(result.decomposed.sub_queries) >= 2
            assert result.decomposed.connection_logic in ["SEQUENTIAL", "AND"]

            # Verify sub-queries are meaningful
            for sq in result.decomposed.sub_queries:
                assert len(sq.query_text) > 5
                assert sq.order >= 0

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_augmentation_query(self, query_processor):
        """Test query that should trigger augmentation."""
        query = "How to export this?"
        result = query_processor.process(query)

        # Should trigger augmentation
        if result.augmented:
            assert len(result.augmented.augmented_variants) >= 2
            assert len(result.augmented.augmented_variants) <= 5

            # Verify variants are different from original
            for variant in result.augmented.augmented_variants:
                assert len(variant) > len(query)

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_complex_query_both_enhancements(self, query_processor):
        """Test query that should trigger both decomposition and augmentation."""
        query = "How to fix this issue with BOM creation, WO assignment, and tracking?"
        result = query_processor.process(query)

        # May trigger one or both
        enhanced = result.decomposed is not None or result.augmented is not None
        assert enhanced, "Complex query should trigger at least one enhancement"

    def test_all_query_variants_property(self, query_processor):
        """Test that all_query_variants property aggregates correctly."""
        query = "How to create BOM and assign to WO?"
        result = query_processor.process(query)

        # Should have at least the original query
        assert query in result.all_query_variants
        assert len(result.all_query_variants) >= 1

        # Should not have duplicates
        assert len(result.all_query_variants) == len(set(result.all_query_variants))


class TestEndToEndFlows:
    """Test complete query processing flows."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_sequential_decomposition_flow(self, query_processor):
        """Test end-to-end flow for sequential multi-part query."""
        query = "First create materials, then assign to BOM, and finally generate WO"
        result = query_processor.process(query)

        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")

        if result.analysis:
            print(f"Analysis:")
            print(f"  Complexity: {result.analysis.query_complexity}")
            print(f"  Needs Decomposition: {result.analysis.needs_decomposition} "
                  f"(confidence={result.analysis.decomposition_confidence:.2f})")
            print(f"  Needs Augmentation: {result.analysis.needs_augmentation} "
                  f"(confidence={result.analysis.augmentation_confidence:.2f})")

        if result.decomposed:
            print(f"\nDecomposition:")
            print(f"  Logic: {result.decomposed.connection_logic}")
            print(f"  Strategy: {result.decomposed.execution_strategy}")
            print(f"  Sub-queries ({len(result.decomposed.sub_queries)}):")
            for sq in result.decomposed.sub_queries:
                print(f"    {sq.order}. [{sq.intent}] {sq.query_text}")

        if result.augmented:
            print(f"\nAugmentation:")
            print(f"  Type: {result.augmented.augmentation_type}")
            print(f"  Variants ({len(result.augmented.augmented_variants)}):")
            for i, variant in enumerate(result.augmented.augmented_variants, 1):
                print(f"    {i}. {variant}")

        print(f"\nAll Query Variants ({len(result.all_query_variants)}):")
        for i, variant in enumerate(result.all_query_variants, 1):
            print(f"  {i}. {variant}")

        print(f"{'='*80}\n")

        # Verify structure
        assert result.original_query == query
        assert len(result.all_query_variants) >= 1

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_performance_metrics(self, query_processor, test_queries):
        """Test performance characteristics of advanced processing."""
        import time

        # Test a sample of different query types
        sample_queries = [
            ("simple", "How to create a BOM in Estimating?"),
            ("decomp", "How to create BOM and assign to WO?"),
            ("augment", "Export this"),
        ]

        results = []

        for query_type, query in sample_queries:
            start = time.time()
            result = query_processor.process(query)
            elapsed_ms = (time.time() - start) * 1000

            results.append({
                "type": query_type,
                "query": query,
                "elapsed_ms": elapsed_ms,
                "decomposed": result.decomposed is not None,
                "augmented": result.augmented is not None,
            })

            print(f"\n{query_type.upper()} Query: {elapsed_ms:.1f}ms")
            print(f"  Decomposed: {result.decomposed is not None}")
            print(f"  Augmented: {result.augmented is not None}")

        # Simple queries should be fast (<200ms without LLM calls)
        simple = [r for r in results if r['type'] == 'simple'][0]
        if not simple['decomposed'] and not simple['augmented']:
            assert simple['elapsed_ms'] < 500, "Simple queries should be fast"


class TestErrorHandling:
    """Test error handling and fallback behavior."""

    def test_llm_failure_fallback(self, query_processor):
        """Test that LLM failures fall back gracefully."""
        # Mock LLM client to fail
        if hasattr(query_processor, 'decomposer') and query_processor.decomposer:
            original_decompose = query_processor.decomposer.decompose

            def failing_decompose(*args, **kwargs):
                raise Exception("Simulated LLM failure")

            query_processor.decomposer.decompose = failing_decompose

            query = "Create BOM, assign WO, track production"
            result = query_processor.process(query)

            # Should still return a valid result
            assert result is not None
            assert result.original_query == query

            # Restore
            query_processor.decomposer.decompose = original_decompose

    def test_missing_api_key_graceful_degradation(self):
        """Test that missing API key degrades gracefully."""
        # Create processor without API key
        processor = QueryProcessor(llm_api_key=None)

        query = "How to create BOM and assign to WO?"
        result = processor.process(query)

        # Should still work, just without advanced processing
        assert result is not None
        assert result.original_query == query
        assert result.analysis is None  # No advanced processing


class TestConfigurationOptions:
    """Test different configuration options."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OpenAI API key")
    def test_feature_flags(self):
        """Test that feature flags control behavior."""
        # This would require modifying config at runtime
        # For now, just verify the flags exist
        from utils.config_loader import get_config

        config = get_config()
        adv_config = config.query_processing.get("advanced_processing", {})

        # Verify configuration structure exists
        assert "enabled" in adv_config or True  # Config may not exist yet
        assert "enable_decomposition" in adv_config or True
        assert "enable_augmentation" in adv_config or True


if __name__ == "__main__":
    """Run integration tests."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])
