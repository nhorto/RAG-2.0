#!/usr/bin/env python3
"""
Test script for database querying and retrieval.

Run with: python tests/test_database_query.py

Modes:
  --interactive    : Interactive mode - type questions and see results
  --preset         : Run preset test questions (default)
  --both           : Run preset questions, then enter interactive mode

This script tests:
- Database connection and collection info
- Hybrid search (dense + sparse)
- Dense-only search
- Sparse-only (keyword/BM25) search
- Metadata filtering
- Result inspection
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.qdrant_client import QdrantManager, SearchResult, build_metadata_filter
from src.retrieval.hybrid_searcher import HybridSearcher
from src.ingestion.embedding_generator import HybridEmbeddingGenerator
from src.utils.config_loader import get_config


def print_separator(title: str, char: str = "="):
    """Print a section separator."""
    print(f"\n{char*60}")
    print(f"  {title}")
    print(f"{char*60}\n")


def print_result(result: SearchResult, index: int, show_full_text: bool = False):
    """Print a single search result."""
    print(f"\n  --- Result {index} ---")
    print(f"  Score: {result.score:.4f}")
    print(f"  Chunk ID: {result.chunk_id}")

    # Show text
    if show_full_text:
        print(f"  Text:\n    {result.text}")
    else:
        text_preview = result.text[:300].replace('\n', ' ')
        print(f"  Text: {text_preview}...")

    # Show metadata
    metadata = result.metadata
    if metadata:
        print(f"  Metadata:")
        for key, value in metadata.items():
            if key != "text":  # Skip text in metadata since we already show it
                if isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")


def test_database_connection():
    """Test database connection and show collection info."""
    print_separator("TEST: Database Connection")

    try:
        qdrant = QdrantManager()
        info = qdrant.get_collection_info()

        print("Database connected successfully!")
        print(f"\nCollection info:")
        print(f"  Exists: {info.get('exists', False)}")
        print(f"  Points count: {info.get('points_count', 0):,}")
        print(f"  Indexed vectors: {info.get('indexed_vectors_count', 0):,}")
        print(f"  Status: {info.get('status', 'unknown')}")

        return info.get('exists', False) and info.get('points_count', 0) > 0

    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        return False


def run_search(
    searcher: HybridSearcher,
    query: str,
    search_type: str = "hybrid",
    top_k: int = 5,
    filters: Optional[dict] = None,
    show_full_text: bool = False
) -> List[SearchResult]:
    """Run a search and display results."""

    print(f"\nQuery: \"{query}\"")
    print(f"Search type: {search_type}")
    print(f"Top K: {top_k}")
    if filters:
        print(f"Filters: {filters}")

    # Perform search based on type
    if search_type == "dense":
        results = searcher.search(query, filters=filters, top_k=top_k, dense_only=True)
    elif search_type == "sparse":
        results = searcher.search(query, filters=filters, top_k=top_k, sparse_only=True)
    else:  # hybrid
        results = searcher.search(query, filters=filters, top_k=top_k)

    print(f"\nFound {len(results)} results:")

    for i, result in enumerate(results, 1):
        print_result(result, i, show_full_text)

    return results


def run_preset_tests(searcher: HybridSearcher):
    """Run a set of preset test queries."""
    print_separator("PRESET TEST QUERIES")

    # Define preset test queries
    preset_queries = [
        {
            "query": "What are the main topics discussed?",
            "search_type": "hybrid",
            "top_k": 5,
            "description": "General topic query - hybrid search"
        },
        {
            "query": "problems issues errors bugs",
            "search_type": "sparse",
            "top_k": 5,
            "description": "Keyword-based query - sparse/BM25 search"
        },
        {
            "query": "How do I configure the system?",
            "search_type": "dense",
            "top_k": 5,
            "description": "Semantic query - dense vector search"
        },
        {
            "query": "What recommendations were made?",
            "search_type": "hybrid",
            "top_k": 10,
            "description": "Recommendations query - more results"
        },
    ]

    results_summary = []

    for i, test in enumerate(preset_queries, 1):
        print_separator(f"Preset Query {i}: {test['description']}", char="-")

        try:
            results = run_search(
                searcher,
                query=test["query"],
                search_type=test["search_type"],
                top_k=test["top_k"]
            )
            results_summary.append({
                "query": test["query"],
                "type": test["search_type"],
                "count": len(results),
                "success": True
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results_summary.append({
                "query": test["query"],
                "type": test["search_type"],
                "count": 0,
                "success": False
            })

    # Print summary
    print_separator("PRESET TESTS SUMMARY")
    for item in results_summary:
        status = "OK" if item["success"] else "FAILED"
        print(f"  [{status}] {item['type']:8} | {item['count']:2} results | {item['query'][:40]}...")

    return all(item["success"] for item in results_summary)


def interactive_mode(searcher: HybridSearcher):
    """Run interactive query mode."""
    print_separator("INTERACTIVE QUERY MODE")

    print("Enter your questions to search the database.")
    print("Commands:")
    print("  /dense   - Switch to dense-only search")
    print("  /sparse  - Switch to sparse-only (BM25) search")
    print("  /hybrid  - Switch to hybrid search (default)")
    print("  /full    - Toggle full text display")
    print("  /topk N  - Set number of results (e.g., /topk 10)")
    print("  /info    - Show database info")
    print("  /help    - Show this help")
    print("  /quit    - Exit interactive mode")
    print()

    search_type = "hybrid"
    top_k = 5
    show_full_text = False

    while True:
        try:
            query = input(f"\n[{search_type}|top{top_k}] Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break

        if not query:
            continue

        # Handle commands
        if query.startswith("/"):
            cmd = query.lower()

            if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
                print("Exiting interactive mode.")
                break
            elif cmd == "/dense":
                search_type = "dense"
                print("Switched to DENSE (semantic) search")
            elif cmd == "/sparse":
                search_type = "sparse"
                print("Switched to SPARSE (BM25/keyword) search")
            elif cmd == "/hybrid":
                search_type = "hybrid"
                print("Switched to HYBRID (dense + sparse) search")
            elif cmd == "/full":
                show_full_text = not show_full_text
                print(f"Full text display: {'ON' if show_full_text else 'OFF'}")
            elif cmd.startswith("/topk"):
                try:
                    n = int(cmd.split()[1])
                    top_k = max(1, min(50, n))
                    print(f"Top K set to {top_k}")
                except (IndexError, ValueError):
                    print("Usage: /topk N (e.g., /topk 10)")
            elif cmd == "/info":
                qdrant = QdrantManager()
                info = qdrant.get_collection_info()
                print(f"Collection: {info.get('points_count', 0):,} points, status: {info.get('status', 'unknown')}")
            elif cmd == "/help":
                print("Commands: /dense, /sparse, /hybrid, /full, /topk N, /info, /quit")
            else:
                print(f"Unknown command: {query}")
            continue

        # Run the search
        try:
            run_search(
                searcher,
                query=query,
                search_type=search_type,
                top_k=top_k,
                show_full_text=show_full_text
            )
        except Exception as e:
            print(f"Search error: {e}")


def test_search_comparison(searcher: HybridSearcher):
    """Compare different search types on the same query."""
    print_separator("TEST: Search Type Comparison")

    query = "What were the key findings or recommendations?"

    print(f"Comparing search types for: \"{query}\"")
    print("\nThis shows how different search methods return different results.")

    # Dense search
    print("\n--- DENSE (Semantic) Search ---")
    dense_results = searcher.search(query, top_k=3, dense_only=True)
    for i, r in enumerate(dense_results, 1):
        print(f"  {i}. [score:{r.score:.4f}] {r.text[:100]}...")

    # Sparse search
    print("\n--- SPARSE (Keyword/BM25) Search ---")
    sparse_results = searcher.search(query, top_k=3, sparse_only=True)
    for i, r in enumerate(sparse_results, 1):
        print(f"  {i}. [score:{r.score:.4f}] {r.text[:100]}...")

    # Hybrid search
    print("\n--- HYBRID (Combined) Search ---")
    hybrid_results = searcher.search(query, top_k=3)
    for i, r in enumerate(hybrid_results, 1):
        print(f"  {i}. [score:{r.score:.4f}] {r.text[:100]}...")

    # Show overlap
    dense_ids = {r.chunk_id for r in dense_results}
    sparse_ids = {r.chunk_id for r in sparse_results}
    hybrid_ids = {r.chunk_id for r in hybrid_results}

    print(f"\n--- Result Overlap ---")
    print(f"  Dense ∩ Sparse: {len(dense_ids & sparse_ids)} common results")
    print(f"  Dense ∩ Hybrid: {len(dense_ids & hybrid_ids)} common results")
    print(f"  Sparse ∩ Hybrid: {len(sparse_ids & hybrid_ids)} common results")

    return True


def main():
    """Run database query tests."""
    parser = argparse.ArgumentParser(description="Test database querying and retrieval")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive query mode")
    parser.add_argument("--preset", "-p", action="store_true", help="Run preset test queries")
    parser.add_argument("--both", "-b", action="store_true", help="Preset tests + interactive mode")
    args = parser.parse_args()

    # Default to preset if no mode specified
    if not args.interactive and not args.preset and not args.both:
        args.preset = True

    print("\n" + "="*60)
    print("  DATABASE QUERY TEST SCRIPT")
    print("="*60)

    # Test database connection first
    if not test_database_connection():
        print("\nERROR: Database is not ready. Please ensure:")
        print("  1. Qdrant is running (docker-compose up -d)")
        print("  2. Documents have been ingested (python scripts/ingest_documents.py)")
        sys.exit(1)

    # Initialize searcher
    print_separator("Initializing Search Components")
    try:
        embedder = HybridEmbeddingGenerator()
        qdrant = QdrantManager()
        searcher = HybridSearcher(qdrant_manager=qdrant, embedding_generator=embedder)
        print("Search components initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize search components: {e}")
        sys.exit(1)

    # Run tests based on mode
    if args.both:
        run_preset_tests(searcher)
        test_search_comparison(searcher)
        interactive_mode(searcher)
    elif args.interactive:
        interactive_mode(searcher)
    else:  # preset
        success = run_preset_tests(searcher)
        test_search_comparison(searcher)

        print_separator("FINAL SUMMARY")
        if success:
            print("All preset tests completed successfully!")
        else:
            print("Some tests failed. Check the output above.")

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
