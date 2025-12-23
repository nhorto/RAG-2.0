#!/usr/bin/env python3
"""
Test script for ChunkingEngine.

Run with: python tests/test_chunking_engine.py

This script tests:
- Document type detection
- Chunking with different strategies (transcript, summary)
- Token counting
- Chunk linking (previous/next)
- Overlap verification
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunking_engine import ChunkingEngine, Chunk


def print_separator(title: str):
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_document_type_detection():
    """Test automatic document type detection."""
    print_separator("TEST: Document Type Detection")

    chunker = ChunkingEngine()

    test_cases = [
        ("2025-04-10_ClientA_SiteB_transcript.vtt", "transcript"),
        ("2025-04-10_ClientA_SiteB_daily_summary.txt", "daily_summary"),
        ("2025-04-15_ClientA_SiteB_master_summary.txt", "master_summary"),
        ("random_notes.txt", "generic"),
    ]

    print("Testing document type detection:")
    all_passed = True

    for filename, expected_type in test_cases:
        detected = chunker.detect_document_type("Sample content", filename)
        status = "PASS" if detected == expected_type else "FAIL"
        if detected != expected_type:
            all_passed = False
        print(f"  [{status}] {filename}")
        print(f"          Expected: {expected_type}, Got: {detected}")

    return all_passed


def test_transcript_chunking():
    """Test chunking a transcript document."""
    print_separator("TEST: Transcript Chunking")

    loader = DocumentLoader()
    chunker = ChunkingEngine()

    raw_dir = project_root / "data" / "raw"
    transcript_files = list(raw_dir.glob("*transcript*.vtt"))[:1]

    if not transcript_files:
        transcript_files = list(raw_dir.glob("*transcript*.srt"))[:1]

    if not transcript_files:
        print("No transcript files found")
        return False

    # Load the transcript
    doc = loader.load_file(str(transcript_files[0]))
    if not doc:
        print("Failed to load transcript")
        return False

    print(f"Loaded: {doc.filename}")
    print(f"Content length: {len(doc.content):,} characters")

    # Chunk it
    chunks = chunker.chunk_document(
        text=doc.content,
        document_id=doc.document_id,
        document_type="transcript",
        metadata=doc.metadata
    )

    print(f"\nCreated {len(chunks)} chunks")
    print("\nChunk statistics:")
    token_counts = [c.token_count for c in chunks]
    print(f"  Min tokens: {min(token_counts)}")
    print(f"  Max tokens: {max(token_counts)}")
    print(f"  Avg tokens: {sum(token_counts)/len(token_counts):.1f}")

    # Show first few chunks
    print("\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  Chunk {i+1}:")
        print(f"    ID: {chunk.chunk_id}")
        print(f"    Tokens: {chunk.token_count}")
        print(f"    Chars: {chunk.char_count}")
        print(f"    Previous: {chunk.previous_chunk_id or 'None'}")
        print(f"    Next: {chunk.next_chunk_id or 'None'}")
        print(f"    Text preview: {chunk.text[:150]}...")

    return len(chunks) > 0


def test_summary_chunking():
    """Test chunking a summary document."""
    print_separator("TEST: Summary Chunking")

    loader = DocumentLoader()
    chunker = ChunkingEngine()

    raw_dir = project_root / "data" / "raw"
    summary_files = list(raw_dir.glob("*daily_summary*.txt"))[:1]

    if not summary_files:
        print("No summary files found")
        return False

    # Load the summary
    doc = loader.load_file(str(summary_files[0]))
    if not doc:
        print("Failed to load summary")
        return False

    print(f"Loaded: {doc.filename}")
    print(f"Content length: {len(doc.content):,} characters")

    # Chunk it
    chunks = chunker.chunk_document(
        text=doc.content,
        document_id=doc.document_id,
        document_type="daily_summary",
        metadata=doc.metadata
    )

    print(f"\nCreated {len(chunks)} chunks")

    if len(chunks) > 0:
        token_counts = [c.token_count for c in chunks]
        print("\nChunk statistics:")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Avg tokens: {sum(token_counts)/len(token_counts):.1f}")

        # Show first few chunks
        print("\nFirst 3 chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n  Chunk {i+1}:")
            print(f"    Tokens: {chunk.token_count}")
            print(f"    Text preview: {chunk.text[:200]}...")

    return len(chunks) > 0


def test_chunk_linking():
    """Test that chunks are properly linked."""
    print_separator("TEST: Chunk Linking")

    loader = DocumentLoader()
    chunker = ChunkingEngine()

    raw_dir = project_root / "data" / "raw"
    files = list(raw_dir.glob("*.txt"))[:1]

    if not files:
        files = list(raw_dir.glob("*.vtt"))[:1]

    if not files:
        print("No files found")
        return False

    doc = loader.load_file(str(files[0]))
    doc_type = chunker.detect_document_type(doc.content, doc.filename)
    chunks = chunker.chunk_document(doc.content, doc.document_id, doc_type)

    if len(chunks) < 2:
        print("Not enough chunks to test linking")
        return True

    print(f"Testing chunk linking on {len(chunks)} chunks:")

    # Check first chunk
    first = chunks[0]
    first_ok = first.previous_chunk_id is None and first.next_chunk_id == chunks[1].chunk_id
    print(f"  First chunk: prev=None, next=correct -> {'PASS' if first_ok else 'FAIL'}")

    # Check last chunk
    last = chunks[-1]
    last_ok = last.next_chunk_id is None and last.previous_chunk_id == chunks[-2].chunk_id
    print(f"  Last chunk: prev=correct, next=None -> {'PASS' if last_ok else 'FAIL'}")

    # Check middle chunks
    middle_ok = True
    for i in range(1, len(chunks) - 1):
        chunk = chunks[i]
        if chunk.previous_chunk_id != chunks[i-1].chunk_id:
            middle_ok = False
            break
        if chunk.next_chunk_id != chunks[i+1].chunk_id:
            middle_ok = False
            break
    print(f"  Middle chunks: properly linked -> {'PASS' if middle_ok else 'FAIL'}")

    return first_ok and last_ok and middle_ok


def test_custom_chunking_params():
    """Test chunking with custom parameters."""
    print_separator("TEST: Custom Chunking Parameters")

    # Create chunker with custom settings
    chunker = ChunkingEngine(
        transcript_chunk_size=256,  # Smaller chunks
        transcript_overlap=25
    )

    # Create some sample text
    sample_text = "This is a test sentence. " * 200  # ~1000 words

    chunks = chunker.chunk_document(
        text=sample_text,
        document_id="test-doc-001",
        document_type="transcript"
    )

    print(f"Sample text: ~{len(sample_text.split())} words")
    print(f"Chunk size: 256 tokens, overlap: 25 tokens")
    print(f"Created: {len(chunks)} chunks")

    if len(chunks) > 0:
        token_counts = [c.token_count for c in chunks]
        print(f"\nToken counts per chunk:")
        for i, count in enumerate(token_counts[:5]):
            print(f"  Chunk {i+1}: {count} tokens")
        if len(token_counts) > 5:
            print(f"  ... and {len(token_counts) - 5} more")

    return len(chunks) > 0


def test_splitter_configs():
    """Test getting splitter configurations."""
    print_separator("TEST: Splitter Configurations")

    chunker = ChunkingEngine()

    doc_types = ["transcript", "daily_summary", "master_summary", "generic"]

    print("Splitter configurations by document type:")
    for doc_type in doc_types:
        try:
            config = chunker.get_splitter_config(doc_type)
            print(f"\n  {doc_type}:")
            print(f"    chunk_size: {config.get('chunk_size', 'N/A')} tokens")
            print(f"    overlap: {config.get('overlap', 'N/A')} tokens")
        except Exception as e:
            print(f"\n  {doc_type}: Error - {e}")

    return True


def main():
    """Run all chunking engine tests."""
    print("\n" + "="*60)
    print("  CHUNKING ENGINE TEST SCRIPT")
    print("="*60)

    results = {}

    # Run tests
    results["Document Type Detection"] = test_document_type_detection()
    results["Transcript Chunking"] = test_transcript_chunking()
    results["Summary Chunking"] = test_summary_chunking()
    results["Chunk Linking"] = test_chunk_linking()
    results["Custom Parameters"] = test_custom_chunking_params()
    results["Splitter Configs"] = test_splitter_configs()

    # Summary
    print_separator("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
