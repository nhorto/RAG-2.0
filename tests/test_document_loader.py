#!/usr/bin/env python3
"""
Test script for DocumentLoader.

Run with: python tests/test_document_loader.py

This script tests:
- Loading single files (txt, srt, vtt)
- Loading directories with patterns
- Metadata extraction from filenames
- Subtitle parsing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.document_loader import DocumentLoader, Document


def print_separator(title: str):
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_load_single_file():
    """Test loading a single file."""
    print_separator("TEST: Load Single File")

    loader = DocumentLoader()

    # Find a file to test with
    raw_dir = project_root / "data" / "raw"
    test_files = list(raw_dir.glob("*.txt"))[:1]

    if not test_files:
        test_files = list(raw_dir.glob("*.vtt"))[:1]

    if not test_files:
        print("No test files found in data/raw/")
        return False

    test_file = test_files[0]
    print(f"Loading: {test_file.name}")

    doc = loader.load_file(str(test_file))

    if doc is None:
        print("ERROR: Failed to load document")
        return False

    print(f"\nDocument loaded successfully!")
    print(f"  Document ID: {doc.document_id}")
    print(f"  Filename: {doc.filename}")
    print(f"  File size: {doc.file_size:,} bytes")
    print(f"  Content length: {len(doc.content):,} characters")
    print(f"  Metadata: {doc.metadata}")
    print(f"\nFirst 500 chars of content:")
    print(f"  {doc.content[:500]}...")

    return True


def test_load_directory():
    """Test loading a directory with patterns."""
    print_separator("TEST: Load Directory")

    loader = DocumentLoader()
    raw_dir = project_root / "data" / "raw"

    print(f"Loading from: {raw_dir}")
    print(f"Patterns: ['*.txt', '*.vtt', '*.srt']")

    documents = loader.load_directory(
        str(raw_dir),
        patterns=["*.txt", "*.vtt", "*.srt"],
        recursive=False
    )

    print(f"\nLoaded {len(documents)} documents:")
    for doc in documents:
        print(f"\n  File: {doc.filename}")
        print(f"    ID: {doc.document_id}")
        print(f"    Size: {doc.file_size:,} bytes")
        print(f"    Content: {len(doc.content):,} chars")
        print(f"    Metadata: {doc.metadata}")

    return len(documents) > 0


def test_subtitle_parsing():
    """Test VTT and SRT subtitle parsing."""
    print_separator("TEST: Subtitle Parsing")

    loader = DocumentLoader()
    raw_dir = project_root / "data" / "raw"

    # Test VTT
    vtt_files = list(raw_dir.glob("*.vtt"))[:1]
    if vtt_files:
        print(f"Testing VTT: {vtt_files[0].name}")
        doc = loader.load_file(str(vtt_files[0]))
        if doc:
            print(f"  Loaded {len(doc.content):,} chars of dialogue")
            print(f"  Sample (first 300 chars):")
            print(f"    {doc.content[:300]}...")
    else:
        print("No VTT files found")

    # Test SRT
    srt_files = list(raw_dir.glob("*.srt"))[:1]
    if srt_files:
        print(f"\nTesting SRT: {srt_files[0].name}")
        doc = loader.load_file(str(srt_files[0]))
        if doc:
            print(f"  Loaded {len(doc.content):,} chars of dialogue")
            print(f"  Sample (first 300 chars):")
            print(f"    {doc.content[:300]}...")
    else:
        print("No SRT files found")

    return True


def test_metadata_extraction():
    """Test metadata extraction from filenames."""
    print_separator("TEST: Metadata Extraction")

    loader = DocumentLoader()
    raw_dir = project_root / "data" / "raw"

    # Load a few different file types
    documents = loader.load_directory(
        str(raw_dir),
        patterns=["*daily_summary*.txt", "*transcript*.vtt", "*master_summary*.txt"],
        recursive=False
    )

    print("Metadata extracted from filenames:")
    for doc in documents[:5]:  # Show first 5
        print(f"\n  File: {doc.filename}")
        for key, value in doc.metadata.items():
            print(f"    {key}: {value}")

    return len(documents) > 0


def main():
    """Run all document loader tests."""
    print("\n" + "="*60)
    print("  DOCUMENT LOADER TEST SCRIPT")
    print("="*60)

    results = {}

    # Run tests
    results["Load Single File"] = test_load_single_file()
    results["Load Directory"] = test_load_directory()
    results["Subtitle Parsing"] = test_subtitle_parsing()
    results["Metadata Extraction"] = test_metadata_extraction()

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
