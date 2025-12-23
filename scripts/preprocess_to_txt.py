#!/usr/bin/env python3
"""
Preprocess VTT/SRT files and save as TXT for inspection.

This script converts subtitle files to plain text so you can
verify the parsing is working correctly.

Usage:
    python scripts/preprocess_to_txt.py
    python scripts/preprocess_to_txt.py --input data/raw --output data/processed
    python scripts/preprocess_to_txt.py --show-raw  # Also save raw content for comparison

Output:
    data/processed/
        2025-04-10_ClientA_SiteB_transcript.txt      # Parsed content
        2025-04-10_ClientA_SiteB_transcript.raw.txt  # Raw content (with --show-raw)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.document_loader import DocumentLoader, SubtitleParser


def preprocess_files(
    input_dir: str,
    output_dir: str,
    patterns: list = None,
    show_raw: bool = False,
    verbose: bool = True
):
    """
    Preprocess VTT/SRT files and save as TXT.

    Args:
        input_dir: Directory containing source files
        output_dir: Directory to save processed TXT files
        patterns: Glob patterns to match (default: VTT and SRT)
        show_raw: Also save raw (unparsed) content for comparison
        verbose: Print progress
    """
    if patterns is None:
        patterns = ["*.vtt", "*.srt"]

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nPreprocessing subtitle files")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Patterns: {patterns}")
        print()

    # Find all matching files
    all_files = set()
    for pattern in patterns:
        all_files.update(input_path.glob(pattern))

    if not all_files:
        print(f"No files found matching {patterns} in {input_path}")
        return

    print(f"Found {len(all_files)} files to process\n")

    loader = DocumentLoader()
    parser = SubtitleParser()

    for filepath in sorted(all_files):
        filename = filepath.name
        base_name = filepath.stem  # filename without extension
        ext = filepath.suffix.lower()

        print(f"Processing: {filename}")

        try:
            # Read raw content
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()

            # Parse based on file type
            if ext == '.vtt':
                parsed_content = parser.parse_vtt(raw_content)
            elif ext == '.srt':
                parsed_content = parser.parse_srt(raw_content)
            else:
                parsed_content = raw_content

            # Save parsed content
            output_file = output_path / f"{base_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(parsed_content)

            print(f"  -> Saved: {output_file.name}")
            print(f"     Raw: {len(raw_content):,} chars -> Parsed: {len(parsed_content):,} chars")

            # Optionally save raw content for comparison
            if show_raw:
                raw_output_file = output_path / f"{base_name}.raw.txt"
                with open(raw_output_file, 'w', encoding='utf-8') as f:
                    f.write(raw_content)
                print(f"  -> Saved raw: {raw_output_file.name}")

            # Show a preview
            preview_lines = parsed_content.split('\n')[:5]
            print(f"     Preview (first 5 lines):")
            for line in preview_lines:
                print(f"       {line[:80]}{'...' if len(line) > 80 else ''}")
            print()

        except Exception as e:
            print(f"  ERROR: {e}\n")

    print(f"\nDone! Check {output_path} for the processed files.")
    print(f"Compare the .txt files against the original .vtt/.srt files to verify parsing.")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess VTT/SRT files to TXT for inspection"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Input directory containing VTT/SRT files (default: data/raw)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed",
        help="Output directory for TXT files (default: data/processed)"
    )
    parser.add_argument(
        "--show-raw", "-r",
        action="store_true",
        help="Also save raw (unparsed) content as .raw.txt for comparison"
    )
    parser.add_argument(
        "--patterns", "-p",
        nargs="+",
        default=["*.vtt", "*.srt"],
        help="Glob patterns to match (default: *.vtt *.srt)"
    )

    args = parser.parse_args()

    preprocess_files(
        input_dir=args.input,
        output_dir=args.output,
        patterns=args.patterns,
        show_raw=args.show_raw
    )


if __name__ == "__main__":
    main()
