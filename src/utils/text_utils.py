"""Text processing utilities for RAG system.

Note: Token counting and text splitting are now handled by LangChain's
RecursiveCharacterTextSplitter.from_tiktoken_encoder() in chunking_engine.py.
This module retains utility functions still used by other components.
"""

import re
from typing import List, Tuple
from pathlib import Path


class TextNormalizer:
    """Text normalization utilities."""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r"\n\n+", "\n\n", text)

        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]

        return "\n".join(lines).strip()

    @staticmethod
    def detect_speaker_changes(text: str) -> List[Tuple[int, str]]:
        """Detect speaker changes in transcript.

        Args:
            text: Transcript text

        Returns:
            List of (position, speaker_name) tuples
        """
        # Pattern for speaker labels like "John:" or "[John]:" or "JOHN:"
        speaker_pattern = re.compile(
            r"^([A-Z][a-zA-Z\s]+|\[[A-Z][a-zA-Z\s]+\]):\s*", flags=re.MULTILINE
        )

        speakers = []
        for match in speaker_pattern.finditer(text):
            speaker = match.group(1).strip("[]").strip()
            position = match.start()
            speakers.append((position, speaker))

        return speakers


def load_domain_vocabulary(vocab_path: str = None) -> dict:
    """Load domain vocabulary from JSON file.

    Args:
        vocab_path: Path to domain_vocabulary.json

    Returns:
        Dictionary with modules, features, topics, abbreviations
    """
    import json

    if vocab_path is None:
        # Default location
        project_root = Path(__file__).parent.parent.parent
        vocab_path = project_root / "config" / "domain_vocabulary.json"

    with open(vocab_path, "r") as f:
        return json.load(f)
