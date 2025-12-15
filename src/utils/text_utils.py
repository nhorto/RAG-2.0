"""Text processing utilities for RAG system."""

import re
import tiktoken
from typing import List, Tuple
from pathlib import Path


class TextTokenizer:
    """Tokenizer for text using tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize tokenizer.

        Args:
            encoding_name: Tiktoken encoding name (cl100k_base for GPT-4/3.5)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        return self.encoding.decode(tokens)

    def split_by_tokens(
        self, text: str, chunk_size: int, overlap: int = 0
    ) -> List[Tuple[str, int, int]]:
        """Split text into chunks by token count.

        Args:
            text: Input text
            chunk_size: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Returns:
            List of (chunk_text, start_char, end_char) tuples
        """
        tokens = self.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.decode(chunk_tokens)

            # Find character positions
            # Approximate - decode from beginning to get accurate char position
            prefix_tokens = tokens[:start]
            prefix_text = self.decode(prefix_tokens) if prefix_tokens else ""
            start_char = len(prefix_text)
            end_char = start_char + len(chunk_text)

            chunks.append((chunk_text, start_char, end_char))

            # Move to next chunk with overlap
            start = end - overlap if overlap > 0 else end

        return chunks


class SentenceSplitter:
    """Simple sentence splitter using regex."""

    def __init__(self):
        """Initialize sentence splitter."""
        # Pattern for sentence boundaries
        self.sentence_pattern = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+\s*"
        )

    def split(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Split on sentence boundaries
        sentences = self.sentence_pattern.split(text)

        # Filter empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]

    def find_sentence_boundary(self, text: str, position: int) -> int:
        """Find nearest sentence boundary after given position.

        Args:
            text: Input text
            position: Character position to start from

        Returns:
            Character position of sentence boundary (or end of text)
        """
        if position >= len(text):
            return len(text)

        # Look for sentence ending after position
        remaining = text[position:]
        match = re.search(r"[.!?](?=\s+[A-Z]|\s*$)", remaining)

        if match:
            return position + match.end()
        else:
            return len(text)


class TextNormalizer:
    """Text normalization utilities."""

    @staticmethod
    def remove_srt_timestamps(text: str) -> str:
        """Remove .srt timestamp lines.

        Args:
            text: SRT file content

        Returns:
            Text without timestamps
        """
        # Pattern: sequence number, timestamp line, then text
        # Example:
        # 1
        # 00:00:00,000 --> 00:00:05,000
        # Speaker text here

        # Remove sequence numbers (standalone digits)
        text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)

        # Remove timestamp lines
        text = re.sub(
            r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$",
            "",
            text,
            flags=re.MULTILINE,
        )

        return text

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
