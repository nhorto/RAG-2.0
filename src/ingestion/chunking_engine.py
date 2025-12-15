"""Chunking engine with type-aware strategies for RAG system."""

import uuid
from dataclasses import dataclass
from typing import List, Literal, Optional
from pathlib import Path

from ..utils.text_utils import TextTokenizer, SentenceSplitter, TextNormalizer
from ..utils.config_loader import get_config


DocumentType = Literal["transcript", "daily_summary", "master_summary", "generic"]


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    chunk_id: str
    text: str
    document_id: str
    chunk_index: int
    token_count: int
    char_count: int
    previous_chunk_id: Optional[str]
    next_chunk_id: Optional[str]


class ChunkingEngine:
    """Chunking engine with type-aware strategies."""

    def __init__(
        self,
        transcript_chunk_size: int = None,
        transcript_overlap: int = None,
        summary_min_size: int = None,
        summary_max_size: int = None,
        config: dict = None,
    ):
        """Initialize chunking engine.

        Args:
            transcript_chunk_size: Chunk size for transcripts (tokens)
            transcript_overlap: Overlap for transcripts (tokens)
            summary_min_size: Minimum chunk size for summaries (tokens)
            summary_max_size: Maximum chunk size for summaries (tokens)
            config: Configuration dict (optional)
        """
        if config is None:
            config = get_config()

        # Load config values
        chunking_config = config.chunking if hasattr(config, "chunking") else config.get("chunking", {})

        self.transcript_chunk_size = (
            transcript_chunk_size
            or chunking_config.get("transcript", {}).get("chunk_size", 512)
        )
        self.transcript_overlap = (
            transcript_overlap
            or chunking_config.get("transcript", {}).get("overlap", 50)
        )
        self.summary_min_size = (
            summary_min_size
            or chunking_config.get("summary", {}).get("min_size", 100)
        )
        self.summary_max_size = (
            summary_max_size
            or chunking_config.get("summary", {}).get("max_size", 800)
        )

        # Initialize utilities
        self.tokenizer = TextTokenizer()
        self.sentence_splitter = SentenceSplitter()
        self.normalizer = TextNormalizer()

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_type: DocumentType,
    ) -> List[Chunk]:
        """Chunk document based on type-specific strategy.

        Args:
            text: Document text
            document_id: Unique document identifier
            document_type: Type of document

        Returns:
            List of Chunk objects
        """
        if document_type == "transcript":
            return self._chunk_transcript(text, document_id)
        elif document_type in ["daily_summary", "master_summary"]:
            return self._chunk_summary(text, document_id)
        else:
            return self._chunk_generic(text, document_id)

    def _chunk_transcript(self, text: str, document_id: str) -> List[Chunk]:
        """Fixed-size chunking with overlap, sentence-boundary aware.

        Args:
            text: Transcript text
            document_id: Document identifier

        Returns:
            List of Chunk objects
        """
        chunks = []
        sentences = self.sentence_splitter.split(text)

        if not sentences:
            return chunks

        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.count_tokens(sentence)

            # If adding this sentence exceeds chunk size, create a chunk
            if current_tokens + sentence_tokens > self.transcript_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_tokens = 0
                overlap_sentences = []

                for s in reversed(current_chunk):
                    s_tokens = self.tokenizer.count_tokens(s)
                    if overlap_tokens + s_tokens <= self.transcript_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

        # Convert to Chunk objects with linking
        return self._create_chunk_objects(chunks, document_id)

    def _chunk_summary(self, text: str, document_id: str) -> List[Chunk]:
        """Paragraph-based chunking with size constraints.

        Args:
            text: Summary text
            document_id: Document identifier

        Returns:
            List of Chunk objects
        """
        # Split on paragraph boundaries
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for paragraph in paragraphs:
            para_tokens = self.tokenizer.count_tokens(paragraph)

            # If paragraph is too large, split by sentences
            if para_tokens > self.summary_max_size:
                # Save current chunk if any
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                sentences = self.sentence_splitter.split(paragraph)
                temp_chunk = []
                temp_tokens = 0

                for sentence in sentences:
                    sent_tokens = self.tokenizer.count_tokens(sentence)

                    if temp_tokens + sent_tokens > self.summary_max_size and temp_chunk:
                        chunks.append(" ".join(temp_chunk))
                        temp_chunk = [sentence]
                        temp_tokens = sent_tokens
                    else:
                        temp_chunk.append(sentence)
                        temp_tokens += sent_tokens

                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))

            # If adding paragraph keeps us under max size
            elif current_tokens + para_tokens <= self.summary_max_size:
                current_chunk.append(paragraph)
                current_tokens += para_tokens

            # If we have content and adding would exceed max
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = para_tokens

        # Add final chunk if it meets minimum size
        if current_chunk:
            final_text = "\n\n".join(current_chunk)
            final_tokens = self.tokenizer.count_tokens(final_text)

            # Merge with previous if too small
            if final_tokens < self.summary_min_size and chunks:
                chunks[-1] = chunks[-1] + "\n\n" + final_text
            else:
                chunks.append(final_text)

        return self._create_chunk_objects(chunks, document_id)

    def _chunk_generic(self, text: str, document_id: str) -> List[Chunk]:
        """Fallback: sentence-based chunking.

        Args:
            text: Generic text
            document_id: Document identifier

        Returns:
            List of Chunk objects
        """
        # Use transcript strategy as fallback
        return self._chunk_transcript(text, document_id)

    def _create_chunk_objects(
        self, chunk_texts: List[str], document_id: str
    ) -> List[Chunk]:
        """Convert chunk texts to Chunk objects with linking.

        Args:
            chunk_texts: List of chunk text strings
            document_id: Document identifier

        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_ids = [str(uuid.uuid4()) for _ in chunk_texts]

        for i, text in enumerate(chunk_texts):
            chunk = Chunk(
                chunk_id=chunk_ids[i],
                text=text,
                document_id=document_id,
                chunk_index=i,
                token_count=self.tokenizer.count_tokens(text),
                char_count=len(text),
                previous_chunk_id=chunk_ids[i - 1] if i > 0 else None,
                next_chunk_id=chunk_ids[i + 1] if i < len(chunk_texts) - 1 else None,
            )
            chunks.append(chunk)

        return chunks

    def detect_document_type(self, text: str, filename: str = "") -> DocumentType:
        """Detect document type from content and filename.

        Args:
            text: Document text
            filename: Document filename

        Returns:
            Document type
        """
        filename_lower = filename.lower()

        # Check filename
        if "transcript" in filename_lower or ".srt" in filename_lower:
            return "transcript"
        elif "daily_summary" in filename_lower or "daily summary" in text.lower()[:500]:
            return "daily_summary"
        elif "master_summary" in filename_lower or "weekly_summary" in filename_lower:
            return "master_summary"
        elif "weekly summary" in text.lower()[:500] or "master summary" in text.lower()[:500]:
            return "master_summary"

        # Check content for SRT markers
        if re.search(r"^\d+$", text, flags=re.MULTILINE):
            if re.search(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->", text):
                return "transcript"

        return "generic"


# For testing
if __name__ == "__main__":
    import re

    # Test with sample text
    engine = ChunkingEngine()

    sample_transcript = """
    This is the first sentence. This is the second sentence. This is the third sentence.
    This continues for many sentences to test chunking. Each sentence should be processed correctly.
    The chunker should respect sentence boundaries. It should also handle overlaps properly.
    """ * 20

    chunks = engine.chunk_document(sample_transcript, "test-doc-1", "transcript")

    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Preview: {chunk.text[:100]}...")
        print(f"  Previous: {chunk.previous_chunk_id}")
        print(f"  Next: {chunk.next_chunk_id}")
