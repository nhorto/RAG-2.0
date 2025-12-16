"""Chunking engine using LangChain text splitters with chunk linking."""

import uuid
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    start_index: int  # Character position in original document
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class ChunkingEngine:
    """Chunking engine using LangChain with type-aware strategies."""

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

        # Initialize LangChain splitters with tiktoken encoding for accurate token counting
        # Using cl100k_base which is used by GPT-4 and text-embedding-3 models
        self._transcript_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.transcript_chunk_size,
            chunk_overlap=self.transcript_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )

        self._summary_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.summary_max_size,
            chunk_overlap=int(self.summary_max_size * 0.1),  # 10% overlap for summaries
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # For token counting
        import tiktoken
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self._encoding.encode(text))

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_type: DocumentType,
        metadata: dict = None,
    ) -> List[Chunk]:
        """Chunk document based on type-specific strategy.

        Args:
            text: Document text
            document_id: Unique document identifier
            document_type: Type of document
            metadata: Optional metadata to include in chunks

        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}

        # Select appropriate splitter based on document type
        if document_type == "transcript":
            splitter = self._transcript_splitter
        elif document_type in ["daily_summary", "master_summary"]:
            splitter = self._summary_splitter
        else:
            # Use transcript splitter as default for generic
            splitter = self._transcript_splitter

        # Split the text using LangChain
        lc_docs = splitter.create_documents(
            texts=[text],
            metadatas=[{"document_type": document_type, **metadata}]
        )

        # Convert to our Chunk format with linking
        return self._create_linked_chunks(lc_docs, document_id, document_type, metadata)

    def _create_linked_chunks(
        self,
        lc_docs: List,
        document_id: str,
        document_type: DocumentType,
        base_metadata: dict,
    ) -> List[Chunk]:
        """Convert LangChain documents to Chunk objects with linking.

        Args:
            lc_docs: List of LangChain Document objects
            document_id: Document identifier
            document_type: Type of document
            base_metadata: Base metadata to include

        Returns:
            List of linked Chunk objects
        """
        if not lc_docs:
            return []

        # Generate chunk IDs upfront for linking
        chunk_ids = [str(uuid.uuid4()) for _ in lc_docs]

        chunks = []
        for i, lc_doc in enumerate(lc_docs):
            text = lc_doc.page_content
            start_index = lc_doc.metadata.get("start_index", 0)

            # Merge metadata
            chunk_metadata = {
                **base_metadata,
                "document_type": document_type,
                **lc_doc.metadata,
            }

            chunk = Chunk(
                chunk_id=chunk_ids[i],
                text=text,
                document_id=document_id,
                chunk_index=i,
                token_count=self._count_tokens(text),
                char_count=len(text),
                start_index=start_index,
                previous_chunk_id=chunk_ids[i - 1] if i > 0 else None,
                next_chunk_id=chunk_ids[i + 1] if i < len(lc_docs) - 1 else None,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

        # Handle minimum size for summaries - merge small final chunks
        if document_type in ["daily_summary", "master_summary"] and len(chunks) > 1:
            chunks = self._merge_small_chunks(chunks, self.summary_min_size)

        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk], min_tokens: int) -> List[Chunk]:
        """Merge chunks that are below minimum token threshold.

        Args:
            chunks: List of chunks
            min_tokens: Minimum token count

        Returns:
            List of chunks with small ones merged
        """
        if not chunks or len(chunks) < 2:
            return chunks

        # Check if last chunk is too small
        if chunks[-1].token_count < min_tokens:
            # Merge with previous chunk
            merged_text = chunks[-2].text + "\n\n" + chunks[-1].text
            chunks[-2].text = merged_text
            chunks[-2].token_count = self._count_tokens(merged_text)
            chunks[-2].char_count = len(merged_text)
            chunks[-2].next_chunk_id = None

            # Remove the last chunk
            chunks = chunks[:-1]

        return chunks

    def detect_document_type(self, text: str, filename: str = "") -> DocumentType:
        """Detect document type from content and filename.

        Args:
            text: Document text
            filename: Document filename

        Returns:
            Document type
        """
        import re

        filename_lower = filename.lower()

        # Check filename first
        if "transcript" in filename_lower or filename_lower.endswith(('.srt', '.vtt')):
            return "transcript"
        elif "daily_summary" in filename_lower or "daily-summary" in filename_lower:
            return "daily_summary"
        elif "master_summary" in filename_lower or "weekly_summary" in filename_lower:
            return "master_summary"

        # Check content for document type indicators
        text_preview = text.lower()[:500]
        if "daily summary" in text_preview:
            return "daily_summary"
        elif "weekly summary" in text_preview or "master summary" in text_preview:
            return "master_summary"

        # Check content for SRT markers (in case extension wasn't checked)
        if re.search(r"^\d+$", text, flags=re.MULTILINE):
            if re.search(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->", text):
                return "transcript"

        # Check for VTT markers
        if text.strip().startswith("WEBVTT"):
            return "transcript"

        return "generic"

    def get_splitter_config(self, document_type: DocumentType) -> dict:
        """Get the current splitter configuration for a document type.

        Args:
            document_type: Type of document

        Returns:
            Dictionary with chunk_size and chunk_overlap
        """
        if document_type == "transcript":
            return {
                "chunk_size": self.transcript_chunk_size,
                "chunk_overlap": self.transcript_overlap,
            }
        elif document_type in ["daily_summary", "master_summary"]:
            return {
                "chunk_size": self.summary_max_size,
                "chunk_overlap": int(self.summary_max_size * 0.1),
                "min_size": self.summary_min_size,
            }
        else:
            return {
                "chunk_size": self.transcript_chunk_size,
                "chunk_overlap": self.transcript_overlap,
            }


# For testing
if __name__ == "__main__":
    # Test with sample text
    engine = ChunkingEngine()

    sample_transcript = """
    This is the first sentence of our consulting session. We discussed the Estimating module today.
    The client had questions about BOM creation and how to properly set up their workflow.

    Moving on to the next topic, we looked at Project Management features. The team wanted to understand
    how to track job progress and generate reports for their management team.

    Finally, we covered Production Control settings. This included reviewing the routing setup
    and making sure the work orders were configured correctly for their shop floor operations.
    """ * 5

    print("Testing ChunkingEngine with LangChain...")
    print(f"Transcript chunk size: {engine.transcript_chunk_size} tokens")
    print(f"Transcript overlap: {engine.transcript_overlap} tokens")

    chunks = engine.chunk_document(sample_transcript, "test-doc-1", "transcript")

    print(f"\nCreated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i}:")
        print(f"  ID: {chunk.chunk_id[:8]}...")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Chars: {chunk.char_count}")
        print(f"  Start Index: {chunk.start_index}")
        print(f"  Previous: {chunk.previous_chunk_id[:8] if chunk.previous_chunk_id else None}...")
        print(f"  Next: {chunk.next_chunk_id[:8] if chunk.next_chunk_id else None}...")
        print(f"  Preview: {chunk.text[:80]}...")
