"""Document loader for RAG system."""

import uuid
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from ..utils.text_utils import TextNormalizer


@dataclass
class Document:
    """Represents a loaded document."""

    document_id: str
    filename: str
    filepath: str
    content: str
    file_size: int


class DocumentLoader:
    """Load documents from filesystem."""

    def __init__(self, encoding: str = "utf-8"):
        """Initialize document loader.

        Args:
            encoding: Text encoding (default: utf-8)
        """
        self.encoding = encoding
        self.normalizer = TextNormalizer()

    def load_file(self, filepath: str) -> Optional[Document]:
        """Load a single file.

        Args:
            filepath: Path to file

        Returns:
            Document object or None if failed
        """
        filepath = Path(filepath)

        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            return None

        if not filepath.is_file():
            print(f"Warning: Not a file: {filepath}")
            return None

        try:
            # Read file
            with open(filepath, "r", encoding=self.encoding) as f:
                content = f.read()

            # Normalize content
            content = self._process_content(content, filepath)

            return Document(
                document_id=str(uuid.uuid4()),
                filename=filepath.name,
                filepath=str(filepath),
                content=content,
                file_size=filepath.stat().st_size,
            )

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def load_directory(
        self,
        directory: str,
        pattern: str = "*.txt",
        recursive: bool = False,
    ) -> List[Document]:
        """Load all matching files from directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for files (default: *.txt)
            recursive: Whether to search recursively

        Returns:
            List of Document objects
        """
        directory = Path(directory)

        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            return []

        # Find matching files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        # Load each file
        documents = []
        for filepath in sorted(files):
            doc = self.load_file(filepath)
            if doc:
                documents.append(doc)

        return documents

    def _process_content(self, content: str, filepath: Path) -> str:
        """Process and normalize file content.

        Args:
            content: Raw file content
            filepath: File path (for extension detection)

        Returns:
            Processed content
        """
        # Handle .srt files
        if filepath.suffix.lower() == ".srt":
            content = self.normalizer.remove_srt_timestamps(content)

        # Normalize whitespace
        content = self.normalizer.normalize_whitespace(content)

        return content


# For testing
if __name__ == "__main__":
    loader = DocumentLoader()

    # Test loading a directory
    docs = loader.load_directory("../data/raw", pattern="*.txt")

    print(f"Loaded {len(docs)} documents")
    for doc in docs[:3]:
        print(f"\n{doc.filename}")
        print(f"  ID: {doc.document_id}")
        print(f"  Size: {doc.file_size} bytes")
        print(f"  Preview: {doc.content[:100]}...")
