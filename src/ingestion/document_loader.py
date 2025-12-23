"""Document loader for RAG system using LangChain loaders."""

import re
import uuid
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document as LCDocument


@dataclass
class Document:
    """Represents a loaded document."""

    document_id: str
    filename: str
    filepath: str
    content: str
    file_size: int
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SubtitleParser:
    """Parser for .srt and .vtt subtitle files."""

    @staticmethod
    def parse_srt(content: str) -> str:
        """Parse .srt content, extracting only the dialogue text.

        Args:
            content: Raw .srt file content

        Returns:
            Cleaned text with only dialogue
        """
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Skip sequence numbers (pure digits)
            if line.isdigit():
                continue
            # Skip timestamp lines (contain -->)
            if '-->' in line:
                continue
            lines.append(line)
        return '\n'.join(lines)

    @staticmethod
    def parse_vtt(content: str) -> str:
        """Parse .vtt (WebVTT) content, extracting only the dialogue text.

        Args:
            content: Raw .vtt file content

        Returns:
            Cleaned text with only dialogue
        """
        lines = []
        skip_header = True

        for line in content.split('\n'):
            line = line.strip()

            # Skip the WEBVTT header and any metadata before first cue
            if skip_header:
                if line.startswith('WEBVTT'):
                    continue
                if not line:
                    continue
                # Check if we've hit a timestamp line (indicates start of cues)
                if '-->' in line:
                    skip_header = False
                    continue
                # Skip NOTE blocks and other metadata
                if line.startswith('NOTE') or line.startswith('STYLE'):
                    continue
                # Skip any other header content
                continue

            # Now processing cue content
            if not line:
                continue
            # Skip cue identifiers (lines before timestamps, often numbers or IDs)
            if '-->' in line:
                continue
            # Skip timestamp lines already handled above
            # Skip lines that look like cue settings (contain positioning info)
            if re.match(r'^[\d:\.]+\s*-->', line):
                continue

            # Remove VTT tags like <v Speaker>, <c>, etc.
            line = re.sub(r'<[^>]+>', '', line)
            # Remove speaker identifiers in format "Speaker: "
            line = re.sub(r'^[A-Za-z\s]+:\s*', '', line)

            if line:
                lines.append(line)

        return '\n'.join(lines)


class DocumentLoader:
    """Load documents from filesystem using appropriate loaders."""

    def __init__(self, encoding: str = "utf-8"):
        """Initialize document loader.

        Args:
            encoding: Text encoding (default: utf-8)
        """
        self.encoding = encoding
        self.subtitle_parser = SubtitleParser()

    def load_file(self, filepath: str) -> Optional[Document]:
        """Load a single file using the appropriate loader.

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
            ext = filepath.suffix.lower()

            # Handle subtitle files (.srt, .vtt)
            if ext in ['.srt', '.vtt']:
                content = self._load_subtitle_file(filepath, ext)
            else:
                # Use LangChain TextLoader for other text files
                content = self._load_text_file(filepath)

            if content is None:
                return None

            # Extract metadata from filename
            metadata = self._extract_metadata_from_filename(filepath.name)

            return Document(
                document_id=str(uuid.uuid4()),
                filename=filepath.name,
                filepath=str(filepath),
                content=content,
                file_size=filepath.stat().st_size,
                metadata=metadata,
            )

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def _load_subtitle_file(self, filepath: Path, ext: str) -> Optional[str]:
        """Load and parse a subtitle file.

        Args:
            filepath: Path to subtitle file
            ext: File extension (.srt or .vtt)

        Returns:
            Parsed text content
        """
        try:
            with open(filepath, 'r', encoding=self.encoding, errors='ignore') as f:
                raw_content = f.read()

            if ext == '.srt':
                return self.subtitle_parser.parse_srt(raw_content)
            elif ext == '.vtt':
                return self.subtitle_parser.parse_vtt(raw_content)

        except Exception as e:
            print(f"Error parsing subtitle file {filepath}: {e}")
            return None

    def _load_text_file(self, filepath: Path) -> Optional[str]:
        """Load a text file using LangChain TextLoader.

        Args:
            filepath: Path to text file

        Returns:
            Text content
        """
        # Try multiple encodings in order of preference
        encodings_to_try = [self.encoding, 'utf-8', 'iso-8859-1', 'cp1252', 'latin-1']

        for encoding in encodings_to_try:
            try:
                loader = TextLoader(str(filepath), encoding=encoding)
                docs = loader.load()
                if docs:
                    return docs[0].page_content
                return None
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # If it's not an encoding error, try with errors='ignore'
                try:
                    with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
                        return f.read()
                except Exception:
                    continue

        print(f"Error loading text file {filepath}: Could not decode with any supported encoding")
        return None

    def _extract_metadata_from_filename(self, filename: str) -> dict:
        """Extract metadata from filename.

        Expected format: YYYY-MM-DD_ClientName_SiteName_doctype.ext

        Args:
            filename: Filename to parse

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}

        # Try to extract date
        date_match = re.match(r'^(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            metadata['date'] = date_match.group(1)

        # Try to extract document type
        filename_lower = filename.lower()
        if 'transcript' in filename_lower or filename.endswith(('.srt', '.vtt')):
            metadata['document_type'] = 'transcript'
        elif 'daily_summary' in filename_lower or 'daily-summary' in filename_lower:
            metadata['document_type'] = 'daily_summary'
        elif 'master_summary' in filename_lower or 'weekly_summary' in filename_lower:
            metadata['document_type'] = 'master_summary'
        else:
            metadata['document_type'] = 'generic'

        # Try to extract client name (second part after date)
        parts = filename.replace('.srt', '').replace('.vtt', '').replace('.txt', '').split('_')
        if len(parts) >= 2 and date_match:
            metadata['client_name'] = parts[1]
        if len(parts) >= 3 and date_match:
            metadata['site_name'] = parts[2]

        # Check for session number in filename
        session_match = re.search(r'session[_-]?(\d+)', filename_lower)
        if session_match:
            metadata['session_number'] = int(session_match.group(1))

        return metadata

    def load_directory(
        self,
        directory: str,
        patterns: List[str] = None,
        recursive: bool = False,
    ) -> List[Document]:
        """Load all matching files from directory.

        Args:
            directory: Directory path
            patterns: List of glob patterns (default: ["*.txt", "*.srt", "*.vtt"])
            recursive: Whether to search recursively

        Returns:
            List of Document objects
        """
        if patterns is None:
            patterns = ["*.txt", "*.srt", "*.vtt"]

        directory = Path(directory)

        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            return []

        # Find matching files
        all_files = set()
        for pattern in patterns:
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)
            all_files.update(files)

        # Load each file
        documents = []
        for filepath in sorted(all_files):
            doc = self.load_file(filepath)
            if doc:
                documents.append(doc)
                print(f"  Loaded: {filepath.name}")

        return documents


# For testing
if __name__ == "__main__":
    loader = DocumentLoader()

    # Test loading a directory
    import sys
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    else:
        test_dir = "../data/raw"

    docs = loader.load_directory(test_dir)

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs[:3]:
        print(f"\n{doc.filename}")
        print(f"  ID: {doc.document_id}")
        print(f"  Size: {doc.file_size} bytes")
        print(f"  Type: {doc.metadata.get('document_type', 'unknown')}")
        print(f"  Preview: {doc.content[:100]}...")
