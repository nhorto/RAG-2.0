"""Metadata extraction using NER and keyword matching."""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from ..utils.text_utils import load_domain_vocabulary
from ..utils.config_loader import get_config


class MetadataExtractor:
    """Extract metadata from documents and chunks."""

    def __init__(self, domain_vocabulary_path: str = None, enable_ner: bool = True):
        """Initialize metadata extractor.

        Args:
            domain_vocabulary_path: Path to domain vocabulary JSON
            enable_ner: Whether to enable NER (requires spaCy)
        """
        # Load domain vocabulary
        self.vocab = load_domain_vocabulary(domain_vocabulary_path)

        # Initialize spaCy if available and enabled
        self.nlp = None
        self.enable_ner = enable_ner and SPACY_AVAILABLE

        if self.enable_ner:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print(
                    "Warning: spaCy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                self.enable_ner = False

    def extract_file_metadata(self, filename: str, file_content: str = None) -> Dict:
        """Extract metadata from filename and optionally file content.

        Args:
            filename: Document filename
            file_content: Optional document content for duration estimation

        Returns:
            Dictionary with file-level metadata
        """
        metadata = {}

        # Pattern: YYYY-MM-DD_ClientName_SiteName_doctype.txt
        pattern = r"(\d{4}-\d{2}-\d{2})_(.+?)_(.+?)_(transcript|daily_summary|master_summary)"
        match = re.match(pattern, filename)

        if match:
            date_str, client, site, doc_type = match.groups()

            try:
                date = datetime.fromisoformat(date_str)
                metadata["date"] = date_str
                metadata["meeting_day"] = date.strftime("%A")
            except ValueError:
                pass

            metadata["client_name"] = client
            metadata["site_name"] = site
            metadata["document_type"] = doc_type

        # Estimate duration for transcripts
        if file_content and metadata.get("document_type") == "transcript":
            word_count = len(file_content.split())
            # Assume 150 words per minute average speaking rate
            metadata["duration_minutes"] = int(word_count / 150)

        return metadata

    def extract_content_metadata(self, text: str) -> Dict:
        """Extract entities, keywords, topics from content.

        Args:
            text: Document or chunk text

        Returns:
            Dictionary with content-level metadata
        """
        metadata = {
            "entities": {},
            "powerfab": {},
            "action_items": [],
            "decisions_made": [],
        }

        # Named Entity Recognition
        if self.enable_ner and self.nlp:
            doc = self.nlp(text[:10000])  # Limit to first 10k chars for performance

            metadata["entities"] = {
                "organizations": list(set(ent.text for ent in doc.ents if ent.label_ == "ORG")),
                "persons": list(set(ent.text for ent in doc.ents if ent.label_ == "PERSON")),
                "locations": list(set(ent.text for ent in doc.ents if ent.label_ == "GPE")),
            }

        # Domain keyword matching
        metadata["powerfab"] = {
            "modules": self._match_keywords(text, self.vocab.get("modules", [])),
            "features": self._match_keywords(text, self.vocab.get("features", [])),
            "topics": self._match_keywords(text, self.vocab.get("topics", [])),
        }

        # Extract action items
        metadata["action_items"] = self._extract_action_items(text)

        # Extract decisions
        metadata["decisions_made"] = self._extract_decisions(text)

        return metadata

    def _match_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Case-insensitive keyword matching.

        Args:
            text: Text to search
            keywords: List of keywords

        Returns:
            List of matched keywords
        """
        text_lower = text.lower()
        matched = []

        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                matched.append(keyword)

        return matched

    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items using pattern matching.

        Args:
            text: Text to search

        Returns:
            List of action items
        """
        patterns = [
            r"TODO:\s*(.+)",
            r"Action:\s*(.+)",
            r"Action Item:\s*(.+)",
            r"Follow-up:\s*(.+)",
            r"Next steps?:\s*(.+)",
        ]

        items = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            items.extend(m.strip() for m in matches)

        # Also look for bullet points starting with action verbs
        action_verbs = ["Send", "Schedule", "Create", "Update", "Review", "Complete"]
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            # Check for bullet points or numbered lists
            if re.match(r'^[-*•]\s+', line) or re.match(r'^\d+\.\s+', line):
                # Check if starts with action verb
                for verb in action_verbs:
                    if line.lower().startswith(verb.lower()):
                        # Extract the action item
                        item = re.sub(r'^[-*•]\s+', '', line)
                        item = re.sub(r'^\d+\.\s+', '', item)
                        if item and item not in items:
                            items.append(item.strip())
                        break

        return items[:10]  # Limit to 10 action items

    def _extract_decisions(self, text: str) -> List[str]:
        """Extract decisions using pattern matching.

        Args:
            text: Text to search

        Returns:
            List of decisions
        """
        patterns = [
            r"Decided:\s*(.+)",
            r"Decision:\s*(.+)",
            r"Agreement:\s*(.+)",
            r"Resolved:\s*(.+)",
            r"Agreed to:\s*(.+)",
            r"We decided to:\s*(.+)",
        ]

        decisions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            decisions.extend(m.strip() for m in matches)

        return decisions[:10]  # Limit to 10 decisions

    def create_chunk_payload(
        self,
        chunk_text: str,
        document_metadata: Dict,
        chunk_metadata: Dict,
    ) -> Dict:
        """Create complete payload for chunk storage in Qdrant.

        Args:
            chunk_text: Chunk text
            document_metadata: Document-level metadata
            chunk_metadata: Chunk-level metadata (index, tokens, etc.)

        Returns:
            Complete payload dictionary
        """
        # Extract content metadata from chunk
        content_metadata = self.extract_content_metadata(chunk_text)

        payload = {
            "text": chunk_text,
            "document_metadata": document_metadata,
            "chunk_metadata": chunk_metadata,
            "content_metadata": content_metadata,
        }

        return payload


# For testing
if __name__ == "__main__":
    extractor = MetadataExtractor()

    # Test filename extraction
    filename = "2024-11-15_ClientA_Site1_transcript.txt"
    file_meta = extractor.extract_file_metadata(filename)
    print("File metadata:")
    print(json.dumps(file_meta, indent=2))

    # Test content extraction
    sample_text = """
    We discussed the Estimating module and how to create BOMs.
    John Smith from TrimbleCAD mentioned the Production Control integration.

    Action: Send updated BOM template to client
    TODO: Schedule follow-up training session

    Decision: Agreed to customize the shipping workflow
    """

    content_meta = extractor.extract_content_metadata(sample_text)
    print("\nContent metadata:")
    print(json.dumps(content_meta, indent=2))
