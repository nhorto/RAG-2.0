"""Query processing including expansion, rewriting, and metadata extraction."""

import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..database.qdrant_client import build_metadata_filter
from ..utils.text_utils import load_domain_vocabulary
from ..utils.config_loader import get_config


@dataclass
class ProcessedQuery:
    """Represents a processed query with expansions and metadata."""

    original_query: str
    expanded_queries: List[str]
    entities: Dict[str, any]
    filter: Optional[any]  # Qdrant Filter object
    intent: str  # "factual", "procedural", "temporal", "troubleshooting"


class QueryProcessor:
    """Process queries for retrieval including expansion and rewriting."""

    def __init__(self, llm_api_key: str = None, llm_model: str = None):
        """Initialize query processor.

        Args:
            llm_api_key: OpenAI API key for query rewriting
            llm_model: LLM model for query rewriting
        """
        config = get_config()

        # Load domain vocabulary
        self.vocab = load_domain_vocabulary()

        # Initialize LLM for query rewriting
        self.enable_llm = OPENAI_AVAILABLE
        if self.enable_llm:
            if llm_api_key is None:
                try:
                    llm_api_key = config.get_api_key("openai")
                except ValueError:
                    self.enable_llm = False

        if self.enable_llm:
            self.llm_client = OpenAI(api_key=llm_api_key)
            self.llm_model = llm_model or config.get("llm.model", "gpt-4")
        else:
            self.llm_client = None
            self.llm_model = None

        # Load query processing config
        self.query_config = config.query_processing

    def process(self, query: str) -> ProcessedQuery:
        """Process query through full pipeline.

        Args:
            query: User query string

        Returns:
            ProcessedQuery object
        """
        # 1. Extract entities
        entities = self._extract_entities(query)

        # 2. Classify intent
        intent = self._classify_intent(query)

        # 3. Query expansion (abbreviations, synonyms)
        expanded = []
        if self.query_config.get("enable_expansion", True):
            expanded = self._expand_query(query)

        # 4. Query rewriting (LLM-based alternatives)
        rewritten = []
        if self.query_config.get("enable_rewriting", True) and self.enable_llm:
            num_rewrites = self.query_config.get("num_rewrites", 2)
            rewritten = self._rewrite_query(query, num_rewrites)

        # Combine all query variations
        all_queries = [query] + expanded + rewritten

        # 5. Build metadata filter
        metadata_filter = None
        if self.query_config.get("enable_metadata_extraction", True):
            metadata_filter = self._build_filter(entities)

        return ProcessedQuery(
            original_query=query,
            expanded_queries=all_queries,
            entities=entities,
            filter=metadata_filter,
            intent=intent,
        )

    def _extract_entities(self, query: str) -> Dict:
        """Extract entities from query (dates, clients, etc.).

        Args:
            query: User query

        Returns:
            Dictionary of extracted entities
        """
        entities = {}

        # Date extraction
        date_range = self._extract_date_range(query)
        if date_range:
            entities["date_range"] = date_range

        # Client name extraction (simple pattern matching)
        # Look for capitalized multi-word phrases that might be client names
        # This is a simplified heuristic - could be enhanced with NER
        client_pattern = r"(?:client|with|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        client_match = re.search(client_pattern, query)
        if client_match:
            entities["client"] = client_match.group(1)

        # PowerFab module extraction
        modules = []
        for module in self.vocab.get("modules", []):
            if module.lower() in query.lower():
                modules.append(module)
        if modules:
            entities["powerfab_modules"] = modules

        return entities

    def _extract_date_range(self, query: str) -> Optional[tuple]:
        """Extract date range from query.

        Args:
            query: User query

        Returns:
            Tuple of (start_date, end_date) in ISO format, or None
        """
        query_lower = query.lower()
        today = datetime.now()

        # Relative dates
        if "yesterday" in query_lower:
            date = today - timedelta(days=1)
            return (date.isoformat()[:10], date.isoformat()[:10])

        if "last week" in query_lower or "past week" in query_lower:
            start = today - timedelta(days=7)
            return (start.isoformat()[:10], today.isoformat()[:10])

        if "last month" in query_lower or "past month" in query_lower:
            start = today - timedelta(days=30)
            return (start.isoformat()[:10], today.isoformat()[:10])

        if "this week" in query_lower:
            # Start of current week (Monday)
            days_since_monday = today.weekday()
            start = today - timedelta(days=days_since_monday)
            return (start.isoformat()[:10], today.isoformat()[:10])

        if "this month" in query_lower:
            start = today.replace(day=1)
            return (start.isoformat()[:10], today.isoformat()[:10])

        # Absolute dates (YYYY-MM-DD format)
        date_pattern = r"\b(\d{4}-\d{2}-\d{2})\b"
        dates = re.findall(date_pattern, query)
        if dates:
            if len(dates) == 1:
                return (dates[0], dates[0])
            else:
                return (min(dates), max(dates))

        return None

    def _classify_intent(self, query: str) -> str:
        """Classify query intent.

        Args:
            query: User query

        Returns:
            Intent classification string
        """
        query_lower = query.lower()

        # Procedural queries
        if any(word in query_lower for word in ["how", "steps", "process", "procedure", "way to"]):
            return "procedural"

        # Temporal queries
        if any(word in query_lower for word in ["when", "date", "time", "yesterday", "last", "recent"]):
            return "temporal"

        # Troubleshooting queries
        if any(word in query_lower for word in ["issue", "problem", "error", "bug", "not working", "fix"]):
            return "troubleshooting"

        # Decision/agreement queries
        if any(word in query_lower for word in ["decided", "agreed", "decision", "agreement"]):
            return "decision"

        # Default to factual
        return "factual"

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with abbreviations and synonyms.

        Args:
            query: Original query

        Returns:
            List of expanded query variants
        """
        expanded = []
        abbreviations = self.vocab.get("abbreviations", {})

        # Replace abbreviations
        for abbrev, full_form in abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                # Create variant with full form
                expanded_query = re.sub(
                    pattern, full_form, query, flags=re.IGNORECASE
                )
                if expanded_query != query:
                    expanded.append(expanded_query)

        return expanded[:3]  # Limit to 3 expansions

    def _rewrite_query(self, query: str, num_rewrites: int = 2) -> List[str]:
        """Rewrite query using LLM.

        Args:
            query: Original query
            num_rewrites: Number of alternative queries to generate

        Returns:
            List of rewritten queries
        """
        if not self.enable_llm:
            return []

        prompt = f"""Given the user query about steel fabrication consulting and Tekla PowerFab software:

Query: "{query}"

Generate {num_rewrites} alternative phrasings that preserve the same intent but use different vocabulary or structure.
Focus on maintaining the technical context and any specific modules or features mentioned.

Return ONLY a JSON array of strings: ["alternative1", "alternative2"]"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rephrases technical queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON response
            # Handle both raw JSON and markdown code blocks
            if "```json" in content:
                content = re.search(r"```json\s*(\[.*?\])\s*```", content, re.DOTALL)
                if content:
                    content = content.group(1)

            alternatives = json.loads(content)

            if isinstance(alternatives, list):
                return alternatives[:num_rewrites]

        except Exception as e:
            print(f"Warning: Query rewriting failed: {e}")

        return []

    def _build_filter(self, entities: Dict) -> Optional[any]:
        """Build Qdrant filter from extracted entities.

        Args:
            entities: Extracted entities

        Returns:
            Qdrant Filter object or None
        """
        filter_kwargs = {}

        if "date_range" in entities:
            start, end = entities["date_range"]
            filter_kwargs["date_start"] = start
            filter_kwargs["date_end"] = end

        if "client" in entities:
            filter_kwargs["client_name"] = entities["client"]

        if "powerfab_modules" in entities:
            filter_kwargs["powerfab_modules"] = entities["powerfab_modules"]

        if not filter_kwargs:
            return None

        return build_metadata_filter(**filter_kwargs)


# For testing
if __name__ == "__main__":
    processor = QueryProcessor()

    # Test queries
    test_queries = [
        "How do I create a BOM in the Estimating module?",
        "What did we discuss with ClientA last week?",
        "Issues with WO creation yesterday",
        "Show me decisions about Production Control this month",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        processed = processor.process(query)
        print(f"  Intent: {processed.intent}")
        print(f"  Entities: {processed.entities}")
        print(f"  Expanded: {processed.expanded_queries}")
        print(f"  Has Filter: {processed.filter is not None}")
