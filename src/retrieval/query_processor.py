"""Query processing including expansion, rewriting, and metadata extraction.

Enhanced with advanced query processing capabilities:
- Query decomposition for multi-part queries
- Query augmentation for vague/underspecified queries
- Intelligent routing based on query complexity
"""

import re
import json
import logging
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
from ..utils.llm_client import LLMClient

# Import advanced query processing components
try:
    from .query_analyzer import QueryAnalyzer, QueryAnalysis
    from .query_decomposer import QueryDecomposer, DecomposedQuery
    from .query_augmenter import QueryAugmenter, AugmentedQuery
    from .types import QueryIntent
    ADVANCED_PROCESSING_AVAILABLE = True
except ImportError:
    ADVANCED_PROCESSING_AVAILABLE = False
    QueryAnalysis = None
    DecomposedQuery = None
    AugmentedQuery = None
    QueryIntent = None

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Represents a processed query with expansions and metadata.

    Enhanced with advanced processing fields for decomposition and augmentation.
    """

    original_query: str
    expanded_queries: List[str]
    entities: Dict[str, any]
    filter: Optional[Dict]  # Qdrant Filter object
    intent: str  # Query intent classification

    # Advanced processing fields (NEW)
    analysis: Optional[QueryAnalysis] = None
    decomposed: Optional[DecomposedQuery] = None
    augmented: Optional[AugmentedQuery] = None

    @property
    def all_query_variants(self) -> List[str]:
        """Get all query variants (expanded + augmented + sub-queries).

        Returns:
            List of all unique query variants for retrieval
        """
        variants = [self.original_query] + self.expanded_queries

        # Add augmented variants
        if self.augmented:
            variants.extend(self.augmented.augmented_variants)

        # Add sub-queries
        if self.decomposed:
            variants.extend([sq.query_text for sq in self.decomposed.sub_queries])

        # Deduplicate while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)

        return unique_variants


class QueryProcessor:
    """Process queries for retrieval including expansion and rewriting.

    Enhanced with advanced query processing capabilities including decomposition
    and augmentation based on query complexity analysis.
    """

    def __init__(self, llm_api_key: str = None, llm_model: str = None):
        """Initialize query processor.

        Args:
            llm_api_key: OpenAI API key for query rewriting and advanced processing
            llm_model: LLM model for query rewriting
        """
        config = get_config()
        self.vocab = load_domain_vocabulary()
        self.query_config = config.query_processing

        # Initialize LLM client for query rewriting
        self.llm_client, self.llm_model = self._init_llm_client(llm_api_key, llm_model, config)
        self.enable_llm = self.llm_client is not None

        # Initialize advanced processing components
        self._init_advanced_processing(config)

    def _init_llm_client(self, llm_api_key: Optional[str], llm_model: Optional[str], config) -> tuple:
        """Initialize LLM client for query rewriting.

        Args:
            llm_api_key: Optional API key
            llm_model: Optional model name
            config: Config object

        Returns:
            Tuple of (llm_client, llm_model)
        """
        if not OPENAI_AVAILABLE:
            return None, None

        # Get API key
        if llm_api_key is None:
            try:
                llm_api_key = config.get_api_key("openai")
            except ValueError:
                return None, None

        # Create OpenAI client for rewriting
        client = OpenAI(api_key=llm_api_key)
        model = llm_model or config.get("llm.model", "gpt-4")
        return client, model

    def _init_advanced_processing(self, config):
        """Initialize advanced query processing components.

        Args:
            config: Config object
        """
        self.enable_advanced_processing = (
            ADVANCED_PROCESSING_AVAILABLE and
            self.enable_llm and
            self.query_config.get("advanced_processing", {}).get("enabled", False)
        )

        if not self.enable_advanced_processing:
            self.analyzer = None
            self.decomposer = None
            self.augmenter = None
            self.enable_decomposition = False
            self.enable_augmentation = False
            return

        # Get advanced processing config
        adv_config = self.query_config.get("advanced_processing", {})

        # Feature flags
        self.enable_decomposition = adv_config.get("enable_decomposition", True)
        self.enable_augmentation = adv_config.get("enable_augmentation", True)

        # Initialize analyzer
        analysis_config = adv_config.get("analysis", {})
        self.analyzer = QueryAnalyzer(config=analysis_config)

        # Initialize decomposer with LLMClient
        if self.enable_decomposition:
            decomp_config = adv_config.get("decomposition", {})
            decomp_llm_client = LLMClient(
                client=self.llm_client,
                model=decomp_config.get("llm_model", "gpt-4"),
                temperature=decomp_config.get("temperature", 0.3),
                max_tokens=decomp_config.get("max_tokens", 500)
            )
            self.decomposer = QueryDecomposer(
                llm_client=decomp_llm_client,
                max_sub_queries=decomp_config.get("max_sub_queries", 5)
            )
        else:
            self.decomposer = None

        # Initialize augmenter with LLMClient
        if self.enable_augmentation:
            aug_config = adv_config.get("augmentation", {})
            aug_llm_client = LLMClient(
                client=self.llm_client,
                model=aug_config.get("llm_model", "gpt-3.5-turbo"),
                temperature=aug_config.get("temperature", 0.5),
                max_tokens=aug_config.get("max_tokens", 300)
            )
            self.augmenter = QueryAugmenter(
                llm_client=aug_llm_client,
                max_variants=aug_config.get("max_variants", 5),
                domain_vocab=self.vocab
            )
        else:
            self.augmenter = None

    def process(self, query: str, context: Optional[Dict] = None) -> ProcessedQuery:
        """Process query through enhanced pipeline with advanced capabilities.

        This method implements a multi-stage processing pipeline:
        1. Query analysis (detect if decomposition/augmentation needed)
        2. Query decomposition (if complex multi-part query)
        3. Query augmentation (if vague/underspecified)
        4. Entity extraction
        5. Intent classification
        6. Query expansion (abbreviations)
        7. Query rewriting (LLM alternatives)
        8. Metadata filter building

        Args:
            query: User query string
            context: Optional conversation context for pronoun resolution

        Returns:
            ProcessedQuery object with all enhancements
        """
        # STEP 1: Analyze query for advanced processing needs (NEW)
        analysis = None
        decomposed = None
        augmented = None

        if self.enable_advanced_processing:
            analysis = self.analyzer.analyze(query)

            # STEP 2: Query decomposition (NEW - conditional)
            if self.enable_decomposition and analysis.needs_decomposition:
                try:
                    decomposed = self.decomposer.decompose(query, analysis)
                except Exception as e:
                    logger.warning(f"Query decomposition failed: {e}")
                    decomposed = None

            # STEP 3: Query augmentation (NEW - conditional)
            if self.enable_augmentation and analysis.needs_augmentation:
                try:
                    augmented = self.augmenter.augment(query, analysis, context)
                except Exception as e:
                    logger.warning(f"Query augmentation failed: {e}")
                    augmented = None

        # STEP 4: Extract entities (EXISTING)
        entities = self._extract_entities(query)

        # STEP 5: Classify intent (EXISTING)
        intent = self._classify_intent(query)

        # STEP 6: Query expansion (EXISTING - abbreviations, synonyms)
        expanded = []
        if self.query_config.get("enable_expansion", True):
            expanded = self._expand_query(query)

        # STEP 7: Query rewriting (EXISTING - LLM-based alternatives)
        rewritten = []
        if self.query_config.get("enable_rewriting", True) and self.enable_llm:
            num_rewrites = self.query_config.get("num_rewrites", 2)
            rewritten = self._rewrite_query(query, num_rewrites)

        # Combine traditional query variations
        expanded_queries = expanded + rewritten

        # STEP 8: Build metadata filter (EXISTING)
        metadata_filter = None
        if self.query_config.get("enable_metadata_extraction", True):
            metadata_filter = self._build_filter(entities)

        return ProcessedQuery(
            original_query=query,
            expanded_queries=expanded_queries,
            entities=entities,
            filter=metadata_filter,
            intent=intent,
            analysis=analysis,
            decomposed=decomposed,
            augmented=augmented,
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
            Intent classification string (QueryIntent enum value when available)
        """
        query_lower = query.lower()

        # Procedural queries
        if any(word in query_lower for word in ["how", "steps", "process", "procedure", "way to"]):
            return QueryIntent.PROCEDURAL.value if QueryIntent else "procedural"

        # Temporal queries
        if any(word in query_lower for word in ["when", "date", "time", "yesterday", "last", "recent"]):
            return QueryIntent.TEMPORAL.value if QueryIntent else "temporal"

        # Troubleshooting queries
        if any(word in query_lower for word in ["issue", "problem", "error", "bug", "not working", "fix"]):
            return QueryIntent.TROUBLESHOOTING.value if QueryIntent else "troubleshooting"

        # Decision/agreement queries
        if any(word in query_lower for word in ["decided", "agreed", "decision", "agreement"]):
            return QueryIntent.DECISION.value if QueryIntent else "decision"

        # Default to factual
        return QueryIntent.FACTUAL.value if QueryIntent else "factual"

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
            logger.warning(f"Query rewriting failed: {e}")

        return []

    def _build_filter(self, entities: Dict) -> Optional[Dict]:
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
