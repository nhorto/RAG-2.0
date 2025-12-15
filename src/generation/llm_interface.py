"""LLM interface for response generation."""

from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..database.qdrant_client import SearchResult
from ..utils.config_loader import get_config


@dataclass
class GeneratedResponse:
    """Represents a generated response with metadata."""

    answer: str
    sources: List[Dict]
    query: str
    model: str
    tokens_used: int


class LLMInterface:
    """Interface for LLM-based response generation."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """Initialize LLM interface.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4)
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

        config = get_config()

        # Get API key
        if api_key is None:
            api_key = config.get_api_key("openai")

        self.client = OpenAI(api_key=api_key)

        # Model configuration
        llm_config = config.llm
        self.model = model or llm_config.get("model", "gpt-4")
        self.temperature = (
            temperature
            if temperature is not None
            else llm_config.get("temperature", 0.0)
        )
        self.max_tokens = max_tokens or llm_config.get("max_tokens", 1000)

        # System prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt for RAG responses.

        Returns:
            System prompt string
        """
        return """You are an expert assistant helping steel fabrication consultants retrieve information from their consulting session transcripts and summaries.

Your task is to:
1. Answer questions based ONLY on the provided context from transcripts/summaries
2. Cite specific sources when making claims
3. If the context doesn't contain enough information, say so clearly
4. Maintain technical accuracy about Tekla PowerFab features and workflows
5. Be concise but comprehensive in your responses

Format your citations as [Source N] where N corresponds to the source number provided."""

    def generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        include_metadata: bool = True,
    ) -> GeneratedResponse:
        """Generate response based on query and retrieved context.

        Args:
            query: User query
            search_results: Retrieved search results
            include_metadata: Whether to include source metadata

        Returns:
            GeneratedResponse object
        """
        # Build context from search results
        context = self._build_context(search_results, include_metadata)

        # Build user message
        user_message = self._build_user_message(query, context)

        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

        except Exception as e:
            answer = f"Error generating response: {e}"
            tokens_used = 0

        # Extract source information
        sources = self._extract_sources(search_results)

        return GeneratedResponse(
            answer=answer,
            sources=sources,
            query=query,
            model=self.model,
            tokens_used=tokens_used,
        )

    def _build_context(
        self, search_results: List[SearchResult], include_metadata: bool
    ) -> str:
        """Build context string from search results.

        Args:
            search_results: Search results to include
            include_metadata: Whether to include metadata

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, result in enumerate(search_results, 1):
            # Source header
            source_header = f"[Source {i}]"

            # Add metadata if requested
            if include_metadata:
                doc_meta = result.metadata.get("document_metadata", {})
                meta_parts = []

                if "date" in doc_meta:
                    meta_parts.append(f"Date: {doc_meta['date']}")
                if "client_name" in doc_meta:
                    meta_parts.append(f"Client: {doc_meta['client_name']}")
                if "document_type" in doc_meta:
                    meta_parts.append(f"Type: {doc_meta['document_type']}")

                if meta_parts:
                    source_header += f" ({', '.join(meta_parts)})"

            # Add text content
            context_parts.append(f"{source_header}\n{result.text}\n")

        return "\n".join(context_parts)

    def _build_user_message(self, query: str, context: str) -> str:
        """Build user message with query and context.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted user message
        """
        return f"""Context from consulting sessions:

{context}

---

Question: {query}

Please answer the question based on the context above. Cite sources using [Source N] format."""

    def _extract_sources(self, search_results: List[SearchResult]) -> List[Dict]:
        """Extract source information from search results.

        Args:
            search_results: Search results

        Returns:
            List of source dictionaries
        """
        sources = []

        for i, result in enumerate(search_results, 1):
            doc_meta = result.metadata.get("document_metadata", {})

            source = {
                "source_number": i,
                "chunk_id": result.chunk_id,
                "text_preview": result.text[:200] + "..."
                if len(result.text) > 200
                else result.text,
                "score": result.score,
                "date": doc_meta.get("date"),
                "client": doc_meta.get("client_name"),
                "filename": doc_meta.get("filename"),
                "document_type": doc_meta.get("document_type"),
            }

            sources.append(source)

        return sources


# For testing
if __name__ == "__main__":
    from ..database.qdrant_client import SearchResult

    # Mock search results for testing
    mock_results = [
        SearchResult(
            chunk_id="test-1",
            text="The Estimating module allows you to create BOMs by navigating to the BOM section and clicking New BOM. You can then add items and materials.",
            score=0.95,
            metadata={
                "document_metadata": {
                    "date": "2024-11-15",
                    "client_name": "ClientA",
                    "document_type": "transcript",
                }
            },
        ),
        SearchResult(
            chunk_id="test-2",
            text="When creating BOMs, make sure to validate all measurements and material specifications before saving.",
            score=0.87,
            metadata={
                "document_metadata": {
                    "date": "2024-11-16",
                    "client_name": "ClientA",
                    "document_type": "daily_summary",
                }
            },
        ),
    ]

    try:
        llm = LLMInterface()

        response = llm.generate_response(
            query="How do I create a BOM in Estimating?",
            search_results=mock_results,
        )

        print(f"Query: {response.query}\n")
        print(f"Answer:\n{response.answer}\n")
        print(f"Sources: {len(response.sources)}")
        print(f"Tokens used: {response.tokens_used}")

    except Exception as e:
        print(f"Test failed (API key may not be configured): {e}")
