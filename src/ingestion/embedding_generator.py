"""Embedding generation using OpenAI API."""

import time
from typing import List, Union, Optional
import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

from ..utils.config_loader import get_config


class EmbeddingGenerator:
    """Generate embeddings using OpenAI's embedding models."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        dimensions: int = None,
        batch_size: int = None,
    ):
        """Initialize embedding generator.

        Args:
            api_key: OpenAI API key (if None, loads from config)
            model: Embedding model name (default: text-embedding-3-large)
            dimensions: Embedding dimensions (default: 3072 for -large, 1536 for -small)
            batch_size: Batch size for API calls (default: 100)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not installed. Run: pip install openai"
            )

        # Load config
        config = get_config()

        # Get API key
        if api_key is None:
            api_key = config.get_api_key("openai")

        self.client = OpenAI(api_key=api_key)

        # Model configuration
        self.model = model or config.get("embeddings.model", "text-embedding-3-large")
        self.dimensions = dimensions or config.get("embeddings.dimensions", 3072)
        self.batch_size = batch_size or config.get("embeddings.batch_size", 100)

        # Validate dimensions for text-embedding-3 models
        if "text-embedding-3" in self.model:
            valid_dims = [256, 768, 1536, 3072]
            if self.dimensions not in valid_dims:
                raise ValueError(
                    f"Invalid dimensions for {self.model}. "
                    f"Must be one of {valid_dims}"
                )

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.dimensions

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            )

            return response.data[0].embedding

        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector on error
            return [0.0] * self.dimensions

    def generate_embeddings(
        self, texts: List[str], show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts
            show_progress: Whether to print progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            if show_progress:
                print(f"Processing batch {batch_num}/{total_batches}...")

            try:
                # Replace empty texts with placeholder
                processed_batch = [
                    text if text and text.strip() else "empty"
                    for text in batch
                ]

                response = self.client.embeddings.create(
                    model=self.model,
                    input=processed_batch,
                    dimensions=self.dimensions,
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                # Rate limiting: small delay between batches
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                # Add zero vectors for failed batch
                embeddings.extend([[0.0] * self.dimensions] * len(batch))

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this generator.

        Returns:
            Embedding dimension
        """
        return self.dimensions

    def cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity (-1 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


class CachedEmbeddingGenerator:
    """Embedding generator with caching support."""

    def __init__(
        self,
        cache_dir: str = None,
        **embedding_kwargs,
    ):
        """Initialize cached embedding generator.

        Args:
            cache_dir: Directory for embedding cache
            **embedding_kwargs: Arguments passed to EmbeddingGenerator
        """
        import hashlib
        import pickle
        from pathlib import Path

        self.generator = EmbeddingGenerator(**embedding_kwargs)

        # Setup cache directory
        if cache_dir is None:
            config = get_config()
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "data" / "embeddings_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.hashlib = hashlib
        self.pickle = pickle

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: Input text

        Returns:
            Cache key (hash)
        """
        # Include model name in hash to avoid collisions
        cache_input = f"{self.generator.model}:{text}"
        return self.hashlib.md5(cache_input.encode()).hexdigest()

    def generate_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding with optional caching.

        Args:
            text: Input text
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            # Try to load from cache
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        return self.pickle.load(f)
                except Exception:
                    pass  # Cache read failed, regenerate

        # Generate embedding
        embedding = self.generator.generate_embedding(text)

        # Save to cache
        if use_cache:
            try:
                cache_key = self._get_cache_key(text)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, "wb") as f:
                    self.pickle.dump(embedding, f)
            except Exception as e:
                print(f"Warning: Failed to cache embedding: {e}")

        return embedding

    def generate_embeddings(
        self, texts: List[str], use_cache: bool = True, show_progress: bool = True
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with caching.

        Args:
            texts: List of input texts
            use_cache: Whether to use cache
            show_progress: Whether to show progress

        Returns:
            List of embedding vectors
        """
        if not use_cache:
            return self.generator.generate_embeddings(texts, show_progress)

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        embedding = self.pickle.load(f)
                        embeddings.append(embedding)
                except Exception:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    embeddings.append(None)  # Placeholder
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder

        # Generate embeddings for uncached texts
        if uncached_texts:
            if show_progress:
                print(
                    f"Generating {len(uncached_texts)} embeddings "
                    f"({len(texts) - len(uncached_texts)} cached)..."
                )

            new_embeddings = self.generator.generate_embeddings(
                uncached_texts, show_progress
            )

            # Fill in uncached embeddings and save to cache
            for i, (idx, text, embedding) in enumerate(
                zip(uncached_indices, uncached_texts, new_embeddings)
            ):
                embeddings[idx] = embedding

                # Save to cache
                try:
                    cache_key = self._get_cache_key(text)
                    cache_file = self.cache_dir / f"{cache_key}.pkl"
                    with open(cache_file, "wb") as f:
                        self.pickle.dump(embedding, f)
                except Exception:
                    pass  # Cache save failed, continue

        return embeddings


class HybridEmbeddingGenerator:
    """Generate both dense (OpenAI) and sparse (BM25) embeddings."""

    def __init__(
        self,
        dense_generator: EmbeddingGenerator = None,
        sparse_model: str = "Qdrant/bm25",
    ):
        """Initialize hybrid embedding generator.

        Args:
            dense_generator: EmbeddingGenerator instance (OpenAI)
            sparse_model: FastEmbed sparse model name
        """
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "FastEmbed library not installed. Run: pip install fastembed"
            )

        # Dense embeddings (OpenAI)
        self.dense_generator = dense_generator or EmbeddingGenerator()

        # Sparse embeddings (BM25)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model)

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, any]]:
        """Generate both dense and sparse embeddings.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress

        Returns:
            List of dicts with 'dense' and 'sparse' keys
        """
        if show_progress:
            print(f"Generating dense embeddings for {len(texts)} texts...")

        # Generate dense embeddings (OpenAI API)
        dense_embeddings = self.dense_generator.generate_embeddings(
            texts, show_progress=show_progress
        )

        if show_progress:
            print(f"Generating sparse embeddings for {len(texts)} texts...")

        # Generate sparse embeddings (local BM25)
        # FastEmbed returns generator, convert to list
        sparse_embeddings = list(self.sparse_model.embed(texts))

        # Combine into hybrid format
        hybrid_embeddings = []
        for dense, sparse in zip(dense_embeddings, sparse_embeddings):
            hybrid_embeddings.append({
                "dense": dense,
                "sparse": sparse  # This is already in Qdrant sparse format
            })

        return hybrid_embeddings

    def generate_query_embeddings(self, query: str) -> Dict[str, any]:
        """Generate embeddings for a single query.

        Args:
            query: Query text

        Returns:
            Dict with 'dense' and 'sparse' embeddings
        """
        # Dense embedding
        dense = self.dense_generator.generate_embedding(query)

        # Sparse embedding
        sparse = list(self.sparse_model.embed([query]))[0]

        return {
            "dense": dense,
            "sparse": sparse
        }


# For testing
if __name__ == "__main__":
    # Test basic embedding generation
    generator = EmbeddingGenerator()

    text = "This is a test sentence for embedding generation."
    embedding = generator.generate_embedding(text)

    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First 10 values: {embedding[:10]}")

    # Test batch generation
    texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence with more words to test batching",
    ]

    embeddings = generator.generate_embeddings(texts)
    print(f"\nGenerated {len(embeddings)} embeddings")

    # Test similarity
    sim = generator.cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first two: {sim:.4f}")
