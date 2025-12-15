"""Qdrant client wrapper for RAG system."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient as QdrantClientBase
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
        SearchRequest,
        ScoredPoint,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ..utils.config_loader import get_config


@dataclass
class SearchResult:
    """Represents a search result from Qdrant."""

    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

    def __repr__(self) -> str:
        return (
            f"SearchResult(chunk_id={self.chunk_id[:8]}..., "
            f"score={self.score:.4f}, text={self.text[:50]}...)"
        )


class QdrantManager:
    """Manager for Qdrant vector database operations."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        api_key: str = None,
    ):
        """Initialize Qdrant manager.

        Args:
            host: Qdrant host (default: localhost)
            port: Qdrant port (default: 6333)
            collection_name: Collection name (default: consulting_transcripts)
            api_key: API key for Qdrant Cloud (optional)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not installed. Run: pip install qdrant-client"
            )

        # Load config
        config = get_config()
        qdrant_config = config.qdrant

        self.host = host or qdrant_config.get("host", "localhost")
        self.port = port or qdrant_config.get("port", 6333)
        self.collection_name = collection_name or qdrant_config.get(
            "collection_name", "consulting_transcripts"
        )

        # Initialize client
        if api_key:
            self.client = QdrantClientBase(url=f"https://{self.host}", api_key=api_key)
        else:
            self.client = QdrantClientBase(host=self.host, port=self.port)

        # Store vector config
        self.vector_size = qdrant_config.get("vector_config", {}).get("size", 3072)
        self.distance = qdrant_config.get("vector_config", {}).get("distance", "Cosine")

    def create_collection(
        self, vector_size: int = None, distance: str = None, recreate: bool = False
    ) -> bool:
        """Create Qdrant collection.

        Args:
            vector_size: Dimension of vectors (default from config)
            distance: Distance metric (Cosine, Euclid, Dot)
            recreate: If True, delete existing collection

        Returns:
            True if created successfully
        """
        vector_size = vector_size or self.vector_size

        # Map distance string to Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        distance_metric = distance_map.get(distance or self.distance, Distance.COSINE)

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists:
            if recreate:
                print(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"Collection already exists: {self.collection_name}")
                return True

        # Create collection
        print(f"Creating collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_metric,
            ),
        )

        return True

    def collection_exists(self) -> bool:
        """Check if collection exists.

        Returns:
            True if collection exists
        """
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception:
            return False

    def get_collection_info(self) -> Dict:
        """Get collection information.

        Returns:
            Dictionary with collection stats
        """
        if not self.collection_exists():
            return {"exists": False}

        info = self.client.get_collection(self.collection_name)

        return {
            "exists": True,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }

    def upsert_points(
        self, points: List[Dict[str, Any]], show_progress: bool = True
    ) -> bool:
        """Insert or update points in collection.

        Args:
            points: List of point dictionaries with 'id', 'vector', 'payload'
            show_progress: Whether to show progress

        Returns:
            True if successful
        """
        if not self.collection_exists():
            raise ValueError(
                f"Collection {self.collection_name} does not exist. "
                "Create it first with create_collection()"
            )

        # Convert to PointStruct objects
        point_structs = []
        for point in points:
            point_structs.append(
                PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point.get("payload", {}),
                )
            )

        # Batch upsert
        batch_size = 100
        total = len(point_structs)

        for i in range(0, total, batch_size):
            batch = point_structs[i : i + batch_size]

            if show_progress:
                print(f"Upserting points {i+1}-{min(i+batch_size, total)}/{total}")

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return True

    def search(
        self,
        query_vector: List[float],
        filters: Optional[Filter] = None,
        limit: int = 10,
        score_threshold: float = None,
    ) -> List[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            filters: Qdrant filter conditions
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": limit,
        }

        if filters:
            search_params["query_filter"] = filters

        if score_threshold:
            search_params["score_threshold"] = score_threshold

        results = self.client.search(**search_params)

        return self._convert_to_search_results(results)

    def _convert_to_search_results(
        self, qdrant_results: List[ScoredPoint]
    ) -> List[SearchResult]:
        """Convert Qdrant results to SearchResult objects.

        Args:
            qdrant_results: Results from Qdrant

        Returns:
            List of SearchResult objects
        """
        search_results = []

        for result in qdrant_results:
            search_results.append(
                SearchResult(
                    chunk_id=str(result.id),
                    text=result.payload.get("text", ""),
                    score=result.score,
                    metadata=result.payload,
                )
            )

        return search_results

    def get_point(self, point_id: str) -> Optional[Dict]:
        """Retrieve a specific point by ID.

        Args:
            point_id: Point ID

        Returns:
            Point data or None if not found
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
            )

            if result:
                return {
                    "id": result[0].id,
                    "payload": result[0].payload,
                }
            return None

        except Exception:
            return None

    def delete_points(self, point_ids: List[str]) -> bool:
        """Delete points from collection.

        Args:
            point_ids: List of point IDs to delete

        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )
            return True
        except Exception as e:
            print(f"Error deleting points: {e}")
            return False

    def count_points(self, filters: Optional[Filter] = None) -> int:
        """Count points in collection.

        Args:
            filters: Optional filters

        Returns:
            Number of points
        """
        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=filters,
        )
        return result.count

    def scroll_points(
        self, limit: int = 100, offset: str = None, with_vectors: bool = False
    ) -> tuple[List[Dict], Optional[str]]:
        """Scroll through points in collection.

        Args:
            limit: Number of points to retrieve
            offset: Offset for pagination
            with_vectors: Whether to include vectors

        Returns:
            Tuple of (points, next_offset)
        """
        result = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            offset=offset,
            with_vectors=with_vectors,
        )

        points = [
            {
                "id": p.id,
                "payload": p.payload,
                "vector": p.vector if with_vectors else None,
            }
            for p in result[0]
        ]

        next_offset = result[1]

        return points, next_offset


def build_metadata_filter(
    client_name: str = None,
    date_start: str = None,
    date_end: str = None,
    document_type: str = None,
    powerfab_modules: List[str] = None,
) -> Optional[Filter]:
    """Build Qdrant filter from metadata criteria.

    Args:
        client_name: Filter by client name
        date_start: Filter by start date (ISO format)
        date_end: Filter by end date (ISO format)
        document_type: Filter by document type
        powerfab_modules: Filter by PowerFab modules

    Returns:
        Qdrant Filter object or None
    """
    conditions = []

    if client_name:
        conditions.append(
            FieldCondition(
                key="document_metadata.client_name",
                match=MatchValue(value=client_name),
            )
        )

    if date_start or date_end:
        range_params = {}
        if date_start:
            range_params["gte"] = date_start
        if date_end:
            range_params["lte"] = date_end

        conditions.append(
            FieldCondition(
                key="document_metadata.date",
                range=Range(**range_params),
            )
        )

    if document_type:
        conditions.append(
            FieldCondition(
                key="document_metadata.document_type",
                match=MatchValue(value=document_type),
            )
        )

    if powerfab_modules:
        # Match any of the modules
        for module in powerfab_modules:
            conditions.append(
                FieldCondition(
                    key="content_metadata.powerfab.modules",
                    match=MatchValue(value=module),
                )
            )

    if not conditions:
        return None

    return Filter(must=conditions)


# Global instance
_qdrant_manager = None


def get_qdrant_manager(
    host: str = None, port: int = None, collection_name: str = None
) -> QdrantManager:
    """Get global Qdrant manager instance.

    Args:
        host: Qdrant host
        port: Qdrant port
        collection_name: Collection name

    Returns:
        QdrantManager instance
    """
    global _qdrant_manager
    if _qdrant_manager is None:
        _qdrant_manager = QdrantManager(host, port, collection_name)
    return _qdrant_manager
