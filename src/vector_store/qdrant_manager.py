"""
Qdrant vector store client for RAG system.

Manages connections, collections, and vector operations with Qdrant.
Requires a running Qdrant container (see scripts/setup_qdrant.sh).

Supports:
- Collection creation and management
- Vector insertion with metadata
- Similarity search
- Filtering by metadata
- Batch operations

Usage:
    from src.vector_store.qdrant_client import QdrantManager

    manager = QdrantManager()
    manager.create_collection("ww2_events", vector_size=384)
    manager.upsert_vectors(collection_name="ww2_events", vectors=embeddings, payloads=metadata)
    results = manager.search(collection_name="ww2_events", query_vector=query_embedding, limit=5)
"""

from typing import Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantManager:
    """Manages Qdrant vector database operations for RAG."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize Qdrant client (container-only).

        Args:
            host: Qdrant server host (default: from settings)
            port: Qdrant server port (default: from settings)
        """
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT

        logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            # Test connection
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            logger.info("Tip: Start Qdrant with: bash scripts/setup_qdrant.sh start")
            raise

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        recreate: bool = False,
    ) -> bool:
        """
        Create a new vector collection.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: Delete existing collection if it exists

        Returns:
            True if successful
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)

            if collection_exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    return True

            # Create collection
            logger.info(
                f"Creating collection: {collection_name} "
                f"(vector_size={vector_size}, distance={distance})"
            )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )

            logger.info(f"Successfully created collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    def upsert_vectors(
        self,
        collection_name: str,
        vectors: Union[List[List[float]], np.ndarray],
        payloads: List[Dict],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Insert or update vectors in the collection.

        Args:
            collection_name: Target collection
            vectors: List of vectors or numpy array
            payloads: Metadata for each vector
            ids: Optional IDs for each vector (auto-generated if not provided)
            batch_size: Batch size for upload

        Returns:
            Number of vectors upserted
        """
        if len(vectors) != len(payloads):
            raise ValueError(f"Vectors ({len(vectors)}) and payloads ({len(payloads)}) must have same length")

        # Convert numpy array to list if needed
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        logger.info(f"Upserting {len(vectors)} vectors to {collection_name}")

        try:
            # Upsert in batches
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i : i + batch_size]
                batch_payloads = payloads[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]

                # Create points
                points = [
                    PointStruct(id=point_id, vector=vector, payload=payload)
                    for point_id, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
                ]

                # Upsert batch
                self.client.upsert(collection_name=collection_name, points=points)

                total_upserted += len(points)
                logger.debug(f"Upserted batch {i // batch_size + 1}: {len(points)} vectors")

            logger.info(f"Successfully upserted {total_upserted} vectors")
            return total_upserted

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search for similar vectors.

        Args:
            collection_name: Collection to search
            query_vector: Query vector
            limit: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Metadata filters

        Returns:
            List of search results with scores and payloads
        """
        # Convert numpy array to list if needed
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        try:
            # Build filter
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)

            # Search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )

            # Format results
            results = [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in search_result
            ]

            logger.debug(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _build_filter(self, filter_conditions: Dict) -> Filter:
        """
        Build Qdrant filter from conditions dictionary.

        Args:
            filter_conditions: Dictionary of field -> value/range

        Returns:
            Qdrant Filter object

        Example:
            {
                "article_title": "Normandy landings",
                "year": {"gte": 1944, "lte": 1945},
                "type": ["battle", "campaign"]
            }
        """
        must_conditions = []

        for field, value in filter_conditions.items():
            if isinstance(value, dict):
                # Range condition
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    must_conditions.append(
                        FieldCondition(
                            key=field,
                            range=Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                                gt=value.get("gt"),
                                lt=value.get("lt"),
                            ),
                        )
                    )
            elif isinstance(value, list):
                # Match any in list
                for val in value:
                    must_conditions.append(FieldCondition(key=field, match=MatchValue(value=val)))
            else:
                # Exact match
                must_conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))

        return Filter(must=must_conditions) if must_conditions else None

    def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a collection."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.name,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def count_vectors(self, collection_name: str, filter_conditions: Optional[Dict] = None) -> int:
        """Count vectors in collection, optionally with filters."""
        try:
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)

            count = self.client.count(collection_name=collection_name, count_filter=query_filter)
            return count.count

        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0

    def delete_vectors(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> bool:
        """
        Delete vectors by IDs or filter.

        Args:
            collection_name: Collection name
            ids: List of IDs to delete
            filter_conditions: Delete all matching filter

        Returns:
            True if successful
        """
        try:
            if ids:
                self.client.delete(collection_name=collection_name, points_selector=ids)
                logger.info(f"Deleted {len(ids)} vectors by ID")
            elif filter_conditions:
                query_filter = self._build_filter(filter_conditions)
                self.client.delete(collection_name=collection_name, points_selector=query_filter)
                logger.info("Deleted vectors by filter")
            else:
                raise ValueError("Must provide either ids or filter_conditions")

            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Get overall statistics about Qdrant instance."""
        try:
            collections = self.client.get_collections().collections

            stats = {
                "host": self.host,
                "port": self.port,
                "total_collections": len(collections),
                "collections": {},
            }

            for collection in collections:
                info = self.get_collection_info(collection.name)
                stats["collections"][collection.name] = info

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


if __name__ == "__main__":
    # Test the Qdrant manager
    print("Testing QdrantManager...")

    manager = QdrantManager()

    # Create test collection
    print("\n=== Creating test collection ===")
    manager.create_collection("test_collection", vector_size=384)

    # Insert test vectors
    print("\n=== Inserting test vectors ===")
    test_vectors = np.random.rand(10, 384).tolist()
    test_payloads = [
        {
            "content": f"Test chunk {i}",
            "article_title": f"Article {i % 3}",
            "chunk_index": i,
            "year": 1940 + i,
        }
        for i in range(10)
    ]

    manager.upsert_vectors(
        collection_name="test_collection",
        vectors=test_vectors,
        payloads=test_payloads,
    )

    # Get collection info
    print("\n=== Collection info ===")
    info = manager.get_collection_info("test_collection")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test search
    print("\n=== Testing search ===")
    query_vector = np.random.rand(384)
    results = manager.search(
        collection_name="test_collection",
        query_vector=query_vector,
        limit=3,
    )

    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    Score: {result['score']:.4f}")
        print(f"    Content: {result['payload']['content']}")
        print(f"    Article: {result['payload']['article_title']}")

    # Test filtered search
    print("\n=== Testing filtered search (year >= 1945) ===")
    results = manager.search(
        collection_name="test_collection",
        query_vector=query_vector,
        limit=5,
        filter_conditions={"year": {"gte": 1945}},
    )

    print(f"Found {len(results)} results:")
    for result in results:
        print(f"  - {result['payload']['content']} (year: {result['payload']['year']})")

    # Get statistics
    print("\n=== Statistics ===")
    stats = manager.get_statistics()
    print(f"Total collections: {stats['total_collections']}")
    for name, coll_info in stats["collections"].items():
        print(f"\nCollection '{name}':")
        print(f"  Points: {coll_info['points_count']}")
        print(f"  Vector size: {coll_info['vector_size']}")
        print(f"  Distance: {coll_info['distance']}")

    print("\n=== All tests completed successfully ===")
