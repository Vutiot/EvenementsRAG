"""
QdrantAdapter — thin wrapper that makes QdrantManager conform to BaseVectorStore.

Delegates every call to the underlying QdrantManager; the only translations
are DistanceMetric → Qdrant's Distance enum and PointStruct → plain dicts
in scroll().

Usage:
    from src.vector_store.qdrant_adapter import QdrantAdapter

    store = QdrantAdapter(use_memory=True)
    store.create_collection("test", vector_size=384)
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from qdrant_client.models import Distance

from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.vector_store.qdrant_manager import QdrantManager

_DISTANCE_MAP: Dict[DistanceMetric, Distance] = {
    DistanceMetric.COSINE: Distance.COSINE,
    DistanceMetric.EUCLIDEAN: Distance.EUCLID,
    DistanceMetric.DOT_PRODUCT: Distance.DOT,
}


class QdrantAdapter(BaseVectorStore):
    """Adapts :class:`QdrantManager` to the :class:`BaseVectorStore` interface."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        default_distance: DistanceMetric = DistanceMetric.COSINE,
        **kwargs,
    ):
        """Wrap an existing QdrantManager or create one from *kwargs*.

        Args:
            qdrant_manager: Pre-built manager instance (takes precedence).
            default_distance: Metric used when ``create_collection`` is called
                              without an explicit ``distance``.
            **kwargs: Forwarded to ``QdrantManager(...)`` when *qdrant_manager*
                      is ``None``.  Common keys: ``use_memory``, ``host``,
                      ``port``, ``path``.
        """
        super().__init__(default_distance=default_distance)
        if qdrant_manager is not None:
            self._mgr = qdrant_manager
        else:
            self._mgr = QdrantManager(**kwargs)

    # Expose the underlying manager for callers that need raw Qdrant access
    @property
    def manager(self) -> QdrantManager:
        return self._mgr

    # Expose the raw client for callers that need it (e.g., BenchmarkRunner)
    @property
    def client(self):
        return self._mgr.client

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Optional[DistanceMetric] = None,
        recreate: bool = False,
    ) -> bool:
        distance = distance or self._default_distance
        qdrant_distance = _DISTANCE_MAP.get(distance)
        if qdrant_distance is None:
            raise ValueError(
                f"Qdrant does not support distance metric '{distance.value}'. "
                f"Supported: {list(_DISTANCE_MAP.keys())}"
            )
        return self._mgr.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance=qdrant_distance,
            recreate=recreate,
        )

    def collection_exists(self, collection_name: str) -> bool:
        return self._mgr.collection_exists(collection_name)

    def delete_collection(self, collection_name: str) -> bool:
        return self._mgr.delete_collection(collection_name)

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def upsert_vectors(
        self,
        collection_name: str,
        vectors: Union[List[List[float]], np.ndarray],
        payloads: List[Dict],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> int:
        return self._mgr.upsert_vectors(
            collection_name=collection_name,
            vectors=vectors,
            payloads=payloads,
            ids=ids,
            batch_size=batch_size,
        )

    def search(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict]:
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        query_filter = None
        if filter_conditions:
            query_filter = self._mgr._build_filter(filter_conditions)

        # qdrant-client >= 1.17 removed client.search(); use query_points()
        response = self._mgr.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

        return [
            {"id": pt.id, "score": pt.score, "payload": pt.payload}
            for pt in response.points
        ]

    def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        filter_conditions: Optional[Dict] = None,
    ) -> Tuple[List[Dict], Optional[str]]:
        query_filter = None
        if filter_conditions:
            query_filter = self._mgr._build_filter(filter_conditions)

        points, next_offset = self._mgr.client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            scroll_filter=query_filter,
        )

        records = []
        for pt in points:
            record: Dict = {"id": pt.id}
            if with_payload and pt.payload is not None:
                record["payload"] = pt.payload
            if with_vectors and pt.vector is not None:
                record["vector"] = pt.vector
            records.append(record)

        return records, next_offset

    # ------------------------------------------------------------------
    # Metadata / statistics
    # ------------------------------------------------------------------

    def get_collection_info(self, collection_name: str) -> Dict:
        return self._mgr.get_collection_info(collection_name)

    def count_vectors(
        self,
        collection_name: str,
        filter_conditions: Optional[Dict] = None,
    ) -> int:
        return self._mgr.count_vectors(collection_name, filter_conditions)

    def delete_vectors(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> bool:
        return self._mgr.delete_vectors(collection_name, ids, filter_conditions)

    def get_statistics(self) -> Dict:
        return self._mgr.get_statistics()
