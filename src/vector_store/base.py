"""
BaseVectorStore — abstract interface for vector database backends.

Defines the contract that all vector store implementations (Qdrant, FAISS,
pgvector) must fulfil.  Filter format mirrors QdrantManager._build_filter
input so existing callers work unchanged.

Usage:
    from src.vector_store.base import BaseVectorStore, DistanceMetric
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class DistanceMetric(str, Enum):
    """Backend-agnostic distance / similarity metric."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class BaseVectorStore(ABC):
    """Abstract base class for vector store backends.

    Filter format (backend-agnostic dict):
        {"field": "value", "field": {"gte": 1, "lte": 10}, "field": ["a", "b"]}
    """

    def __init__(self, default_distance: DistanceMetric = DistanceMetric.COSINE):
        self._default_distance = default_distance

    @property
    def default_distance(self) -> DistanceMetric:
        """The distance metric used when ``create_collection`` is called without an explicit ``distance``."""
        return self._default_distance

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Optional[DistanceMetric] = None,
        recreate: bool = False,
    ) -> bool:
        """Create a vector collection.  Returns True on success."""
        ...

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check whether *collection_name* exists."""
        ...

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.  Returns True on success."""
        ...

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_vectors(
        self,
        collection_name: str,
        vectors: Union[List[List[float]], np.ndarray],
        payloads: List[Dict],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> int:
        """Insert or update vectors.  Returns number upserted."""
        ...

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict]:
        """Similarity search.  Returns list of ``{"id", "score", "payload"}``."""
        ...

    @abstractmethod
    def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        filter_conditions: Optional[Dict] = None,
    ) -> Tuple[List[Dict], Optional[str]]:
        """Paginated iteration over stored points.

        Returns ``(records, next_offset)`` where each record is a dict with
        keys ``id``, and optionally ``payload`` and ``vector``.  *next_offset*
        is ``None`` when there are no more pages.
        """
        ...

    # ------------------------------------------------------------------
    # Metadata / statistics
    # ------------------------------------------------------------------

    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict:
        """Return metadata about a collection (name, vector_size, points_count, …)."""
        ...

    @abstractmethod
    def count_vectors(
        self,
        collection_name: str,
        filter_conditions: Optional[Dict] = None,
    ) -> int:
        """Count vectors, optionally filtered."""
        ...

    @abstractmethod
    def delete_vectors(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> bool:
        """Delete vectors by IDs or filter.  Returns True on success."""
        ...

    def get_statistics(self) -> Dict:
        """Return backend-level statistics (non-abstract, default ``{}``)."""
        return {}
