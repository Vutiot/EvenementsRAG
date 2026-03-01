"""
VectorStoreFactory — registry-based creation of vector store backends.

Follows the same lazy-import pattern as ``_RAG_REGISTRY`` in ``runner.py``.

Usage:
    from src.vector_store.factory import VectorStoreFactory

    store = VectorStoreFactory.create("faiss", persist_dir="/tmp/faiss")
    store = VectorStoreFactory.from_config(vector_db_config)
    print(VectorStoreFactory.available_backends())
"""

import importlib
from typing import Optional

from src.vector_store.base import BaseVectorStore, DistanceMetric

_BACKEND_REGISTRY = {
    "qdrant": "src.vector_store.qdrant_adapter.QdrantAdapter",
    "faiss": "src.vector_store.faiss_store.FAISSStore",
    "pgvector": "src.vector_store.pgvector_store.PgVectorStore",
}


class VectorStoreFactory:
    """Create :class:`BaseVectorStore` instances by backend name."""

    @staticmethod
    def create(backend: str, **kwargs) -> BaseVectorStore:
        """Instantiate a vector store backend.

        Args:
            backend: One of ``"qdrant"``, ``"faiss"``, ``"pgvector"``.
            **kwargs: Forwarded to the backend constructor.

        Raises:
            ValueError: If *backend* is not in the registry.
        """
        target = _BACKEND_REGISTRY.get(backend)
        if target is None:
            raise ValueError(
                f"Unknown vector store backend '{backend}'. "
                f"Available: {sorted(_BACKEND_REGISTRY.keys())}"
            )

        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    @staticmethod
    def from_config(vector_db_config) -> BaseVectorStore:
        """Create a store from a :class:`VectorDBConfig` instance.

        Passes ``connection_params`` (if set) as keyword arguments to the
        backend constructor.
        """
        kwargs = {}
        if vector_db_config.connection_params:
            kwargs.update(vector_db_config.connection_params)
        kwargs["default_distance"] = DistanceMetric(vector_db_config.distance_metric)
        return VectorStoreFactory.create(vector_db_config.backend, **kwargs)

    @staticmethod
    def available_backends() -> list:
        """Return sorted list of registered backend names."""
        return sorted(_BACKEND_REGISTRY.keys())
