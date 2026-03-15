"""Tests for VectorStoreFactory — dispatch, from_config, error handling."""

import pytest

from src.vector_store.base import BaseVectorStore
from src.vector_store.factory import VectorStoreFactory


def _skip_if_no_qdrant():
    """Skip the test if Qdrant container is not running."""
    try:
        store = VectorStoreFactory.create("qdrant")
        return store
    except Exception:
        pytest.skip("Qdrant container not available")


class TestCreate:
    def test_create_qdrant(self):
        store = _skip_if_no_qdrant()
        assert isinstance(store, BaseVectorStore)

    def test_create_faiss(self):
        store = VectorStoreFactory.create("faiss")
        assert isinstance(store, BaseVectorStore)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            VectorStoreFactory.create("redis")


class TestFromConfig:
    def test_from_config_qdrant(self):
        from src.benchmarks.config import VectorDBConfig

        cfg = VectorDBConfig(backend="qdrant", connection_params=None)
        try:
            store = VectorStoreFactory.from_config(cfg)
        except Exception:
            pytest.skip("Qdrant container not available")
        assert isinstance(store, BaseVectorStore)

    def test_from_config_faiss(self):
        from src.benchmarks.config import VectorDBConfig

        cfg = VectorDBConfig(backend="faiss")
        store = VectorStoreFactory.from_config(cfg)
        assert isinstance(store, BaseVectorStore)

    def test_from_config_defaults(self):
        from src.benchmarks.config import VectorDBConfig

        cfg = VectorDBConfig()
        # Default is qdrant — skip if container not available
        try:
            store = VectorStoreFactory.from_config(cfg)
        except Exception:
            pytest.skip("Qdrant container not available")
        assert isinstance(store, BaseVectorStore)


class TestFromConfigDistance:
    def test_from_config_passes_distance_metric(self):
        from src.benchmarks.config import VectorDBConfig
        from src.vector_store.base import DistanceMetric

        cfg = VectorDBConfig(backend="faiss", distance_metric="euclidean")
        store = VectorStoreFactory.from_config(cfg)
        assert store.default_distance == DistanceMetric.EUCLIDEAN

    def test_from_config_default_distance_is_cosine(self):
        from src.benchmarks.config import VectorDBConfig
        from src.vector_store.base import DistanceMetric

        cfg = VectorDBConfig(backend="faiss")
        store = VectorStoreFactory.from_config(cfg)
        assert store.default_distance == DistanceMetric.COSINE


class TestAvailableBackends:
    def test_returns_sorted_list(self):
        backends = VectorStoreFactory.available_backends()
        assert isinstance(backends, list)
        assert backends == sorted(backends)

    def test_contains_all_three(self):
        backends = VectorStoreFactory.available_backends()
        assert "qdrant" in backends
        assert "faiss" in backends
        assert "pgvector" in backends

    def test_length(self):
        assert len(VectorStoreFactory.available_backends()) == 3
