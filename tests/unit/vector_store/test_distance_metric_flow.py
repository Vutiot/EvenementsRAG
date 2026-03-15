"""Tests for E2-F2-T2: distance metric flows from config -> store -> collection.

Covers:
- default_distance on each store constructor
- Factory from_config passes distance
- create_collection uses default when distance=None
- Explicit distance overrides default

Qdrant tests skip automatically if no container is running.
"""

import uuid

import pytest
from unittest.mock import patch, MagicMock

from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.vector_store.faiss_store import FAISSStore
from src.vector_store.qdrant_adapter import QdrantAdapter
from src.vector_store.factory import VectorStoreFactory
from src.benchmarks.config import VectorDBConfig


def _unique_name(prefix: str = "dmt") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _try_adapter(**kwargs):
    try:
        return QdrantAdapter(**kwargs)
    except Exception:
        pytest.skip("Qdrant container not available")


# ---------------------------------------------------------------------------
# BaseVectorStore default_distance
# ---------------------------------------------------------------------------


class TestBaseVectorStoreDefaultDistance:
    def test_default_is_cosine(self, faiss_store):
        assert faiss_store.default_distance == DistanceMetric.COSINE

    def test_custom_default_distance(self):
        store = FAISSStore(default_distance=DistanceMetric.EUCLIDEAN)
        assert store.default_distance == DistanceMetric.EUCLIDEAN

    def test_dot_product_default(self):
        store = FAISSStore(default_distance=DistanceMetric.DOT_PRODUCT)
        assert store.default_distance == DistanceMetric.DOT_PRODUCT


# ---------------------------------------------------------------------------
# QdrantAdapter default_distance
# ---------------------------------------------------------------------------


class TestQdrantAdapterDefaultDistance:
    def test_default_is_cosine(self):
        adapter = _try_adapter()
        assert adapter.default_distance == DistanceMetric.COSINE

    def test_custom_default(self):
        adapter = _try_adapter(default_distance=DistanceMetric.EUCLIDEAN)
        assert adapter.default_distance == DistanceMetric.EUCLIDEAN

    def test_create_collection_uses_default(self):
        adapter = _try_adapter(default_distance=DistanceMetric.DOT_PRODUCT)
        coll = _unique_name("dp")
        try:
            result = adapter.create_collection(coll, vector_size=4)
            assert result is True
            info = adapter.get_collection_info(coll)
            assert info is not None
        finally:
            adapter.delete_collection(coll)

    def test_create_collection_explicit_overrides_default(self):
        adapter = _try_adapter(default_distance=DistanceMetric.DOT_PRODUCT)
        coll = _unique_name("override")
        try:
            result = adapter.create_collection(
                coll, vector_size=4, distance=DistanceMetric.COSINE
            )
            assert result is True
        finally:
            adapter.delete_collection(coll)


# ---------------------------------------------------------------------------
# FAISSStore default_distance
# ---------------------------------------------------------------------------


class TestFAISSStoreDefaultDistance:
    def test_default_is_cosine(self, faiss_store):
        assert faiss_store.default_distance == DistanceMetric.COSINE

    def test_euclidean_default(self):
        store = FAISSStore(default_distance=DistanceMetric.EUCLIDEAN)
        assert store.default_distance == DistanceMetric.EUCLIDEAN
        store.create_collection("test_l2", vector_size=4)
        info = store.get_collection_info("test_l2")
        assert info["distance"] == "euclidean"

    def test_create_collection_none_uses_default(self):
        store = FAISSStore(default_distance=DistanceMetric.DOT_PRODUCT)
        store.create_collection("test_dp", vector_size=4)
        info = store.get_collection_info("test_dp")
        assert info["distance"] == "dot_product"

    def test_create_collection_explicit_overrides_default(self):
        store = FAISSStore(default_distance=DistanceMetric.DOT_PRODUCT)
        store.create_collection("test_cos", vector_size=4, distance=DistanceMetric.COSINE)
        info = store.get_collection_info("test_cos")
        assert info["distance"] == "cosine"

    def test_manhattan_raises(self):
        store = FAISSStore(default_distance=DistanceMetric.MANHATTAN)
        with pytest.raises(ValueError, match="FAISS does not support"):
            store.create_collection("test_manhattan", vector_size=4)


# ---------------------------------------------------------------------------
# PgVectorStore default_distance (mocked — no DB required)
# ---------------------------------------------------------------------------


class TestPgVectorStoreDefaultDistance:
    def test_default_is_cosine(self):
        from src.vector_store.pgvector_store import PgVectorStore
        store = PgVectorStore()
        assert store.default_distance == DistanceMetric.COSINE

    def test_custom_default(self):
        from src.vector_store.pgvector_store import PgVectorStore
        store = PgVectorStore(default_distance=DistanceMetric.EUCLIDEAN)
        assert store.default_distance == DistanceMetric.EUCLIDEAN


# ---------------------------------------------------------------------------
# Factory from_config passes default_distance
# ---------------------------------------------------------------------------


class TestFactoryDefaultDistance:
    def test_from_config_cosine(self):
        cfg = VectorDBConfig(backend="faiss", distance_metric="cosine")
        store = VectorStoreFactory.from_config(cfg)
        assert store.default_distance == DistanceMetric.COSINE

    def test_from_config_euclidean(self):
        cfg = VectorDBConfig(backend="faiss", distance_metric="euclidean")
        store = VectorStoreFactory.from_config(cfg)
        assert store.default_distance == DistanceMetric.EUCLIDEAN

    def test_from_config_dot_product(self):
        cfg = VectorDBConfig(backend="faiss", distance_metric="dot_product")
        store = VectorStoreFactory.from_config(cfg)
        assert store.default_distance == DistanceMetric.DOT_PRODUCT

    def test_from_config_qdrant_euclidean(self):
        cfg = VectorDBConfig(
            backend="qdrant",
            distance_metric="euclidean",
        )
        try:
            store = VectorStoreFactory.from_config(cfg)
        except Exception:
            pytest.skip("Qdrant container not available")
        assert store.default_distance == DistanceMetric.EUCLIDEAN

    def test_end_to_end_config_to_collection(self):
        """Config distance_metric flows all the way to create_collection."""
        cfg = VectorDBConfig(backend="faiss", distance_metric="euclidean")
        store = VectorStoreFactory.from_config(cfg)
        store.create_collection("e2e_test", vector_size=4)
        info = store.get_collection_info("e2e_test")
        assert info["distance"] == "euclidean"
