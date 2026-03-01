"""Tests for QdrantAdapter — full lifecycle with in-memory QdrantManager."""

from uuid import uuid5, NAMESPACE_DNS

import numpy as np
import pytest

from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.vector_store.qdrant_adapter import QdrantAdapter
from src.vector_store.qdrant_manager import QdrantManager


def _make_uuid(i: int) -> str:
    """Deterministic UUID from an integer (Qdrant requires valid UUIDs)."""
    return str(uuid5(NAMESPACE_DNS, f"test-vector-{i}"))


@pytest.fixture
def adapter():
    mgr = QdrantManager(use_memory=True)
    return QdrantAdapter(qdrant_manager=mgr)


@pytest.fixture
def populated_adapter(adapter):
    """Adapter with a 'test' collection containing 10 vectors."""
    adapter.create_collection("test", vector_size=4, distance=DistanceMetric.COSINE)
    vectors = np.random.default_rng(42).random((10, 4)).tolist()
    payloads = [
        {"title": f"doc_{i}", "year": 1940 + i, "type": "article"}
        for i in range(10)
    ]
    ids = [_make_uuid(i) for i in range(10)]
    adapter.upsert_vectors("test", vectors, payloads, ids=ids)
    return adapter


@pytest.fixture
def populated_ids():
    """Return the same IDs used in populated_adapter."""
    return [_make_uuid(i) for i in range(10)]


class TestAdapterIsBaseVectorStore:
    def test_isinstance(self, adapter):
        assert isinstance(adapter, BaseVectorStore)


class TestCollectionManagement:
    def test_create_and_exists(self, adapter):
        assert adapter.create_collection("c1", vector_size=4)
        assert adapter.collection_exists("c1")

    def test_create_idempotent(self, adapter):
        adapter.create_collection("c1", vector_size=4)
        assert adapter.create_collection("c1", vector_size=4)

    def test_create_with_recreate(self, adapter):
        adapter.create_collection("c1", vector_size=4)
        assert adapter.create_collection("c1", vector_size=4, recreate=True)

    def test_delete_collection(self, adapter):
        adapter.create_collection("c1", vector_size=4)
        assert adapter.delete_collection("c1")
        assert not adapter.collection_exists("c1")

    def test_unsupported_distance_raises(self, adapter):
        with pytest.raises(ValueError, match="does not support"):
            adapter.create_collection("c1", vector_size=4, distance=DistanceMetric.MANHATTAN)


class TestUpsertAndSearch:
    def test_upsert_count(self, populated_adapter):
        assert populated_adapter.count_vectors("test") == 10

    def test_search_returns_results(self, populated_adapter):
        query = np.random.default_rng(99).random(4).tolist()
        results = populated_adapter.search("test", query, limit=3)
        assert len(results) == 3
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "payload" in r

    def test_search_with_filter(self, populated_adapter):
        query = np.random.default_rng(99).random(4).tolist()
        results = populated_adapter.search(
            "test", query, limit=10, filter_conditions={"year": {"gte": 1945}}
        )
        for r in results:
            assert r["payload"]["year"] >= 1945


class TestScroll:
    def test_scroll_returns_all_records(self, populated_adapter):
        records, next_offset = populated_adapter.scroll("test", limit=100)
        assert len(records) == 10
        for r in records:
            assert "id" in r
            assert "payload" in r

    def test_scroll_pagination(self, populated_adapter):
        all_records = []
        offset = None
        while True:
            records, offset = populated_adapter.scroll("test", limit=3, offset=offset)
            all_records.extend(records)
            if offset is None:
                break
        assert len(all_records) == 10

    def test_scroll_with_vectors(self, populated_adapter):
        records, _ = populated_adapter.scroll("test", limit=2, with_vectors=True)
        assert len(records) == 2
        for r in records:
            assert "vector" in r


class TestCollectionInfo:
    def test_get_collection_info(self, populated_adapter):
        info = populated_adapter.get_collection_info("test")
        assert info["name"] == "test"
        assert info["vector_size"] == 4
        assert info["points_count"] == 10

    def test_count_with_filter(self, populated_adapter):
        count = populated_adapter.count_vectors("test", {"year": {"gte": 1945}})
        assert 0 < count < 10


class TestDeleteVectors:
    def test_delete_by_ids(self, populated_adapter, populated_ids):
        assert populated_adapter.delete_vectors("test", ids=populated_ids[:2])
        assert populated_adapter.count_vectors("test") == 8


class TestStatistics:
    def test_get_statistics(self, populated_adapter):
        stats = populated_adapter.get_statistics()
        assert "total_collections" in stats
        assert stats["total_collections"] >= 1


class TestManagerAccess:
    def test_manager_property(self, adapter):
        assert isinstance(adapter.manager, QdrantManager)

    def test_client_property(self, adapter):
        assert adapter.client is adapter.manager.client

    def test_create_without_manager(self):
        adapter = QdrantAdapter(use_memory=True)
        assert isinstance(adapter.manager, QdrantManager)
