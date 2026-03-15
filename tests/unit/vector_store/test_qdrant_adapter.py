"""Tests for QdrantAdapter — full lifecycle against Qdrant container.

All tests skip automatically if the Qdrant container is not running.
Each test uses a unique collection name to avoid cross-test contamination.
"""

import uuid

from uuid import uuid5, NAMESPACE_DNS

import numpy as np
import pytest

from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.vector_store.qdrant_adapter import QdrantAdapter
from src.vector_store.qdrant_manager import QdrantManager


def _make_uuid(i: int) -> str:
    """Deterministic UUID from an integer (Qdrant requires valid UUIDs)."""
    return str(uuid5(NAMESPACE_DNS, f"test-vector-{i}"))


def _unique_name(prefix: str = "test") -> str:
    """Generate a unique collection name for test isolation."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _try_adapter():
    """Create a QdrantAdapter, skipping if container is unavailable."""
    try:
        return QdrantAdapter()
    except Exception:
        pytest.skip("Qdrant container not available")


@pytest.fixture
def adapter(request):
    adapter = _try_adapter()
    created_collections = []
    adapter._test_collections = created_collections
    yield adapter
    # Teardown: delete all collections created during the test
    for name in created_collections:
        try:
            adapter.delete_collection(name)
        except Exception:
            pass


@pytest.fixture
def populated_adapter(adapter):
    """Adapter with a unique 'test_*' collection containing 10 vectors."""
    coll = _unique_name("pop")
    adapter._test_collections.append(coll)
    adapter.create_collection(coll, vector_size=4, distance=DistanceMetric.COSINE)
    vectors = np.random.default_rng(42).random((10, 4)).tolist()
    payloads = [
        {"title": f"doc_{i}", "year": 1940 + i, "type": "article"}
        for i in range(10)
    ]
    ids = [_make_uuid(i) for i in range(10)]
    adapter.upsert_vectors(coll, vectors, payloads, ids=ids)
    adapter._test_coll = coll
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
        coll = _unique_name("c1")
        adapter._test_collections.append(coll)
        assert adapter.create_collection(coll, vector_size=4)
        assert adapter.collection_exists(coll)

    def test_create_idempotent(self, adapter):
        coll = _unique_name("c2")
        adapter._test_collections.append(coll)
        adapter.create_collection(coll, vector_size=4)
        assert adapter.create_collection(coll, vector_size=4)

    def test_create_with_recreate(self, adapter):
        coll = _unique_name("c3")
        adapter._test_collections.append(coll)
        adapter.create_collection(coll, vector_size=4)
        assert adapter.create_collection(coll, vector_size=4, recreate=True)

    def test_delete_collection(self, adapter):
        coll = _unique_name("c4")
        adapter.create_collection(coll, vector_size=4)
        assert adapter.delete_collection(coll)
        assert not adapter.collection_exists(coll)

    def test_unsupported_distance_raises(self, adapter):
        with pytest.raises(ValueError, match="does not support"):
            adapter.create_collection(
                _unique_name("c5"), vector_size=4, distance=DistanceMetric.MANHATTAN
            )


class TestUpsertAndSearch:
    def test_upsert_count(self, populated_adapter):
        assert populated_adapter.count_vectors(populated_adapter._test_coll) == 10

    def test_search_returns_results(self, populated_adapter):
        query = np.random.default_rng(99).random(4).tolist()
        results = populated_adapter.search(populated_adapter._test_coll, query, limit=3)
        assert len(results) == 3
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "payload" in r

    def test_search_with_filter(self, populated_adapter):
        query = np.random.default_rng(99).random(4).tolist()
        results = populated_adapter.search(
            populated_adapter._test_coll, query, limit=10,
            filter_conditions={"year": {"gte": 1945}},
        )
        for r in results:
            assert r["payload"]["year"] >= 1945


class TestScroll:
    def test_scroll_returns_all_records(self, populated_adapter):
        records, next_offset = populated_adapter.scroll(
            populated_adapter._test_coll, limit=100
        )
        assert len(records) == 10
        for r in records:
            assert "id" in r
            assert "payload" in r

    def test_scroll_pagination(self, populated_adapter):
        all_records = []
        offset = None
        while True:
            records, offset = populated_adapter.scroll(
                populated_adapter._test_coll, limit=3, offset=offset
            )
            all_records.extend(records)
            if offset is None:
                break
        assert len(all_records) == 10

    def test_scroll_with_vectors(self, populated_adapter):
        records, _ = populated_adapter.scroll(
            populated_adapter._test_coll, limit=2, with_vectors=True
        )
        assert len(records) == 2
        for r in records:
            assert "vector" in r


class TestCollectionInfo:
    def test_get_collection_info(self, populated_adapter):
        info = populated_adapter.get_collection_info(populated_adapter._test_coll)
        assert info["name"] == populated_adapter._test_coll
        assert info["vector_size"] == 4
        assert info["points_count"] == 10

    def test_count_with_filter(self, populated_adapter):
        count = populated_adapter.count_vectors(
            populated_adapter._test_coll, {"year": {"gte": 1945}}
        )
        assert 0 < count < 10


class TestDeleteVectors:
    def test_delete_by_ids(self, populated_adapter, populated_ids):
        assert populated_adapter.delete_vectors(
            populated_adapter._test_coll, ids=populated_ids[:2]
        )
        assert populated_adapter.count_vectors(populated_adapter._test_coll) == 8


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
        try:
            adapter = QdrantAdapter()
        except Exception:
            pytest.skip("Qdrant container not available")
        assert isinstance(adapter.manager, QdrantManager)
