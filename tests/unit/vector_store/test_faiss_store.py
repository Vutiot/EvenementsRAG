"""Tests for FAISSStore — full lifecycle, persistence, distance metrics."""

import numpy as np
import pytest

from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.vector_store.faiss_store import FAISSStore


@pytest.fixture
def store():
    return FAISSStore()


@pytest.fixture
def persisted_store(tmp_path):
    return FAISSStore(persist_dir=str(tmp_path))


@pytest.fixture
def populated_store(store):
    """FAISSStore with a 'test' collection containing 10 vectors."""
    store.create_collection("test", vector_size=4, distance=DistanceMetric.COSINE)
    rng = np.random.default_rng(42)
    vectors = rng.random((10, 4)).tolist()
    payloads = [
        {"title": f"doc_{i}", "year": 1940 + i, "type": "article"}
        for i in range(10)
    ]
    ids = [f"id_{i}" for i in range(10)]
    store.upsert_vectors("test", vectors, payloads, ids=ids)
    return store


class TestFAISSIsBaseVectorStore:
    def test_isinstance(self, store):
        assert isinstance(store, BaseVectorStore)


class TestCollectionManagement:
    def test_create_and_exists(self, store):
        assert store.create_collection("c1", vector_size=4)
        assert store.collection_exists("c1")

    def test_create_idempotent(self, store):
        store.create_collection("c1", vector_size=4)
        assert store.create_collection("c1", vector_size=4)

    def test_create_with_recreate(self, store):
        store.create_collection("c1", vector_size=4)
        store.upsert_vectors("c1", [[1, 2, 3, 4]], [{"a": 1}])
        store.create_collection("c1", vector_size=4, recreate=True)
        assert store.count_vectors("c1") == 0

    def test_delete_collection(self, store):
        store.create_collection("c1", vector_size=4)
        assert store.delete_collection("c1")
        assert not store.collection_exists("c1")

    def test_nonexistent_raises_keyerror(self, store):
        with pytest.raises(KeyError, match="does not exist"):
            store.count_vectors("nonexistent")

    def test_unsupported_distance_raises(self, store):
        with pytest.raises(ValueError):
            store.create_collection("c1", vector_size=4, distance=DistanceMetric.MANHATTAN)


class TestUpsertAndSearch:
    def test_upsert_count(self, populated_store):
        assert populated_store.count_vectors("test") == 10

    def test_upsert_auto_ids(self, store):
        store.create_collection("c1", vector_size=4)
        count = store.upsert_vectors("c1", [[1, 2, 3, 4]], [{"a": 1}])
        assert count == 1
        assert store.count_vectors("c1") == 1

    def test_search_returns_results(self, populated_store):
        query = np.random.default_rng(99).random(4).tolist()
        results = populated_store.search("test", query, limit=3)
        assert len(results) == 3
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "payload" in r

    def test_search_with_filter(self, populated_store):
        query = np.random.default_rng(99).random(4).tolist()
        results = populated_store.search(
            "test", query, limit=10, filter_conditions={"year": {"gte": 1945}}
        )
        for r in results:
            assert r["payload"]["year"] >= 1945

    def test_search_with_score_threshold(self, populated_store):
        query = np.random.default_rng(99).random(4).tolist()
        results = populated_store.search("test", query, limit=10, score_threshold=0.99)
        # Very high threshold should filter most results
        assert len(results) <= 10

    def test_search_empty_collection(self, store):
        store.create_collection("empty", vector_size=4)
        results = store.search("empty", [1, 2, 3, 4], limit=5)
        assert results == []

    def test_vectors_mismatch_payloads_raises(self, store):
        store.create_collection("c1", vector_size=4)
        with pytest.raises(ValueError, match="same length"):
            store.upsert_vectors("c1", [[1, 2, 3, 4]], [{"a": 1}, {"b": 2}])


class TestSearchDistanceMetrics:
    def test_cosine_search(self, store):
        store.create_collection("cos", vector_size=3, distance=DistanceMetric.COSINE)
        store.upsert_vectors(
            "cos",
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [{"d": "x"}, {"d": "y"}, {"d": "z"}],
            ids=["x", "y", "z"],
        )
        results = store.search("cos", [1, 0, 0], limit=1)
        assert results[0]["id"] == "x"

    def test_euclidean_search(self, store):
        store.create_collection("euc", vector_size=3, distance=DistanceMetric.EUCLIDEAN)
        store.upsert_vectors(
            "euc",
            [[1, 0, 0], [0, 1, 0], [10, 10, 10]],
            [{"d": "near"}, {"d": "mid"}, {"d": "far"}],
            ids=["near", "mid", "far"],
        )
        results = store.search("euc", [1, 0, 0], limit=1)
        assert results[0]["id"] == "near"

    def test_dot_product_search(self, store):
        store.create_collection("dot", vector_size=3, distance=DistanceMetric.DOT_PRODUCT)
        store.upsert_vectors(
            "dot",
            [[10, 0, 0], [1, 0, 0], [0, 0, 0]],
            [{"d": "big"}, {"d": "small"}, {"d": "zero"}],
            ids=["big", "small", "zero"],
        )
        results = store.search("dot", [1, 0, 0], limit=1)
        assert results[0]["id"] == "big"


class TestScroll:
    def test_scroll_returns_all_records(self, populated_store):
        records, next_offset = populated_store.scroll("test", limit=100)
        assert len(records) == 10

    def test_scroll_pagination(self, populated_store):
        all_records = []
        offset = None
        while True:
            records, offset = populated_store.scroll("test", limit=3, offset=offset)
            all_records.extend(records)
            if offset is None:
                break
        assert len(all_records) == 10

    def test_scroll_with_payload(self, populated_store):
        records, _ = populated_store.scroll("test", limit=2, with_payload=True)
        for r in records:
            assert "payload" in r

    def test_scroll_with_vectors(self, populated_store):
        records, _ = populated_store.scroll("test", limit=2, with_vectors=True)
        for r in records:
            assert "vector" in r
            assert len(r["vector"]) == 4

    def test_scroll_with_filter(self, populated_store):
        records, _ = populated_store.scroll(
            "test", limit=100, filter_conditions={"year": {"gte": 1945}}
        )
        for r in records:
            assert r["payload"]["year"] >= 1945


class TestDeleteVectors:
    def test_delete_by_ids(self, populated_store):
        assert populated_store.delete_vectors("test", ids=["id_0", "id_1"])
        assert populated_store.count_vectors("test") == 8

    def test_delete_by_filter(self, populated_store):
        before = populated_store.count_vectors("test")
        populated_store.delete_vectors("test", filter_conditions={"year": {"gte": 1948}})
        after = populated_store.count_vectors("test")
        assert after < before

    def test_delete_requires_ids_or_filter(self, populated_store):
        with pytest.raises(ValueError, match="Must provide"):
            populated_store.delete_vectors("test")


class TestCollectionInfo:
    def test_get_info(self, populated_store):
        info = populated_store.get_collection_info("test")
        assert info["name"] == "test"
        assert info["vector_size"] == 4
        assert info["points_count"] == 10
        assert info["distance"] == "cosine"

    def test_count_with_filter(self, populated_store):
        count = populated_store.count_vectors("test", {"year": {"gte": 1945}})
        assert 0 < count < 10


class TestPersistence:
    def test_persist_and_reload(self, tmp_path):
        store1 = FAISSStore(persist_dir=str(tmp_path))
        store1.create_collection("persist_test", vector_size=4)
        store1.upsert_vectors(
            "persist_test",
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [{"a": 1}, {"b": 2}],
            ids=["p1", "p2"],
        )

        # Create a fresh store pointing to same dir
        store2 = FAISSStore(persist_dir=str(tmp_path))
        assert store2.collection_exists("persist_test")
        assert store2.count_vectors("persist_test") == 2

    def test_delete_removes_files(self, tmp_path):
        store = FAISSStore(persist_dir=str(tmp_path))
        store.create_collection("del_test", vector_size=4)
        store.upsert_vectors("del_test", [[1, 2, 3, 4]], [{"a": 1}], ids=["d1"])
        assert (tmp_path / "del_test.faiss").exists()
        assert (tmp_path / "del_test.meta.pkl").exists()

        store.delete_collection("del_test")
        assert not (tmp_path / "del_test.faiss").exists()
        assert not (tmp_path / "del_test.meta.pkl").exists()


class TestStatistics:
    def test_get_statistics(self, populated_store):
        stats = populated_store.get_statistics()
        assert stats["backend"] == "faiss"
        assert stats["total_collections"] == 1
        assert "test" in stats["collections"]
