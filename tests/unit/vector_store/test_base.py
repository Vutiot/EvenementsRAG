"""Tests for src/vector_store/base.py — ABC constraints and DistanceMetric enum."""

import pytest

from src.vector_store.base import BaseVectorStore, DistanceMetric


class TestDistanceMetric:
    def test_cosine_value(self):
        assert DistanceMetric.COSINE.value == "cosine"

    def test_euclidean_value(self):
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"

    def test_dot_product_value(self):
        assert DistanceMetric.DOT_PRODUCT.value == "dot_product"

    def test_manhattan_value(self):
        assert DistanceMetric.MANHATTAN.value == "manhattan"

    def test_from_string(self):
        assert DistanceMetric("cosine") is DistanceMetric.COSINE


class TestBaseVectorStoreABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_incomplete_subclass_raises_type_error(self):
        class Incomplete(BaseVectorStore):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_can_instantiate(self):
        class Complete(BaseVectorStore):
            def create_collection(self, *a, **kw):
                return True

            def collection_exists(self, *a, **kw):
                return False

            def delete_collection(self, *a, **kw):
                return True

            def upsert_vectors(self, *a, **kw):
                return 0

            def search(self, *a, **kw):
                return []

            def scroll(self, *a, **kw):
                return [], None

            def get_collection_info(self, *a, **kw):
                return {}

            def count_vectors(self, *a, **kw):
                return 0

            def delete_vectors(self, *a, **kw):
                return True

        store = Complete()
        assert isinstance(store, BaseVectorStore)

    def test_get_statistics_default_returns_empty_dict(self):
        class Minimal(BaseVectorStore):
            def create_collection(self, *a, **kw):
                return True

            def collection_exists(self, *a, **kw):
                return False

            def delete_collection(self, *a, **kw):
                return True

            def upsert_vectors(self, *a, **kw):
                return 0

            def search(self, *a, **kw):
                return []

            def scroll(self, *a, **kw):
                return [], None

            def get_collection_info(self, *a, **kw):
                return {}

            def count_vectors(self, *a, **kw):
                return 0

            def delete_vectors(self, *a, **kw):
                return True

        store = Minimal()
        assert store.get_statistics() == {}
