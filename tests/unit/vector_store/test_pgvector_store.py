"""Tests for PgVectorStore — marked @pytest.mark.pgvector (skip unless DB available)."""

import pytest

pytestmark = pytest.mark.pgvector


@pytest.fixture
def pg_store():
    """Attempt to create a PgVectorStore with default local connection.

    Skips the entire module if psycopg2 or PostgreSQL is unavailable.
    """
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        pytest.skip("psycopg2 not installed")

    from src.vector_store.pgvector_store import PgVectorStore

    try:
        store = PgVectorStore(connection_params={
            "dbname": "rag_test",
            "user": "postgres",
            "host": "localhost",
            "port": 5432,
        })
        # Test connection
        store._get_conn()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")

    yield store

    # Cleanup
    try:
        store.delete_collection("pgtest")
    except Exception:
        pass


class TestPgVectorCollections:
    def test_create_and_exists(self, pg_store):
        pg_store.create_collection("pgtest", vector_size=4)
        assert pg_store.collection_exists("pgtest")

    def test_delete_collection(self, pg_store):
        pg_store.create_collection("pgtest", vector_size=4)
        pg_store.delete_collection("pgtest")
        assert not pg_store.collection_exists("pgtest")


class TestPgVectorUpsertSearch:
    def test_upsert_and_search(self, pg_store):
        pg_store.create_collection("pgtest", vector_size=4, recreate=True)
        pg_store.upsert_vectors(
            "pgtest",
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            [{"title": "a"}, {"title": "b"}],
            ids=["a", "b"],
        )
        results = pg_store.search("pgtest", [1, 0, 0, 0], limit=1)
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_count(self, pg_store):
        pg_store.create_collection("pgtest", vector_size=4, recreate=True)
        pg_store.upsert_vectors(
            "pgtest",
            [[1, 0, 0, 0]],
            [{"title": "a"}],
            ids=["a"],
        )
        assert pg_store.count_vectors("pgtest") == 1


class TestPgVectorScroll:
    def test_scroll(self, pg_store):
        pg_store.create_collection("pgtest", vector_size=4, recreate=True)
        pg_store.upsert_vectors(
            "pgtest",
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            [{"t": "a"}, {"t": "b"}, {"t": "c"}],
            ids=["a", "b", "c"],
        )
        records, next_offset = pg_store.scroll("pgtest", limit=2)
        assert len(records) == 2
        assert next_offset is not None

        records2, next_offset2 = pg_store.scroll("pgtest", limit=2, offset=next_offset)
        assert len(records2) == 1
        assert next_offset2 is None
