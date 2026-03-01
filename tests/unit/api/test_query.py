"""Tests for POST /api/query endpoint (E3-F1-T2)."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.rag.base_rag import RetrievedChunk as RAGChunk


def _make_rag_chunks(n=3):
    """Create *n* fake RAGChunk objects."""
    return [
        RAGChunk(
            chunk_id=f"chunk_{i:03d}",
            content=f"Content for chunk {i}.",
            score=round(0.95 - i * 0.1, 2),
            metadata={
                "article_title": f"Article {i}",
                "source_url": f"https://en.wikipedia.org/wiki/Article_{i}",
                "chunk_index": i,
            },
        )
        for i in range(n)
    ]


def _mock_execute(query, config):
    """Default mock return value for QueryService.execute_query."""
    return {
        "chunks": _make_rag_chunks(3),
        "answer": "This is a generated answer.",
        "retrieval_time_ms": 42.5,
        "generation_time_ms": 1250.0,
        "config_hash": config.config_hash(),
    }


# Patch at module level so every test gets a mocked QueryService
@pytest.fixture(autouse=True)
def _patch_query_service():
    mock_svc = MagicMock()
    mock_svc.execute_query.side_effect = _mock_execute
    with patch("src.api.routers.query._query_service", mock_svc):
        yield mock_svc


@pytest.fixture()
def client():
    from src.api.main import app
    return TestClient(app)


class TestQueryEndpoint:
    def test_returns_200_with_expected_fields(self, client):
        resp = client.post(
            "/api/query",
            json={"query": "What happened on D-Day?", "preset": "phase1_vanilla.yaml"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What happened on D-Day?"
        assert isinstance(data["generated_answer"], str)
        assert len(data["generated_answer"]) > 0
        assert isinstance(data["retrieved_chunks"], list)
        assert isinstance(data["retrieval_time_ms"], float)
        assert isinstance(data["generation_time_ms"], float)
        assert isinstance(data["config_hash"], str)

    def test_chunk_structure_has_flat_fields(self, client):
        resp = client.post(
            "/api/query",
            json={"query": "Test", "preset": "phase1_vanilla.yaml"},
        )
        chunk = resp.json()["retrieved_chunks"][0]
        assert "chunk_id" in chunk
        assert "content" in chunk
        assert "score" in chunk
        assert "article_title" in chunk
        assert "source_url" in chunk
        assert "chunk_index" in chunk

    def test_latency_fields_present_and_numeric(self, client):
        resp = client.post(
            "/api/query",
            json={"query": "Test", "preset": "phase1_vanilla.yaml"},
        )
        data = resp.json()
        assert data["retrieval_time_ms"] > 0
        assert data["generation_time_ms"] > 0

    def test_config_hash_is_16_hex_chars(self, client):
        resp = client.post(
            "/api/query",
            json={"query": "Test", "preset": "phase1_vanilla.yaml"},
        )
        h = resp.json()["config_hash"]
        assert len(h) == 16
        int(h, 16)  # should not raise

    def test_invalid_preset_returns_404(self, client):
        resp = client.post(
            "/api/query",
            json={"query": "Test", "preset": "nonexistent.yaml"},
        )
        assert resp.status_code == 404

    def test_missing_query_field_returns_422(self, client):
        resp = client.post(
            "/api/query",
            json={"preset": "phase1_vanilla.yaml"},
        )
        assert resp.status_code == 422

    def test_missing_preset_field_returns_422(self, client):
        resp = client.post(
            "/api/query",
            json={"query": "Test"},
        )
        assert resp.status_code == 422

    def test_collection_not_indexed_returns_409(self, client, _patch_query_service):
        from src.api.query_service import CollectionNotIndexedError

        _patch_query_service.execute_query.side_effect = CollectionNotIndexedError(
            "Collection 'xyz' does not exist."
        )
        resp = client.post(
            "/api/query",
            json={"query": "Test", "preset": "phase1_vanilla.yaml"},
        )
        assert resp.status_code == 409
        assert "does not exist" in resp.json()["detail"]

    def test_generic_pipeline_error_returns_502(self, client, _patch_query_service):
        _patch_query_service.execute_query.side_effect = RuntimeError("boom")
        resp = client.post(
            "/api/query",
            json={"query": "Test", "preset": "phase1_vanilla.yaml"},
        )
        assert resp.status_code == 502
        assert "Pipeline error" in resp.json()["detail"]
