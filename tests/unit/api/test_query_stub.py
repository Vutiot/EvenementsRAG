"""Tests for POST /api/query stub endpoint."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestQueryStub:
    def test_returns_mock_result(self):
        response = client.post(
            "/api/query",
            json={"query": "What happened on D-Day?", "preset": "phase1_vanilla.yaml"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What happened on D-Day?"
        assert isinstance(data["generated_answer"], str)
        assert len(data["generated_answer"]) > 0

    def test_returns_chunks(self):
        response = client.post(
            "/api/query",
            json={"query": "Test query", "preset": "phase1_vanilla.yaml"},
        )
        data = response.json()
        chunks = data["retrieved_chunks"]
        assert isinstance(chunks, list)
        assert len(chunks) == 5
        # Verify chunk structure
        chunk = chunks[0]
        assert "chunk_id" in chunk
        assert "content" in chunk
        assert "score" in chunk
        assert "article_title" in chunk
        assert "source_url" in chunk
        assert "chunk_index" in chunk

    def test_scores_are_descending(self):
        response = client.post(
            "/api/query",
            json={"query": "Test", "preset": "phase1_vanilla.yaml"},
        )
        chunks = response.json()["retrieved_chunks"]
        scores = [c["score"] for c in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_returns_latency_metrics(self):
        response = client.post(
            "/api/query",
            json={"query": "Test", "preset": "phase1_vanilla.yaml"},
        )
        data = response.json()
        assert "retrieval_time_ms" in data
        assert "generation_time_ms" in data
        assert data["retrieval_time_ms"] > 0
        assert data["generation_time_ms"] > 0

    def test_returns_config_hash(self):
        response = client.post(
            "/api/query",
            json={"query": "Test", "preset": "phase1_vanilla.yaml"},
        )
        data = response.json()
        assert "config_hash" in data
        assert len(data["config_hash"]) == 16

    def test_invalid_preset_returns_404(self):
        response = client.post(
            "/api/query",
            json={"query": "Test", "preset": "nonexistent.yaml"},
        )
        assert response.status_code == 404

    def test_missing_query_field(self):
        response = client.post(
            "/api/query",
            json={"preset": "phase1_vanilla.yaml"},
        )
        assert response.status_code == 422

    def test_missing_preset_field(self):
        response = client.post(
            "/api/query",
            json={"query": "Test"},
        )
        assert response.status_code == 422
