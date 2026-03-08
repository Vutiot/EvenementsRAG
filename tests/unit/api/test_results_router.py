"""Tests for benchmark result file endpoints."""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Fixtures — mock result files in a temp directory
# ---------------------------------------------------------------------------

LEGACY_RESULT = {
    "avg_recall_at_k": {"1": 0.5, "3": 0.7, "5": 0.8, "10": 0.9},
    "avg_mrr": 0.65,
    "avg_ndcg": {"5": 0.6, "10": 0.7},
    "avg_article_hit_at_k": {"1": 0.9, "5": 1.0, "10": 1.0},
    "avg_chunk_hit_at_k": {"1": 0.5, "5": 0.8, "10": 0.9},
    "metrics_by_type": {
        "factual": {"recall_at_1": 0.6, "recall_at_5": 0.9, "mrr": 0.7, "ndcg_at_5": 0.65},
        "temporal": {"recall_at_1": 0.4, "recall_at_5": 0.7, "mrr": 0.5, "ndcg_at_5": 0.5},
    },
    "per_question_metrics": [
        {
            "question_id": "q001",
            "question": "Who started WW2?",
            "type": "factual",
            "difficulty": "easy",
            "source_article": "World War II",
            "ground_truth_count": 1,
            "retrieved_count": 10,
            "retrieval_time_ms": 15.5,
            "metrics": {"recall_at_1": 1.0, "recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_5": 1.0},
        },
        {
            "question_id": "q002",
            "question": "When did D-Day happen?",
            "type": "temporal",
            "difficulty": "medium",
            "source_article": "D-Day",
            "ground_truth_count": 2,
            "retrieved_count": 10,
            "retrieval_time_ms": 20.3,
            "metrics": {"recall_at_1": 0.0, "recall_at_5": 0.5, "mrr": 0.3, "ndcg_at_5": 0.4},
        },
    ],
    "total_questions": 2,
    "avg_retrieval_time_ms": 17.9,
}

BENCHMARK_RESULT = {
    "config": {
        "name": "test_run",
        "description": "test",
        "dataset": {"dataset_name": "wiki_10k", "collection_name": "test"},
    },
    "config_hash": "abc123",
    "phase_name": "phase1_vanilla",
    "timestamp": "2026-01-15T12:00:00Z",
    "evaluation": {
        "avg_recall_at_k": {"1": 0.6, "3": 0.8, "5": 0.85, "10": 0.95},
        "avg_mrr": 0.75,
        "avg_ndcg": {"5": 0.7, "10": 0.8},
        "avg_article_hit_at_k": {"1": 0.95, "5": 1.0, "10": 1.0},
        "avg_chunk_hit_at_k": {"1": 0.6, "5": 0.85, "10": 0.95},
        "metrics_by_type": {
            "factual": {"recall_at_1": 0.7, "mrr": 0.8},
        },
        "per_question_metrics": [],
        "total_questions": 1,
        "avg_retrieval_time_ms": 12.0,
    },
    "per_question_full": [
        {
            "question_id": "q001",
            "question": "Who started WW2?",
            "type": "factual",
            "metrics": {"recall_at_1": 1.0, "mrr": 1.0},
            "generated_answer": "Germany invaded Poland in September 1939.",
            "generation_time_ms": 450.0,
            "retrieved_contexts": ["Context chunk 1", "Context chunk 2"],
            "generation_metrics": {"rouge_l_f1": 0.45},
            "ragas_metrics": {"faithfulness": 0.8, "answer_relevancy": 0.9},
        },
    ],
    "total_wall_time_s": 30.5,
    "metrics_summary": {
        "latency": {"retrieval_p50_ms": 12.0, "generation_p50_ms": 450.0},
        "ragas": {"avg_faithfulness": 0.8, "avg_answer_relevancy": 0.9, "num_questions_scored": 1},
    },
}

SUBDIR_RESULT = {
    "config": {
        "name": "wiki_hybrid_w50",
        "description": "hybrid weight 50",
        "dataset": {"dataset_name": "wiki_10k", "collection_name": "ww2_hybrid_w50"},
        "retrieval": {"technique": "hybrid"},
    },
    "config_hash": "deadbeef99",
    "phase_name": "wiki_hybrid_w50",
    "timestamp": "2026-03-09T01:00:00Z",
    "evaluation": {
        "avg_recall_at_k": {"5": 0.9},
        "avg_mrr": 0.8,
        "avg_ndcg": {"5": 0.75},
        "metrics_by_type": {},
        "per_question_metrics": [],
        "total_questions": 1,
        "avg_retrieval_time_ms": 10.0,
    },
    "per_question_full": [
        {
            "question_id": "q010",
            "question": "What was the Atlantic Charter?",
            "type": "factual",
            "metrics": {"recall_at_1": 1.0, "mrr": 1.0},
        },
    ],
    "total_wall_time_s": 5.0,
}


@pytest.fixture()
def mock_results_dir(tmp_path):
    """Write mock result files and patch RESULTS_DIR."""
    # Flat (legacy) files
    (tmp_path / "legacy_30q.json").write_text(json.dumps(LEGACY_RESULT))
    (tmp_path / "new_benchmark.json").write_text(json.dumps(BENCHMARK_RESULT))
    (tmp_path / "not_json.txt").write_text("hello")
    (tmp_path / "bad.json").write_text("{invalid json")

    # Subdirectory result
    hybrid_dir = tmp_path / "hybrid"
    hybrid_dir.mkdir()
    (hybrid_dir / "wiki_hybrid_w50_deadbeef_20260309T010000Z.json").write_text(
        json.dumps(SUBDIR_RESULT)
    )

    # Clear cache between tests
    from src.api.routers.results import _file_info_cache
    _file_info_cache.clear()

    with patch("src.api.routers.results.RESULTS_DIR", tmp_path):
        yield tmp_path


# ---------------------------------------------------------------------------
# GET /api/results
# ---------------------------------------------------------------------------


class TestListResults:
    def test_returns_list(self, mock_results_dir):
        response = client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # legacy + new_benchmark (flat) + hybrid subdir file = 3
        assert len(data) == 3

    def test_legacy_format_detected(self, mock_results_dir):
        response = client.get("/api/results")
        data = response.json()
        legacy = next(r for r in data if r["filename"] == "legacy_30q.json")
        assert legacy["format"] == "legacy"
        assert legacy["total_questions"] == 2
        assert legacy["avg_mrr"] == 0.65
        assert legacy["avg_recall_at_5"] == 0.8

    def test_new_format_detected(self, mock_results_dir):
        response = client.get("/api/results")
        data = response.json()
        new = next(r for r in data if r["filename"] == "new_benchmark.json")
        assert new["format"] == "benchmark_result"
        assert new["phase_name"] == "phase1_vanilla"
        assert new["timestamp"] == "2026-01-15T12:00:00Z"
        assert new["avg_mrr"] == 0.75

    def test_subdirectory_files_found(self, mock_results_dir):
        response = client.get("/api/results")
        data = response.json()
        filenames = [r["filename"] for r in data]
        assert any("hybrid/" in f for f in filenames)

    def test_subdir_file_has_relative_path(self, mock_results_dir):
        response = client.get("/api/results")
        data = response.json()
        hybrid = next(r for r in data if "hybrid/" in r["filename"])
        assert hybrid["filename"] == "hybrid/wiki_hybrid_w50_deadbeef_20260309T010000Z.json"
        assert hybrid["phase_name"] == "wiki_hybrid_w50"
        assert hybrid["avg_mrr"] == 0.8

    def test_empty_dir(self, tmp_path):
        from src.api.routers.results import _file_info_cache
        _file_info_cache.clear()
        with patch("src.api.routers.results.RESULTS_DIR", tmp_path):
            response = client.get("/api/results")
            assert response.status_code == 200
            assert response.json() == []


# ---------------------------------------------------------------------------
# GET /api/results/{filename}
# ---------------------------------------------------------------------------


class TestGetResult:
    def test_load_legacy(self, mock_results_dir):
        response = client.get("/api/results/legacy_30q.json")
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "legacy"
        assert data["avg_mrr"] == 0.65
        assert len(data["per_question"]) == 2
        q1 = data["per_question"][0]
        assert q1["question_id"] == "q001"
        assert q1["generated_answer"] is None
        assert q1["retrieved_contexts"] is None

    def test_load_new_format(self, mock_results_dir):
        response = client.get("/api/results/new_benchmark.json")
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "benchmark_result"
        assert data["config"] is not None
        assert data["total_wall_time_s"] == 30.5
        q1 = data["per_question"][0]
        assert q1["generated_answer"] == "Germany invaded Poland in September 1939."
        assert len(q1["retrieved_contexts"]) == 2
        assert q1["ragas_metrics"]["faithfulness"] == 0.8

    def test_load_from_subdirectory(self, mock_results_dir):
        response = client.get(
            "/api/results/hybrid/wiki_hybrid_w50_deadbeef_20260309T010000Z.json"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["phase_name"] == "wiki_hybrid_w50"
        assert data["avg_mrr"] == 0.8

    def test_not_found(self, mock_results_dir):
        response = client.get("/api/results/nonexistent.json")
        assert response.status_code == 404

    def test_not_found_in_subdirectory(self, mock_results_dir):
        response = client.get("/api/results/vanilla/nonexistent.json")
        assert response.status_code == 404

    def test_non_json_rejected(self, mock_results_dir):
        response = client.get("/api/results/not_json.txt")
        assert response.status_code == 400

    def test_path_traversal_blocked(self, mock_results_dir):
        response = client.get("/api/results/..%2F..%2Fetc%2Fpasswd.json")
        assert response.status_code in (400, 404)

    def test_path_traversal_via_subdir_blocked(self, mock_results_dir):
        response = client.get("/api/results/hybrid/../../etc/passwd.json")
        assert response.status_code in (400, 404)

    def test_metrics_summary_present(self, mock_results_dir):
        response = client.get("/api/results/new_benchmark.json")
        data = response.json()
        assert data["metrics_summary"] is not None
        assert "latency" in data["metrics_summary"]
        assert "ragas" in data["metrics_summary"]

    def test_metrics_by_type_keys(self, mock_results_dir):
        response = client.get("/api/results/legacy_30q.json")
        data = response.json()
        assert "factual" in data["metrics_by_type"]
        assert "temporal" in data["metrics_by_type"]
