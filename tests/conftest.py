"""Shared pytest fixtures for the EvenementsRAG test suite."""

import json

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.benchmarks.config import BenchmarkConfig, GenerationConfig
from src.benchmarks.runner import BenchmarkResult
from src.evaluation.metrics import EvaluationResults, RetrievalMetrics
from src.vector_store.base import DistanceMetric
from src.vector_store.qdrant_adapter import QdrantAdapter
from src.vector_store.qdrant_manager import QdrantManager


@pytest.fixture
def vanilla_config() -> BenchmarkConfig:
    return BenchmarkConfig.phase1_vanilla()


@pytest.fixture
def hybrid_config() -> BenchmarkConfig:
    return BenchmarkConfig.phase2_hybrid()


@pytest.fixture
def vanilla_config_no_gen(vanilla_config) -> BenchmarkConfig:
    """Vanilla config with generation disabled — avoids requiring a real questions file."""
    return vanilla_config.model_copy(
        update={"generation": GenerationConfig(enabled=False)}
    )


@pytest.fixture
def fake_evaluation_results() -> EvaluationResults:
    return EvaluationResults(
        avg_recall_at_k={1: 0.5, 3: 0.6, 5: 0.7, 10: 0.8},
        avg_mrr=0.65,
        avg_ndcg={5: 0.70, 10: 0.75},
        avg_article_hit_at_k={1: 0.6, 3: 0.7, 5: 0.8, 10: 0.9},
        avg_chunk_hit_at_k={1: 0.5, 3: 0.65, 5: 0.72, 10: 0.85},
        metrics_by_type={},
        per_question_metrics=[
            {
                "question_id": "q1",
                "question": "What was D-Day?",
                "type": "factual",
                "difficulty": "medium",
                "source_article": "Normandy",
                "metrics": RetrievalMetrics(),
                "retrieval_time_ms": 12.5,
                "ground_truth_count": 1,
                "retrieved_count": 10,
            },
        ],
        total_questions=1,
        questions_with_recall_at_5_gt_50=1,
        avg_retrieval_time_ms=12.5,
    )


@pytest.fixture
def fake_benchmark_result(vanilla_config, fake_evaluation_results) -> BenchmarkResult:
    return BenchmarkResult(
        config=vanilla_config,
        config_hash=vanilla_config.config_hash(),
        phase_name=vanilla_config.name,
        timestamp="2026-02-28T12:00:00Z",
        evaluation=fake_evaluation_results,
        per_question_full=[
            {
                "question_id": "q1",
                "question": "What was D-Day?",
                "type": "factual",
                "difficulty": "medium",
            },
        ],
        total_wall_time_s=1.5,
    )


@pytest.fixture
def tmp_questions_file(tmp_path):
    """A minimal questions JSON file for generation-pass tests."""
    data = {
        "questions": [
            {
                "id": "q1",
                "question": "What was D-Day?",
                "type": "factual",
                "difficulty": "medium",
            },
        ]
    }
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Vector store fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qdrant_adapter():
    """QdrantAdapter backed by Qdrant container. Skips if container unavailable."""
    try:
        adapter = QdrantAdapter()
    except Exception:
        pytest.skip("Qdrant container not available")
    return adapter


# Keep old name as alias for backward compat during transition
qdrant_memory_adapter = qdrant_adapter


@pytest.fixture
def faiss_store():
    """In-memory FAISSStore (no persistence)."""
    from src.vector_store.faiss_store import FAISSStore
    return FAISSStore()


@pytest.fixture
def faiss_persisted_store(tmp_path):
    """FAISSStore with tmp_path persistence."""
    from src.vector_store.faiss_store import FAISSStore
    return FAISSStore(persist_dir=str(tmp_path))
