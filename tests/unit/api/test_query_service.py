"""Unit tests for QueryService (E3-F1-T2).

All heavy deps (ML models, Qdrant, LLM) are mocked — no real inference.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.api.query_service import (
    CollectionNotIndexedError,
    QueryService,
)
from src.benchmarks.config import BenchmarkConfig
from src.rag.base_rag import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_chunks(n=2):
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content=f"text {i}",
            score=0.9 - i * 0.1,
            metadata={"article_title": f"Art {i}", "source_url": "", "chunk_index": i},
        )
        for i in range(n)
    ]


def _make_mock_pipeline():
    pipeline = MagicMock()
    pipeline.retrieve.return_value = _fake_chunks(2)
    pipeline.generate.return_value = "Mock answer"
    return pipeline


@pytest.fixture()
def vanilla_config():
    return BenchmarkConfig.phase1_vanilla()


@pytest.fixture()
def hybrid_config():
    return BenchmarkConfig.phase2_hybrid()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecuteQuery:
    """execute_query returns correct dict structure."""

    @patch("src.api.query_service.QueryService._build_pipeline")
    def test_returns_correct_structure(self, mock_build, vanilla_config):
        mock_build.return_value = _make_mock_pipeline()
        svc = QueryService()

        result = svc.execute_query("What was D-Day?", vanilla_config)

        assert set(result.keys()) == {
            "chunks", "answer", "retrieval_time_ms",
            "generation_time_ms", "config_hash",
        }
        assert len(result["chunks"]) == 2
        assert result["answer"] == "Mock answer"
        assert result["retrieval_time_ms"] >= 0
        assert result["generation_time_ms"] >= 0
        assert result["config_hash"] == vanilla_config.config_hash()

    @patch("src.api.query_service.QueryService._build_pipeline")
    def test_generation_disabled_returns_placeholder(self, mock_build, vanilla_config):
        cfg = vanilla_config.model_copy(
            deep=True,
            update={"generation": vanilla_config.generation.model_copy(update={"enabled": False})},
        )
        pipeline = _make_mock_pipeline()
        mock_build.return_value = pipeline
        svc = QueryService()

        result = svc.execute_query("Test", cfg)

        assert result["answer"] == "[Generation disabled]"
        assert result["generation_time_ms"] == 0.0
        pipeline.generate.assert_not_called()


class TestPipelineCaching:
    """Pipeline cached on second call; FIFO eviction at max capacity."""

    @patch("src.api.query_service.QueryService._build_pipeline")
    def test_pipeline_cached_on_second_call(self, mock_build, vanilla_config):
        mock_build.return_value = _make_mock_pipeline()
        svc = QueryService()

        svc.execute_query("q1", vanilla_config)
        svc.execute_query("q2", vanilla_config)

        # _build_pipeline called only once for the same config
        assert mock_build.call_count == 1

    @patch("src.api.query_service.QueryService._build_pipeline")
    def test_fifo_eviction_at_max_capacity(self, mock_build):
        mock_build.return_value = _make_mock_pipeline()
        svc = QueryService(max_cached=2)

        configs = []
        for i in range(3):
            cfg = BenchmarkConfig.phase1_vanilla().model_copy(
                update={"name": f"run_{i}", "description": f"desc_{i}"},
            )
            # Ensure distinct hashes by varying a functional param
            cfg = cfg.model_copy(
                deep=True,
                update={"chunking": cfg.chunking.model_copy(update={"chunk_size": 256 + i * 256})},
            )
            configs.append(cfg)

        svc.execute_query("q", configs[0])
        svc.execute_query("q", configs[1])
        svc.execute_query("q", configs[2])

        # First config should have been evicted (max_cached=2)
        assert configs[0].config_hash() not in svc._cache
        assert configs[1].config_hash() in svc._cache
        assert configs[2].config_hash() in svc._cache

    @patch("src.api.query_service.QueryService._build_pipeline")
    def test_clear_cache_empties(self, mock_build, vanilla_config):
        mock_build.return_value = _make_mock_pipeline()
        svc = QueryService()

        svc.execute_query("q", vanilla_config)
        assert len(svc._cache) == 1

        svc.clear_cache()
        assert len(svc._cache) == 0


class TestBuildPipelineErrors:
    """_build_pipeline raises on missing collection or unknown technique."""

    @patch("src.api.query_service.importlib.import_module")
    @patch("src.embeddings.embedding_generator.EmbeddingGenerator", autospec=True)
    @patch("src.vector_store.factory.VectorStoreFactory.from_config")
    def test_collection_not_indexed_error(
        self, mock_from_config, mock_emb_cls, mock_import, vanilla_config
    ):
        mock_store = MagicMock()
        mock_store.collection_exists.return_value = False
        mock_from_config.return_value = mock_store

        svc = QueryService()
        with pytest.raises(CollectionNotIndexedError, match="does not exist"):
            svc.execute_query("q", vanilla_config)

    def test_unknown_technique_raises_value_error(self):
        cfg = BenchmarkConfig.phase1_vanilla()

        svc = QueryService()

        # Temporarily patch the registry to empty
        with patch.dict("src.api.query_service._RAG_REGISTRY", {}, clear=True):
            with pytest.raises(ValueError, match="Unknown RAG technique"):
                svc._build_pipeline(cfg)


class TestHybridPipeline:
    """Hybrid pipeline passes config= kwarg."""

    @patch("src.api.query_service.importlib.import_module")
    @patch("src.embeddings.embedding_generator.EmbeddingGenerator", autospec=True)
    @patch("src.vector_store.factory.VectorStoreFactory.from_config")
    def test_hybrid_passes_config_kwarg(
        self, mock_from_config, mock_emb_cls, mock_import, hybrid_config
    ):
        mock_store = MagicMock()
        mock_store.collection_exists.return_value = True
        mock_from_config.return_value = mock_store

        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.HybridRetriever = mock_cls
        mock_import.return_value = mock_module

        svc = QueryService()
        svc._build_pipeline(hybrid_config)

        # The cls() call should include config=
        _, kwargs = mock_cls.call_args
        assert "config" in kwargs
        assert kwargs["config"] is hybrid_config

    @patch("src.api.query_service.importlib.import_module")
    @patch("src.embeddings.embedding_generator.EmbeddingGenerator", autospec=True)
    @patch("src.vector_store.factory.VectorStoreFactory.from_config")
    def test_vanilla_does_not_pass_config_kwarg(
        self, mock_from_config, mock_emb_cls, mock_import, vanilla_config
    ):
        mock_store = MagicMock()
        mock_store.collection_exists.return_value = True
        mock_from_config.return_value = mock_store

        mock_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.VanillaRetriever = mock_cls
        mock_import.return_value = mock_module

        svc = QueryService()
        svc._build_pipeline(vanilla_config)

        _, kwargs = mock_cls.call_args
        assert "config" not in kwargs
