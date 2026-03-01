"""Unit tests for RerankerFactory and reranker implementations."""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.base_rag import RetrievedChunk
from src.retrieval.reranker import (
    BaseReranker,
    BGEReranker,
    CohereReranker,
    CrossEncoderReranker,
    NoOpReranker,
)
from src.retrieval.reranker_factory import RerankerFactory


def _make_chunks(n: int = 5) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"c{i}",
            content=f"Content {i} about topic",
            score=float(n - i),
            metadata={"article_title": f"Article {i}"},
        )
        for i in range(n)
    ]


class TestNoOpReranker:
    def test_returns_first_top_k(self):
        chunks = _make_chunks(10)
        r = NoOpReranker()
        result = r.rerank("query", chunks, top_k=3)
        assert result == chunks[:3]

    def test_top_k_larger_than_chunks_returns_all(self):
        chunks = _make_chunks(3)
        r = NoOpReranker()
        result = r.rerank("query", chunks, top_k=10)
        assert result == chunks

    def test_empty_chunks(self):
        r = NoOpReranker()
        assert r.rerank("query", [], top_k=5) == []

    def test_is_base_reranker_subclass(self):
        assert isinstance(NoOpReranker(), BaseReranker)


class TestRerankerFactory:
    def test_create_none_returns_noop(self):
        r = RerankerFactory.create("none")
        assert isinstance(r, NoOpReranker)

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker type"):
            RerankerFactory.create("nonexistent_type")

    def test_available_types_includes_all(self):
        types = RerankerFactory.available_types()
        assert "none" in types
        assert "cohere" in types
        assert "bge" in types
        assert "cross_encoder" in types

    def test_available_types_sorted(self):
        types = RerankerFactory.available_types()
        assert types == sorted(types)

    def test_from_config_none(self):
        cfg = MagicMock()
        cfg.type = "none"
        cfg.model_name = None
        r = RerankerFactory.from_config(cfg)
        assert isinstance(r, NoOpReranker)

    def test_from_config_bge_forwards_model_name(self):
        cfg = MagicMock()
        cfg.type = "bge"
        cfg.model_name = "BAAI/bge-reranker-v2-m3"
        # BGEReranker lazy-loads CrossEncoder inside _get_model(); just check attrs
        r = RerankerFactory.from_config(cfg)
        assert isinstance(r, BGEReranker)
        assert r.model_name == "BAAI/bge-reranker-v2-m3"
        assert r._model is None  # not loaded until rerank() is called

    def test_from_config_cross_encoder_forwards_model_name(self):
        cfg = MagicMock()
        cfg.type = "cross_encoder"
        cfg.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # CrossEncoderReranker lazy-loads CrossEncoder inside _get_model()
        r = RerankerFactory.from_config(cfg)
        assert isinstance(r, CrossEncoderReranker)
        assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert r._model is None  # not loaded until rerank() is called

    def test_from_config_no_model_name_for_none(self):
        cfg = MagicMock()
        cfg.type = "none"
        cfg.model_name = None
        r = RerankerFactory.from_config(cfg)
        assert isinstance(r, NoOpReranker)


class TestBGERerankerRerank:
    def test_rerank_returns_top_k(self):
        chunks = _make_chunks(5)
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.2, 0.7, 0.1, 0.5]

        # Bypass lazy import by pre-setting _model
        r = BGEReranker(model_name="BAAI/bge-reranker-v2-m3")
        r._model = mock_model
        result = r.rerank("query", chunks, top_k=3)

        assert len(result) == 3
        # Highest score (0.9) should be first
        assert result[0].score == pytest.approx(0.9)

    def test_rerank_empty_returns_empty(self):
        r = BGEReranker(model_name="BAAI/bge-reranker-v2-m3")
        assert r.rerank("query", [], top_k=5) == []


class TestCohereRerankerRerank:
    def test_rerank_calls_cohere_client(self):
        chunks = _make_chunks(3)

        mock_cohere = MagicMock()
        mock_result = MagicMock()
        mock_result.index = 2
        mock_result.relevance_score = 0.95
        mock_cohere.rerank.return_value.results = [mock_result]

        r = CohereReranker(model_name="rerank-english-v3.0")
        r._client = mock_cohere

        result = r.rerank("query", chunks, top_k=1)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.95)
        assert result[0].content == chunks[2].content
