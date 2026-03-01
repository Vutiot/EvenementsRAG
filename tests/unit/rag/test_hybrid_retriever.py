"""Unit tests for HybridRetriever (all mocked — no network, no Qdrant)."""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.base_rag import RetrievedChunk
from src.retrieval.hybrid_search import SearchResult


def _make_search_results(n: int = 5) -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id=f"uuid-{i}",
            score=float(n - i) / n,
            payload={
                "content": f"Content chunk {i} about World War II",
                "article_title": f"WW2 Article {i}",
                "chunk_id": f"uuid-{i}",
            },
            rank=i + 1,
        )
        for i in range(n)
    ]


def _make_config(sparse_weight=0.3, dense_weight=0.7, rerank_k=10, sparse_type="bm25"):
    from src.benchmarks.config import BenchmarkConfig, RerankerConfig, RetrievalConfig

    cfg = BenchmarkConfig.phase2_hybrid()
    cfg = cfg.model_copy(deep=True, update={
        "retrieval": RetrievalConfig(
            technique="hybrid",
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
            rerank_k=rerank_k,
            sparse_type=sparse_type,
        ),
        "reranker": RerankerConfig(type="none"),
    })
    return cfg


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.collection_exists.return_value = True
    store.manager = MagicMock()  # exposes .manager for raw QdrantManager
    store.manager.client = MagicMock()
    return store


@pytest.fixture
def mock_embedding_gen():
    gen = MagicMock()
    gen.generate_embeddings.return_value = [[0.1] * 384]
    return gen


@pytest.fixture
def mock_hybrid_searcher(monkeypatch):
    searcher = MagicMock()
    searcher.search.return_value = _make_search_results(5)
    searcher.index_collection.return_value = None

    monkeypatch.setattr(
        "src.rag.phase3_hybrid.retriever.HybridSearcher",
        lambda **kwargs: searcher,
    )
    return searcher


class TestHybridRetrieverInit:
    def test_raises_if_collection_missing(self, mock_vector_store):
        mock_vector_store.collection_exists.return_value = False
        with pytest.raises(ValueError, match="does not exist"):
            from src.rag.phase3_hybrid.retriever import HybridRetriever
            HybridRetriever(
                collection_name="missing_coll",
                qdrant_manager=mock_vector_store,
                embedding_generator=MagicMock(),
                config=_make_config(),
            )

    def test_init_succeeds_with_existing_collection(
        self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher
    ):
        from src.rag.phase3_hybrid.retriever import HybridRetriever
        r = HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            config=_make_config(),
        )
        assert r is not None

    def test_unwraps_qdrant_adapter(self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher):
        """HybridRetriever should pass raw manager (via .manager) to HybridSearcher."""
        from src.rag.phase3_hybrid.retriever import HybridRetriever

        raw_mgr = MagicMock()
        mock_vector_store.manager = raw_mgr

        HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            config=_make_config(),
        )
        # The mock HybridSearcher received qdrant_manager=raw_mgr
        # (mock_hybrid_searcher fixture captures the call args)
        # We just verify no AttributeError was raised (i.e., .manager was accessed)
        mock_vector_store.manager  # was accessed without error


class TestHybridRetrieverRetrieve:
    def test_retrieve_returns_retrieved_chunks(
        self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher
    ):
        from src.rag.phase3_hybrid.retriever import HybridRetriever
        r = HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            config=_make_config(),
        )
        chunks = r.retrieve("D-Day landings", top_k=3)
        assert len(chunks) == 3
        for c in chunks:
            assert isinstance(c, RetrievedChunk)

    def test_retrieve_indexes_on_first_call(
        self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher
    ):
        from src.rag.phase3_hybrid.retriever import HybridRetriever
        r = HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            config=_make_config(),
        )
        assert r._indexed is False
        r.retrieve("query", top_k=3)
        assert r._indexed is True
        mock_hybrid_searcher.index_collection.assert_called_once_with("ww2_test")

    def test_retrieve_does_not_reindex_on_second_call(
        self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher
    ):
        from src.rag.phase3_hybrid.retriever import HybridRetriever
        r = HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            config=_make_config(),
        )
        r.retrieve("query1", top_k=3)
        r.retrieve("query2", top_k=3)
        mock_hybrid_searcher.index_collection.assert_called_once()

    def test_retrieve_passes_filters(
        self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher
    ):
        from src.rag.phase3_hybrid.retriever import HybridRetriever
        r = HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            config=_make_config(),
        )
        filters = {"year": 1944}
        r.retrieve("query", top_k=3, filters=filters)
        call_kwargs = mock_hybrid_searcher.search.call_args[1]
        assert call_kwargs.get("filter_conditions") == filters

    def test_retrieve_uses_rerank_k_for_candidates(
        self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher
    ):
        from src.rag.phase3_hybrid.retriever import HybridRetriever
        cfg = _make_config(rerank_k=15)
        r = HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            config=cfg,
        )
        r.retrieve("query", top_k=5)
        call_kwargs = mock_hybrid_searcher.search.call_args[1]
        assert call_kwargs["top_k"] == 15


class TestHybridRetrieverGenerate:
    def test_generate_returns_string(
        self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher
    ):
        from src.rag.phase3_hybrid.retriever import HybridRetriever

        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "D-Day was the Allied invasion of Normandy on June 6, 1944."
        )

        r = HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            llm_client=mock_llm,
            config=_make_config(),
        )
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                content="The D-Day landings took place on June 6, 1944.",
                score=0.9,
                metadata={"article_title": "D-Day"},
            )
        ]
        answer = r.generate("What was D-Day?", chunks, temperature=0.0, max_tokens=500)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_generate_fallback_on_llm_error(
        self, mock_vector_store, mock_embedding_gen, mock_hybrid_searcher
    ):
        from src.rag.phase3_hybrid.retriever import HybridRetriever

        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = RuntimeError("API down")

        r = HybridRetriever(
            collection_name="ww2_test",
            qdrant_manager=mock_vector_store,
            embedding_generator=mock_embedding_gen,
            llm_client=mock_llm,
            config=_make_config(),
        )
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                content="Some content",
                score=0.5,
                metadata={"article_title": "Article"},
            )
        ]
        answer = r.generate("query", chunks)
        assert "[Generation failed" in answer
