"""HybridSearcher indexing and fusion (sparse side reads the indexed text)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.retrieval.hybrid_search import HybridSearcher

DOCS = {
    "a": "The Battle of Midway was a naval battle in the Pacific",
    "b": "Stalingrad was besieged by the German Sixth Army",
    "c": "The Yalta Conference took place in February 1945",
}


@pytest.fixture(params=["bm25", "tfidf"])
def searcher(request):
    qdrant = MagicMock()
    points = [SimpleNamespace(id=k, payload={"content": v, "article_title": k}) for k, v in DOCS.items()]
    qdrant.client.scroll.return_value = (points, None)
    # Dense side returns only "c" (deliberately wrong) so the sparse side must find "a"
    qdrant.search.return_value = [{"id": "c", "score": 0.9, "payload": {"content": DOCS["c"]}}]
    emb = MagicMock()
    emb.generate_embeddings.return_value = [[0.0, 1.0]]
    s = HybridSearcher(qdrant, emb, bm25_weight=0.5, sparse_type=request.param)
    s.index_collection("coll")
    return s, qdrant


def test_sparse_side_indexes_content_and_brings_its_payload(searcher):
    s, qdrant = searcher
    results = s.search("naval battle Midway", "coll", top_k=3)
    ids = [r.chunk_id for r in results]
    assert "a" in ids
    hit = next(r for r in results if r.chunk_id == "a")
    assert hit.payload["content"] == DOCS["a"]
    qdrant.client.retrieve.assert_not_called()


def test_zero_score_documents_do_not_enter_the_fusion(searcher):
    s, _ = searcher
    results = s.search("zzzz unrelated", "coll", top_k=3)
    assert [r.chunk_id for r in results] == ["c"]
