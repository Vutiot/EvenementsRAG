"""BenchmarkRunner uses the technique's retrieval when a retrieve_fn is given."""

from unittest.mock import MagicMock

from src.evaluation.benchmark_runner import BenchmarkRunner
from src.rag.base_rag import RetrievedChunk


def test_query_for_question_uses_retrieve_fn_point_ids():
    chunks = [
        RetrievedChunk(chunk_id="orig_1", content="a", score=1.0, metadata={"x": 1}, point_id="uuid-1"),
        RetrievedChunk(chunk_id="uuid-2", content="b", score=0.5, metadata={}),
    ]
    fn = MagicMock(return_value=chunks)
    qdrant = MagicMock()
    r = BenchmarkRunner(qdrant_manager=qdrant, embedding_generator=MagicMock(), retrieve_fn=fn)
    out = r.query_for_question({"id": "q1", "question": "Who?"}, "coll", top_k=50)
    fn.assert_called_once_with("Who?", 50)
    qdrant.search.assert_not_called()
    assert out["retrieved_chunks"] == ["uuid-1", "uuid-2"]
    assert out["retrieved_payloads"] == [{"x": 1}, {}]
