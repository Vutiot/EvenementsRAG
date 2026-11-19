"""LazyGraphRAG: concept graph and retriever (E5-F1-T1)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.benchmarks.config import BenchmarkConfig, RetrievalConfig
from src.graph_store.concept_graph import (
    ConceptGraph,
    graph_path,
    heuristic_concepts,
    load_or_build,
)

CHUNKS = {
    "p1": ("Battle of Midway", "In June 1942 Admiral Chester Nimitz ambushed the Japanese carriers near Midway Atoll."),
    "p2": ("Chester Nimitz", "Chester Nimitz commanded the Pacific Fleet. He planned the Battle of Midway with Raymond Spruance."),
    "p3": ("Raymond Spruance", "Raymond Spruance led Task Force 16 and later commanded at the Battle of the Philippine Sea."),
    "p4": ("Battle of Stalingrad", "The Red Army encircled the German Sixth Army under Friedrich Paulus at Stalingrad."),
    "p5": ("Friedrich Paulus", "Friedrich Paulus surrendered at Stalingrad in February 1943."),
}


def _records():
    return [
        {"id": pid, "payload": {"content": text, "article_title": title, "chunk_id": f"orig_{pid}"}}
        for pid, (title, text) in CHUNKS.items()
    ]


@pytest.fixture
def graph():
    return ConceptGraph.build(_records(), min_max_df=50)


def test_heuristic_concepts():
    cs = heuristic_concepts("The Red Army encircled the German Sixth Army under Friedrich Paulus.")
    assert "red army" in cs and "friedrich paulus" in cs
    assert "the" not in cs


def test_build_graph(graph):
    assert graph.n_points == 5
    assert set(graph.concept_chunks["chester nimitz"]) == {"p1", "p2"}
    assert "battle of midway" in graph.chunk_concepts["p1"]  # article title as concept
    # nimitz and spruance co-occur in p2
    assert graph.cooccurrence["chester nimitz"]["raymond spruance"] >= 1


def test_match_query_prefers_longest_phrase(graph):
    assert graph.match_query("Who planned the Battle of Midway with Nimitz?") == ["battle of midway"]
    assert "chester nimitz" in graph.match_query("What did Chester Nimitz command?")


def test_score_chunks_expands_through_neighbours(graph):
    direct = graph.score_chunks({"chester nimitz": 1.0}, hops=0)
    assert set(direct) == {"p1", "p2"}
    expanded = graph.score_chunks({"chester nimitz": 1.0}, hops=1)
    assert "p3" in expanded  # reached through raymond spruance
    assert expanded["p2"] > expanded["p3"]
    assert "p4" not in expanded  # unrelated front


def test_over_frequent_concepts_dropped():
    recs = [{"id": f"x{i}", "payload": {"content": "Winston Churchill spoke.", "article_title": ""}} for i in range(10)]
    g = ConceptGraph.build(recs, max_df_ratio=0.5, min_max_df=1)
    assert "winston churchill" not in g.concept_chunks


def test_load_or_build_caches_and_rebuilds_on_size_change(tmp_path):
    store = MagicMock()
    store.scroll.return_value = (_records(), None)
    store.count_vectors.return_value = 5
    g1 = load_or_build(store, "coll", graph_dir=tmp_path)
    assert graph_path("coll", "heuristic", tmp_path).exists()
    g2 = load_or_build(store, "coll", graph_dir=tmp_path)
    assert store.scroll.call_count == 1 and g2.concept_chunks == g1.concept_chunks
    store.count_vectors.return_value = 6
    load_or_build(store, "coll", graph_dir=tmp_path)
    assert store.scroll.call_count == 2


def _retriever(graph, dense_ids, graph_weight=0.5, hops=1):
    from src.rag.phase4_graph.retriever import LazyGraphRetriever

    store = MagicMock()
    store.collection_exists.return_value = True
    store.get_collection_info.return_value = {"points_count": 5}
    store.search.return_value = [
        {"id": pid, "score": 0.9 - i * 0.1, "payload": graph.payloads[pid]}
        for i, pid in enumerate(dense_ids)
    ]
    emb = MagicMock()
    emb.generate_embedding.return_value = [0.0]
    cfg = BenchmarkConfig(
        retrieval=RetrievalConfig(technique="lazygraph", rerank_k=10, graph_weight=graph_weight, graph_hops=hops)
    )
    return LazyGraphRetriever(
        collection_name="coll",
        qdrant_manager=store,
        embedding_generator=emb,
        llm_client=MagicMock(),
        config=cfg,
        graph=graph,
    )


def test_retriever_adds_graph_only_chunks(graph):
    # Dense only finds the Stalingrad chunk; the query names Nimitz
    r = _retriever(graph, ["p4"])
    out = r.retrieve("What fleet did Chester Nimitz command?", top_k=5)
    ids = [c.point_id for c in out]
    assert {"p1", "p2", "p4"} <= set(ids)
    hit = next(c for c in out if c.point_id == "p2")
    assert hit.chunk_id == "orig_p2" and "Pacific Fleet" in hit.content


def test_graph_weight_zero_is_dense_order(graph):
    r = _retriever(graph, ["p4", "p5"], graph_weight=0.0)
    out = r.retrieve("What fleet did Chester Nimitz command?", top_k=2)
    assert [c.point_id for c in out] == ["p4", "p5"]


def test_registered_and_eval_uses_pipeline(monkeypatch):
    from src.benchmarks.runner import _RAG_REGISTRY, ParameterizedBenchmarkRunner

    assert _RAG_REGISTRY["lazygraph"].endswith("LazyGraphRetriever")
    cfg = BenchmarkConfig(retrieval=RetrievalConfig(technique="lazygraph"))
    runner = ParameterizedBenchmarkRunner(config=cfg, qdrant_manager=MagicMock(), embedding_generator=MagicMock())
    runner._rag_pipeline = MagicMock()
    runner._rag_pipeline.retrieve.return_value = [SimpleNamespace(point_id="p1")]
    fn = runner._eval_retrieve_fn()
    assert fn("q", 7)[0].point_id == "p1"
    runner._rag_pipeline.retrieve.assert_called_once_with("q", top_k=7)

    vanilla = ParameterizedBenchmarkRunner(config=BenchmarkConfig(), qdrant_manager=MagicMock())
    assert vanilla._eval_retrieve_fn() is None
