"""Tests for src.evaluation.metrics — rank-based retrieval metrics."""

import pytest

from src.evaluation.metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    article_hit_at_k,
    chunk_hit_at_k,
    compute_retrieval_metrics,
    find_article_rank,
    find_chunk_rank,
)


# ---------------------------------------------------------------------------
# find_chunk_rank
# ---------------------------------------------------------------------------


class TestFindChunkRank:
    def test_found_first(self):
        assert find_chunk_rank(["a", "b", "c"], "a") == 1

    def test_found_middle(self):
        assert find_chunk_rank(["a", "b", "c"], "b") == 2

    def test_found_last(self):
        assert find_chunk_rank(["a", "b", "c"], "c") == 3

    def test_not_found(self):
        assert find_chunk_rank(["a", "b", "c"], "d") is None

    def test_empty_list(self):
        assert find_chunk_rank([], "a") is None

    def test_empty_source(self):
        assert find_chunk_rank(["a", "b"], "") is None

    def test_none_source(self):
        assert find_chunk_rank(["a", "b"], None) is None


# ---------------------------------------------------------------------------
# find_article_rank
# ---------------------------------------------------------------------------


class TestFindArticleRank:
    def test_found_by_pageid(self):
        payloads = [
            {"pageid": "100", "article_title": "Art A"},
            {"pageid": "200", "article_title": "Art B"},
        ]
        assert find_article_rank(["c1", "c2"], payloads, "200") == 2

    def test_found_by_title(self):
        payloads = [
            {"pageid": "100", "article_title": "Art A"},
            {"pageid": "200", "article_title": "Art B"},
        ]
        assert find_article_rank(["c1", "c2"], payloads, "Art B") == 2

    def test_pageid_int_vs_str(self):
        """pageid stored as int should still match str source_article_id."""
        payloads = [{"pageid": 42, "article_title": "X"}]
        assert find_article_rank(["c1"], payloads, "42") == 1

    def test_not_found(self):
        payloads = [{"pageid": "100", "article_title": "A"}]
        assert find_article_rank(["c1"], payloads, "999") is None

    def test_empty_payloads(self):
        assert find_article_rank([], [], "100") is None

    def test_empty_source(self):
        payloads = [{"pageid": "100"}]
        assert find_article_rank(["c1"], payloads, "") is None

    def test_none_source(self):
        payloads = [{"pageid": "100"}]
        assert find_article_rank(["c1"], payloads, None) is None


# ---------------------------------------------------------------------------
# RetrievalMetrics — rank fields & derived properties
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    def test_defaults(self):
        m = RetrievalMetrics()
        assert m.ground_truth_chunk_rank is None
        assert m.ground_truth_article_rank is None

    def test_chunk_hit_properties(self):
        m = RetrievalMetrics(ground_truth_chunk_rank=3)
        assert m.chunk_hit_at_1 == 0.0
        assert m.chunk_hit_at_3 == 1.0
        assert m.chunk_hit_at_5 == 1.0
        assert m.chunk_hit_at_10 == 1.0

    def test_article_hit_properties(self):
        m = RetrievalMetrics(ground_truth_article_rank=5)
        assert m.article_hit_at_1 == 0.0
        assert m.article_hit_at_3 == 0.0
        assert m.article_hit_at_5 == 1.0
        assert m.article_hit_at_10 == 1.0

    def test_hit_properties_none_rank(self):
        m = RetrievalMetrics()
        assert m.chunk_hit_at_1 == 0.0
        assert m.chunk_hit_at_10 == 0.0
        assert m.article_hit_at_1 == 0.0
        assert m.article_hit_at_10 == 0.0

    def test_hit_properties_rank_1(self):
        m = RetrievalMetrics(ground_truth_chunk_rank=1, ground_truth_article_rank=1)
        assert m.chunk_hit_at_1 == 1.0
        assert m.article_hit_at_1 == 1.0

    def test_to_dict_includes_rank_and_hits(self):
        m = RetrievalMetrics(ground_truth_chunk_rank=2, ground_truth_article_rank=4)
        d = m.to_dict()

        # Rank fields present
        assert d["ground_truth_chunk_rank"] == 2
        assert d["ground_truth_article_rank"] == 4

        # Derived hit fields present
        assert d["chunk_hit_at_1"] == 0.0
        assert d["chunk_hit_at_3"] == 1.0
        assert d["chunk_hit_at_5"] == 1.0
        assert d["chunk_hit_at_10"] == 1.0

        assert d["article_hit_at_1"] == 0.0
        assert d["article_hit_at_3"] == 0.0
        assert d["article_hit_at_5"] == 1.0
        assert d["article_hit_at_10"] == 1.0

    def test_to_dict_none_ranks(self):
        d = RetrievalMetrics().to_dict()
        assert d["ground_truth_chunk_rank"] is None
        assert d["ground_truth_article_rank"] is None
        for k in [1, 3, 5, 10]:
            assert d[f"chunk_hit_at_{k}"] == 0.0
            assert d[f"article_hit_at_{k}"] == 0.0


# ---------------------------------------------------------------------------
# chunk_hit_at_k / article_hit_at_k (standalone functions)
# ---------------------------------------------------------------------------


class TestChunkHitAtKFunction:
    def test_found_within_k(self):
        assert chunk_hit_at_k(["a", "b", "c"], "b", k=3) == 1.0

    def test_found_beyond_k(self):
        assert chunk_hit_at_k(["a", "b", "c"], "c", k=2) == 0.0

    def test_not_found(self):
        assert chunk_hit_at_k(["a", "b", "c"], "z", k=10) == 0.0

    def test_empty_source(self):
        assert chunk_hit_at_k(["a"], "", k=5) == 0.0


class TestArticleHitAtKFunction:
    def test_found_within_k(self):
        payloads = [{"pageid": "1"}, {"pageid": "2"}]
        assert article_hit_at_k(["c1", "c2"], payloads, "2", k=3) == 1.0

    def test_found_beyond_k(self):
        payloads = [{"pageid": "1"}, {"pageid": "2"}, {"pageid": "3"}]
        assert article_hit_at_k(["c1", "c2", "c3"], payloads, "3", k=2) == 0.0

    def test_not_found(self):
        payloads = [{"pageid": "1"}]
        assert article_hit_at_k(["c1"], payloads, "99", k=5) == 0.0


# ---------------------------------------------------------------------------
# compute_retrieval_metrics — rank fields
# ---------------------------------------------------------------------------


class TestComputeRetrievalMetricsRank:
    def test_chunk_rank_set(self):
        m = compute_retrieval_metrics(
            retrieved_chunks=["a", "b", "c"],
            ground_truth_chunks=["c"],
            k_values=[1, 3, 5, 10],
            source_chunk_id="b",
        )
        assert m.ground_truth_chunk_rank == 2

    def test_chunk_rank_not_found(self):
        m = compute_retrieval_metrics(
            retrieved_chunks=["a", "b"],
            ground_truth_chunks=["a"],
            k_values=[1, 3, 5, 10],
            source_chunk_id="z",
        )
        assert m.ground_truth_chunk_rank is None

    def test_article_rank_set(self):
        payloads = [
            {"pageid": "10", "article_title": "X"},
            {"pageid": "20", "article_title": "Y"},
        ]
        m = compute_retrieval_metrics(
            retrieved_chunks=["c1", "c2"],
            ground_truth_chunks=["c1"],
            k_values=[1, 3, 5, 10],
            retrieved_payloads=payloads,
            source_article_id="20",
        )
        assert m.ground_truth_article_rank == 2

    def test_article_rank_not_found(self):
        payloads = [{"pageid": "10"}]
        m = compute_retrieval_metrics(
            retrieved_chunks=["c1"],
            ground_truth_chunks=["c1"],
            k_values=[1, 3, 5, 10],
            retrieved_payloads=payloads,
            source_article_id="99",
        )
        assert m.ground_truth_article_rank is None

    def test_no_source_ids_gives_none(self):
        m = compute_retrieval_metrics(
            retrieved_chunks=["a", "b"],
            ground_truth_chunks=["a"],
            k_values=[1, 3, 5, 10],
        )
        assert m.ground_truth_chunk_rank is None
        assert m.ground_truth_article_rank is None

    def test_derived_hit_consistent_with_rank(self):
        """chunk_hit_at_K via property matches what the old per-K loop produced."""
        m = compute_retrieval_metrics(
            retrieved_chunks=["a", "b", "c", "d", "e"],
            ground_truth_chunks=["c"],
            k_values=[1, 3, 5, 10],
            source_chunk_id="c",
        )
        assert m.ground_truth_chunk_rank == 3
        assert m.chunk_hit_at_1 == 0.0
        assert m.chunk_hit_at_3 == 1.0
        assert m.chunk_hit_at_5 == 1.0
        assert m.chunk_hit_at_10 == 1.0


# ---------------------------------------------------------------------------
# aggregate_metrics — avg rank stats
# ---------------------------------------------------------------------------


class TestAggregateMetricsRank:
    def test_avg_ranks_included(self):
        metrics = [
            RetrievalMetrics(ground_truth_chunk_rank=2, ground_truth_article_rank=1),
            RetrievalMetrics(ground_truth_chunk_rank=4, ground_truth_article_rank=3),
        ]
        agg = aggregate_metrics(metrics)
        assert agg["avg_ground_truth_chunk_rank"] == 3.0
        assert agg["avg_ground_truth_article_rank"] == 2.0

    def test_avg_ranks_only_over_found(self):
        """None ranks are excluded from the average."""
        metrics = [
            RetrievalMetrics(ground_truth_chunk_rank=2),
            RetrievalMetrics(ground_truth_chunk_rank=None),
        ]
        agg = aggregate_metrics(metrics)
        assert agg["avg_ground_truth_chunk_rank"] == 2.0
        assert "avg_ground_truth_article_rank" not in agg

    def test_no_ranks_at_all(self):
        metrics = [RetrievalMetrics(), RetrievalMetrics()]
        agg = aggregate_metrics(metrics)
        assert "avg_ground_truth_chunk_rank" not in agg
        assert "avg_ground_truth_article_rank" not in agg

    def test_empty_metrics(self):
        assert aggregate_metrics([]) == {}
