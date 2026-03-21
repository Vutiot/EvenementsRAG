"""Tests for document-level and chunk-level precision metrics."""

import pytest

from src.evaluation.metrics import (
    RetrievalMetrics,
    aggregate_metrics,
    compute_retrieval_metrics,
    doc_mrr_score,
    doc_precision_at_k,
    doc_recall_at_k,
    precision_at_k,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _payloads(*pageids):
    """Create payload dicts with the given pageids."""
    return [{"pageid": pid, "article_title": f"Art_{pid}"} for pid in pageids]


# ---------------------------------------------------------------------------
# doc_precision_at_k
# ---------------------------------------------------------------------------


class TestDocPrecisionAtK:
    def test_all_from_source(self):
        payloads = _payloads(100, 100, 100)
        assert doc_precision_at_k(payloads, 100, k=3) == 1.0

    def test_half_from_source(self):
        payloads = _payloads(100, 200, 100, 200)
        assert doc_precision_at_k(payloads, 100, k=4) == 0.5

    def test_none_from_source(self):
        payloads = _payloads(200, 300, 400)
        assert doc_precision_at_k(payloads, 100, k=3) == 0.0

    def test_k_zero(self):
        payloads = _payloads(100)
        assert doc_precision_at_k(payloads, 100, k=0) == 0.0

    def test_k_larger_than_results(self):
        payloads = _payloads(100, 200)
        # 1 match out of k=5 (only 2 payloads, so count=1)
        assert doc_precision_at_k(payloads, 100, k=5) == pytest.approx(1 / 5)

    def test_int_vs_str_pageid(self):
        payloads = [{"pageid": 100}]
        assert doc_precision_at_k(payloads, "100", k=1) == 1.0

    def test_no_source_article_id(self):
        payloads = _payloads(100)
        assert doc_precision_at_k(payloads, None, k=1) == 0.0

    def test_empty_payloads(self):
        assert doc_precision_at_k([], 100, k=3) == 0.0


# ---------------------------------------------------------------------------
# doc_recall_at_k
# ---------------------------------------------------------------------------


class TestDocRecallAtK:
    def test_found_in_top_k(self):
        payloads = _payloads(200, 100, 300)
        assert doc_recall_at_k(payloads, 100, k=3) == 1.0

    def test_not_found_in_top_k(self):
        payloads = _payloads(200, 300, 100)
        assert doc_recall_at_k(payloads, 100, k=2) == 0.0

    def test_found_at_boundary(self):
        payloads = _payloads(200, 300, 100)
        assert doc_recall_at_k(payloads, 100, k=3) == 1.0

    def test_empty_payloads(self):
        assert doc_recall_at_k([], 100, k=3) == 0.0

    def test_no_source_article_id(self):
        payloads = _payloads(100)
        assert doc_recall_at_k(payloads, None, k=1) == 0.0


# ---------------------------------------------------------------------------
# doc_mrr_score
# ---------------------------------------------------------------------------


class TestDocMRR:
    def test_first_position(self):
        payloads = _payloads(100, 200, 300)
        assert doc_mrr_score(payloads, 100) == 1.0

    def test_third_position(self):
        payloads = _payloads(200, 300, 100)
        assert doc_mrr_score(payloads, 100) == pytest.approx(1 / 3)

    def test_not_found(self):
        payloads = _payloads(200, 300, 400)
        assert doc_mrr_score(payloads, 100) == 0.0

    def test_empty_payloads(self):
        assert doc_mrr_score([], 100) == 0.0

    def test_no_source_article_id(self):
        payloads = _payloads(100)
        assert doc_mrr_score(payloads, None) == 0.0


# ---------------------------------------------------------------------------
# compute_retrieval_metrics — doc & chunk precision fields
# ---------------------------------------------------------------------------


class TestComputeRetrievalMetricsDocFields:
    def test_doc_fields_populated_with_source(self):
        payloads = _payloads(100, 200, 100, 300, 100)
        chunks = ["c1", "c2", "c3", "c4", "c5"]
        m = compute_retrieval_metrics(
            retrieved_chunks=chunks,
            ground_truth_chunks=["c1"],
            k_values=[1, 3, 5, 10],
            retrieved_payloads=payloads,
            source_article_id=100,
            source_chunk_id="c1",
        )
        # doc_precision: 1/1, 2/3, 3/5, 3/5 (only 5 payloads for k=10)
        assert m.doc_precision_at_1 == pytest.approx(1.0)
        assert m.doc_precision_at_3 == pytest.approx(2 / 3)
        assert m.doc_precision_at_5 == pytest.approx(3 / 5)
        # doc_recall: all 1.0 since article 100 is at position 1
        assert m.doc_recall_at_1 == 1.0
        assert m.doc_recall_at_5 == 1.0
        # doc_mrr
        assert m.doc_mrr == 1.0

    def test_doc_fields_zero_without_source(self):
        m = compute_retrieval_metrics(
            retrieved_chunks=["c1", "c2"],
            ground_truth_chunks=["c1"],
            k_values=[1, 3, 5, 10],
        )
        assert m.doc_precision_at_1 == 0.0
        assert m.doc_precision_at_5 == 0.0
        assert m.doc_recall_at_1 == 0.0
        assert m.doc_mrr == 0.0

    def test_chunk_precision_populated(self):
        m = compute_retrieval_metrics(
            retrieved_chunks=["c1", "c2", "c3", "c4", "c5"],
            ground_truth_chunks=["c1", "c3"],
            k_values=[1, 3, 5, 10],
        )
        # chunk_precision: 1/1=1.0, 2/3≈0.667, 2/5=0.4, 2/5=0.4
        assert m.chunk_precision_at_1 == 1.0
        assert m.chunk_precision_at_3 == pytest.approx(2 / 3)
        assert m.chunk_precision_at_5 == pytest.approx(2 / 5)

    def test_chunk_precision_zero_without_ground_truth(self):
        m = compute_retrieval_metrics(
            retrieved_chunks=["c1", "c2"],
            ground_truth_chunks=[],
            k_values=[1, 5],
        )
        assert m.chunk_precision_at_1 == 0.0
        assert m.chunk_precision_at_5 == 0.0


# ---------------------------------------------------------------------------
# RetrievalMetrics.to_dict — doc fields present
# ---------------------------------------------------------------------------


class TestRetrievalMetricsToDictDocFields:
    def test_new_fields_in_to_dict(self):
        m = RetrievalMetrics(
            doc_precision_at_5=0.6,
            doc_recall_at_5=1.0,
            doc_mrr=0.5,
            chunk_precision_at_5=0.4,
        )
        d = m.to_dict()
        assert d["doc_precision_at_5"] == 0.6
        assert d["doc_recall_at_5"] == 1.0
        assert d["doc_mrr"] == 0.5
        assert d["chunk_precision_at_5"] == 0.4
        # All k variants exist
        for k in (1, 3, 5, 10):
            assert f"doc_precision_at_{k}" in d
            assert f"doc_recall_at_{k}" in d
            assert f"chunk_precision_at_{k}" in d


# ---------------------------------------------------------------------------
# aggregate_metrics — doc fields
# ---------------------------------------------------------------------------


class TestAggregateMetricsDoc:
    def test_doc_averages_computed(self):
        m1 = RetrievalMetrics(
            doc_precision_at_5=0.6, doc_recall_at_5=1.0, doc_mrr=1.0,
            chunk_precision_at_5=0.4,
        )
        m2 = RetrievalMetrics(
            doc_precision_at_5=0.2, doc_recall_at_5=0.0, doc_mrr=0.0,
            chunk_precision_at_5=0.6,
        )
        agg = aggregate_metrics([m1, m2])
        assert agg["avg_doc_precision_at_5"] == pytest.approx(0.4)
        assert agg["avg_doc_recall_at_5"] == pytest.approx(0.5)
        assert agg["avg_doc_mrr"] == pytest.approx(0.5)
        assert agg["avg_chunk_precision_at_5"] == pytest.approx(0.5)
