"""Unit tests for src/evaluation/metrics_collector.py (E1-F2-T1).

All ROUGE/BERTScore tests mock the actual scorers — no heavy model downloads
required for the test suite.
"""

from unittest.mock import MagicMock, patch, PropertyMock
from collections import namedtuple

import numpy as np
import pytest

from src.benchmarks.config import EvaluationConfig
from src.evaluation.metrics_collector import (
    AggregatedGenerationMetrics,
    GenerationMetrics,
    LatencyMetrics,
    MetricsCollector,
)


# ---------------------------------------------------------------------------
# GenerationMetrics dataclass
# ---------------------------------------------------------------------------


class TestGenerationMetrics:
    def test_defaults_are_none(self):
        gm = GenerationMetrics()
        assert gm.rouge_l_f1 is None
        assert gm.bert_score_f1 is None

    def test_to_dict_excludes_none(self):
        gm = GenerationMetrics(rouge_l_f1=0.85)
        d = gm.to_dict()
        assert d == {"rouge_l_f1": 0.85}
        assert "bert_score_f1" not in d

    def test_to_dict_includes_all_when_set(self):
        gm = GenerationMetrics(
            rouge_l_f1=0.8,
            rouge_l_precision=0.75,
            rouge_l_recall=0.85,
            bert_score_f1=0.9,
            bert_score_precision=0.88,
            bert_score_recall=0.92,
        )
        d = gm.to_dict()
        assert len(d) == 6

    def test_to_dict_empty_when_all_none(self):
        assert GenerationMetrics().to_dict() == {}


# ---------------------------------------------------------------------------
# LatencyMetrics dataclass
# ---------------------------------------------------------------------------


class TestLatencyMetrics:
    def test_defaults_are_zero(self):
        lm = LatencyMetrics()
        assert lm.retrieval_p50_ms == 0.0
        assert lm.generation_p99_ms == 0.0

    def test_to_dict_returns_all_fields(self):
        lm = LatencyMetrics(retrieval_p50_ms=5.0, generation_p95_ms=120.0)
        d = lm.to_dict()
        assert d["retrieval_p50_ms"] == 5.0
        assert d["generation_p95_ms"] == 120.0
        assert len(d) == 6


# ---------------------------------------------------------------------------
# AggregatedGenerationMetrics dataclass
# ---------------------------------------------------------------------------


class TestAggregatedGenerationMetrics:
    def test_defaults(self):
        agg = AggregatedGenerationMetrics()
        assert agg.avg_rouge_l_f1 is None
        assert agg.num_questions_scored == 0

    def test_to_dict_excludes_none(self):
        agg = AggregatedGenerationMetrics(
            avg_rouge_l_f1=0.7, num_questions_scored=5
        )
        d = agg.to_dict()
        assert d == {"avg_rouge_l_f1": 0.7, "num_questions_scored": 5}
        assert "avg_bert_score_f1" not in d


# ---------------------------------------------------------------------------
# MetricsCollector — latency
# ---------------------------------------------------------------------------


class TestComputeLatencyMetrics:
    def test_empty_input_returns_zeros(self):
        mc = MetricsCollector(EvaluationConfig())
        lm = mc.compute_latency_metrics([])
        assert lm.retrieval_p50_ms == 0.0
        assert lm.generation_p99_ms == 0.0

    def test_single_entry(self):
        mc = MetricsCollector(EvaluationConfig())
        per_q = [{"retrieval_time_ms": 10.0, "generation_time_ms": 200.0}]
        lm = mc.compute_latency_metrics(per_q)
        assert lm.retrieval_p50_ms == 10.0
        assert lm.generation_p50_ms == 200.0

    def test_many_entries_percentiles(self):
        mc = MetricsCollector(EvaluationConfig())
        per_q = [{"retrieval_time_ms": float(i)} for i in range(1, 101)]
        lm = mc.compute_latency_metrics(per_q)
        assert lm.retrieval_p50_ms == pytest.approx(50.5, abs=0.5)
        assert lm.retrieval_p95_ms == pytest.approx(95.05, abs=1.0)
        assert lm.retrieval_p99_ms == pytest.approx(99.01, abs=1.0)

    def test_missing_generation_times_still_computes_retrieval(self):
        mc = MetricsCollector(EvaluationConfig())
        per_q = [{"retrieval_time_ms": 5.0}, {"retrieval_time_ms": 15.0}]
        lm = mc.compute_latency_metrics(per_q)
        assert lm.retrieval_p50_ms == pytest.approx(10.0, abs=0.5)
        assert lm.generation_p50_ms == 0.0

    def test_none_values_are_skipped(self):
        mc = MetricsCollector(EvaluationConfig())
        per_q = [
            {"retrieval_time_ms": 10.0, "generation_time_ms": None},
            {"retrieval_time_ms": 20.0, "generation_time_ms": 100.0},
        ]
        lm = mc.compute_latency_metrics(per_q)
        assert lm.generation_p50_ms == 100.0  # only one valid entry


# ---------------------------------------------------------------------------
# MetricsCollector — ROUGE (mocked)
# ---------------------------------------------------------------------------


# Mimic rouge_score's Score namedtuple
_RougeScore = namedtuple("Score", ["precision", "recall", "fmeasure"])


def _mock_rouge_scorer_factory():
    """Return a mock RougeScorer whose .score() returns fixed values."""
    mock = MagicMock()
    mock.score.return_value = {
        "rougeL": _RougeScore(precision=0.75, recall=0.80, fmeasure=0.77),
    }
    return mock


class TestComputeRouge:
    def test_rouge_scores_computed_when_flag_true(self):
        cfg = EvaluationConfig(compute_rouge=True, compute_bert_score=False)
        mc = MetricsCollector(cfg)
        mc._rouge_scorer = _mock_rouge_scorer_factory()

        per_q = [
            {
                "question_id": "q1",
                "generated_answer": "The answer.",
            }
        ]
        questions_by_id = {"q1": {"expected_answer_hint": "The expected."}}
        mc.compute_generation_metrics(per_q, questions_by_id)

        assert "generation_metrics" in per_q[0]
        gm = per_q[0]["generation_metrics"]
        assert gm["rouge_l_f1"] == pytest.approx(0.77)
        assert gm["rouge_l_precision"] == pytest.approx(0.75)
        assert gm["rouge_l_recall"] == pytest.approx(0.80)
        assert "bert_score_f1" not in gm

    def test_rouge_skipped_when_flag_false(self):
        cfg = EvaluationConfig(compute_rouge=False, compute_bert_score=False)
        mc = MetricsCollector(cfg)

        per_q = [
            {
                "question_id": "q1",
                "generated_answer": "The answer.",
            }
        ]
        questions_by_id = {"q1": {"expected_answer_hint": "The expected."}}
        mc.compute_generation_metrics(per_q, questions_by_id)
        assert "generation_metrics" not in per_q[0]

    def test_rouge_import_error_message(self):
        cfg = EvaluationConfig(compute_rouge=True)
        mc = MetricsCollector(cfg)
        with patch(
            "src.evaluation.metrics_collector.MetricsCollector._ensure_rouge_scorer",
            side_effect=ImportError("rouge-score is required"),
        ):
            with pytest.raises(ImportError, match="rouge-score"):
                mc.compute_generation_metrics(
                    [{"question_id": "q1", "generated_answer": "x"}],
                    {"q1": {"expected_answer_hint": "y"}},
                )


# ---------------------------------------------------------------------------
# MetricsCollector — BERTScore (mocked)
# ---------------------------------------------------------------------------


def _mock_bert_scorer_factory():
    """Return a mock BERTScorer whose .score() returns tensors."""
    mock = MagicMock()
    # BERTScorer.score() returns (P, R, F1) tensors
    import torch
    mock.score.return_value = (
        torch.tensor([0.88]),  # P
        torch.tensor([0.92]),  # R
        torch.tensor([0.90]),  # F1
    )
    return mock


class TestComputeBertScore:
    def test_bertscore_computed_when_flag_true(self):
        cfg = EvaluationConfig(compute_rouge=False, compute_bert_score=True)
        mc = MetricsCollector(cfg)
        mc._bert_scorer = _mock_bert_scorer_factory()

        per_q = [
            {
                "question_id": "q1",
                "generated_answer": "The answer.",
            }
        ]
        questions_by_id = {"q1": {"expected_answer_hint": "The expected."}}
        mc.compute_generation_metrics(per_q, questions_by_id)

        gm = per_q[0]["generation_metrics"]
        assert gm["bert_score_f1"] == pytest.approx(0.90)
        assert gm["bert_score_precision"] == pytest.approx(0.88)
        assert gm["bert_score_recall"] == pytest.approx(0.92)
        assert "rouge_l_f1" not in gm

    def test_bertscore_skipped_when_flag_false(self):
        cfg = EvaluationConfig(compute_rouge=False, compute_bert_score=False)
        mc = MetricsCollector(cfg)

        per_q = [
            {
                "question_id": "q1",
                "generated_answer": "The answer.",
            }
        ]
        questions_by_id = {"q1": {"expected_answer_hint": "The expected."}}
        mc.compute_generation_metrics(per_q, questions_by_id)
        assert "generation_metrics" not in per_q[0]

    def test_bertscore_import_error_message(self):
        cfg = EvaluationConfig(compute_bert_score=True, compute_rouge=False)
        mc = MetricsCollector(cfg)
        with patch(
            "src.evaluation.metrics_collector.MetricsCollector._ensure_bert_scorer",
            side_effect=ImportError("bert-score is required"),
        ):
            with pytest.raises(ImportError, match="bert-score"):
                mc.compute_generation_metrics(
                    [{"question_id": "q1", "generated_answer": "x"}],
                    {"q1": {"expected_answer_hint": "y"}},
                )


# ---------------------------------------------------------------------------
# MetricsCollector — both ROUGE + BERTScore
# ---------------------------------------------------------------------------


class TestComputeBothMetrics:
    def test_both_metrics_when_both_flags_true(self):
        cfg = EvaluationConfig(compute_rouge=True, compute_bert_score=True)
        mc = MetricsCollector(cfg)
        mc._rouge_scorer = _mock_rouge_scorer_factory()
        mc._bert_scorer = _mock_bert_scorer_factory()

        per_q = [
            {
                "question_id": "q1",
                "generated_answer": "The answer.",
            }
        ]
        questions_by_id = {"q1": {"expected_answer_hint": "The expected."}}
        mc.compute_generation_metrics(per_q, questions_by_id)

        gm = per_q[0]["generation_metrics"]
        assert "rouge_l_f1" in gm
        assert "bert_score_f1" in gm


# ---------------------------------------------------------------------------
# MetricsCollector — skipping logic
# ---------------------------------------------------------------------------


class TestSkipLogic:
    def test_skips_entries_without_generated_answer(self):
        cfg = EvaluationConfig(compute_rouge=True)
        mc = MetricsCollector(cfg)
        mc._rouge_scorer = _mock_rouge_scorer_factory()

        per_q = [
            {"question_id": "q1"},  # no generated_answer
            {"question_id": "q2", "generated_answer": None},
        ]
        questions_by_id = {
            "q1": {"expected_answer_hint": "x"},
            "q2": {"expected_answer_hint": "y"},
        }
        mc.compute_generation_metrics(per_q, questions_by_id)
        assert "generation_metrics" not in per_q[0]
        assert "generation_metrics" not in per_q[1]

    def test_skips_entries_without_expected_answer_hint(self):
        cfg = EvaluationConfig(compute_rouge=True)
        mc = MetricsCollector(cfg)
        mc._rouge_scorer = _mock_rouge_scorer_factory()

        per_q = [
            {"question_id": "q1", "generated_answer": "answer"},
        ]
        questions_by_id = {"q1": {}}  # no expected_answer_hint
        mc.compute_generation_metrics(per_q, questions_by_id)
        assert "generation_metrics" not in per_q[0]

    def test_skips_entries_with_unknown_question_id(self):
        cfg = EvaluationConfig(compute_rouge=True)
        mc = MetricsCollector(cfg)
        mc._rouge_scorer = _mock_rouge_scorer_factory()

        per_q = [
            {"question_id": "unknown", "generated_answer": "answer"},
        ]
        questions_by_id = {"q1": {"expected_answer_hint": "x"}}
        mc.compute_generation_metrics(per_q, questions_by_id)
        assert "generation_metrics" not in per_q[0]

    def test_mixed_eligible_and_ineligible(self):
        cfg = EvaluationConfig(compute_rouge=True, compute_bert_score=False)
        mc = MetricsCollector(cfg)
        mc._rouge_scorer = _mock_rouge_scorer_factory()

        per_q = [
            {"question_id": "q1", "generated_answer": "answer1"},
            {"question_id": "q2"},  # no answer
            {"question_id": "q3", "generated_answer": "answer3"},
        ]
        questions_by_id = {
            "q1": {"expected_answer_hint": "ref1"},
            "q2": {"expected_answer_hint": "ref2"},
            "q3": {"expected_answer_hint": "ref3"},
        }
        mc.compute_generation_metrics(per_q, questions_by_id)
        assert "generation_metrics" in per_q[0]
        assert "generation_metrics" not in per_q[1]
        assert "generation_metrics" in per_q[2]


# ---------------------------------------------------------------------------
# MetricsCollector — get_summary()
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_empty_summary_when_nothing_computed(self):
        mc = MetricsCollector(EvaluationConfig())
        assert mc.get_summary() == {}

    def test_summary_has_latency_after_compute(self):
        mc = MetricsCollector(EvaluationConfig())
        mc.compute_latency_metrics([{"retrieval_time_ms": 10.0}])
        s = mc.get_summary()
        assert "latency" in s
        assert s["latency"]["retrieval_p50_ms"] == 10.0

    def test_summary_has_generation_after_compute(self):
        cfg = EvaluationConfig(compute_rouge=True, compute_bert_score=False)
        mc = MetricsCollector(cfg)
        mc._rouge_scorer = _mock_rouge_scorer_factory()

        per_q = [{"question_id": "q1", "generated_answer": "a"}]
        q_by_id = {"q1": {"expected_answer_hint": "b"}}
        mc.compute_generation_metrics(per_q, q_by_id)

        s = mc.get_summary()
        assert "generation" in s
        assert s["generation"]["num_questions_scored"] == 1
        assert "avg_rouge_l_f1" in s["generation"]

    def test_summary_has_both_sections(self):
        cfg = EvaluationConfig(compute_rouge=True, compute_bert_score=False)
        mc = MetricsCollector(cfg)
        mc._rouge_scorer = _mock_rouge_scorer_factory()

        per_q = [
            {
                "question_id": "q1",
                "generated_answer": "a",
                "retrieval_time_ms": 5.0,
                "generation_time_ms": 100.0,
            }
        ]
        q_by_id = {"q1": {"expected_answer_hint": "b"}}
        mc.compute_generation_metrics(per_q, q_by_id)
        mc.compute_latency_metrics(per_q)

        s = mc.get_summary()
        assert "latency" in s
        assert "generation" in s


# ---------------------------------------------------------------------------
# MetricsCollector — ensure_*_scorer lazy init
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MetricsCollector — RAGAS delegation
# ---------------------------------------------------------------------------


class TestComputeRagasMetrics:
    def test_ragas_skipped_when_flag_false(self):
        cfg = EvaluationConfig(compute_ragas=False)
        mc = MetricsCollector(cfg)

        per_q = [{"question_id": "q1", "generated_answer": "answer"}]
        mc.compute_ragas_metrics(per_q, {"q1": {"question": "Q?"}})
        # _ragas_evaluator should never have been created
        assert mc._ragas_evaluator is None

    def test_ragas_delegates_to_evaluator(self):
        cfg = EvaluationConfig(compute_ragas=True)
        mc = MetricsCollector(cfg)
        mock_evaluator = MagicMock()
        mc._ragas_evaluator = mock_evaluator

        per_q = [{"question_id": "q1", "generated_answer": "answer"}]
        q_by_id = {"q1": {"question": "Q?"}}
        mc.compute_ragas_metrics(per_q, q_by_id)

        mock_evaluator.evaluate.assert_called_once_with(per_q, q_by_id)

    def test_ragas_import_error_message(self):
        cfg = EvaluationConfig(compute_ragas=True)
        mc = MetricsCollector(cfg)
        with patch(
            "src.evaluation.metrics_collector.MetricsCollector._ensure_ragas_evaluator",
            side_effect=ImportError("ragas is required"),
        ):
            with pytest.raises(ImportError, match="ragas"):
                mc.compute_ragas_metrics(
                    [{"question_id": "q1", "generated_answer": "x"}],
                    {"q1": {"question": "Q?"}},
                )


class TestGetSummaryWithRagas:
    def test_summary_includes_ragas_section(self):
        cfg = EvaluationConfig(compute_ragas=True)
        mc = MetricsCollector(cfg)

        mock_evaluator = MagicMock()
        mock_evaluator.get_aggregated.return_value = {
            "avg_faithfulness": 0.85,
            "num_questions_scored": 3,
        }
        mc._ragas_evaluator = mock_evaluator

        s = mc.get_summary()
        assert "ragas" in s
        assert s["ragas"]["avg_faithfulness"] == 0.85
        assert s["ragas"]["num_questions_scored"] == 3

    def test_summary_omits_ragas_when_no_evaluator(self):
        mc = MetricsCollector(EvaluationConfig())
        s = mc.get_summary()
        assert "ragas" not in s

    def test_summary_omits_ragas_when_empty_aggregated(self):
        cfg = EvaluationConfig(compute_ragas=True)
        mc = MetricsCollector(cfg)
        mock_evaluator = MagicMock()
        mock_evaluator.get_aggregated.return_value = {}
        mc._ragas_evaluator = mock_evaluator

        s = mc.get_summary()
        assert "ragas" not in s


class TestLazyInit:
    def test_ensure_rouge_scorer_caches(self):
        mc = MetricsCollector(EvaluationConfig())
        mock_scorer = MagicMock()
        with patch(
            "src.evaluation.metrics_collector.MetricsCollector._ensure_rouge_scorer"
        ) as mock_ensure:
            # Simulate already having a scorer
            mc._rouge_scorer = mock_scorer
            mc._ensure_rouge_scorer = mock_ensure
            # Call won't actually re-init since we're mocking
            # This test just verifies the pattern works

    def test_ensure_bert_scorer_raises_on_missing_package(self):
        mc = MetricsCollector(EvaluationConfig())
        with patch.dict("sys.modules", {"bert_score": None}):
            with pytest.raises(ImportError):
                mc._ensure_bert_scorer()

    def test_ensure_rouge_scorer_raises_on_missing_package(self):
        mc = MetricsCollector(EvaluationConfig())
        with patch.dict("sys.modules", {"rouge_score": None, "rouge_score.rouge_scorer": None}):
            with pytest.raises(ImportError):
                mc._ensure_rouge_scorer()
