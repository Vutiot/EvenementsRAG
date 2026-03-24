"""Unit tests for src/evaluation/metrics_collector.py.

Tests cover latency metrics and RAGAS delegation. ROUGE/BERTScore ML metrics
have been removed from the project — only LLM-based metrics remain.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.benchmarks.config import EvaluationConfig
from src.evaluation.metrics_collector import (
    LatencyMetrics,
    MetricsCollector,
)


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
