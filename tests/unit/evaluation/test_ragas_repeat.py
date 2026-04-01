"""Unit tests for RAGAS repeat-count averaging logic (E7-F5-T1).

Tests that:
- repeat_count=1 preserves original behaviour (no std computed)
- repeat_count>1 calls the evaluator N times and averages results
- Standard deviations are stored in ``ragas_metrics_std``

All tests mock RagasEvaluator — no real LLM calls.
"""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.benchmarks.config import EvaluationConfig
from src.evaluation.metrics_collector import MetricsCollector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_per_question(n=2):
    """Return a list of per-question dicts suitable for RAGAS evaluation."""
    return [
        {
            "question_id": f"q{i}",
            "generated_answer": f"Answer {i}",
            "retrieved_contexts": [f"context {i}"],
        }
        for i in range(n)
    ]


def _make_questions_by_id(n=2):
    return {
        f"q{i}": {
            "question": f"Question {i}?",
            "expected_answer_hint": f"Expected {i}",
        }
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# Single-run (repeat_count=1) — behaviour unchanged
# ---------------------------------------------------------------------------


class TestRepeatCountOne:
    """With repeat_count=1 (default), behaviour is identical to before."""

    def test_default_repeat_count_is_one(self):
        cfg = EvaluationConfig()
        assert cfg.ragas_repeat_count == 1

    def test_single_run_no_std(self):
        """When repeat_count=1, ragas_metrics_std should NOT be added."""
        cfg = EvaluationConfig(
            compute_ragas=True,
            ragas_metrics=["faithfulness", "coherence"],
            ragas_repeat_count=1,
        )
        collector = MetricsCollector(cfg)

        per_q = _make_per_question(2)
        q_by_id = _make_questions_by_id(2)

        # Mock the evaluator so it adds ragas_metrics in-place
        mock_evaluator = MagicMock()

        def fake_evaluate(pq, qbi):
            for entry in pq:
                entry["ragas_metrics"] = {"faithfulness": 0.9, "coherence": 0.8}
            return []

        mock_evaluator.evaluate.side_effect = fake_evaluate

        with patch.object(collector, "_ensure_ragas_evaluator"):
            collector._ragas_evaluator = mock_evaluator
            collector.compute_ragas_metrics(per_q, q_by_id)

        # ragas_metrics should be set
        for entry in per_q:
            assert "ragas_metrics" in entry
            assert entry["ragas_metrics"]["faithfulness"] == 0.9

        # ragas_metrics_std should NOT be present (single run)
        for entry in per_q:
            assert "ragas_metrics_std" not in entry

        # Evaluator called exactly once
        mock_evaluator.evaluate.assert_called_once()

    def test_single_run_skipped_when_compute_ragas_false(self):
        cfg = EvaluationConfig(compute_ragas=False, ragas_repeat_count=1)
        collector = MetricsCollector(cfg)

        per_q = _make_per_question(1)
        q_by_id = _make_questions_by_id(1)

        collector.compute_ragas_metrics(per_q, q_by_id)

        assert "ragas_metrics" not in per_q[0]


# ---------------------------------------------------------------------------
# Multi-run (repeat_count>1) — averaging + std
# ---------------------------------------------------------------------------


class TestRepeatCountMultiple:
    """With repeat_count>1, evaluator is called N times and results are averaged."""

    def _run_with_repeat(self, repeat_count, run_scores_sequence):
        """Helper: run compute_ragas_metrics with mocked evaluator returning
        different scores for each repeat.

        Args:
            repeat_count: Number of repeats
            run_scores_sequence: List of dicts, one per run. Each dict maps
                question index to {metric: score} dict.

        Returns:
            per_q after compute_ragas_metrics
        """
        n_questions = len(run_scores_sequence[0])
        cfg = EvaluationConfig(
            compute_ragas=True,
            ragas_metrics=["faithfulness", "coherence"],
            ragas_repeat_count=repeat_count,
        )
        collector = MetricsCollector(cfg)

        per_q = _make_per_question(n_questions)
        q_by_id = _make_questions_by_id(n_questions)

        call_count = [0]

        def fake_ensure():
            pass

        def make_fake_evaluator(run_idx):
            mock = MagicMock()

            def fake_evaluate(pq, qbi):
                scores = run_scores_sequence[run_idx]
                for i, entry in enumerate(pq):
                    entry["ragas_metrics"] = dict(scores[i])
                return []

            mock.evaluate.side_effect = fake_evaluate
            return mock

        # Patch _ensure_ragas_evaluator to set a fresh mock each time
        original_ensure = collector._ensure_ragas_evaluator

        def patched_ensure():
            idx = call_count[0]
            call_count[0] += 1
            collector._ragas_evaluator = make_fake_evaluator(idx)

        collector._ensure_ragas_evaluator = patched_ensure
        collector.compute_ragas_metrics(per_q, q_by_id)

        assert call_count[0] == repeat_count
        return per_q

    def test_three_runs_averaged(self):
        """3 runs: scores are averaged, std is computed."""
        run_scores = [
            # Run 0
            {0: {"faithfulness": 0.6, "coherence": 0.8},
             1: {"faithfulness": 0.7, "coherence": 0.9}},
            # Run 1
            {0: {"faithfulness": 0.8, "coherence": 0.7},
             1: {"faithfulness": 0.9, "coherence": 0.8}},
            # Run 2
            {0: {"faithfulness": 0.7, "coherence": 0.9},
             1: {"faithfulness": 0.8, "coherence": 0.7}},
        ]

        per_q = self._run_with_repeat(3, run_scores)

        # Q0 faithfulness: mean(0.6, 0.8, 0.7) = 0.7
        assert per_q[0]["ragas_metrics"]["faithfulness"] == pytest.approx(0.7, abs=1e-6)
        # Q0 coherence: mean(0.8, 0.7, 0.9) = 0.8
        assert per_q[0]["ragas_metrics"]["coherence"] == pytest.approx(0.8, abs=1e-6)

        # Q1 faithfulness: mean(0.7, 0.9, 0.8) = 0.8
        assert per_q[1]["ragas_metrics"]["faithfulness"] == pytest.approx(0.8, abs=1e-6)
        # Q1 coherence: mean(0.9, 0.8, 0.7) = 0.8
        assert per_q[1]["ragas_metrics"]["coherence"] == pytest.approx(0.8, abs=1e-6)

        # Standard deviations should be present
        assert "ragas_metrics_std" in per_q[0]
        assert "ragas_metrics_std" in per_q[1]

        # Q0 faithfulness std: std([0.6, 0.8, 0.7])
        expected_std = float(np.std([0.6, 0.8, 0.7]))
        assert per_q[0]["ragas_metrics_std"]["faithfulness"] == pytest.approx(
            expected_std, abs=1e-6
        )

    def test_two_runs_std_computed(self):
        """2 runs: std is non-zero when scores differ."""
        run_scores = [
            {0: {"faithfulness": 0.6, "coherence": 0.8}},
            {0: {"faithfulness": 0.8, "coherence": 0.8}},
        ]

        per_q = self._run_with_repeat(2, run_scores)

        # Mean faithfulness: 0.7
        assert per_q[0]["ragas_metrics"]["faithfulness"] == pytest.approx(0.7, abs=1e-6)
        # Mean coherence: 0.8
        assert per_q[0]["ragas_metrics"]["coherence"] == pytest.approx(0.8, abs=1e-6)

        # Std faithfulness: std([0.6, 0.8]) = 0.1
        assert per_q[0]["ragas_metrics_std"]["faithfulness"] == pytest.approx(0.1, abs=1e-6)
        # Std coherence: std([0.8, 0.8]) = 0.0
        assert per_q[0]["ragas_metrics_std"]["coherence"] == pytest.approx(0.0, abs=1e-6)

    def test_identical_runs_zero_std(self):
        """When all runs produce the same score, std is 0."""
        run_scores = [
            {0: {"faithfulness": 0.9}},
            {0: {"faithfulness": 0.9}},
            {0: {"faithfulness": 0.9}},
        ]

        per_q = self._run_with_repeat(3, run_scores)

        assert per_q[0]["ragas_metrics"]["faithfulness"] == pytest.approx(0.9, abs=1e-6)
        assert per_q[0]["ragas_metrics_std"]["faithfulness"] == pytest.approx(0.0, abs=1e-6)

    def test_no_eligible_entries_with_repeat(self):
        """When no entries are eligible, repeat still works without error."""
        cfg = EvaluationConfig(
            compute_ragas=True,
            ragas_metrics=["faithfulness"],
            ragas_repeat_count=3,
        )
        collector = MetricsCollector(cfg)

        per_q = _make_per_question(1)
        q_by_id = _make_questions_by_id(1)

        call_count = [0]

        def patched_ensure():
            mock = MagicMock()
            # Evaluator does NOT add ragas_metrics (no eligible entries)
            mock.evaluate.return_value = []
            collector._ragas_evaluator = mock
            call_count[0] += 1

        collector._ensure_ragas_evaluator = patched_ensure
        collector.compute_ragas_metrics(per_q, q_by_id)

        # No ragas_metrics or std should be added (empty dicts from runs)
        assert per_q[0].get("ragas_metrics") is None or per_q[0].get("ragas_metrics") == {}

    def test_repeat_does_not_mutate_original_per_question(self):
        """Each repeat run operates on a shadow copy, not the original."""
        run_scores = [
            {0: {"faithfulness": 0.6}},
            {0: {"faithfulness": 0.8}},
        ]

        cfg = EvaluationConfig(
            compute_ragas=True,
            ragas_metrics=["faithfulness"],
            ragas_repeat_count=2,
        )
        collector = MetricsCollector(cfg)
        per_q = _make_per_question(1)
        q_by_id = _make_questions_by_id(1)

        call_count = [0]
        received_entries = []

        def patched_ensure():
            mock = MagicMock()
            idx = call_count[0]
            call_count[0] += 1

            def fake_eval(pq, qbi):
                # Record that we received a different list than per_q
                received_entries.append(id(pq))
                scores = run_scores[idx]
                for i, entry in enumerate(pq):
                    entry["ragas_metrics"] = dict(scores[i])
                return []

            mock.evaluate.side_effect = fake_eval
            collector._ragas_evaluator = mock

        collector._ensure_ragas_evaluator = patched_ensure
        collector.compute_ragas_metrics(per_q, q_by_id)

        # The shadow lists should NOT be the original per_q
        assert all(eid != id(per_q) for eid in received_entries)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestRepeatCountValidation:
    def test_repeat_count_min_1(self):
        with pytest.raises(Exception):
            EvaluationConfig(ragas_repeat_count=0)

    def test_repeat_count_max_10(self):
        with pytest.raises(Exception):
            EvaluationConfig(ragas_repeat_count=11)

    def test_repeat_count_valid_range(self):
        for n in [1, 5, 10]:
            cfg = EvaluationConfig(ragas_repeat_count=n)
            assert cfg.ragas_repeat_count == n
