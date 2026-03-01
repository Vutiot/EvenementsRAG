"""Unit tests for src/evaluation/ragas_evaluator.py (E1-F2-T2).

All tests mock ``ragas.evaluate()`` entirely — no actual LLM calls.
"""

import math
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from src.benchmarks.config import EvaluationConfig
from src.evaluation.ragas_evaluator import (
    RagasEvaluator,
    RagasResult,
    _load_ragas_metrics,
    _RAGAS_METRIC_REGISTRY,
)


# ---------------------------------------------------------------------------
# RagasResult dataclass
# ---------------------------------------------------------------------------


class TestRagasResult:
    def test_to_dict_excludes_none(self):
        r = RagasResult(scores={"faithfulness": 0.9, "coherence": None})
        d = r.to_dict()
        assert d == {"faithfulness": 0.9}

    def test_to_dict_empty_when_all_none(self):
        r = RagasResult(scores={"faithfulness": None})
        assert r.to_dict() == {}

    def test_to_dict_full(self):
        r = RagasResult(scores={"faithfulness": 0.9, "coherence": 0.8, "conciseness": 0.7})
        d = r.to_dict()
        assert len(d) == 3
        assert d["faithfulness"] == 0.9

    def test_to_dict_default_empty(self):
        r = RagasResult()
        assert r.to_dict() == {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_per_q(
    q_id="q1",
    answer="The answer.",
    contexts=None,
    question=None,
):
    """Create a per-question entry for tests."""
    entry = {"question_id": q_id}
    if answer is not None:
        entry["generated_answer"] = answer
    if contexts is not None:
        entry["retrieved_contexts"] = contexts
    if question is not None:
        entry["question"] = question
    return entry


def _make_questions_by_id(
    q_id="q1",
    question="What was D-Day?",
    hint="D-Day was June 6, 1944.",
):
    """Create questions_by_id dict for tests."""
    q = {"question": question}
    if hint is not None:
        q["expected_answer_hint"] = hint
    return {q_id: q}


def _mock_ragas_evaluate(scores_dict, num_rows=1):
    """Return a mock for ragas.evaluate that returns a DataFrame."""
    rows = []
    for _ in range(num_rows):
        row = {"user_input": "q", "response": "a", "retrieved_contexts": ["c"], "reference": "r"}
        row.update(scores_dict)
        rows.append(row)
    df = pd.DataFrame(rows)

    mock_result = MagicMock()
    mock_result.to_pandas.return_value = df
    return mock_result


# ---------------------------------------------------------------------------
# RagasEvaluator — evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    @patch("src.evaluation.ragas_evaluator._build_evaluator_embeddings")
    @patch("src.evaluation.ragas_evaluator._build_evaluator_llm")
    @patch("src.evaluation.ragas_evaluator._build_run_config")
    @patch("src.evaluation.ragas_evaluator._load_ragas_metrics")
    @patch("ragas.evaluate")
    def test_evaluate_adds_ragas_metrics_to_entry(
        self, mock_ragas_eval, mock_load, mock_run_cfg, mock_llm, mock_emb
    ):
        mock_load.return_value = {"faithfulness": MagicMock(), "coherence": MagicMock()}
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        mock_run_cfg.return_value = MagicMock()

        mock_eval_result = _mock_ragas_evaluate({"faithfulness": 0.9, "coherence": 0.85})
        mock_ragas_eval.return_value = mock_eval_result

        cfg = EvaluationConfig(
            compute_ragas=True,
            ragas_metrics=["faithfulness", "coherence"],
        )
        evaluator = RagasEvaluator(cfg)

        per_q = [_make_per_q(contexts=["context text"])]
        q_by_id = _make_questions_by_id()

        results = evaluator.evaluate(per_q, q_by_id)

        assert len(results) == 1
        assert "ragas_metrics" in per_q[0]
        assert per_q[0]["ragas_metrics"]["faithfulness"] == 0.9
        assert per_q[0]["ragas_metrics"]["coherence"] == 0.85

    @patch("src.evaluation.ragas_evaluator._build_evaluator_embeddings")
    @patch("src.evaluation.ragas_evaluator._build_evaluator_llm")
    @patch("src.evaluation.ragas_evaluator._build_run_config")
    @patch("src.evaluation.ragas_evaluator._load_ragas_metrics")
    @patch("ragas.evaluate")
    def test_evaluate_handles_nan_values(
        self, mock_ragas_eval, mock_load, mock_run_cfg, mock_llm, mock_emb
    ):
        mock_load.return_value = {"faithfulness": MagicMock()}
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        mock_run_cfg.return_value = MagicMock()

        mock_eval_result = _mock_ragas_evaluate({"faithfulness": float("nan")})
        mock_ragas_eval.return_value = mock_eval_result

        cfg = EvaluationConfig(compute_ragas=True, ragas_metrics=["faithfulness"])
        evaluator = RagasEvaluator(cfg)

        per_q = [_make_per_q(contexts=["ctx"])]
        q_by_id = _make_questions_by_id()

        results = evaluator.evaluate(per_q, q_by_id)

        # NaN should be converted to None
        assert results[0].scores["faithfulness"] is None
        # to_dict excludes None
        assert "faithfulness" not in per_q[0]["ragas_metrics"]


# ---------------------------------------------------------------------------
# RagasEvaluator — skip logic
# ---------------------------------------------------------------------------


class TestEvaluateSkipLogic:
    def test_skip_no_generated_answer(self):
        cfg = EvaluationConfig(compute_ragas=True, ragas_metrics=["faithfulness"])
        evaluator = RagasEvaluator(cfg)

        per_q = [_make_per_q(answer=None, contexts=["ctx"])]
        q_by_id = _make_questions_by_id()

        # Should not call ragas.evaluate at all
        results = evaluator.evaluate(per_q, q_by_id)
        assert results == []
        assert "ragas_metrics" not in per_q[0]

    def test_skip_empty_contexts(self):
        cfg = EvaluationConfig(compute_ragas=True, ragas_metrics=["faithfulness"])
        evaluator = RagasEvaluator(cfg)

        per_q = [_make_per_q(contexts=[])]
        q_by_id = _make_questions_by_id()

        results = evaluator.evaluate(per_q, q_by_id)
        assert results == []

    def test_skip_no_question_text(self):
        cfg = EvaluationConfig(compute_ragas=True, ragas_metrics=["faithfulness"])
        evaluator = RagasEvaluator(cfg)

        per_q = [_make_per_q(contexts=["ctx"])]
        # No question text in questions_by_id and no "question" key in entry
        q_by_id = {"q1": {}}

        results = evaluator.evaluate(per_q, q_by_id)
        assert results == []

    def test_skip_no_eligible_entries(self):
        cfg = EvaluationConfig(compute_ragas=True, ragas_metrics=["faithfulness"])
        evaluator = RagasEvaluator(cfg)

        # Entry missing both answer and contexts
        per_q = [{"question_id": "q1"}]
        q_by_id = _make_questions_by_id()

        results = evaluator.evaluate(per_q, q_by_id)
        assert results == []


# ---------------------------------------------------------------------------
# RagasEvaluator — ground_truths fallback
# ---------------------------------------------------------------------------


class TestGroundTruthsFallback:
    @patch("src.evaluation.ragas_evaluator._build_evaluator_embeddings")
    @patch("src.evaluation.ragas_evaluator._build_evaluator_llm")
    @patch("src.evaluation.ragas_evaluator._build_run_config")
    @patch("src.evaluation.ragas_evaluator._load_ragas_metrics")
    @patch("ragas.evaluate")
    def test_fallback_to_answer_when_no_hint(
        self, mock_ragas_eval, mock_load, mock_run_cfg, mock_llm, mock_emb
    ):
        """When expected_answer_hint is absent, reference falls back to the answer."""
        mock_load.return_value = {"faithfulness": MagicMock()}
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        mock_run_cfg.return_value = MagicMock()

        mock_eval_result = _mock_ragas_evaluate({"faithfulness": 0.8})
        mock_ragas_eval.return_value = mock_eval_result

        cfg = EvaluationConfig(compute_ragas=True, ragas_metrics=["faithfulness"])
        evaluator = RagasEvaluator(cfg)

        per_q = [_make_per_q(answer="The generated answer", contexts=["ctx"])]
        q_by_id = {"q1": {"question": "What was D-Day?"}}  # no expected_answer_hint

        captured_samples = []

        original_sample = None
        try:
            from ragas.dataset_schema import SingleTurnSample as _OrigSample
            original_sample = _OrigSample
        except ImportError:
            pass

        def capture_sample(**kwargs):
            captured_samples.append(kwargs)
            if original_sample is not None:
                return original_sample(**kwargs)
            return MagicMock()

        with patch(
            "ragas.dataset_schema.SingleTurnSample",
            side_effect=capture_sample,
        ):
            evaluator.evaluate(per_q, q_by_id)

        assert len(captured_samples) == 1
        # reference should fall back to the generated answer
        assert captured_samples[0]["reference"] == "The generated answer"


# ---------------------------------------------------------------------------
# RagasEvaluator — get_aggregated()
# ---------------------------------------------------------------------------


class TestGetAggregated:
    def test_empty_when_no_results(self):
        cfg = EvaluationConfig(compute_ragas=True)
        evaluator = RagasEvaluator(cfg)
        assert evaluator.get_aggregated() == {}

    def test_averages_computed(self):
        cfg = EvaluationConfig(compute_ragas=True)
        evaluator = RagasEvaluator(cfg)

        evaluator._results = [
            RagasResult(scores={"faithfulness": 0.8, "coherence": 0.9}),
            RagasResult(scores={"faithfulness": 0.6, "coherence": 0.7}),
        ]

        agg = evaluator.get_aggregated()
        assert agg["avg_faithfulness"] == pytest.approx(0.7)
        assert agg["avg_coherence"] == pytest.approx(0.8)
        assert agg["num_questions_scored"] == 2

    def test_averages_skip_none(self):
        cfg = EvaluationConfig(compute_ragas=True)
        evaluator = RagasEvaluator(cfg)

        evaluator._results = [
            RagasResult(scores={"faithfulness": 0.8, "coherence": None}),
            RagasResult(scores={"faithfulness": 0.6, "coherence": 0.7}),
        ]

        agg = evaluator.get_aggregated()
        assert agg["avg_faithfulness"] == pytest.approx(0.7)
        assert agg["avg_coherence"] == pytest.approx(0.7)  # only 1 value
        assert agg["num_questions_scored"] == 2


# ---------------------------------------------------------------------------
# Lazy import error handling
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_evaluate_raises_on_missing_ragas(self):
        cfg = EvaluationConfig(compute_ragas=True, ragas_metrics=["faithfulness"])
        evaluator = RagasEvaluator(cfg)

        per_q = [_make_per_q(contexts=["ctx"])]
        q_by_id = _make_questions_by_id()

        with patch.dict("sys.modules", {"ragas": None, "ragas.evaluate": None}):
            with pytest.raises(ImportError, match="ragas"):
                evaluator.evaluate(per_q, q_by_id)

    def test_load_ragas_metrics_raises_on_missing_ragas(self):
        import src.evaluation.ragas_evaluator as mod
        old = mod._RAGAS_METRIC_REGISTRY
        mod._RAGAS_METRIC_REGISTRY = None
        try:
            with patch.dict("sys.modules", {
                "ragas": None,
                "ragas.metrics": None,
                "ragas.metrics._faithfulness": None,
            }):
                with pytest.raises(ImportError, match="ragas"):
                    _load_ragas_metrics()
        finally:
            mod._RAGAS_METRIC_REGISTRY = old
