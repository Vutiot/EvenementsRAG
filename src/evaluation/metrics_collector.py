"""
MetricsCollector — generation quality (ROUGE-L, BERTScore) and latency percentiles.

Complements the retrieval metrics computed by the legacy BenchmarkRunner.
ROUGE and BERTScore scorers are lazy-imported so the module stays light when
the flags are disabled.

Usage:
    from src.evaluation.metrics_collector import MetricsCollector
    from src.benchmarks.config import EvaluationConfig

    collector = MetricsCollector(EvaluationConfig(compute_rouge=True))
    collector.compute_generation_metrics(per_question, questions_by_id)
    latency = collector.compute_latency_metrics(per_question)
    summary = collector.get_summary()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.benchmarks.config import EvaluationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GenerationMetrics:
    """Per-question generation quality scores."""

    rouge_l_f1: Optional[float] = None
    rouge_l_precision: Optional[float] = None
    rouge_l_recall: Optional[float] = None
    bert_score_f1: Optional[float] = None
    bert_score_precision: Optional[float] = None
    bert_score_recall: Optional[float] = None

    def to_dict(self) -> dict:
        """Return a dict excluding ``None`` values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class LatencyMetrics:
    """Aggregated latency percentiles (milliseconds)."""

    retrieval_p50_ms: float = 0.0
    retrieval_p95_ms: float = 0.0
    retrieval_p99_ms: float = 0.0
    generation_p50_ms: float = 0.0
    generation_p95_ms: float = 0.0
    generation_p99_ms: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class AggregatedGenerationMetrics:
    """Averaged generation scores across all scored questions."""

    avg_rouge_l_f1: Optional[float] = None
    avg_bert_score_f1: Optional[float] = None
    num_questions_scored: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Computes generation-quality and latency metrics on top of retrieval results.

    Does **not** wrap ``compute_retrieval_metrics()`` — retrieval metrics stay
    in the legacy ``BenchmarkRunner``.  This collector only adds generation +
    latency on top.
    """

    def __init__(self, eval_config: EvaluationConfig) -> None:
        self._config = eval_config
        self._rouge_scorer = None
        self._bert_scorer = None
        self._ragas_evaluator = None
        self._latency: Optional[LatencyMetrics] = None
        self._generation_scores: List[GenerationMetrics] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_generation_metrics(
        self,
        per_question: List[dict],
        questions_by_id: Dict[str, dict],
    ) -> None:
        """Compute ROUGE-L and/or BERTScore per question (in-place mutation).

        Adds a ``"generation_metrics"`` key to each entry that has both
        ``generated_answer`` and a matching ``expected_answer_hint`` in
        *questions_by_id*.  Entries without either are silently skipped.
        """
        predictions: List[str] = []
        references: List[str] = []
        indices: List[int] = []

        for i, entry in enumerate(per_question):
            answer = entry.get("generated_answer")
            if not answer:
                continue
            q_id = entry.get("question_id")
            q_data = questions_by_id.get(q_id, {})
            ref = q_data.get("expected_answer_hint")
            if not ref:
                continue
            predictions.append(answer)
            references.append(ref)
            indices.append(i)

        if not predictions:
            logger.info("No question/answer pairs eligible for generation metrics")
            return

        if not self._config.compute_rouge and not self._config.compute_bert_score:
            logger.info("Both compute_rouge and compute_bert_score are False — skipping")
            return

        rouge_results: Optional[List[dict]] = None
        bert_results: Optional[List[dict]] = None

        if self._config.compute_rouge:
            rouge_results = self._compute_rouge_batch(predictions, references)

        if self._config.compute_bert_score:
            bert_results = self._compute_bertscore_batch(predictions, references)

        self._generation_scores = []
        for j, idx in enumerate(indices):
            gm = GenerationMetrics()
            if rouge_results is not None:
                gm.rouge_l_f1 = rouge_results[j]["f1"]
                gm.rouge_l_precision = rouge_results[j]["precision"]
                gm.rouge_l_recall = rouge_results[j]["recall"]
            if bert_results is not None:
                gm.bert_score_f1 = bert_results[j]["f1"]
                gm.bert_score_precision = bert_results[j]["precision"]
                gm.bert_score_recall = bert_results[j]["recall"]
            per_question[idx]["generation_metrics"] = gm.to_dict()
            self._generation_scores.append(gm)

    def compute_latency_metrics(self, per_question: List[dict]) -> LatencyMetrics:
        """Compute p50/p95/p99 latency from per-question timing fields."""
        retrieval_times = [
            e["retrieval_time_ms"]
            for e in per_question
            if "retrieval_time_ms" in e and e["retrieval_time_ms"] is not None
        ]
        generation_times = [
            e["generation_time_ms"]
            for e in per_question
            if "generation_time_ms" in e and e["generation_time_ms"] is not None
        ]

        def _percentiles(values):
            if not values:
                return 0.0, 0.0, 0.0
            arr = np.array(values, dtype=float)
            return (
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 95)),
                float(np.percentile(arr, 99)),
            )

        r50, r95, r99 = _percentiles(retrieval_times)
        g50, g95, g99 = _percentiles(generation_times)

        self._latency = LatencyMetrics(
            retrieval_p50_ms=r50,
            retrieval_p95_ms=r95,
            retrieval_p99_ms=r99,
            generation_p50_ms=g50,
            generation_p95_ms=g95,
            generation_p99_ms=g99,
        )
        return self._latency

    def compute_ragas_metrics(
        self,
        per_question: List[dict],
        questions_by_id: Dict[str, dict],
    ) -> None:
        """Compute RAGAS metrics if the ``compute_ragas`` flag is enabled.

        Delegates to ``RagasEvaluator.evaluate()`` which mutates *per_question*
        in-place (adds ``"ragas_metrics"`` key to eligible entries).
        """
        if not self._config.compute_ragas:
            logger.info("compute_ragas is False — skipping RAGAS evaluation")
            return

        self._ensure_ragas_evaluator()
        self._ragas_evaluator.evaluate(per_question, questions_by_id)

    def get_summary(self) -> dict:
        """Return ``{"latency": {...}, "generation": {...}, "ragas": {...}}`` for serialization."""
        summary: dict = {}

        if self._latency is not None:
            summary["latency"] = self._latency.to_dict()

        if self._generation_scores:
            rouge_f1s = [
                g.rouge_l_f1 for g in self._generation_scores if g.rouge_l_f1 is not None
            ]
            bert_f1s = [
                g.bert_score_f1 for g in self._generation_scores if g.bert_score_f1 is not None
            ]
            agg = AggregatedGenerationMetrics(
                num_questions_scored=len(self._generation_scores),
            )
            if rouge_f1s:
                agg.avg_rouge_l_f1 = float(np.mean(rouge_f1s))
            if bert_f1s:
                agg.avg_bert_score_f1 = float(np.mean(bert_f1s))
            summary["generation"] = agg.to_dict()

        if self._ragas_evaluator is not None:
            ragas_agg = self._ragas_evaluator.get_aggregated()
            if ragas_agg:
                summary["ragas"] = ragas_agg

        return summary

    # ------------------------------------------------------------------
    # Lazy scorer initialisation
    # ------------------------------------------------------------------

    def _ensure_ragas_evaluator(self):
        """Lazy-import and cache ``RagasEvaluator``."""
        if self._ragas_evaluator is not None:
            return
        try:
            from src.evaluation.ragas_evaluator import RagasEvaluator
        except ImportError as exc:
            raise ImportError(
                "ragas is required for RAGAS metrics. "
                "Install with: uv pip install ragas --python .venv/bin/python"
            ) from exc
        self._ragas_evaluator = RagasEvaluator(self._config)
        logger.debug("RagasEvaluator initialised")

    def _ensure_rouge_scorer(self):
        """Lazy-import and cache ``rouge_score.rouge_scorer.RougeScorer``."""
        if self._rouge_scorer is not None:
            return
        try:
            from rouge_score.rouge_scorer import RougeScorer
        except ImportError as exc:
            raise ImportError(
                "rouge-score is required for ROUGE metrics. "
                "Install with: uv pip install rouge-score --python .venv/bin/python"
            ) from exc
        self._rouge_scorer = RougeScorer(["rougeL"], use_stemmer=True)
        logger.debug("RougeScorer initialised")

    def _ensure_bert_scorer(self):
        """Lazy-import and cache ``bert_score.BERTScorer``."""
        if self._bert_scorer is not None:
            return
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ImportError(
                "bert-score is required for BERTScore metrics. "
                "Install with: uv pip install bert-score --python .venv/bin/python"
            ) from exc
        self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        logger.debug("BERTScorer initialised")

    # ------------------------------------------------------------------
    # Batch computation helpers
    # ------------------------------------------------------------------

    def _compute_rouge_batch(
        self, predictions: List[str], references: List[str]
    ) -> List[dict]:
        """Per-pair ROUGE-L computation (CPU-only, fast)."""
        self._ensure_rouge_scorer()
        results = []
        for pred, ref in zip(predictions, references):
            scores = self._rouge_scorer.score(ref, pred)
            rl = scores["rougeL"]
            results.append({
                "f1": float(rl.fmeasure),
                "precision": float(rl.precision),
                "recall": float(rl.recall),
            })
        return results

    def _compute_bertscore_batch(
        self, predictions: List[str], references: List[str]
    ) -> List[dict]:
        """Single vectorized ``BERTScorer.score()`` call."""
        self._ensure_bert_scorer()
        P, R, F1 = self._bert_scorer.score(predictions, references)
        results = []
        for i in range(len(predictions)):
            results.append({
                "f1": float(F1[i]),
                "precision": float(P[i]),
                "recall": float(R[i]),
            })
        return results
