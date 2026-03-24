"""
MetricsCollector — latency percentiles, RAGAS, and entity metrics.

Complements the retrieval metrics computed by the legacy BenchmarkRunner.
RAGAS and entity scorers are lazy-imported so the module stays light when
the flags are disabled.

Usage:
    from src.evaluation.metrics_collector import MetricsCollector
    from src.benchmarks.config import EvaluationConfig

    collector = MetricsCollector(EvaluationConfig(compute_ragas=True))
    collector.compute_ragas_metrics(per_question, questions_by_id)
    latency = collector.compute_latency_metrics(per_question)
    summary = collector.get_summary()
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.benchmarks.config import EvaluationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Computes latency, RAGAS, and entity metrics on top of retrieval results.

    Does **not** wrap ``compute_retrieval_metrics()`` — retrieval metrics stay
    in the legacy ``BenchmarkRunner``.  This collector only adds LLM-based
    evaluation + latency on top.
    """

    def __init__(self, eval_config: EvaluationConfig) -> None:
        self._config = eval_config
        self._ragas_evaluator = None
        self._entity_extractor = None
        self._entity_scores: Optional[dict] = None
        self._latency: Optional[LatencyMetrics] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def compute_context_precision_only(
        self,
        per_question: List[dict],
        questions_by_id: Dict[str, dict],
    ) -> None:
        """Run only ``context_precision`` from RAGAS (lightweight alternative).

        Called when ``compute_context_precision`` is True but ``compute_ragas``
        is False, so the full RAGAS suite is not needed.
        """
        self._ensure_ragas_evaluator_for_context_precision()
        self._ragas_evaluator.evaluate(per_question, questions_by_id)

    def compute_entity_metrics(
        self,
        per_question: List[dict],
        questions_by_id: Dict[str, dict],
    ) -> None:
        """Compute entity precision/recall/MRR per question (in-place mutation).

        Adds ``"entity_metrics"`` key to eligible entries. Entries without
        ``retrieved_contexts`` or ``expected_answer_hint`` are skipped.
        """
        if not self._config.compute_entity_metrics:
            logger.info("compute_entity_metrics is False — skipping")
            return

        self._ensure_entity_extractor()

        from src.evaluation.metrics import (
            entity_precision_at_k,
            entity_recall_at_k,
            entity_mrr_score,
        )

        precisions: List[float] = []
        recalls: List[float] = []
        mrrs: List[float] = []

        for entry in per_question:
            q_id = entry.get("question_id")
            q_data = questions_by_id.get(q_id, {})
            ground_truth = q_data.get("expected_answer_hint")
            retrieved_texts = entry.get("retrieved_contexts")

            if not ground_truth or not retrieved_texts:
                continue

            ep = entity_precision_at_k(retrieved_texts, ground_truth, k=5, extractor=self._entity_extractor)
            er = entity_recall_at_k(retrieved_texts, ground_truth, k=5, extractor=self._entity_extractor)
            em = entity_mrr_score(retrieved_texts, ground_truth, extractor=self._entity_extractor)

            entry["entity_metrics"] = {
                "entity_precision_at_5": ep,
                "entity_recall_at_5": er,
                "entity_mrr": em,
            }
            precisions.append(ep)
            recalls.append(er)
            mrrs.append(em)

        if precisions:
            self._entity_scores = {
                "avg_entity_precision_at_5": float(np.mean(precisions)),
                "avg_entity_recall_at_5": float(np.mean(recalls)),
                "avg_entity_mrr": float(np.mean(mrrs)),
                "num_questions_scored": len(precisions),
            }
            logger.info(
                "Entity metrics computed for %d questions: P@5=%.3f, R@5=%.3f, MRR=%.3f",
                len(precisions),
                self._entity_scores["avg_entity_precision_at_5"],
                self._entity_scores["avg_entity_recall_at_5"],
                self._entity_scores["avg_entity_mrr"],
            )
        else:
            logger.info("No entries eligible for entity metrics")

    def get_summary(self) -> dict:
        """Return ``{"latency": {...}, "ragas": {...}, "entity": {...}}`` for serialization."""
        summary: dict = {}

        if self._latency is not None:
            summary["latency"] = self._latency.to_dict()

        if self._ragas_evaluator is not None:
            ragas_agg = self._ragas_evaluator.get_aggregated()
            if ragas_agg:
                summary["ragas"] = ragas_agg

        if self._entity_scores is not None:
            summary["entity"] = self._entity_scores

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

    def _ensure_ragas_evaluator_for_context_precision(self):
        """Lazy-import ``RagasEvaluator`` configured for context_precision only."""
        if self._ragas_evaluator is not None:
            return
        try:
            from src.evaluation.ragas_evaluator import RagasEvaluator
        except ImportError as exc:
            raise ImportError(
                "ragas is required for context_precision. "
                "Install with: uv pip install ragas --python .venv/bin/python"
            ) from exc
        cfg = self._config.model_copy(update={"ragas_metrics": ["context_precision"]})
        self._ragas_evaluator = RagasEvaluator(cfg)
        logger.debug("RagasEvaluator initialised (context_precision only)")

    def _ensure_entity_extractor(self):
        """Lazy-import and cache ``EntityExtractor``."""
        if self._entity_extractor is not None:
            return
        from src.evaluation.entity_extractor import EntityExtractor
        self._entity_extractor = EntityExtractor()
        logger.debug("EntityExtractor initialised")
