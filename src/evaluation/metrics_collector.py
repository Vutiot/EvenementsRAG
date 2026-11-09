"""
MetricsCollector — latency percentiles, answer correctness (two judges),
cost per question, and the optional RAGAS / entity metrics.

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
        self._correctness: Optional[dict] = None
        self._cost: Optional[dict] = None

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

        When ``ragas_repeat_count > 1``, the evaluator runs N times and the
        per-question scores are averaged.  Standard deviations are stored in a
        ``"ragas_metrics_std"`` key on each entry.  When ``ragas_repeat_count``
        is 1 (default) the behavior is unchanged — no std is computed.
        """
        if not self._config.compute_ragas:
            logger.info("compute_ragas is False — skipping RAGAS evaluation")
            return

        repeat = self._config.ragas_repeat_count

        if repeat <= 1:
            # Single run — original behavior
            self._ensure_ragas_evaluator()
            self._ragas_evaluator.evaluate(per_question, questions_by_id)
            return

        # --- Multi-run averaging ---
        # Collect per-question RAGAS scores from each run.
        # Each element is a list of dicts (one per question) from that run.
        all_run_scores: List[List[dict]] = []

        for run_idx in range(repeat):
            logger.info("RAGAS repeat run %d/%d", run_idx + 1, repeat)
            # Create a fresh evaluator for each run so internal state is clean
            self._ragas_evaluator = None
            self._ensure_ragas_evaluator()

            # Work on a shallow copy so each run writes its own ragas_metrics
            shadow = [{**entry} for entry in per_question]
            self._ragas_evaluator.evaluate(shadow, questions_by_id)

            run_scores = [entry.get("ragas_metrics", {}) for entry in shadow]
            all_run_scores.append(run_scores)

        # Average across runs for each question
        n_questions = len(per_question)
        for q_idx in range(n_questions):
            # Gather all metric names seen for this question across runs
            metric_names: set = set()
            for run_scores in all_run_scores:
                metric_names.update(run_scores[q_idx].keys())

            if not metric_names:
                continue

            averaged: Dict[str, float] = {}
            stds: Dict[str, float] = {}

            for metric in sorted(metric_names):
                values = []
                for run_scores in all_run_scores:
                    val = run_scores[q_idx].get(metric)
                    if val is not None:
                        values.append(float(val))
                if values:
                    arr = np.array(values)
                    averaged[metric] = float(np.mean(arr))
                    stds[metric] = float(np.std(arr))

            per_question[q_idx]["ragas_metrics"] = averaged
            per_question[q_idx]["ragas_metrics_std"] = stds

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

    def compute_answer_correctness(
        self,
        per_question: List[dict],
        questions_by_id: Dict[str, dict],
        llm_judge=None,
        jev_judge=None,
    ) -> None:
        """Binary answer correctness by the LLM judge and Jev (in-place, E10-F2).

        Adds ``entry["correctness"]`` = ``{"llm": {...}, "jev": {...}, "agree": bool}``
        and the 0/1 scores ``answer_correctness`` / ``answer_correctness_jev`` to
        ``entry["metrics"]`` so per-question views can read them like any metric.
        Questions without a reference answer are skipped. Failed or empty
        generations score 0 for both judges without an API call.
        """
        if not self._config.compute_answer_correctness:
            logger.info("compute_answer_correctness is False — skipping")
            return

        from concurrent.futures import ThreadPoolExecutor

        from src.evaluation.correctness_judge import (
            JevJudge,
            JudgeVerdict,
            LLMCorrectnessJudge,
            gold_answer,
            is_failed_generation,
            judge_agreement,
        )

        cfg = self._config
        if llm_judge is None:
            llm_judge = LLMCorrectnessJudge(
                model=cfg.correctness_judge_model, provider=cfg.correctness_judge_provider
            )
        if jev_judge is None and cfg.use_jev_judge:
            jev_judge = JevJudge(
                threshold=cfg.jev_threshold,
                snapshot=cfg.jev_snapshot,
                price_per_mtok=cfg.jev_input_price_per_mtok,
            )

        def _score(entry: dict) -> None:
            q = questions_by_id.get(entry.get("question_id"))
            if not q or not gold_answer(q):
                return
            answer = entry.get("generated_answer")
            if is_failed_generation(answer):
                verdicts = {
                    "llm": JudgeVerdict(judge="llm", correct=0.0, error="no answer generated"),
                }
                if jev_judge is not None:
                    verdicts["jev"] = JudgeVerdict(judge="jev", correct=0.0, error="no answer generated")
            else:
                verdicts = {"llm": llm_judge.judge(q, answer)}
                if jev_judge is not None:
                    verdicts["jev"] = jev_judge.judge(q, answer)

            block = {name: v.to_dict() for name, v in verdicts.items()}
            llm_v, jev_v = verdicts["llm"].correct, verdicts.get("jev") and verdicts["jev"].correct
            if llm_v is not None and jev_v is not None:
                block["agree"] = llm_v == jev_v
            entry["correctness"] = block
            entry["judge_cost_usd"] = sum(v.cost_usd or 0.0 for v in verdicts.values())

            metrics = entry.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
                entry["metrics"] = metrics
            if llm_v is not None:
                metrics["answer_correctness"] = llm_v
            if jev_v is not None:
                metrics["answer_correctness_jev"] = jev_v

        with ThreadPoolExecutor(max_workers=cfg.judge_max_workers) as pool:
            list(pool.map(_score, per_question))

        scored = [e for e in per_question if "correctness" in e]
        llm_scores = [e["correctness"].get("llm", {}).get("correct") for e in scored]
        jev_scores = [e["correctness"].get("jev", {}).get("correct") for e in scored]

        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        summary = {
            "num_questions_scored": len(scored),
            "judge_model": cfg.correctness_judge_model,
            "answer_correctness": _mean(llm_scores),
            "llm_judge_errors": sum(1 for v in llm_scores if v is None),
        }
        if jev_judge is not None:
            agreement = judge_agreement(llm_scores, jev_scores)
            summary.update({
                "jev_snapshot": cfg.jev_snapshot,
                "jev_threshold": getattr(jev_judge, "threshold", cfg.jev_threshold),
                "answer_correctness_jev": _mean(jev_scores),
                "jev_errors": sum(1 for v in jev_scores if v is None),
                "judge_agreement": agreement.get("agreement"),
                "judge_kappa": agreement.get("kappa"),
                "agreement_detail": agreement,
            })
        self._correctness = summary
        logger.info(
            "Answer correctness: llm=%s jev=%s agreement=%s (%d questions)",
            summary.get("answer_correctness"),
            summary.get("answer_correctness_jev"),
            summary.get("judge_agreement"),
            len(scored),
        )

    def compute_cost_metrics(self, per_question: List[dict]) -> dict:
        """Per-question cost (generation + judges, USD) and end-to-end latency.

        Sets ``entry["cost_usd"]`` and ``entry["latency_ms"]`` on each entry and
        returns the summary stored under ``"cost"``.
        """
        costs: List[float] = []
        gen_costs: List[float] = []
        judge_costs: List[float] = []
        latencies: List[float] = []
        missing_cost = 0
        for entry in per_question:
            gen = (entry.get("generation_usage") or {}).get("cost_usd")
            judge = entry.get("judge_cost_usd")
            if gen is None and entry.get("generated_answer"):
                missing_cost += 1
            total = (gen or 0.0) + (judge or 0.0)
            entry["cost_usd"] = total
            costs.append(total)
            gen_costs.append(gen or 0.0)
            judge_costs.append(judge or 0.0)

            lat = (entry.get("retrieval_time_ms") or 0.0) + (entry.get("generation_time_ms") or 0.0)
            entry["latency_ms"] = lat
            latencies.append(lat)

        n = len(per_question)
        self._cost = {
            "cost_per_question_usd": float(np.mean(costs)) if n else 0.0,
            "generation_cost_per_question_usd": float(np.mean(gen_costs)) if n else 0.0,
            "judge_cost_per_question_usd": float(np.mean(judge_costs)) if n else 0.0,
            "total_cost_usd": float(np.sum(costs)) if n else 0.0,
            "latency_per_question_ms": float(np.mean(latencies)) if n else 0.0,
            "questions_without_reported_cost": missing_cost,
        }
        return self._cost

    @staticmethod
    def summarize_hits(per_question: List[dict]) -> dict:
        """Average the hit@K / hit@C values stored in each entry's metrics."""
        sums: Dict[str, List[float]] = {}
        for entry in per_question:
            for key, val in (entry.get("metrics") or {}).items():
                if key.startswith("hit_at_") and isinstance(val, (int, float)):
                    sums.setdefault(key, []).append(float(val))
        return {key: float(np.mean(vals)) for key, vals in sorted(sums.items())}

    def get_summary(self) -> dict:
        """Return ``{"latency", "ragas", "entity", "correctness", "cost"}`` for serialization."""
        summary: dict = {}

        if self._latency is not None:
            summary["latency"] = self._latency.to_dict()

        if self._correctness is not None:
            summary["correctness"] = self._correctness

        if self._cost is not None:
            summary["cost"] = self._cost

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


# ---------------------------------------------------------------------------
# Core metric set (E10)
# ---------------------------------------------------------------------------

CORE_CORRECTNESS_KEYS = ("answer_correctness", "answer_correctness_jev", "judge_agreement", "judge_kappa")
CORE_COST_KEYS = ("cost_per_question_usd", "latency_per_question_ms")


def core_metrics_summary(metrics_summary: Optional[dict]) -> dict:
    """Flatten the core metrics of a run: hit@5/10/C, both judges, cost and latency."""
    ms = metrics_summary or {}
    out: dict = {}
    for key, val in (ms.get("hits") or {}).items():
        out[key] = round(val, 4)
    corr = ms.get("correctness") or {}
    for key in CORE_CORRECTNESS_KEYS:
        if corr.get(key) is not None:
            out[key] = round(corr[key], 4)
    cost = ms.get("cost") or {}
    for key in CORE_COST_KEYS:
        if cost.get(key) is not None:
            out[key] = round(cost[key], 6)
    return out
