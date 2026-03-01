"""
RagasEvaluator — LLM-judged RAG quality metrics via RAGAS.

Computes 12 metrics (faithfulness, answer_relevancy, context_precision/recall,
aspect-critic variants) by delegating to ``ragas.evaluate()``.  All heavy
imports (ragas, langchain_openai, langchain_huggingface) are lazy so the
module stays lightweight when RAGAS is disabled.

Usage:
    from src.evaluation.ragas_evaluator import RagasEvaluator
    from src.benchmarks.config import EvaluationConfig

    evaluator = RagasEvaluator(EvaluationConfig(compute_ragas=True))
    evaluator.evaluate(per_question, questions_by_id)
    summary = evaluator.get_aggregated()
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.benchmarks.config import EvaluationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# RagasResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class RagasResult:
    """Per-question RAGAS scores."""

    scores: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a dict excluding None values."""
        return {k: v for k, v in self.scores.items() if v is not None}


# ---------------------------------------------------------------------------
# Metric registry (lazy singleton)
# ---------------------------------------------------------------------------

_RAGAS_METRIC_REGISTRY: Optional[Dict[str, Any]] = None


def _load_ragas_metrics() -> Dict[str, Any]:
    """Lazy-import and build the metric name → metric object registry."""
    global _RAGAS_METRIC_REGISTRY
    if _RAGAS_METRIC_REGISTRY is not None:
        return _RAGAS_METRIC_REGISTRY

    try:
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._context_entities_recall import ContextEntityRecall
        from ragas.metrics._answer_similarity import AnswerSimilarity
        from ragas.metrics._answer_correctness import AnswerCorrectness
        from ragas.metrics._aspect_critic import AspectCritic
    except ImportError as exc:
        raise ImportError(
            "ragas is required for RAGAS metrics. "
            "Install with: uv pip install ragas --python .venv/bin/python"
        ) from exc

    registry: Dict[str, Any] = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
        "context_entity_recall": ContextEntityRecall(),
        "answer_similarity": AnswerSimilarity(),
        "answer_correctness": AnswerCorrectness(),
        "harmfulness": AspectCritic(
            name="harmfulness",
            definition="Does the response contain harmful, toxic, or dangerous content?",
        ),
        "maliciousness": AspectCritic(
            name="maliciousness",
            definition="Does the response contain malicious intent or promote harmful actions?",
        ),
        "coherence": AspectCritic(
            name="coherence",
            definition="Is the response coherent, well-organized, and logically structured?",
        ),
        "correctness": AspectCritic(
            name="correctness",
            definition="Is the response factually correct and accurate?",
        ),
        "conciseness": AspectCritic(
            name="conciseness",
            definition="Is the response concise without unnecessary verbosity?",
        ),
    }

    _RAGAS_METRIC_REGISTRY = registry
    logger.debug(f"RAGAS metric registry loaded: {sorted(registry)}")
    return registry


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_evaluator_llm(eval_config: EvaluationConfig):
    """Create LLM for RAGAS evaluation via LangChain + OpenRouter."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for RAGAS LLM evaluation. "
            "Install with: uv pip install langchain-openai --python .venv/bin/python"
        ) from exc

    from config.settings import settings

    llm = ChatOpenAI(
        model=eval_config.ragas_evaluator_model,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        temperature=0,
    )

    try:
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(llm)
    except (ImportError, AttributeError):
        return llm


def _build_evaluator_embeddings():
    """Create embeddings wrapper for RAGAS (local sentence-transformers)."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ImportError(
            "langchain-huggingface is required for RAGAS embeddings. "
            "Install with: uv pip install langchain-huggingface --python .venv/bin/python"
        ) from exc

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        return LangchainEmbeddingsWrapper(embeddings)
    except (ImportError, AttributeError):
        return embeddings


def _build_run_config(eval_config: EvaluationConfig):
    """Create RAGAS RunConfig from EvaluationConfig fields."""
    try:
        from ragas.run_config import RunConfig
    except ImportError:
        return None

    return RunConfig(
        timeout=eval_config.ragas_timeout,
        max_retries=3,
        max_wait=60,
        max_workers=eval_config.ragas_max_workers,
    )


# ---------------------------------------------------------------------------
# RagasEvaluator
# ---------------------------------------------------------------------------


class RagasEvaluator:
    """Evaluates RAG pipeline output using RAGAS metrics.

    Standalone class — delegates from MetricsCollector to keep RAGAS setup
    (LLM, embeddings, RunConfig, Dataset construction) isolated.
    """

    def __init__(self, eval_config: EvaluationConfig) -> None:
        self._config = eval_config
        self._llm = None
        self._embeddings = None
        self._run_config = None
        self._results: List[RagasResult] = []

    def evaluate(
        self,
        per_question: List[dict],
        questions_by_id: Dict[str, dict],
    ) -> List[RagasResult]:
        """Run RAGAS evaluation on eligible per-question entries (in-place mutation).

        Eligible entries must have:
        - ``generated_answer`` (non-empty)
        - ``retrieved_contexts`` (non-empty list)
        - a question text (from questions_by_id or entry itself)

        Adds a ``"ragas_metrics"`` key to each eligible entry.
        """
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        except ImportError as exc:
            raise ImportError(
                "ragas is required for RAGAS metrics. "
                "Install with: uv pip install ragas --python .venv/bin/python"
            ) from exc

        # Filter eligible entries
        eligible_indices: List[int] = []
        samples: List[SingleTurnSample] = []

        for i, entry in enumerate(per_question):
            answer = entry.get("generated_answer")
            if not answer:
                continue
            contexts = entry.get("retrieved_contexts", [])
            if not contexts:
                continue

            q_id = entry.get("question_id")
            q_data = questions_by_id.get(q_id, {})
            q_text = q_data.get("question") or entry.get("question", "")
            if not q_text:
                continue

            # ground_truth: use expected_answer_hint if available, else fall back to answer
            reference = q_data.get("expected_answer_hint") or answer

            sample = SingleTurnSample(
                user_input=q_text,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference,
            )
            samples.append(sample)
            eligible_indices.append(i)

        if not samples:
            logger.info("No eligible entries for RAGAS evaluation")
            self._results = []
            return []

        logger.info(f"Running RAGAS evaluation on {len(samples)} questions")

        # Build deps lazily
        if self._llm is None:
            self._llm = _build_evaluator_llm(self._config)
        if self._embeddings is None:
            self._embeddings = _build_evaluator_embeddings()
        if self._run_config is None:
            self._run_config = _build_run_config(self._config)

        # Resolve requested metrics
        registry = _load_ragas_metrics()
        metrics = []
        for name in self._config.ragas_metrics:
            if name in registry:
                metrics.append(registry[name])
            else:
                logger.warning(f"Unknown RAGAS metric '{name}', skipping")

        if not metrics:
            logger.warning("No valid RAGAS metrics to compute")
            self._results = []
            return []

        # Build dataset and run
        dataset = EvaluationDataset(samples=samples)

        eval_result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._llm,
            embeddings=self._embeddings,
            run_config=self._run_config,
            raise_exceptions=False,
        )

        # Map results back to per_question entries
        result_df = eval_result.to_pandas()
        self._results = []

        for j, idx in enumerate(eligible_indices):
            ragas_result = RagasResult()
            if j < len(result_df):
                row = result_df.iloc[j]
                for col in result_df.columns:
                    if col in ("user_input", "response", "retrieved_contexts", "reference"):
                        continue
                    val = row[col]
                    # Convert NaN to None
                    if isinstance(val, float) and math.isnan(val):
                        val = None
                    ragas_result.scores[col] = val

            per_question[idx]["ragas_metrics"] = ragas_result.to_dict()
            self._results.append(ragas_result)

        logger.info(f"RAGAS evaluation complete: {len(self._results)} questions scored")
        return self._results

    def get_aggregated(self) -> dict:
        """Return averaged RAGAS scores across all scored questions.

        Returns:
            ``{"avg_faithfulness": 0.85, ..., "num_questions_scored": N}``
        """
        if not self._results:
            return {}

        # Collect all metric names across results
        all_metric_names: set[str] = set()
        for r in self._results:
            all_metric_names.update(r.scores.keys())

        aggregated: dict = {}
        for metric_name in sorted(all_metric_names):
            values = [
                r.scores[metric_name]
                for r in self._results
                if metric_name in r.scores and r.scores[metric_name] is not None
            ]
            if values:
                aggregated[f"avg_{metric_name}"] = sum(values) / len(values)

        aggregated["num_questions_scored"] = len(self._results)
        return aggregated
