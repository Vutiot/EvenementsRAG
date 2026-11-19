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
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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
        from ragas.metrics._answer_correctness import AnswerCorrectness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._answer_similarity import AnswerSimilarity
        from ragas.metrics._aspect_critic import AspectCritic
        from ragas.metrics._context_entities_recall import ContextEntityRecall
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness
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


def _resolve_evaluator_provider(model: str) -> tuple[str, str, str]:
    """Return (api_key, api_base, provider_label) for an evaluator model name.

    Routing rules (allowlist-first so past NVIDIA-hosted runs stay reproducible):
      - ``gpt-*``         → OpenAI
      - model in ``NVIDIA_MODELS`` allowlist → NVIDIA
      - looks like ``vendor/name``   → OpenRouter
      - otherwise: ValueError with supported prefixes
    """
    from config.settings import settings
    from src.benchmarks.config import NVIDIA_MODELS

    if model.startswith("gpt-"):
        return settings.OPENAI_API_KEY, "https://api.openai.com/v1", "openai"
    if model in NVIDIA_MODELS:
        return settings.NVIDIA_API_KEY, settings.NVIDIA_BASE_URL, "nvidia"
    if "/" in model:
        key = settings.OPENROUTER_API_KEY or settings.GEMMA_API_KEY
        return key, "https://openrouter.ai/api/v1", "openrouter"
    raise ValueError(
        f"Cannot resolve provider for RAGAS evaluator model '{model}'. "
        "Use 'gpt-*' (OpenAI), a model in NVIDIA_MODELS (NVIDIA), "
        "or a 'vendor/name' identifier (OpenRouter)."
    )


def _build_evaluator_llm(eval_config: EvaluationConfig, usage_tracker=None):
    """Create a RAGAS v0.4 InstructorLLM for evaluation, routing by model prefix.

    Uses ``ragas.llms.llm_factory`` which wraps an openai-compatible client
    with the Instructor adapter. Instructor auto-handles GPT-5/o-series
    constraints (temperature bypass, n bypass) and returns validated
    Pydantic-shaped outputs — no manual ChatOpenAI subclassing required.
    """
    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise ImportError(
            "ragas>=0.4 and openai are required for RAGAS LLM evaluation. "
            "Install with: uv pip install 'ragas>=0.4' openai --python .venv/bin/python"
        ) from exc

    model = eval_config.ragas_evaluator_model
    api_key, api_base, provider = _resolve_evaluator_provider(model)
    if not api_key:
        raise ValueError(
            f"RAGAS evaluator model '{model}' resolves to provider '{provider}' "
            f"but the matching API key is empty. Set it in .env."
        )

    logger.info(f"RAGAS evaluator LLM: provider={provider}, model={model}")

    client = OpenAI(api_key=api_key, base_url=api_base)
    if usage_tracker is not None:
        from src.llm.usage import instrument_client

        # Private client: every RAGAS/Instructor call (incl. retries) is counted
        instrument_client(client, usage_tracker, "ragas", model)
    # RAGAS v0.4 guidance: reasoning models with structured output need ≥4096
    # to avoid truncated JSON. 4096 is safe for non-reasoning too.
    factory_kwargs: dict = {"max_tokens": 4096}
    if model.startswith("gpt-5") or model.startswith("o"):
        factory_kwargs["reasoning_effort"] = eval_config.ragas_evaluator_reasoning_effort

    return llm_factory(model, client=client, **factory_kwargs)


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

    def __init__(self, eval_config: EvaluationConfig, usage_tracker=None) -> None:
        self._config = eval_config
        self._usage_tracker = usage_tracker
        self._llm = None
        self._embeddings = None
        self._run_config = None
        self._results: List[RagasResult] = []

    def evaluate(
        self,
        per_question: List[dict],
        questions_by_id: Dict[str, dict],
        progress_callback: Optional[Callable[[str, dict], None]] = None,
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

        logger.info(
            f"Running RAGAS evaluation on {len(samples)} questions with metrics: {self._config.ragas_metrics}",
        )

        # Build deps lazily
        if self._llm is None:
            self._llm = _build_evaluator_llm(self._config, self._usage_tracker)
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

        if progress_callback:
            progress_callback("ragas_eval_start", {
                "num_questions": len(samples),
                "num_metrics": len(metrics),
            })

        logger.info(
            f"RAGAS LLM call: model={self._config.ragas_evaluator_model}, "
            f"{len(metrics)} metrics × {len(samples)} questions, "
            f"timeout={self._config.ragas_timeout}s, max_workers={self._config.ragas_max_workers}",
        )

        # Track RAGAS progress by intercepting its tqdm updates
        _ragas_progress = {"current": 0, "total": len(samples)}
        stop_heartbeat = threading.Event()

        def _heartbeat():
            tick = 0
            while not stop_heartbeat.wait(15):
                tick += 1
                if progress_callback:
                    progress_callback("ragas_heartbeat", {
                        "elapsed_s": tick * 15,
                        "current": _ragas_progress["current"],
                        "total": _ragas_progress["total"],
                    })

        hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        hb_thread.start()

        # Monkey-patch tqdm to capture RAGAS's internal progress.
        # ragas 0.4 imports `from tqdm.auto import tqdm` at module load in
        # several places — most importantly ragas.utils.ProgressBarManager,
        # which is what actually constructs the bars (ragas.evaluation and
        # ragas.executor only hold tqdm as a type alias). Patching top-level
        # tqdm.tqdm does NOT affect those already-bound references; we must
        # rebind the `tqdm` name inside each ragas module that uses it.
        import ragas.evaluation as _ragas_eval
        import ragas.executor as _ragas_exec
        import ragas.utils as _ragas_utils
        from tqdm.auto import tqdm as _auto_tqdm

        class _ProgressTqdm(_auto_tqdm):
            def update(self, n=1):
                super().update(n)
                _ragas_progress["current"] = int(self.n)
                if self.total:
                    _ragas_progress["total"] = int(self.total)

        _orig_eval_tqdm = _ragas_eval.tqdm
        _orig_exec_tqdm = _ragas_exec.tqdm
        _orig_utils_tqdm = _ragas_utils.tqdm
        _ragas_eval.tqdm = _ProgressTqdm
        _ragas_exec.tqdm = _ProgressTqdm
        _ragas_utils.tqdm = _ProgressTqdm

        try:
            eval_result = ragas_evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self._llm,
                embeddings=self._embeddings,
                run_config=self._run_config,
                raise_exceptions=False,
            )
        finally:
            _ragas_eval.tqdm = _orig_eval_tqdm
            _ragas_exec.tqdm = _orig_exec_tqdm
            _ragas_utils.tqdm = _orig_utils_tqdm
            stop_heartbeat.set()
            hb_thread.join(timeout=2)

        logger.info(f"RAGAS LLM evaluation returned {len(eval_result.to_pandas())} rows")

        if progress_callback:
            progress_callback("ragas_eval_complete", {
                "num_questions": len(samples),
                "num_scored": len(eligible_indices),
            })

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
