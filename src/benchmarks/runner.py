"""
ParameterizedBenchmarkRunner — drives evaluation from a BenchmarkConfig.

Delegates retrieval metrics to the existing BenchmarkRunner, adds an optional
LLM generation pass, and supports batch parameter sweeps.

Usage:
    from src.benchmarks import BenchmarkConfig, ParameterizedBenchmarkRunner
    from pathlib import Path

    cfg = BenchmarkConfig.phase1_vanilla()
    runner = ParameterizedBenchmarkRunner(config=cfg)
    result = runner.run(max_questions=3)
    result.print_summary()
    result.to_json(Path("results/test_e1f1t2.json"))
"""

import importlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.benchmarks.config import BenchmarkConfig
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.metrics import EvaluationResults
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _filter_top_k_articles(chunks, k: int):
    """Keep only chunks belonging to the top-k highest-scored unique articles.

    Articles are ranked by the best (highest) score among their chunks.
    """
    seen: list[str] = []
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        title = chunk.article_title
        if title not in seen:
            seen.append(title)
        if len(seen) >= k:
            break
    return [c for c in chunks if c.article_title in seen]


def _save_result(result: "BenchmarkResult", output_dir: Path) -> Path:
    """Save result to {output_dir}/{phase_name}/{config_hash}_{timestamp}.json."""
    subdir = output_dir / result.phase_name
    subdir.mkdir(parents=True, exist_ok=True)
    ts = result.timestamp.replace(":", "").replace("-", "")  # 20260301T120000Z
    path = subdir / f"{result.config_hash}_{ts}.json"
    result.to_json(path)
    logger.info(f"Result saved: {path}", extra={"config_hash": result.config_hash})
    return path


# ---------------------------------------------------------------------------
# RAG technique registry  (lazy import — avoids errors from empty modules)
# ---------------------------------------------------------------------------

_RAG_REGISTRY = {
    "vanilla": "src.rag.phase1_vanilla.retriever.VanillaRetriever",
    "hybrid": "src.rag.phase3_hybrid.retriever.HybridRetriever",
    "temporal": None, # NotImplementedError until E2-F1 (see ROADMAP.md)
}


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Complete result of a single parameterized benchmark run."""

    config: BenchmarkConfig
    config_hash: str
    phase_name: str
    timestamp: str               # ISO-8601 UTC
    evaluation: EvaluationResults
    per_question_full: list      # retrieval + optional generated_answer per question
    total_wall_time_s: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "config": self.config.model_dump(),
            "config_hash": self.config_hash,
            "phase_name": self.phase_name,
            "timestamp": self.timestamp,
            "evaluation": self.evaluation.to_dict(),
            "per_question_full": self.per_question_full,
            "total_wall_time_s": self.total_wall_time_s,
        }

    def to_json(self, path=None) -> str:
        """Serialize to JSON string, optionally writing to *path*."""
        content = json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return content

    def print_summary(self) -> None:
        """Print a concise summary to stdout."""
        r = self.evaluation
        print("\n" + "=" * 70)
        print(f"BENCHMARK RESULT: {self.phase_name}")
        print("=" * 70)
        print(f"Config Hash : {self.config_hash}")
        print(f"Timestamp   : {self.timestamp}")
        print(f"Wall Time   : {self.total_wall_time_s:.1f}s")
        print(f"Questions   : {r.total_questions}")
        print(f"Avg Recall@5: {r.avg_recall_at_k.get(5, 0.0):.3f}")
        print(f"Avg MRR     : {r.avg_mrr:.3f}")
        for k, ndcg in r.avg_ndcg.items():
            print(f"Avg NDCG@{k}  : {ndcg:.3f}")
        has_gen = any(
            entry.get("generated_answer") is not None
            for entry in self.per_question_full
        )
        print(f"Generation  : {'enabled' if has_gen else 'disabled'}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# ParameterizedBenchmarkRunner
# ---------------------------------------------------------------------------


class ParameterizedBenchmarkRunner:
    """Drives evaluation from a BenchmarkConfig.

    Shared vector store and EmbeddingGenerator instances are reused between
    the RAG pipeline and the legacy BenchmarkRunner to avoid double model
    loading during parameter sweeps.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        vector_store=None,
        embedding_generator=None,
        # Backward compatibility alias
        qdrant_manager=None,
    ):
        self.config = config
        # Accept either name; vector_store takes precedence
        self._vector_store = vector_store or qdrant_manager
        self._embedding_gen = embedding_generator
        self._rag_pipeline = None

        logger.info(
            "ParameterizedBenchmarkRunner created",
            extra={"name": config.name, "technique": config.retrieval.technique},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        questions_file: Optional[Path] = None,
        max_questions: Optional[int] = None,
        output_dir=None,
    ) -> BenchmarkResult:
        """Run a full benchmark and return a BenchmarkResult.

        Args:
            questions_file: Override the questions file from config.
            max_questions: Limit evaluation to the first N questions.
            output_dir: Path | str | None; if set, auto-saves result JSON.

        Returns:
            BenchmarkResult with retrieval metrics and optional generated answers.
        """
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.vector_store.factory import VectorStoreFactory

        wall_start = time.time()

        logger.info(
            f"Benchmark run started: {self.config.name}",
            extra={
                "config_hash": self.config.config_hash(),
                "technique": self.config.retrieval.technique,
            },
        )

        questions_path = Path(questions_file or self.config.dataset.questions_file)

        # Initialise shared deps once (avoids double model loading in sweeps)
        if self._vector_store is None:
            self._vector_store = VectorStoreFactory.from_config(self.config.vector_db)
        if self._embedding_gen is None:
            self._embedding_gen = EmbeddingGenerator(model_name=self.config.embedding.model_name)

        # Ensure dataset is indexed before building pipeline
        from src.benchmarks.dataset_manager import DatasetManager
        DatasetManager().ensure_indexed(self.config, self._vector_store)

        # Build RAG pipeline (raises NotImplementedError for unimplemented techniques)
        self._build_rag_pipeline()

        # Delegate retrieval evaluation to the existing BenchmarkRunner
        legacy_runner = BenchmarkRunner(
            questions_file=questions_path,
            qdrant_manager=self._vector_store,
            embedding_generator=self._embedding_gen,
            k_values=self.config.evaluation.k_values,
        )
        eval_results = legacy_runner.run_benchmark(
            collection_name=self.config.dataset.collection_name,
            phase_name=self.config.name,
            max_questions=max_questions,
        )

        # Work on a mutable copy of per-question results
        per_q = [dict(r) for r in eval_results.per_question_metrics]

        # Optional generation pass
        if self.config.generation.enabled:
            self._run_generation_pass(questions_path, per_q, max_questions)

        result = BenchmarkResult(
            config=self.config,
            config_hash=self.config.config_hash(),
            phase_name=self.config.name,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            evaluation=eval_results,
            per_question_full=per_q,
            total_wall_time_s=time.time() - wall_start,
        )

        if output_dir is not None:
            _save_result(result, Path(output_dir))

        logger.info(
            f"Benchmark run complete: {self.config.name}",
            extra={
                "config_hash": result.config_hash,
                "wall_time_s": result.total_wall_time_s,
                "recall_at_5": result.evaluation.avg_recall_at_k.get(5, 0.0),
                "mrr": result.evaluation.avg_mrr,
            },
        )

        return result

    @staticmethod
    def run_sweep(
        configs: List[BenchmarkConfig],
        questions_file: Optional[Path] = None,
        max_questions: Optional[int] = None,
        output_dir: Optional[Path] = None,
        stop_on_error: bool = False,
    ) -> List[BenchmarkResult]:
        """Run multiple benchmark configs in sequence.

        Args:
            configs: List of BenchmarkConfig objects to evaluate.
            questions_file: Override questions file for all configs.
            max_questions: Limit evaluation to first N questions.
            output_dir: Directory to save per-result JSON files.
            stop_on_error: Re-raise errors (including NotImplementedError).
                           Default False skips unimplemented techniques.

        Returns:
            List of BenchmarkResult (one per successfully completed config).
        """
        results = []

        for cfg in configs:
            logger.info(f"Sweep: running '{cfg.name}' [{cfg.config_hash()[:8]}]")
            runner = ParameterizedBenchmarkRunner(config=cfg)
            try:
                result = runner.run(
                    questions_file=questions_file,
                    max_questions=max_questions,
                )
            except NotImplementedError as exc:
                if stop_on_error:
                    raise
                logger.warning(f"Skipping '{cfg.name}': {exc}")
                continue
            except Exception as exc:
                if stop_on_error:
                    raise
                logger.error(f"Error running '{cfg.name}': {exc}", exc_info=True)
                continue

            results.append(result)

            if output_dir is not None:
                output_dir = Path(output_dir)
                filename = f"{cfg.name}_{result.config_hash}.json"
                result.to_json(output_dir / filename)
                logger.info(f"Saved sweep result to {output_dir / filename}")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_rag_pipeline(self) -> None:
        """Instantiate the RAG pipeline for config.retrieval.technique."""
        technique = self.config.retrieval.technique
        target = _RAG_REGISTRY.get(technique)

        if target is None:
            raise NotImplementedError(
                f"RAG technique '{technique}' is not yet implemented. "
                "See ROADMAP.md for the implementation schedule."
            )

        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        extra = {"config": self.config} if technique == "hybrid" else {}
        self._rag_pipeline = cls(
            collection_name=self.config.dataset.collection_name,
            qdrant_manager=self._vector_store,
            embedding_generator=self._embedding_gen,
            prompt_template=self.config.generation.prompt_template,
            **extra,
        )
        logger.debug(f"Built RAG pipeline: {class_name} for technique='{technique}'")

    def _run_generation_pass(
        self,
        questions_path: Path,
        per_q: list,
        max_questions: Optional[int],
    ) -> None:
        """Add generated_answer to each per-question result entry (in-place).

        Uses retrieve() → optional article filter → generate() so that
        temperature, max_tokens, model, and top_k_articles from config are
        all honoured. Failures are logged as warnings and don't abort the run.
        """
        with open(questions_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        if max_questions:
            questions = questions[:max_questions]

        q_by_id = {q.get("id"): q for q in questions}
        top_k_chunks = self.config.generation.top_k_chunks
        top_k_articles = self.config.generation.top_k_articles
        gen_kwargs = dict(
            temperature=self.config.generation.temperature,
            max_tokens=self.config.generation.max_tokens,
            model=self.config.generation.model,
        )

        for entry in per_q:
            q_id = entry.get("question_id")
            q_text = (q_by_id.get(q_id) or {}).get("question") or entry.get("question", "")

            if not q_text:
                logger.warning(f"No question text for question_id={q_id}, skipping generation")
                entry["generated_answer"] = None
                entry["generation_time_ms"] = 0.0
                continue

            try:
                chunks = self._rag_pipeline.retrieve(q_text, top_k=top_k_chunks)
                if top_k_articles is not None:
                    chunks = _filter_top_k_articles(chunks, top_k_articles)
                gen_start = time.time()
                answer = self._rag_pipeline.generate(q_text, chunks, **gen_kwargs)
                entry["generated_answer"] = answer
                entry["generation_time_ms"] = (time.time() - gen_start) * 1000
            except Exception as exc:
                logger.warning(f"Generation failed for question_id={q_id}: {exc}")
                entry["generated_answer"] = None
                entry["generation_time_ms"] = 0.0
