"""
src.benchmarks — public API for parameterized benchmark runs.

Exposes:
    BenchmarkConfig + sub-models  (E1-F1-T1, E2-F2-T1)
    BenchmarkResult               (E1-F1-T2)
    ParameterizedBenchmarkRunner  (E1-F1-T2)
    BaseReranker, RerankerFactory (E2-F3-T2)
"""

from src.benchmarks.config import (
    BenchmarkConfig,
    ChunkingConfig,
    DatasetConfig,
    EmbeddingConfig,
    EvaluationConfig,
    GenerationConfig,
    RerankerConfig,
    RetrievalConfig,
    VectorDBConfig,
)
from src.benchmarks.dataset_manager import DATASET_REGISTRY, DatasetManager
from src.benchmarks.runner import BenchmarkResult, ParameterizedBenchmarkRunner
from src.evaluation.metrics_collector import (
    GenerationMetrics,
    LatencyMetrics,
    MetricsCollector,
)
from src.evaluation.ragas_evaluator import RagasEvaluator, RagasResult
from src.retrieval.reranker import BaseReranker
from src.retrieval.reranker_factory import RerankerFactory

__all__ = [
    # Config
    "BenchmarkConfig",
    "DatasetConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "RetrievalConfig",
    "RerankerConfig",
    "GenerationConfig",
    "EvaluationConfig",
    "VectorDBConfig",
    # Dataset
    "DatasetManager",
    "DATASET_REGISTRY",
    # Runner
    "BenchmarkResult",
    "ParameterizedBenchmarkRunner",
    # Metrics
    "GenerationMetrics",
    "LatencyMetrics",
    "MetricsCollector",
    # RAGAS
    "RagasEvaluator",
    "RagasResult",
    # Reranker (E2-F3-T2)
    "BaseReranker",
    "RerankerFactory",
]
