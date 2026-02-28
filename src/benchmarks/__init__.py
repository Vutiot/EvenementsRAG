"""
src.benchmarks — public API for parameterized benchmark runs.

Exposes:
    BenchmarkConfig + sub-models  (E1-F1-T1)
    BenchmarkResult               (E1-F1-T2)
    ParameterizedBenchmarkRunner  (E1-F1-T2)
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
)
from src.benchmarks.runner import BenchmarkResult, ParameterizedBenchmarkRunner

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
    # Runner
    "BenchmarkResult",
    "ParameterizedBenchmarkRunner",
]
