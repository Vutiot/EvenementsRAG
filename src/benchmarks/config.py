"""
BenchmarkConfig — Pydantic v2 schema for parameterized benchmark runs.

Defines seven nested sub-models plus the root BenchmarkConfig that drives
ParameterizedBenchmarkRunner. Used by all E2 parameter sweeps.

Usage:
    from src.benchmarks.config import BenchmarkConfig

    cfg = BenchmarkConfig.phase1_vanilla()
    print(cfg.config_hash())

    cfg.to_yaml(Path("config/benchmarks/my_run.yaml"))
    cfg2 = BenchmarkConfig.from_yaml(Path("config/benchmarks/my_run.yaml"))
"""

import hashlib
import json
import warnings
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    dataset_name: str = "wiki_10k"
    collection_name: str = "ww2_events_10000"
    questions_file: str = "data/evaluation/eval_10k_200q.json"
    articles_dir: Optional[str] = None  # override DATASET_REGISTRY path

    @model_validator(mode="after")
    def check_dataset_name_known(self) -> "DatasetConfig":
        from src.benchmarks.dataset_manager import DATASET_REGISTRY
        if self.dataset_name not in DATASET_REGISTRY:
            warnings.warn(
                f"DatasetConfig: unknown dataset_name '{self.dataset_name}'. "
                f"Known datasets: {sorted(DATASET_REGISTRY)}. "
                "Set articles_dir to override the path.",
                stacklevel=2,
            )
        return self


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384

    @model_validator(mode="after")
    def check_model_dimension(self) -> "EmbeddingConfig":
        known = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-MiniLM-L12-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
        }
        expected = known.get(self.model_name)
        if expected is not None and self.dimension != expected:
            warnings.warn(
                f"EmbeddingConfig: model '{self.model_name}' typically uses "
                f"dimension {expected}, got {self.dimension}",
                stacklevel=2,
            )
        return self


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(512, ge=64, le=2048)
    chunk_overlap: int = Field(50, ge=0, le=500)

    @model_validator(mode="after")
    def check_overlap_lt_size(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        return self


class RetrievalConfig(BaseModel):
    technique: Literal["vanilla", "hybrid", "temporal"] = "vanilla"
    top_k: int = Field(10, ge=1, le=100)
    rerank_k: int = Field(20, ge=1, le=100)
    sparse_weight: float = Field(0.3, ge=0.0, le=1.0)
    dense_weight: float = Field(0.7, ge=0.0, le=1.0)
    fusion_method: Literal["rrf", "weighted_sum"] = "rrf"

    @model_validator(mode="after")
    def check_hybrid_weights(self) -> "RetrievalConfig":
        if self.technique == "hybrid":
            total = self.sparse_weight + self.dense_weight
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"For hybrid technique, sparse_weight ({self.sparse_weight}) + "
                    f"dense_weight ({self.dense_weight}) must sum to 1.0, got {total:.4f}"
                )
        return self


class RerankerConfig(BaseModel):
    type: Literal["none", "cohere", "bge", "cross_encoder"] = "none"
    model_name: Optional[str] = None

    @model_validator(mode="after")
    def check_model_name_required(self) -> "RerankerConfig":
        if self.type != "none" and not self.model_name:
            raise ValueError(
                f"model_name is required when reranker type is '{self.type}'"
            )
        return self


OPENROUTER_FREE_MODELS: list[str] = [
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-2-9b-it:free",
]


class GenerationConfig(BaseModel):
    llm_provider: Literal["anthropic", "openai", "openrouter"] = "openrouter"
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=8000)
    top_k_chunks: int = Field(5, ge=1, le=20)
    top_k_articles: Optional[int] = Field(None, ge=1, le=20)
    prompt_template: Optional[str] = None
    enabled: bool = True


class EvaluationConfig(BaseModel):
    k_values: list[int] = [1, 3, 5, 10]
    compute_ragas: bool = False
    compute_bert_score: bool = False
    compute_rouge: bool = True

    @model_validator(mode="after")
    def warn_nonstandard_k_values(self) -> "EvaluationConfig":
        supported = {1, 3, 5, 10}
        unsupported = [k for k in self.k_values if k not in supported]
        if unsupported:
            warnings.warn(
                f"EvaluationConfig: k_values {unsupported} are outside the supported set "
                f"{supported}. BenchmarkRunner aggregation only supports {sorted(supported)}.",
                stacklevel=2,
            )
        return self


class VectorDBConfig(BaseModel):
    backend: Literal["qdrant", "faiss", "pgvector"] = "qdrant"
    distance_metric: Literal["cosine", "euclidean", "dot_product", "manhattan"] = "cosine"
    connection_params: Optional[dict] = None


# ---------------------------------------------------------------------------
# Embedding model sweep registry
# ---------------------------------------------------------------------------

_EMBEDDING_SWEEP_MODELS: dict[str, tuple[str, int]] = {
    "sentence-transformers/all-MiniLM-L6-v2": ("minilm_l6", 384),
    "sentence-transformers/all-MiniLM-L12-v2": ("minilm_l12", 384),
    "BAAI/bge-small-en-v1.5": ("bge_small", 384),
    "BAAI/bge-base-en-v1.5": ("bge_base", 768),
}


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class BenchmarkConfig(BaseModel):
    name: str = "unnamed_benchmark"
    description: str = ""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def config_hash(self) -> str:
        """SHA-256 of model_dump (excluding name/description), first 16 hex chars.

        Renaming a benchmark doesn't bust E4 cache because name/description
        are excluded from the hash.
        """
        data = self.model_dump(exclude={"name", "description"})
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    def to_yaml(self, path: Optional[Path] = None) -> str:
        """Serialize to YAML string, optionally writing to *path*."""
        content = yaml.dump(
            self.model_dump(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=True,
        )
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return content

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        """Load from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    @classmethod
    def from_yaml_string(cls, content: str) -> "BenchmarkConfig":
        """Load from a YAML string (useful in tests)."""
        data = yaml.safe_load(content)
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def phase1_vanilla(cls) -> "BenchmarkConfig":
        """Vanilla semantic search baseline — chunk_size=512, all-MiniLM-L6-v2."""
        return cls(
            name="phase1_vanilla",
            description="Phase 1: Vanilla semantic search baseline",
            embedding=EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
            ),
            chunking=ChunkingConfig(chunk_size=512, chunk_overlap=50),
            retrieval=RetrievalConfig(technique="vanilla"),
            evaluation=EvaluationConfig(k_values=[1, 3, 5, 10]),
        )

    @classmethod
    def phase2_hybrid(cls) -> "BenchmarkConfig":
        """Hybrid search with RRF fusion — sparse=0.3, dense=0.7."""
        return cls(
            name="phase2_hybrid",
            description="Phase 2: Hybrid search with RRF fusion",
            embedding=EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
            ),
            chunking=ChunkingConfig(chunk_size=512, chunk_overlap=50),
            retrieval=RetrievalConfig(
                technique="hybrid",
                sparse_weight=0.3,
                dense_weight=0.7,
                fusion_method="rrf",
            ),
            evaluation=EvaluationConfig(k_values=[1, 3, 5, 10]),
        )

    # ------------------------------------------------------------------
    # Parameter sweeps
    # ------------------------------------------------------------------

    @classmethod
    def chunk_size_sweep(
        cls,
        base: Optional["BenchmarkConfig"] = None,
        sizes: list[int] = (256, 512, 1024),
    ) -> list["BenchmarkConfig"]:
        """Return configs varying chunk_size; one per entry in *sizes*.

        Each config gets a unique collection_name ``{dataset_name}_cs{size}_co{overlap}``
        so Qdrant stores each chunking variant in its own collection.

        Args:
            base: Template config. Defaults to ``phase1_vanilla()``.
            sizes: Chunk sizes to sweep. Default (256, 512, 1024).
        """
        if base is None:
            base = cls.phase1_vanilla()
        configs = []
        for size in sizes:
            overlap = base.chunking.chunk_overlap
            coll = f"{base.dataset.dataset_name}_cs{size}_co{overlap}"
            cfg = base.model_copy(deep=True, update={
                "name": f"chunk_size_cs{size}",
                "description": f"Chunk size sweep: chunk_size={size}, chunk_overlap={overlap}",
                "chunking": ChunkingConfig(chunk_size=size, chunk_overlap=overlap),
                "dataset": base.dataset.model_copy(update={"collection_name": coll}),
            })
            configs.append(cfg)
        return configs

    @classmethod
    def chunk_overlap_sweep(
        cls,
        base: Optional["BenchmarkConfig"] = None,
        overlaps: list[int] = (0, 50, 128, 256),
    ) -> list["BenchmarkConfig"]:
        """Return configs varying chunk_overlap; one per entry in *overlaps*.

        Overlap values >= chunk_size are skipped with a UserWarning.

        Args:
            base: Template config. Defaults to ``phase1_vanilla()``.
            overlaps: Overlap values to sweep. Default (0, 50, 128, 256).
        """
        if base is None:
            base = cls.phase1_vanilla()
        configs = []
        size = base.chunking.chunk_size
        for overlap in overlaps:
            if overlap >= size:
                warnings.warn(
                    f"chunk_overlap_sweep: skipping overlap={overlap} "
                    f"(>= chunk_size={size})",
                    stacklevel=2,
                )
                continue
            coll = f"{base.dataset.dataset_name}_cs{size}_co{overlap}"
            cfg = base.model_copy(deep=True, update={
                "name": f"chunk_overlap_co{overlap}",
                "description": f"Chunk overlap sweep: chunk_size={size}, chunk_overlap={overlap}",
                "chunking": ChunkingConfig(chunk_size=size, chunk_overlap=overlap),
                "dataset": base.dataset.model_copy(update={"collection_name": coll}),
            })
            configs.append(cfg)
        return configs

    @classmethod
    def distance_metric_sweep(cls) -> List["BenchmarkConfig"]:
        """Return 3 configs sweeping cosine / euclidean / dot_product.

        Manhattan is excluded — no backend supports it natively.
        Collection naming: ``ww2_dm_{metric}`` (e.g. ``ww2_dm_cosine``).
        """
        metrics = ["cosine", "euclidean", "dot_product"]
        configs = []
        for metric in metrics:
            configs.append(
                cls(
                    name=f"wiki_dm_{metric}",
                    description=f"Distance-metric sweep: {metric}",
                    dataset=DatasetConfig(
                        collection_name=f"ww2_dm_{metric}",
                    ),
                    embedding=EmbeddingConfig(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        dimension=384,
                    ),
                    chunking=ChunkingConfig(chunk_size=512, chunk_overlap=50),
                    retrieval=RetrievalConfig(technique="vanilla"),
                    evaluation=EvaluationConfig(k_values=[1, 3, 5, 10]),
                    vector_db=VectorDBConfig(distance_metric=metric),
                )
            )
        return configs

    @classmethod
    def embedding_model_sweep(cls) -> List["BenchmarkConfig"]:
        """Return 4 configs sweeping embedding models.

        Models: all-MiniLM-L6-v2, all-MiniLM-L12-v2, bge-small-en-v1.5,
        bge-base-en-v1.5.  Collection naming: ``ww2_em_{short_name}``.
        """
        configs = []
        for model_name, (short_name, dim) in _EMBEDDING_SWEEP_MODELS.items():
            configs.append(
                cls(
                    name=f"wiki_em_{short_name}",
                    description=f"Embedding-model sweep: {model_name}",
                    dataset=DatasetConfig(
                        collection_name=f"ww2_em_{short_name}",
                    ),
                    embedding=EmbeddingConfig(
                        model_name=model_name,
                        dimension=dim,
                    ),
                    chunking=ChunkingConfig(chunk_size=512, chunk_overlap=50),
                    retrieval=RetrievalConfig(technique="vanilla"),
                    evaluation=EvaluationConfig(k_values=[1, 3, 5, 10]),
                )
            )
        return configs

    # ------------------------------------------------------------------
    # Generation parameter sweeps
    # ------------------------------------------------------------------

    def temperature_sweep(self) -> list["BenchmarkConfig"]:
        """Return 3 configs varying temperature: 0.0, 0.3, 0.7."""
        configs = []
        for t in [0.0, 0.3, 0.7]:
            gen = self.generation.model_copy(update={"temperature": t})
            configs.append(self.model_copy(update={
                "name": f"{self.name}_temp{t}",
                "generation": gen,
            }))
        return configs

    def top_k_chunks_sweep(self) -> list["BenchmarkConfig"]:
        """Return 3 configs varying top_k_chunks: 3, 5, 10."""
        configs = []
        for k in [3, 5, 10]:
            gen = self.generation.model_copy(update={"top_k_chunks": k})
            configs.append(self.model_copy(update={
                "name": f"{self.name}_topk{k}",
                "generation": gen,
            }))
        return configs

    def model_sweep(self) -> list["BenchmarkConfig"]:
        """Return 3 configs, one per OPENROUTER_FREE_MODELS."""
        configs = []
        for model in OPENROUTER_FREE_MODELS:
            sanitized = model.replace("/", "_").replace(":", "_").replace(".", "_").replace("-", "_")
            gen = self.generation.model_copy(
                update={"model": model, "llm_provider": "openrouter"}
            )
            configs.append(self.model_copy(update={
                "name": f"{self.name}_model_{sanitized}",
                "generation": gen,
            }))
        return configs
