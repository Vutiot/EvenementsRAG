# Plan: E1-F1-T1 + E1-F1-T2 — Benchmark Config Schema & Parameterized Runner

## Context

The project has an existing `BenchmarkRunner` (`src/evaluation/benchmark_runner.py`) that takes
direct dependencies (QdrantManager, EmbeddingGenerator) and is hardcoded to vanilla semantic
search. It cannot swap datasets, embedding models, chunk sizes, RAG techniques, or rerankers.

**Goal**: Add a `BenchmarkConfig` Pydantic schema (E1-F1-T1) and a `ParameterizedBenchmarkRunner`
(E1-F1-T2) that drives evaluation from config. This is the foundation for all E2 parameter sweeps.

**Constraint**: Do NOT modify any file in `src/` — create new files only.

---

## Files to Create (5 new files)

```
src/benchmarks/__init__.py
src/benchmarks/config.py
src/benchmarks/runner.py
config/benchmarks/phase1_vanilla.yaml
config/benchmarks/phase2_hybrid.yaml
```

---

## E1-F1-T1: `src/benchmarks/config.py`

Seven nested Pydantic `BaseModel` sub-models + root `BenchmarkConfig`.

### Sub-models

```python
class DatasetConfig(BaseModel):
    dataset_name: str = "wiki_10k"
    collection_name: str = "ww2_events_10000"
    questions_file: str = "data/evaluation/eval_10k_200q.json"

class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    # model_validator: known model→dim check

class ChunkingConfig(BaseModel):
    chunk_size: int = Field(512, ge=64, le=2048)
    chunk_overlap: int = Field(50, ge=0, le=500)
    # model_validator: overlap < chunk_size

class RetrievalConfig(BaseModel):
    technique: Literal["vanilla", "hybrid", "temporal"] = "vanilla"
    top_k: int = Field(10, ge=1, le=100)
    rerank_k: int = Field(20, ge=1, le=100)
    sparse_weight: float = Field(0.3, ge=0.0, le=1.0)
    dense_weight: float = Field(0.7, ge=0.0, le=1.0)
    fusion_method: Literal["rrf", "weighted_sum"] = "rrf"
    # model_validator: if hybrid, sparse+dense must sum to 1.0

class RerankerConfig(BaseModel):
    type: Literal["none", "cohere", "bge", "cross_encoder"] = "none"
    model_name: Optional[str] = None
    # model_validator: model_name required if type != "none"

class GenerationConfig(BaseModel):
    llm_provider: Literal["anthropic", "openai", "openrouter"] = "openrouter"
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=1, le=8000)
    top_k_chunks: int = Field(5, ge=1, le=20)
    enabled: bool = True  # False = retrieval-only run

class EvaluationConfig(BaseModel):
    k_values: list[int] = [1, 3, 5, 10]
    compute_ragas: bool = False
    compute_bert_score: bool = False
    compute_rouge: bool = True
    # model_validator: warn if k_values contains values outside {1,3,5,10}
    #   (current BenchmarkRunner aggregation only supports those 4 values)
```

### Root model

```python
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

    def config_hash(self) -> str:
        # SHA-256 of model_dump(exclude={"name","description"}) as sorted JSON
        # Returns first 16 hex chars. Excludes name/description so renaming
        # doesn't bust E4 cache.

    def to_yaml(self, path: Optional[Path] = None) -> str:
        # yaml.dump(self.model_dump(), ...) → write to path if given

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        # yaml.safe_load → model_validate

    @classmethod
    def from_yaml_string(cls, content: str) -> "BenchmarkConfig":
        # for tests

    @classmethod
    def phase1_vanilla(cls) -> "BenchmarkConfig":
        # technique="vanilla", chunk_size=512, all-MiniLM-L6-v2

    @classmethod
    def phase2_hybrid(cls) -> "BenchmarkConfig":
        # technique="hybrid", sparse=0.3, dense=0.7, fusion_method="rrf"
```

---

## YAML Presets

**`config/benchmarks/phase1_vanilla.yaml`** — vanilla technique, chunk_size=512, k_values=[1,3,5,10]
**`config/benchmarks/phase2_hybrid.yaml`** — hybrid technique, sparse=0.3, dense=0.7, rrf fusion

Both use: all-MiniLM-L6-v2, reranker=none, openrouter mistral-small, generation.enabled=true

---

## E1-F1-T2: `src/benchmarks/runner.py`

### BenchmarkResult dataclass

```python
@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    config_hash: str
    phase_name: str
    timestamp: str           # ISO-8601 UTC
    evaluation: EvaluationResults   # from src/evaluation/metrics.py
    per_question_full: list[dict]   # retrieval + optional generated_answer
    total_wall_time_s: float = 0.0

    def to_dict(self) -> dict: ...
    def to_json(self, path=None) -> str: ...
    def print_summary(self) -> None: ...
```

### ParameterizedBenchmarkRunner class

Key design:

1. **RAG registry** (dict, lazy import via importlib to avoid errors from empty modules):
   ```python
   _RAG_REGISTRY = {
       "vanilla":  "src.rag.phase1_vanilla.retriever.VanillaRetriever",
       "hybrid":   None,   # NotImplementedError until E2-F3
       "temporal": None,   # NotImplementedError until E2-F1
   }
   ```

2. **`__init__(config, qdrant_manager=None, embedding_generator=None)`**
   - Stores config, optional injected deps (for testing)
   - Lazy RAG pipeline build

3. **`run(questions_file=None, max_questions=None) -> BenchmarkResult`**
   - Initializes shared dependencies once (QdrantManager + EmbeddingGenerator)
     to avoid double model loading during sweeps
   - Calls `_build_rag_pipeline()` using config.retrieval.technique
   - Delegates to **existing** `BenchmarkRunner` for retrieval metrics
     (passes same qdrant/embedding instances → no duplicate model loading)
   - Optionally runs generation pass (if config.generation.enabled)
   - Returns `BenchmarkResult`

4. **`_build_rag_pipeline()`**
   - Looks up technique in `_RAG_REGISTRY`
   - Dynamic import via `importlib.import_module()`
   - Instantiates: `VanillaRetriever(collection_name, qdrant_manager, embedding_generator)`

5. **`_run_generation_pass(questions_path, legacy_per_q, max_questions)`**
   - Loads question texts from JSON
   - For each per-question result, calls `self._rag_pipeline.query(text, top_k=top_k_chunks)`
   - Adds `generated_answer` and `generation_time_ms` to each result entry
   - Logs warnings for failures, doesn't abort

6. **`@staticmethod run_sweep(configs, questions_file, max_questions, output_dir, stop_on_error)`**
   - Iterates configs, creates fresh runner per config
   - Saves results to `output_dir/{name}_{hash}.json` if provided
   - Skips `NotImplementedError` unless `stop_on_error=True`

---

## `src/benchmarks/__init__.py`

Re-exports all public symbols:
```python
from src.benchmarks.config import (BenchmarkConfig, DatasetConfig, EmbeddingConfig,
    ChunkingConfig, RetrievalConfig, RerankerConfig, GenerationConfig, EvaluationConfig)
from src.benchmarks.runner import BenchmarkResult, ParameterizedBenchmarkRunner
```

---

## Critical Files Referenced (read-only)

| File | Used for |
|------|----------|
| `src/evaluation/benchmark_runner.py` | Delegation target: `BenchmarkRunner(questions_file, qdrant_manager, embedding_generator, k_values)` |
| `src/evaluation/metrics.py` | `EvaluationResults` type returned by BenchmarkRunner, composed into BenchmarkResult |
| `src/rag/phase1_vanilla/retriever.py` | Factory target: `VanillaRetriever(collection_name, qdrant_manager, embedding_generator, llm_client=None, prompt_template=None)` |
| `config/settings.py` | Field pattern reference (Pydantic v2: `Field(ge=, le=)`, `Literal`, `model_validator`) |

---

## Verification

After implementation, verify by running:

```python
from src.benchmarks import BenchmarkConfig, ParameterizedBenchmarkRunner

# 1. Config round-trips
cfg = BenchmarkConfig.phase1_vanilla()
assert cfg.config_hash() == BenchmarkConfig.from_yaml_string(cfg.to_yaml()).config_hash()

# 2. YAML file loading
from pathlib import Path
cfg2 = BenchmarkConfig.from_yaml(Path("config/benchmarks/phase1_vanilla.yaml"))
print(cfg2)

# 3. Runner (requires indexed Qdrant collection)
runner = ParameterizedBenchmarkRunner(config=BenchmarkConfig.phase1_vanilla())
result = runner.run(max_questions=3)
result.print_summary()
result.to_json(Path("results/test_e1f1t2.json"))

# 4. Sweep
results = ParameterizedBenchmarkRunner.run_sweep(
    [BenchmarkConfig.phase1_vanilla()],
    max_questions=3,
    output_dir=Path("results/sweep_test/"),
)
assert len(results) == 1
```
