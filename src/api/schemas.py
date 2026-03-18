"""Response/request models for the EvenementsRAG API."""

from pydantic import BaseModel, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"


class PresetInfo(BaseModel):
    filename: str
    name: str
    description: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    score: float
    article_title: str
    source_url: str
    chunk_index: int


class QueryRequest(BaseModel):
    query: str
    preset: str
    config_overrides: dict | None = None


class QueryResult(BaseModel):
    query: str
    generated_answer: str
    retrieved_chunks: list[RetrievedChunk]
    retrieval_time_ms: float
    generation_time_ms: float
    config_hash: str


class HighlightChunksRequest(BaseModel):
    query: str
    chunks: list[dict]  # [{chunk_id, content}, ...]
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free"


class HighlightedChunk(BaseModel):
    chunk_id: str
    highlighted_content: str
    relevance: str = "not_relevant"  # "exact_answer" | "related" | "not_relevant"


class HighlightChunksResponse(BaseModel):
    highlighted_chunks: list[HighlightedChunk]


# ---------------------------------------------------------------------------
# Benchmark result viewer models
# ---------------------------------------------------------------------------


class ResultFileInfo(BaseModel):
    filename: str
    phase_name: str
    timestamp: str | None = None
    total_questions: int
    format: str  # "legacy" | "benchmark_result"
    avg_mrr: float
    avg_recall_at_5: float | None = None
    avg_recall_at_10: float | None = None
    total_wall_time_s: float | None = None
    config_summary: dict | None = None


class NormalizedQuestion(BaseModel):
    question_id: str
    question: str
    type: str
    difficulty: str | None = None
    source_article: str | None = None
    ground_truth_count: int | None = None
    retrieved_count: int | None = None
    retrieval_time_ms: float | None = None
    metrics: dict[str, float | int | None]
    generated_answer: str | None = None
    generation_time_ms: float | None = None
    retrieved_contexts: list[str] | None = None
    generation_metrics: dict[str, float] | None = None
    ragas_metrics: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Collection management models
# ---------------------------------------------------------------------------


class CollectionInfo(BaseModel):
    name: str
    backend: str
    vector_size: int | None = None
    distance: str | None = None
    points_count: int | None = None


class CollectionListResponse(BaseModel):
    collections: list[CollectionInfo]
    backends_available: list[str]


class EnsureCollectionRequest(BaseModel):
    dataset_name: str
    backend: str = "qdrant"
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    distance_metric: str = "cosine"


class EnsureCollectionResponse(BaseModel):
    status: str  # "exists" | "created"
    collection_name: str
    message: str


class CollectionCreateRequest(BaseModel):
    dataset_name: str
    collection_name: str | None = None
    backend: str = "qdrant"
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    distance_metric: str = "cosine"


class CollectionCreateResponse(BaseModel):
    status: str
    collection_name: str
    message: str


# ---------------------------------------------------------------------------
# Dataset management models
# ---------------------------------------------------------------------------


class DatasetCategoryConfig(BaseModel):
    type: str
    prompt: str
    model: str
    count: int

    @field_validator("type")
    @classmethod
    def check_type_in_taxonomy(cls, v: str) -> str:
        from src.evaluation.question_generator import VALID_QUESTION_TYPES

        if v not in VALID_QUESTION_TYPES:
            raise ValueError(
                f"Invalid question type {v!r}. "
                f"Must be one of: {', '.join(sorted(VALID_QUESTION_TYPES))}"
            )
        return v


class DatasetCreateRequest(BaseModel):
    name: str
    collection_name: str
    categories: list[DatasetCategoryConfig]


class DatasetInfo(BaseModel):
    id: str
    name: str
    created_at: str
    status: str
    collection_name: str
    total_questions: int
    categories: list[dict]


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfo]


class DatasetDetail(BaseModel):
    id: str
    name: str
    created_at: str
    status: str
    collection_name: str
    total_questions: int
    categories: list[dict]
    questions: list[dict]


class BenchmarkRunRequest(BaseModel):
    preset: str
    config_overrides: dict | None = None
    eval_dataset_id: str


class NormalizedBenchmarkResult(BaseModel):
    filename: str
    format: str
    phase_name: str
    timestamp: str | None = None
    config: dict | None = None
    avg_recall_at_k: dict[str, float]
    avg_mrr: float
    avg_ndcg: dict[str, float]
    avg_article_hit_at_k: dict[str, float] | None = None
    avg_chunk_hit_at_k: dict[str, float] | None = None
    metrics_by_type: dict[str, dict[str, float | None]]
    per_question: list[NormalizedQuestion]
    total_questions: int
    avg_retrieval_time_ms: float
    total_wall_time_s: float | None = None
    metrics_summary: dict | None = None
