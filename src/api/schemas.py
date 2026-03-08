"""Response/request models for the EvenementsRAG API."""

from pydantic import BaseModel


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
    metrics_by_type: dict[str, dict[str, float]]
    per_question: list[NormalizedQuestion]
    total_questions: int
    avg_retrieval_time_ms: float
    total_wall_time_s: float | None = None
    metrics_summary: dict | None = None
