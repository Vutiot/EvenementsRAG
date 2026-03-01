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


class QueryResult(BaseModel):
    query: str
    generated_answer: str
    retrieved_chunks: list[RetrievedChunk]
    retrieval_time_ms: float
    generation_time_ms: float
    config_hash: str
