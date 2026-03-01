"""Query execution endpoint — runs real RAG pipelines (E3-F1-T2)."""

import asyncio

from fastapi import APIRouter, HTTPException

from src.api.dependencies import PRESETS_DIR
from src.api.query_service import CollectionNotIndexedError, QueryService
from src.api.schemas import QueryRequest, QueryResult, RetrievedChunk
from src.benchmarks.config import BenchmarkConfig
from src.rag.base_rag import RetrievedChunk as RAGChunk

router = APIRouter()

_query_service = QueryService()


def _map_rag_chunk_to_api(chunk: RAGChunk) -> RetrievedChunk:
    """Convert a ``base_rag.RetrievedChunk`` (metadata dict) to the flat API schema."""
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        score=chunk.score,
        article_title=chunk.article_title,
        source_url=chunk.source_url,
        chunk_index=chunk.chunk_index,
    )


@router.post("/query", response_model=QueryResult)
async def execute_query(request: QueryRequest):
    """Execute a query against the RAG system.

    For the 'default.yaml' preset, also checks for user overrides in 'user-config.yaml'
    and merges them if the file exists.
    """
    # Validate that the preset exists
    preset_path = PRESETS_DIR / request.preset
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{request.preset}' not found")

    # For default preset, also check for user overrides
    if request.preset == "default.yaml":
        user_config_path = PRESETS_DIR / "user-config.yaml"
        cfg = BenchmarkConfig.load_with_user_overrides(preset_path, user_config_path)
    else:
        cfg = BenchmarkConfig.from_yaml(preset_path)

    try:
        result = await asyncio.to_thread(
            _query_service.execute_query, request.query, cfg
        )
    except CollectionNotIndexedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline error: {exc}")

    api_chunks = [_map_rag_chunk_to_api(c) for c in result["chunks"]]

    return QueryResult(
        query=request.query,
        generated_answer=result["answer"],
        retrieved_chunks=api_chunks,
        retrieval_time_ms=result["retrieval_time_ms"],
        generation_time_ms=result["generation_time_ms"],
        config_hash=result["config_hash"],
    )
