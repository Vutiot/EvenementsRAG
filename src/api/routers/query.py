"""Query execution endpoint — runs real RAG pipelines (E3-F1-T2)."""

import asyncio

import openai
from fastapi import APIRouter, HTTPException

from config.settings import settings
from src.api.dependencies import PRESETS_DIR
from src.api.query_service import CollectionNotIndexedError, QueryService
from src.api.schemas import (
    HighlightChunksRequest,
    HighlightChunksResponse,
    HighlightedChunk,
    QueryRequest,
    QueryResult,
    RetrievedChunk,
)
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

    # Apply frontend config overrides (from parameter modal)
    if request.config_overrides:
        from src.benchmarks.config import _deep_merge
        merged = cfg.model_dump()
        _deep_merge(merged, request.config_overrides)
        cfg = BenchmarkConfig.model_validate(merged)

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


# ---------------------------------------------------------------------------
# Chunk highlighting (separate from main query — no impact on generation metrics)
# ---------------------------------------------------------------------------

_HIGHLIGHT_SYSTEM_PROMPT = (
    "You are a text highlighting and relevance assessment assistant.\n"
    "Given a user query and a text passage:\n"
    "1. On the FIRST line, write exactly one of: "
    "RELEVANCE: exact_answer | RELEVANCE: related | RELEVANCE: not_relevant\n"
    "   - exact_answer: passage directly answers the query\n"
    "   - related: passage contains relevant context but not a direct answer\n"
    "   - not_relevant: passage is not relevant to the query\n"
    "2. On the second line, write ---\n"
    "3. Then return the FULL original text with <mark>...</mark> around relevant phrases.\n"
    "Do NOT alter, summarize, or omit any text in part 3."
)

_VALID_RELEVANCE = {"exact_answer", "related", "not_relevant"}


async def _highlight_one_chunk(
    client: openai.AsyncOpenAI,
    model: str,
    query: str,
    chunk_id: str,
    content: str,
) -> HighlightedChunk:
    """Call LLM to highlight relevant passages in a single chunk."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _HIGHLIGHT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Passage:\n{content}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content or content
        raw = raw.strip()

        # Parse relevance tier from first line
        relevance = "not_relevant"
        highlighted = raw
        if "---" in raw:
            header, _, body = raw.partition("---")
            body = body.strip()
            if body:  # only use parsed result if body is non-empty
                highlighted = body
                # Extract tier from header like "RELEVANCE: exact_answer"
                for token in header.replace(":", " ").split():
                    if token.lower() in _VALID_RELEVANCE:
                        relevance = token.lower()
                        break

        return HighlightedChunk(
            chunk_id=chunk_id,
            highlighted_content=highlighted,
            relevance=relevance,
        )
    except Exception:
        return HighlightedChunk(chunk_id=chunk_id, highlighted_content=content)


@router.post("/query/highlight-chunks", response_model=HighlightChunksResponse)
async def highlight_chunks(request: HighlightChunksRequest):
    """Highlight relevant passages in retrieved chunks via batch LLM calls.

    This endpoint is fully independent from /api/query — it does NOT affect
    generation_time_ms or any generation metrics.
    """
    client = openai.AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "EvenementsRAG",
        },
    )

    tasks = [
        _highlight_one_chunk(
            client,
            request.model,
            request.query,
            chunk.get("chunk_id", str(i)),
            chunk.get("content", ""),
        )
        for i, chunk in enumerate(request.chunks)
    ]

    results = await asyncio.gather(*tasks)
    return HighlightChunksResponse(highlighted_chunks=list(results))
