"""Query execution endpoint (stub for E3-F1-T1, real in E3-F1-T2)."""

import hashlib
import time

from fastapi import APIRouter, HTTPException

from src.api.dependencies import PRESETS_DIR
from src.api.schemas import QueryRequest, QueryResult, RetrievedChunk
from src.benchmarks.config import BenchmarkConfig

router = APIRouter()

_MOCK_CHUNKS = [
    RetrievedChunk(
        chunk_id="chunk_001",
        content=(
            "On June 6, 1944, Allied forces launched the largest amphibious "
            "invasion in history on the beaches of Normandy, France. The operation, "
            "codenamed Operation Overlord, involved over 156,000 troops from the "
            "United States, United Kingdom, Canada, and other Allied nations."
        ),
        score=0.92,
        article_title="Normandy landings",
        source_url="https://en.wikipedia.org/wiki/Normandy_landings",
        chunk_index=0,
    ),
    RetrievedChunk(
        chunk_id="chunk_002",
        content=(
            "The D-Day landings were preceded by extensive aerial and naval bombardment "
            "and an airborne assault. The amphibious landings took place on five beaches "
            "designated Utah, Omaha, Gold, Juno, and Sword, stretching over 50 miles "
            "of the Normandy coast."
        ),
        score=0.87,
        article_title="Normandy landings",
        source_url="https://en.wikipedia.org/wiki/Normandy_landings",
        chunk_index=1,
    ),
    RetrievedChunk(
        chunk_id="chunk_003",
        content=(
            "Omaha Beach was the bloodiest of the D-Day beaches, with approximately "
            "2,400 American casualties. The beach was heavily defended by the German "
            "352nd Infantry Division, and the initial waves of troops faced devastating "
            "machine gun and artillery fire."
        ),
        score=0.81,
        article_title="Omaha Beach",
        source_url="https://en.wikipedia.org/wiki/Omaha_Beach",
        chunk_index=0,
    ),
    RetrievedChunk(
        chunk_id="chunk_004",
        content=(
            "The success of D-Day was a turning point in World War II. It established "
            "a Western Front in Europe, forcing Germany to fight a two-front war. Within "
            "a year of the landings, Nazi Germany had surrendered unconditionally."
        ),
        score=0.76,
        article_title="Operation Overlord",
        source_url="https://en.wikipedia.org/wiki/Operation_Overlord",
        chunk_index=3,
    ),
    RetrievedChunk(
        chunk_id="chunk_005",
        content=(
            "General Dwight D. Eisenhower served as Supreme Commander of the Allied "
            "Expeditionary Force. He made the critical decision to proceed with the "
            "invasion despite uncertain weather conditions on June 5, 1944."
        ),
        score=0.71,
        article_title="Dwight D. Eisenhower",
        source_url="https://en.wikipedia.org/wiki/Dwight_D._Eisenhower",
        chunk_index=5,
    ),
]

_MOCK_ANSWER = (
    "D-Day, officially known as Operation Overlord, took place on June 6, 1944. "
    "It was the largest amphibious invasion in history, involving over 156,000 Allied "
    "troops landing on five beaches along the Normandy coast of France: Utah, Omaha, "
    "Gold, Juno, and Sword. The operation was commanded by General Dwight D. Eisenhower "
    "and was preceded by extensive aerial and naval bombardment. Omaha Beach saw the "
    "heaviest casualties, with approximately 2,400 American soldiers killed or wounded. "
    "The success of the landings established a crucial Western Front in Europe, which "
    "was a major turning point in World War II, ultimately leading to the defeat of "
    "Nazi Germany within a year."
)


@router.post("/query", response_model=QueryResult)
def execute_query(request: QueryRequest):
    """Execute a query against the RAG system.

    Stub implementation for E3-F1-T1 — returns mock data.
    Real implementation in E3-F1-T2.
    """
    # Validate that the preset exists
    preset_path = PRESETS_DIR / request.preset
    if not preset_path.exists():
        raise HTTPException(status_code=404, detail=f"Preset '{request.preset}' not found")

    cfg = BenchmarkConfig.from_yaml(preset_path)

    # Simulate some processing time
    time.sleep(0.1)

    return QueryResult(
        query=request.query,
        generated_answer=_MOCK_ANSWER,
        retrieved_chunks=_MOCK_CHUNKS,
        retrieval_time_ms=42.5,
        generation_time_ms=1250.0,
        config_hash=cfg.config_hash(),
    )
