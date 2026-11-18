"""Assistant endpoints (E8/E9). For now: the deterministic dataset profile."""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from src.assistant.dataset_profiler import DatasetProfile, get_profile

router = APIRouter()


@router.get("/assistant/profile/{collection}", response_model=DatasetProfile)
async def get_collection_profile(
    collection: str,
    refresh: bool = Query(False, description="Recompute instead of using the cached profile"),
):
    """Compact, LLM-ready profile of a collection (cached on disk)."""
    try:
        return await asyncio.to_thread(get_profile, collection, refresh)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
