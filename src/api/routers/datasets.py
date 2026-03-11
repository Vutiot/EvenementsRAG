"""Dataset CRUD + SSE generation endpoints."""

import asyncio
import queue
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.dataset_service import DatasetService
from src.api.schemas import (
    DatasetCreateRequest,
    DatasetDetail,
    DatasetInfo,
    DatasetListResponse,
)

router = APIRouter()

_service = DatasetService()


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets():
    """List all saved datasets."""
    datasets = await asyncio.to_thread(_service.list_datasets)
    return DatasetListResponse(datasets=[DatasetInfo(**d) for d in datasets])


@router.get("/datasets/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: str):
    """Get a single dataset with all questions."""
    data = await asyncio.to_thread(_service.get_dataset, dataset_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return DatasetDetail(**data)


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    success = await asyncio.to_thread(_service.delete_dataset, dataset_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return {"status": "deleted", "dataset_id": dataset_id}


@router.post("/datasets/generate")
async def generate_dataset(request: DatasetCreateRequest):
    """Start dataset generation, streaming progress via SSE."""
    q: queue.Queue[str | None] = queue.Queue()

    def _worker():
        try:
            for event in _service.generate_dataset(request):
                q.put(event)
        except Exception as exc:
            import json
            q.put(f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n")
        finally:
            q.put(None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    async def _stream():
        while True:
            try:
                item = await asyncio.to_thread(q.get, timeout=120)
            except Exception:
                break
            if item is None:
                break
            yield item

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
