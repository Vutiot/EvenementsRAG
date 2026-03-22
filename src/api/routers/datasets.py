"""Dataset CRUD + SSE generation endpoints."""

import asyncio
import json
import queue
import threading

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.api.collection_service import CollectionService
from src.api.dataset_service import DatasetService
from src.api.schemas import (
    DatasetCreateRequest,
    DatasetDetail,
    DatasetInfo,
    DatasetListResponse,
    EnsureCollectionsRequest,
)

router = APIRouter()

_service = DatasetService()


@router.get("/datasets/registry")
async def get_dataset_registry():
    """Return known raw datasets with their default collection names."""
    from src.benchmarks.dataset_manager import DATASET_REGISTRY
    from src.api.collection_service import CollectionService

    svc = CollectionService()
    all_collections = await asyncio.to_thread(svc.list_all)
    existing_names = {c["name"] for c in all_collections["collections"]}

    results = []
    for name, info in DATASET_REGISTRY.items():
        default_col = CollectionService.derive_collection_name(name)
        # Find all existing collections that start with this dataset name
        matching = sorted(c for c in existing_names if c.startswith(name) or c == default_col)
        results.append({
            "name": name,
            "description": info.get("description", ""),
            "default_collection": default_col,
            "collections": matching,
        })

    return {"datasets": results}


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(collection_name: str | None = Query(None)):
    """List all saved datasets, optionally filtered by collection_name."""
    datasets = await asyncio.to_thread(_service.list_datasets, collection_name=collection_name)
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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/datasets/ensure-collections")
async def ensure_collections(request: EnsureCollectionsRequest):
    """Create missing collections via SSE, reporting progress per collection."""
    q: queue.Queue[str | None] = queue.Queue()

    def _worker():
        svc = CollectionService()
        total = len(request.collections)
        try:
            for i, col in enumerate(request.collections):
                col_name = CollectionService.derive_collection_name(
                    dataset_name=col.dataset_name,
                    backend=col.backend,
                    chunk_size=col.chunk_size,
                    chunk_overlap=col.chunk_overlap,
                    embedding_model=col.embedding_model,
                    distance_metric=col.distance_metric,
                )
                payload = {"name": col_name, "index": i, "total": total}

                if svc.get_one(col.backend, col_name) is not None:
                    q.put(_sse("collection_exists", payload))
                    continue

                q.put(_sse("collection_creating", payload))
                try:
                    def _on_progress(msg: str, _p=payload):
                        q.put(_sse("collection_progress", {**_p, "step": msg}))

                    svc.create_and_index(
                        dataset_name=col.dataset_name,
                        collection_name=col_name,
                        backend=col.backend,
                        chunk_size=col.chunk_size,
                        chunk_overlap=col.chunk_overlap,
                        embedding_model=col.embedding_model,
                        embedding_dimension=col.embedding_dimension,
                        distance_metric=col.distance_metric,
                        on_progress=_on_progress,
                    )
                    q.put(_sse("collection_created", payload))
                except Exception as exc:
                    q.put(_sse("collection_error", {**payload, "error": str(exc)}))
        except Exception as exc:
            q.put(_sse("error", {"message": str(exc)}))
        finally:
            q.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    async def _stream():
        while True:
            try:
                item = await asyncio.to_thread(q.get, timeout=3600)
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
