"""Collection CRUD endpoints — list, get, create, delete across backends."""

import asyncio

from fastapi import APIRouter, HTTPException

from src.api.collection_service import CollectionService
from src.api.schemas import (
    CollectionCreateRequest,
    CollectionCreateResponse,
    CollectionInfo,
    CollectionListResponse,
    EnsureCollectionRequest,
    EnsureCollectionResponse,
)

router = APIRouter()

_service = CollectionService()


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    """Discover collections across all available vector backends."""
    result = await asyncio.to_thread(_service.list_all)
    return CollectionListResponse(
        collections=[CollectionInfo(**c) for c in result["collections"]],
        backends_available=result["backends_available"],
    )


@router.get("/collections/{backend}/{name}", response_model=CollectionInfo)
async def get_collection(backend: str, name: str):
    """Get details for a single collection."""
    info = await asyncio.to_thread(_service.get_one, backend, name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found on '{backend}'")
    return CollectionInfo(**info)


@router.post("/ensure-collection", response_model=EnsureCollectionResponse)
async def ensure_collection(request: EnsureCollectionRequest):
    """Check if a matching collection exists; if not, create & index it."""
    collection_name = CollectionService.derive_collection_name(
        dataset_name=request.dataset_name,
        backend=request.backend,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        embedding_model=request.embedding_model,
        distance_metric=request.distance_metric,
    )

    # Fast path: collection already exists
    info = await asyncio.to_thread(_service.get_one, request.backend, collection_name)
    if info is not None:
        return EnsureCollectionResponse(
            status="exists",
            collection_name=collection_name,
            message=f"Collection '{collection_name}' already exists.",
        )

    # Slow path: create & index
    try:
        await asyncio.to_thread(
            _service.create_and_index,
            dataset_name=request.dataset_name,
            collection_name=collection_name,
            backend=request.backend,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model,
            embedding_dimension=request.embedding_dimension,
            distance_metric=request.distance_metric,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Indexing error: {exc}")

    return EnsureCollectionResponse(
        status="created",
        collection_name=collection_name,
        message=f"Collection '{collection_name}' created and indexed on {request.backend}.",
    )


@router.post("/collections", response_model=CollectionCreateResponse)
async def create_collection(request: CollectionCreateRequest):
    """Create a new collection and index a dataset into it."""
    collection_name = request.collection_name
    if not collection_name:
        # Auto-generate from params
        collection_name = (
            f"{request.dataset_name}_cs{request.chunk_size}"
            f"_co{request.chunk_overlap}_{request.backend}"
        )

    try:
        result_name = await asyncio.to_thread(
            _service.create_and_index,
            dataset_name=request.dataset_name,
            collection_name=collection_name,
            backend=request.backend,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model,
            embedding_dimension=request.embedding_dimension,
            distance_metric=request.distance_metric,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Indexing error: {exc}")

    return CollectionCreateResponse(
        status="created",
        collection_name=result_name,
        message=f"Collection '{result_name}' created and indexed on {request.backend}.",
    )


@router.delete("/collections/{backend}/{name}")
async def delete_collection(backend: str, name: str):
    """Delete a collection from the specified backend."""
    try:
        success = await asyncio.to_thread(_service.delete, backend, name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Delete error: {exc}")

    if not success:
        raise HTTPException(status_code=500, detail="Delete returned false")

    return {"status": "deleted", "collection_name": name, "backend": backend}
