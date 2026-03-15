"""FastAPI application for EvenementsRAG web UI.

Run with:
    .venv/bin/uvicorn src.api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.ensure_qdrant import ensure_qdrant_running
from src.api.routers import collections, config, datasets, health, query, results


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_qdrant_running()
    yield


app = FastAPI(
    title="EvenementsRAG API",
    description="REST API for interactive RAG query testing and benchmark visualization",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend dev server (Vite on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers under /api prefix
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(config.router, prefix="/api", tags=["config"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(results.router, prefix="/api", tags=["results"])
app.include_router(collections.router, prefix="/api", tags=["collections"])
app.include_router(datasets.router, prefix="/api", tags=["datasets"])
