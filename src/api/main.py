"""FastAPI application for EvenementsRAG web UI.

Run with:
    .venv/bin/uvicorn src.api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import config, health, query

app = FastAPI(
    title="EvenementsRAG API",
    description="REST API for interactive RAG query testing and benchmark visualization",
    version="0.1.0",
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
