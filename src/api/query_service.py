"""QueryService — pipeline cache + query executor for the API layer.

Caches initialised RAG pipelines keyed by ``config_hash()`` (16-char hex).
Two presets with identical parameters share one pipeline.  Max 5 cached
pipelines with FIFO eviction.  Thread-safe via ``threading.Lock``.

Usage (from a FastAPI endpoint):
    from src.api.query_service import QueryService, CollectionNotIndexedError

    service = QueryService()
    result = service.execute_query("What was D-Day?", config)
"""

import importlib
import threading
import time
from collections import OrderedDict
from typing import Any, Dict

from src.benchmarks.config import BenchmarkConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_CACHED_PIPELINES = 5

# Local RAG registry — avoids importing the full benchmark runner stack.
_RAG_REGISTRY: Dict[str, str] = {
    "vanilla": "src.rag.phase1_vanilla.retriever.VanillaRetriever",
    "hybrid": "src.rag.phase3_hybrid.retriever.HybridRetriever",
}


class CollectionNotIndexedError(Exception):
    """Raised when the required vector collection does not exist."""


class QueryService:
    """Manages cached RAG pipelines and executes user queries."""

    def __init__(self, max_cached: int = MAX_CACHED_PIPELINES):
        self._max_cached = max_cached
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_query(self, query: str, config: BenchmarkConfig) -> dict:
        """Execute *query* against the RAG pipeline described by *config*.

        Returns a dict with keys: ``chunks``, ``answer``,
        ``retrieval_time_ms``, ``generation_time_ms``, ``config_hash``.

        Raises:
            CollectionNotIndexedError: if the collection doesn't exist.
            ValueError: if the RAG technique is unknown.
        """
        pipeline = self._get_or_build(config)

        top_k = config.generation.top_k_chunks

        # Retrieve
        t0 = time.perf_counter()
        chunks = pipeline.retrieve(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Generate (if enabled)
        if config.generation.enabled:
            t1 = time.perf_counter()
            answer = pipeline.generate(
                query,
                chunks,
                temperature=config.generation.temperature,
                max_tokens=config.generation.max_tokens,
                model=config.generation.model,
            )
            generation_ms = (time.perf_counter() - t1) * 1000
        else:
            answer = "[Generation disabled]"
            generation_ms = 0.0

        return {
            "chunks": chunks,
            "answer": answer,
            "retrieval_time_ms": retrieval_ms,
            "generation_time_ms": generation_ms,
            "config_hash": config.config_hash(),
        }

    def clear_cache(self) -> None:
        """Remove all cached pipelines."""
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_build(self, config: BenchmarkConfig):
        """Return a cached pipeline or build a new one."""
        key = config.config_hash()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        # Build outside the lock (may take a while for model loading)
        pipeline = self._build_pipeline(config)

        with self._lock:
            self._cache[key] = pipeline
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_cached:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.info(f"Evicted pipeline {evicted_key} from cache")

        return pipeline

    def _build_pipeline(self, config: BenchmarkConfig):
        """Create a new RAG pipeline from *config*.

        Raises:
            CollectionNotIndexedError: collection does not exist.
            ValueError: unknown RAG technique.
        """
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.vector_store.factory import VectorStoreFactory

        technique = config.retrieval.technique
        target = _RAG_REGISTRY.get(technique)
        if target is None:
            raise ValueError(
                f"Unknown RAG technique '{technique}'. "
                f"Available: {sorted(_RAG_REGISTRY)}"
            )

        # Vector store + embeddings
        vector_store = VectorStoreFactory.from_config(config.vector_db)
        embedding_gen = EmbeddingGenerator(model_name=config.embedding.model_name)

        # Fast-fail if collection is missing
        if not vector_store.collection_exists(config.dataset.collection_name):
            raise CollectionNotIndexedError(
                f"Collection '{config.dataset.collection_name}' does not exist. "
                "Index the dataset first."
            )

        # Dynamically import the RAG class
        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)

        extra = {"config": config} if technique == "hybrid" else {}
        pipeline = cls(
            collection_name=config.dataset.collection_name,
            qdrant_manager=vector_store,
            embedding_generator=embedding_gen,
            prompt_template=config.generation.prompt_template,
            **extra,
        )

        logger.info(
            f"Built {class_name} pipeline for collection "
            f"'{config.dataset.collection_name}' [{config.config_hash()[:8]}]"
        )
        return pipeline
