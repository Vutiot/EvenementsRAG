"""CollectionService — multi-backend collection discovery, creation, and deletion."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from src.api.dependencies import FAISS_PERSIST_DIR
from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.vector_store.factory import VectorStoreFactory

logger = logging.getLogger(__name__)

# Known datasets — mirrors DatasetManager.DATASET_REGISTRY
_DATASET_DIRS: Dict[str, str] = {
    "wiki_10k": "data/raw/wikipedia_articles_10000",
    "octank": "data/raw/octank",
}

# Embedding model → short name for collection naming
_EMBEDDING_SHORT_NAMES: Dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "minilm_l6",
    "sentence-transformers/all-MiniLM-L12-v2": "minilm_l12",
    "BAAI/bge-small-en-v1.5": "bge_small",
    "BAAI/bge-base-en-v1.5": "bge_base",
}

# Legacy collection names for backward compatibility (avoid re-indexing)
_LEGACY_NAMES: Dict[tuple, str] = {
    ("wiki_10k", "qdrant", 512, 50, "sentence-transformers/all-MiniLM-L6-v2", "cosine"): "ww2_events_10000",
}


class CollectionService:
    """Discover, create, and delete collections across vector backends."""

    @staticmethod
    def derive_collection_name(
        dataset_name: str,
        backend: str = "qdrant",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_metric: str = "cosine",
    ) -> str:
        """Deterministically map params to a collection name.

        Returns a legacy name if the combination matches a known default,
        otherwise builds ``{dataset}_{backend}_cs{size}_co{overlap}_{emb_short}_{dist}``.
        """
        key = (dataset_name, backend, chunk_size, chunk_overlap, embedding_model, distance_metric)
        legacy = _LEGACY_NAMES.get(key)
        if legacy is not None:
            return legacy

        emb_short = _EMBEDDING_SHORT_NAMES.get(embedding_model, embedding_model.split("/")[-1].lower())
        return f"{dataset_name}_{backend}_cs{chunk_size}_co{chunk_overlap}_{emb_short}_{distance_metric}"

    def _get_qdrant_store(self) -> BaseVectorStore | None:
        try:
            return VectorStoreFactory.create("qdrant")
        except Exception as exc:
            logger.debug("Qdrant unavailable: %s", exc)
            return None

    def _get_faiss_store(self) -> BaseVectorStore:
        return VectorStoreFactory.create(
            "faiss", persist_dir=str(FAISS_PERSIST_DIR)
        )

    def _get_pgvector_store(self) -> BaseVectorStore | None:
        try:
            return VectorStoreFactory.create("pgvector")
        except Exception as exc:
            logger.debug("pgvector unavailable: %s", exc)
            return None

    def list_all(self) -> Dict[str, Any]:
        """Discover collections across all available backends.

        Returns:
            {"collections": [...], "backends_available": [...]}
        """
        collections: List[Dict] = []
        backends_available: List[str] = []

        # Qdrant
        qdrant = self._get_qdrant_store()
        if qdrant is not None:
            backends_available.append("qdrant")
            try:
                for info in qdrant.list_collections():
                    info["backend"] = "qdrant"
                    collections.append(info)
            except Exception as exc:
                logger.warning("Failed to list Qdrant collections: %s", exc)

        # FAISS
        try:
            faiss_store = self._get_faiss_store()
            backends_available.append("faiss")
            for info in faiss_store.list_collections():
                info["backend"] = "faiss"
                collections.append(info)
        except Exception as exc:
            logger.debug("FAISS unavailable: %s", exc)

        # pgvector
        pg = self._get_pgvector_store()
        if pg is not None:
            backends_available.append("pgvector")
            try:
                for info in pg.list_collections():
                    info["backend"] = "pgvector"
                    collections.append(info)
            except Exception as exc:
                logger.warning("Failed to list pgvector collections: %s", exc)

        return {
            "collections": collections,
            "backends_available": backends_available,
        }

    def get_one(self, backend: str, name: str) -> Dict | None:
        """Get info for a single collection."""
        store = self._store_for(backend)
        if store is None or not store.collection_exists(name):
            return None
        info = store.get_collection_info(name)
        info["backend"] = backend
        return info

    def create_and_index(
        self,
        dataset_name: str,
        collection_name: str,
        backend: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        embedding_dimension: int,
        distance_metric: str,
    ) -> str:
        """Create a collection and index the dataset into it.

        Returns the collection name on success.
        Raises ValueError / FileNotFoundError on bad input.
        """
        from src.embeddings.embedding_generator import EmbeddingGenerator
        from src.preprocessing.text_chunker import TextChunker
        from src.vector_store.indexer import DocumentIndexer

        # Resolve articles dir
        if dataset_name not in _DATASET_DIRS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Known: {sorted(_DATASET_DIRS)}"
            )
        articles_dir = Path(_DATASET_DIRS[dataset_name])
        if not articles_dir.exists():
            raise FileNotFoundError(f"Articles directory not found: {articles_dir}")

        # Build store
        store = self._store_for(backend)
        if store is None:
            raise ValueError(f"Backend '{backend}' is not available")

        distance = DistanceMetric(distance_metric)

        # Create the collection
        store.create_collection(
            collection_name=collection_name,
            vector_size=embedding_dimension,
            distance=distance,
            recreate=False,
        )

        # Index articles
        embedding_gen = EmbeddingGenerator(model_name=embedding_model)
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # DocumentIndexer expects a QdrantManager-like object; for non-Qdrant
        # backends we use the BaseVectorStore directly through a thin shim.
        if backend == "qdrant":
            from src.vector_store.qdrant_adapter import QdrantAdapter
            mgr = store.manager if isinstance(store, QdrantAdapter) else store
            indexer = DocumentIndexer(qdrant_manager=mgr, text_chunker=chunker)
            indexer.index_all_articles(
                collection_name=collection_name,
                articles_dir=articles_dir,
            )
        else:
            # Generic indexing path for FAISS / pgvector
            self._generic_index(
                store, collection_name, articles_dir, chunker, embedding_gen
            )

        return collection_name

    def delete(self, backend: str, name: str) -> bool:
        """Delete a collection from the specified backend."""
        store = self._store_for(backend)
        if store is None:
            raise ValueError(f"Backend '{backend}' is not available")
        return store.delete_collection(name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _store_for(self, backend: str) -> BaseVectorStore | None:
        if backend == "qdrant":
            return self._get_qdrant_store()
        elif backend == "faiss":
            return self._get_faiss_store()
        elif backend == "pgvector":
            return self._get_pgvector_store()
        return None

    @staticmethod
    def _generic_index(
        store: BaseVectorStore,
        collection_name: str,
        articles_dir: Path,
        chunker,
        embedding_gen,
    ) -> None:
        """Index articles into a non-Qdrant vector store."""
        import json

        articles = sorted(articles_dir.glob("*.json"))
        if not articles:
            raise FileNotFoundError(f"No JSON articles found in {articles_dir}")

        logger.info("Indexing %d articles into %s", len(articles), collection_name)

        all_chunks = []
        for article_path in articles:
            with open(article_path) as f:
                article = json.load(f)
            content = article.get("content", "")
            if not content:
                continue
            chunks = chunker.chunk_text(content)
            for i, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "payload": {
                        "article_title": article.get("title", ""),
                        "source_url": article.get("url", ""),
                        "chunk_index": i,
                        "content": chunk_text,
                    },
                })

        if not all_chunks:
            return

        # Embed in batches
        batch_size = 256
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            payloads = [c["payload"] for c in batch]
            embeddings = embedding_gen.generate_embeddings(texts)
            store.upsert_vectors(
                collection_name=collection_name,
                vectors=embeddings,
                payloads=payloads,
            )
            logger.info(
                "Indexed batch %d/%d (%d chunks)",
                i // batch_size + 1,
                (len(all_chunks) + batch_size - 1) // batch_size,
                len(batch),
            )
