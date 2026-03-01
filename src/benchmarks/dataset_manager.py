"""DatasetManager — registry + lazy auto-indexing for benchmark datasets."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wiki_10k": {
        "articles_dir": "data/raw/wikipedia_articles_10000",
        "description": "Wikipedia 10k WW2 articles",
    },
    "octank": {
        "articles_dir": "data/raw/octank",
        "description": "OctankFinancial 10-K report (run scripts/convert_octank_pdf_to_json.py to prepare JSON files)",
    },
}


class DatasetManager:
    """Resolves article paths and ensures Qdrant collections are indexed."""

    def get_articles_dir(self, dataset_config) -> Path:
        """Return the articles directory, validating it exists.

        Priority:
          1. dataset_config.articles_dir override (if set)
          2. DATASET_REGISTRY entry for dataset_config.dataset_name
        """
        if dataset_config.articles_dir:
            path = Path(dataset_config.articles_dir)
        else:
            name = dataset_config.dataset_name
            if name not in DATASET_REGISTRY:
                raise ValueError(
                    f"Unknown dataset '{name}'. "
                    f"Known datasets: {sorted(DATASET_REGISTRY)}"
                )
            path = Path(DATASET_REGISTRY[name]["articles_dir"])

        if not path.exists():
            raise FileNotFoundError(
                f"Articles directory not found: {path}. "
                "Download the dataset first or set dataset.articles_dir."
            )
        return path

    def ensure_indexed(self, config, qdrant_manager=None) -> str:
        """Ensure the dataset is indexed in Qdrant; return the collection name.

        If the collection already exists, indexing is skipped. Otherwise the
        full pipeline (load → chunk → embed → upsert) is executed using the
        chunking and embedding params from *config*.

        Args:
            config: BenchmarkConfig driving this run.
            qdrant_manager: Shared QdrantManager (lazy-created if None).

        Returns:
            The effective collection name (same as config.dataset.collection_name).
        """
        from src.preprocessing.text_chunker import TextChunker
        from src.vector_store.indexer import DocumentIndexer
        from src.vector_store.qdrant_manager import QdrantManager

        if qdrant_manager is None:
            qdrant_manager = QdrantManager()

        collection_name = config.dataset.collection_name

        if qdrant_manager.collection_exists(collection_name):
            logger.info(
                "Collection '%s' already exists — skipping indexing.", collection_name
            )
            return collection_name

        articles_dir = self.get_articles_dir(config.dataset)
        logger.info(
            "Indexing '%s' (chunk_size=%d, overlap=%d) → collection '%s'",
            config.dataset.dataset_name,
            config.chunking.chunk_size,
            config.chunking.chunk_overlap,
            collection_name,
        )

        chunker = TextChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        indexer = DocumentIndexer(
            qdrant_manager=qdrant_manager,
            text_chunker=chunker,
        )
        indexer.index_all_articles(
            collection_name=collection_name,
            articles_dir=articles_dir,
        )

        return collection_name
