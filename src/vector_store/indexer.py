"""
Indexer for Wikipedia articles into Qdrant vector store.

Handles the complete pipeline:
- Load articles from disk
- Chunk documents
- Generate embeddings
- Index into Qdrant with metadata

Usage:
    from src.vector_store.indexer import DocumentIndexer

    indexer = DocumentIndexer()
    indexer.index_all_articles(collection_name="ww2_events")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from config.settings import settings
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.preprocessing.text_chunker import TextChunker
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager

logger = get_logger(__name__)


class DocumentIndexer:
    """Indexes Wikipedia documents into Qdrant vector store."""

    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        text_chunker: Optional[TextChunker] = None,
    ):
        """
        Initialize the document indexer.

        Args:
            qdrant_manager: Qdrant manager instance
            embedding_generator: Embedding generator instance
            text_chunker: Text chunker instance
        """
        self.qdrant = qdrant_manager or QdrantManager()
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
        self.chunker = text_chunker or TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        logger.info(
            "DocumentIndexer initialized",
            extra={
                "chunk_size": settings.CHUNK_SIZE,
                "embedding_model": self.embedding_gen.model_name,
                "embedding_dim": self.embedding_gen.embedding_dim,
            },
        )

    def load_articles(self, articles_dir: Optional[Path] = None) -> List[Dict]:
        """
        Load all Wikipedia articles from directory.

        Args:
            articles_dir: Directory containing article JSON files

        Returns:
            List of article dictionaries
        """
        if articles_dir is None:
            articles_dir = settings.DATA_DIR / "raw" / "wikipedia_articles"

        logger.info(f"Loading articles from {articles_dir}")

        articles = []
        json_files = list(articles_dir.glob("*.json"))

        if not json_files:
            logger.warning(f"No articles found in {articles_dir}")
            return []

        for article_file in tqdm(json_files, desc="Loading articles"):
            try:
                with open(article_file, "r", encoding="utf-8") as f:
                    article = json.load(f)
                    articles.append(article)
            except Exception as e:
                logger.error(f"Failed to load {article_file}: {e}")

        logger.info(f"Loaded {len(articles)} articles")
        return articles

    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Process articles into chunks with embeddings.

        Args:
            articles: List of article dictionaries

        Returns:
            List of chunks with embeddings and metadata
        """
        logger.info(f"Processing {len(articles)} articles")

        all_chunks = []

        # Chunk all articles
        for article in tqdm(articles, desc="Chunking articles"):
            chunks = self.chunker.chunk_document(article)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks")

        # Generate embeddings
        logger.info("Generating embeddings for all chunks")
        chunks_with_embeddings = self.embedding_gen.embed_chunks(
            all_chunks,
            content_field="content",
            show_progress=True,
        )

        logger.info(f"Successfully processed {len(chunks_with_embeddings)} chunks")
        return chunks_with_embeddings

    def prepare_for_indexing(self, chunks: List[Dict]) -> tuple[List[List[float]], List[Dict], List[str]]:
        """
        Prepare chunks for Qdrant indexing.

        Args:
            chunks: List of chunks with embeddings

        Returns:
            Tuple of (vectors, payloads, ids)
        """
        import hashlib
        import uuid

        vectors = []
        payloads = []
        ids = []

        for i, chunk in enumerate(chunks):
            # Extract embedding
            embedding = chunk.get("embedding")
            if not embedding:
                logger.warning(f"Chunk {i} missing embedding, skipping")
                continue

            vectors.append(embedding)

            # Create payload (metadata without embedding)
            # Store the original chunk ID in metadata for reference
            chunk_identifier = f"{chunk.get('pageid', 'unknown')}_{chunk.get('chunk_index', i)}"

            payload = {
                "content": chunk.get("content", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 0),
                "article_title": chunk.get("article_title", ""),
                "title": chunk.get("title", ""),
                "source_url": chunk.get("source_url", ""),
                "categories": chunk.get("categories", [])[:10],  # Limit categories
                "pageid": chunk.get("pageid"),
                "token_count": chunk.get("token_count", 0),
                "char_count": chunk.get("char_count", 0),
                "embedding_model": chunk.get("embedding_model", ""),
                "chunk_id": chunk_identifier,  # Store original ID for reference
            }

            payloads.append(payload)

            # Generate UUID from chunk identifier using hash
            # This ensures the same chunk always gets the same UUID
            chunk_hash = hashlib.md5(chunk_identifier.encode()).hexdigest()
            chunk_uuid = str(uuid.UUID(chunk_hash))
            ids.append(chunk_uuid)

        logger.info(f"Prepared {len(vectors)} chunks for indexing")
        return vectors, payloads, ids

    def index_chunks(
        self,
        collection_name: str,
        chunks: List[Dict],
        recreate_collection: bool = False,
        batch_size: int = 100,
    ) -> int:
        """
        Index chunks into Qdrant collection.

        Args:
            collection_name: Name of the collection
            chunks: List of chunks with embeddings
            recreate_collection: Whether to recreate the collection
            batch_size: Batch size for indexing

        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing {len(chunks)} chunks into collection '{collection_name}'")

        # Create collection if needed
        if not self.qdrant.collection_exists(collection_name) or recreate_collection:
            logger.info(f"Creating collection '{collection_name}'")
            self.qdrant.create_collection(
                collection_name=collection_name,
                vector_size=self.embedding_gen.embedding_dim,
                recreate=recreate_collection,
            )

        # Prepare data
        vectors, payloads, ids = self.prepare_for_indexing(chunks)

        if not vectors:
            logger.warning("No vectors to index")
            return 0

        # Index into Qdrant
        count = self.qdrant.upsert_vectors(
            collection_name=collection_name,
            vectors=vectors,
            payloads=payloads,
            ids=ids,
            batch_size=batch_size,
        )

        logger.info(f"Successfully indexed {count} chunks into '{collection_name}'")
        return count

    def index_all_articles(
        self,
        collection_name: Optional[str] = None,
        articles_dir: Optional[Path] = None,
        recreate_collection: bool = False,
    ) -> Dict:
        """
        Complete pipeline: load, process, and index all articles.

        Args:
            collection_name: Name of the collection (default: from settings)
            articles_dir: Directory with articles (default: data/raw/wikipedia_articles)
            recreate_collection: Whether to recreate the collection

        Returns:
            Statistics dictionary
        """
        if collection_name is None:
            collection_name = settings.QDRANT_COLLECTION_NAME

        logger.info(
            f"Starting full indexing pipeline for collection '{collection_name}'",
            extra={"recreate": recreate_collection},
        )

        # Load articles
        articles = self.load_articles(articles_dir)

        if not articles:
            logger.error("No articles to index")
            return {
                "success": False,
                "error": "No articles found",
            }

        # Process articles
        chunks = self.process_articles(articles)

        # Index chunks
        indexed_count = self.index_chunks(
            collection_name=collection_name,
            chunks=chunks,
            recreate_collection=recreate_collection,
        )

        # Get collection info
        collection_info = self.qdrant.get_collection_info(collection_name)

        stats = {
            "success": True,
            "collection_name": collection_name,
            "articles_loaded": len(articles),
            "chunks_created": len(chunks),
            "chunks_indexed": indexed_count,
            "collection_info": collection_info,
        }

        logger.info("Indexing pipeline completed successfully", extra=stats)
        return stats

    def save_chunks(self, chunks: List[Dict], output_path: Optional[Path] = None) -> None:
        """
        Save processed chunks to JSON file.

        Args:
            chunks: List of chunks with embeddings
            output_path: Output file path
        """
        if output_path is None:
            output_path = settings.DATA_DIR / "processed" / "chunks" / "all_chunks.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(chunks)} chunks to {output_path}")

        # Remove embeddings to reduce file size (embeddings are in Qdrant)
        chunks_without_embeddings = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            chunk_copy.pop("embedding", None)
            chunks_without_embeddings.append(chunk_copy)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks_without_embeddings, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved chunks to {output_path}")


if __name__ == "__main__":
    # Test the indexer
    import sys

    print("=" * 70)
    print("Document Indexer Test")
    print("=" * 70)

    # Initialize indexer with in-memory Qdrant for testing
    print("\nInitializing indexer with in-memory Qdrant...")
    qdrant = QdrantManager(use_memory=True)
    indexer = DocumentIndexer(qdrant_manager=qdrant)

    # Check if articles exist
    articles_dir = settings.DATA_DIR / "raw" / "wikipedia_articles"
    if not articles_dir.exists() or not list(articles_dir.glob("*.json")):
        print(f"\n⚠ No articles found in {articles_dir}")
        print("Please run: python scripts/download_wikipedia_data.py --max-articles 5")
        sys.exit(1)

    # Load a few articles for testing
    print(f"\nLoading articles from {articles_dir}...")
    all_articles = indexer.load_articles()
    test_articles = all_articles[:3]  # Use first 3 for testing

    print(f"Loaded {len(test_articles)} articles for testing:")
    for article in test_articles:
        print(f"  - {article.get('title', 'Unknown')} ({article.get('word_count', 0):,} words)")

    # Process articles
    print("\nProcessing articles...")
    chunks = indexer.process_articles(test_articles)

    # Show chunk statistics
    print(f"\nCreated {len(chunks)} chunks:")
    print(f"  Average tokens per chunk: {sum(c['token_count'] for c in chunks) / len(chunks):.1f}")
    print(f"  Total chunks with embeddings: {sum(1 for c in chunks if 'embedding' in c)}")

    # Index into Qdrant
    collection_name = "test_ww2_events"
    print(f"\nIndexing into collection '{collection_name}'...")
    indexed_count = indexer.index_chunks(
        collection_name=collection_name,
        chunks=chunks,
        recreate_collection=True,
    )

    # Get collection info
    print("\nCollection Info:")
    info = qdrant.get_collection_info(collection_name)
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test search
    print("\n" + "=" * 70)
    print("Testing Search")
    print("=" * 70)

    # Use first chunk's content as query
    test_query = chunks[0]["content"][:100] + "..."
    print(f"\nQuery: {test_query}")

    # Generate query embedding
    query_embedding = indexer.embedding_gen.generate_embedding(chunks[0]["content"])

    # Search
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=3,
    )

    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  Result {i} (score: {result['score']:.4f}):")
        print(f"    Title: {result['payload']['article_title']}")
        print(f"    Chunk: {result['payload']['chunk_index']}/{result['payload']['total_chunks']}")
        print(f"    Content: {result['payload']['content'][:100]}...")

    print("\n" + "=" * 70)
    print("✓ All tests completed successfully")
    print("=" * 70)
