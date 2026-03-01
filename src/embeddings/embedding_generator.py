"""
Embedding generation for text chunks.

Generates dense vector embeddings using sentence-transformers models.
Supports batch processing and caching for efficiency.

Usage:
    from src.embeddings.embedding_generator import EmbeddingGenerator

    generator = EmbeddingGenerator()
    embeddings = generator.generate_embeddings(chunks)
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for text chunks using sentence-transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Model identifier (default: from settings)
            batch_size: Batch size for processing (default: from settings)
            cache_dir: Directory for caching embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.cache_dir = cache_dir or (settings.CACHE_DIR_PATH / "embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded successfully",
                extra={
                    "model": self.model_name,
                    "dimension": self.embedding_dim,
                    "device": self.model.device,
                },
            )
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _get_cache_path(self, text_hash: str) -> Path:
        """Get cache file path for a text hash."""
        return self.cache_dir / f"{text_hash}.npy"

    def _hash_text(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        key = f"{self.model_name}::{text}"
        return hashlib.md5(key.encode()).hexdigest()

    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache if exists."""
        if not settings.ENABLE_CACHE:
            return None

        text_hash = self._hash_text(text)
        cache_path = self._get_cache_path(text_hash)

        if cache_path.exists():
            try:
                embedding = np.load(cache_path)
                logger.debug(f"Loaded embedding from cache: {text_hash}")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                return None

        return None

    def _save_to_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        if not settings.ENABLE_CACHE:
            return

        text_hash = self._hash_text(text)
        cache_path = self._get_cache_path(text_hash)

        try:
            np.save(cache_path, embedding)
            logger.debug(f"Saved embedding to cache: {text_hash}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")

    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            use_cache: Whether to use caching

        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if use_cache:
            cached = self._load_from_cache(text)
            if cached is not None:
                return cached

        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Cache if enabled
        if use_cache:
            self._save_to_cache(text, embedding)

        return embedding

    def generate_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts
            use_cache: Whether to use caching
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings (shape: [n_texts, embedding_dim])
        """
        if not texts:
            return np.array([])

        logger.info(f"Generating embeddings for {len(texts)} texts")

        # Check cache for each text
        embeddings = []
        texts_to_encode = []
        text_indices = []

        for i, text in enumerate(texts):
            if use_cache:
                cached = self._load_from_cache(text)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue

            texts_to_encode.append(text)
            text_indices.append(i)

        logger.info(
            f"Cache hits: {len(embeddings)}/{len(texts)}, "
            f"need to encode: {len(texts_to_encode)}"
        )

        # Encode remaining texts in batches
        if texts_to_encode:
            batch_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )

            # Cache new embeddings
            if use_cache:
                for text, embedding in zip(texts_to_encode, batch_embeddings):
                    self._save_to_cache(text, embedding)

            # Add to results
            for idx, embedding in zip(text_indices, batch_embeddings):
                embeddings.append((idx, embedding))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])

        logger.info(
            f"Generated {len(result)} embeddings",
            extra={"shape": result.shape, "cached": len(texts) - len(texts_to_encode)},
        )

        return result

    def embed_chunks(
        self,
        chunks: List[Dict],
        content_field: str = "content",
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of chunk dictionaries
            content_field: Field containing text content
            show_progress: Whether to show progress bar

        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} chunks")

        # Extract texts
        texts = [chunk.get(content_field, "") for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings(
            texts,
            use_cache=True,
            show_progress=show_progress,
        )

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()  # Convert to list for JSON serialization
            chunk["embedding_model"] = self.model_name
            chunk["embedding_dimension"] = self.embedding_dim

        logger.info(f"Successfully embedded {len(chunks)} chunks")

        return chunks

    def save_embeddings(
        self,
        chunks: List[Dict],
        output_path: Path,
        format: str = "npy",
    ) -> None:
        """
        Save embeddings to file.

        Args:
            chunks: List of chunks with embeddings
            output_path: Output file path
            format: Format ('npy' or 'pkl')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "npy":
            # Save as numpy array
            embeddings = np.array([chunk["embedding"] for chunk in chunks])
            np.save(output_path, embeddings)

        elif format == "pkl":
            # Save full chunks with pickle
            with open(output_path, "wb") as f:
                pickle.dump(chunks, f)

        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved embeddings to {output_path}")

    def load_embeddings(
        self,
        input_path: Path,
        format: str = "npy",
    ) -> Union[np.ndarray, List[Dict]]:
        """
        Load embeddings from file.

        Args:
            input_path: Input file path
            format: Format ('npy' or 'pkl')

        Returns:
            Embeddings array or chunks list
        """
        if format == "npy":
            return np.load(input_path)

        elif format == "pkl":
            with open(input_path, "rb") as f:
                return pickle.load(f)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_statistics(self) -> Dict:
        """Get embedding generator statistics."""
        # Count cache files
        cache_files = list(self.cache_dir.glob("*.npy"))
        total_cache_size = sum(f.stat().st_size for f in cache_files)

        stats = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "batch_size": self.batch_size,
            "device": str(self.model.device),
            "cache_enabled": settings.ENABLE_CACHE,
            "cache_dir": str(self.cache_dir),
            "cached_embeddings": len(cache_files),
            "cache_size_mb": round(total_cache_size / (1024 * 1024), 2),
        }

        return stats


if __name__ == "__main__":
    # Test the embedding generator
    import json

    print("Testing EmbeddingGenerator...")

    # Initialize generator
    generator = EmbeddingGenerator()

    print(f"\nModel: {generator.model_name}")
    print(f"Embedding dimension: {generator.embedding_dim}")
    print(f"Device: {generator.model.device}")

    # Test single embedding
    print("\n=== Testing single text ===")
    test_text = "The Normandy landings were a pivotal operation in World War II."
    embedding = generator.generate_embedding(test_text)
    print(f"Text: {test_text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding sample: {embedding[:5]}...")

    # Test batch embeddings
    print("\n=== Testing batch embeddings ===")
    test_texts = [
        "D-Day marked the beginning of the liberation of Western Europe.",
        "The Battle of Stalingrad was a turning point on the Eastern Front.",
        "Pearl Harbor brought the United States into World War II.",
    ]

    embeddings = generator.generate_embeddings(test_texts, show_progress=False)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embeddings shape: {embeddings.shape}")

    # Test with actual chunks
    print("\n=== Testing with document chunks ===")
    test_chunk_file = "data/raw/wikipedia_articles/Normandy_landings.json"

    try:
        # Load article
        with open(test_chunk_file, "r") as f:
            article = json.load(f)

        # Create a few test chunks
        from src.preprocessing.text_chunker import TextChunker

        chunker = TextChunker(chunk_size=256, chunk_overlap=25)
        chunks = chunker.chunk_document(article)[:3]  # Just first 3 chunks

        print(f"Loaded {len(chunks)} test chunks")

        # Generate embeddings
        chunks_with_embeddings = generator.embed_chunks(chunks, show_progress=False)

        print(f"Successfully embedded {len(chunks_with_embeddings)} chunks")
        print(f"First chunk embedding shape: {np.array(chunks_with_embeddings[0]['embedding']).shape}")

        # Show statistics
        print("\n=== Generator Statistics ===")
        stats = generator.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")

    except FileNotFoundError:
        print(f"\nTest file not found: {test_chunk_file}")
        print("Run the Wikipedia downloader first:")
        print("  python scripts/download_wikipedia_data.py --titles 'D-Day'")
