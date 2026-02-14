#!/usr/bin/env python3
"""
Index all Wikipedia articles into Qdrant vector store.

This script runs the complete indexing pipeline:
1. Load all Wikipedia articles
2. Chunk documents
3. Generate embeddings
4. Index into Qdrant

Usage:
    # Index all articles with default settings
    python scripts/index_all_articles.py

    # Recreate collection (delete existing)
    python scripts/index_all_articles.py --recreate

    # Use custom collection name
    python scripts/index_all_articles.py --collection my_collection
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import get_logger
from src.vector_store.indexer import DocumentIndexer
from src.vector_store.qdrant_manager import QdrantManager

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Index Wikipedia articles into Qdrant vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all articles
  python scripts/index_all_articles.py

  # Recreate collection from scratch
  python scripts/index_all_articles.py --recreate

  # Use custom collection
  python scripts/index_all_articles.py --collection ww2_custom

  # Save processed chunks to file
  python scripts/index_all_articles.py --save-chunks
        """,
    )

    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=settings.QDRANT_COLLECTION_NAME,
        help=f"Collection name (default: {settings.QDRANT_COLLECTION_NAME})",
    )

    parser.add_argument(
        "--recreate",
        "-r",
        action="store_true",
        help="Recreate collection (delete if exists)",
    )

    parser.add_argument(
        "--articles-dir",
        "-a",
        type=str,
        default=None,
        help="Articles directory (default: data/raw/wikipedia_articles)",
    )

    parser.add_argument(
        "--save-chunks",
        "-s",
        action="store_true",
        help="Save processed chunks to JSON file",
    )

    parser.add_argument(
        "--use-memory",
        "-m",
        action="store_true",
        help="Use in-memory Qdrant (for testing only)",
    )

    return parser.parse_args()


def print_header():
    """Print script header."""
    print("=" * 70)
    print("Wikipedia Articles Indexer")
    print("=" * 70)
    print()


def main():
    """Main function."""
    args = parse_args()
    print_header()

    # Prepare paths
    articles_dir = Path(args.articles_dir) if args.articles_dir else None

    print("Configuration:")
    print(f"  Collection:       {args.collection}")
    print(f"  Recreate:         {args.recreate}")
    print(f"  Save chunks:      {args.save_chunks}")
    print(f"  Use in-memory:    {args.use_memory}")
    print()

    # Initialize Qdrant
    try:
        if args.use_memory:
            logger.info("Using in-memory Qdrant for testing")
            qdrant = QdrantManager(use_memory=True)
            print("⚠ Using in-memory Qdrant (data will be lost on exit)")
        else:
            logger.info("Connecting to Qdrant...")
            qdrant = QdrantManager()

        logger.info("Connected to Qdrant successfully")

    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        print("❌ Error: Cannot connect to Qdrant")
        print()
        if not args.use_memory:
            print("Start Qdrant first:")
            print("  bash scripts/setup_qdrant.sh start")
            print()
            print("Or use --use-memory flag for testing")
        sys.exit(1)

    # Initialize indexer
    try:
        logger.info("Initializing document indexer...")
        indexer = DocumentIndexer(qdrant_manager=qdrant)
        logger.info("Indexer ready")

    except Exception as e:
        logger.error(f"Failed to initialize indexer: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Run indexing
    try:
        print("=" * 70)
        print("Starting Indexing Pipeline")
        print("=" * 70)
        print()
        print("This may take several minutes...")
        print()

        stats = indexer.index_all_articles(
            collection_name=args.collection,
            articles_dir=articles_dir,
            recreate_collection=args.recreate,
        )

        if not stats["success"]:
            print(f"\n❌ Indexing failed: {stats.get('error')}")
            sys.exit(1)

        # Print results
        print()
        print("=" * 70)
        print("Indexing Complete")
        print("=" * 70)
        print()
        print(f"✓ Collection:      {stats['collection_name']}")
        print(f"✓ Articles loaded: {stats['articles_loaded']}")
        print(f"✓ Chunks created:  {stats['chunks_created']}")
        print(f"✓ Chunks indexed:  {stats['chunks_indexed']}")
        print()

        # Collection info
        coll_info = stats['collection_info']
        print("Collection Info:")
        print(f"  Vector size:       {coll_info['vector_size']}")
        print(f"  Distance metric:   {coll_info['distance']}")
        print(f"  Total points:      {coll_info['points_count']}")
        print()

        print("=" * 70)
        print()

        # Save chunks if requested
        if args.save_chunks:
            print("Saving processed chunks...")
            # We'd need to re-process to get chunks, so skip for now
            print("⚠ Chunks already indexed in Qdrant. Use DocumentIndexer API to save.")

        print("Next steps:")
        print("  1. Generate questions: python scripts/generate_evaluation_questions.py")
        print("  2. Run evaluation: python scripts/run_evaluation.py")
        print("  3. Query RAG: python -m src.rag.phase1_vanilla.retriever")
        print()

    except KeyboardInterrupt:
        print("\n\n⚠ Indexing interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        print(f"\n❌ Error during indexing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
