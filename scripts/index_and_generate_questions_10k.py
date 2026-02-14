#!/usr/bin/env python3
"""
Index 10,000 articles and generate evaluation questions.

This script handles the large 10k dataset with optimizations for memory and speed.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager
from src.vector_store.indexer import DocumentIndexer
from src.evaluation.question_generator import QuestionGenerator

logger = get_logger(__name__)


def main():
    print("=" * 70)
    print("Index 10,000 Articles & Generate Questions")
    print("=" * 70)
    print()

    # Configuration
    articles_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_10000"
    collection_name = "ww2_events_10000"
    questions_file = settings.DATA_DIR / "evaluation" / "eval_10000_articles_50q.json"
    num_questions = 50

    # Check if articles exist
    if not articles_dir.exists():
        print(f"❌ Articles directory not found: {articles_dir}")
        print("Please run download_10000_articles.py first")
        sys.exit(1)

    article_count = len(list(articles_dir.glob("*.json")))
    print(f"Found {article_count} articles in {articles_dir}")
    print()

    if article_count == 0:
        print("❌ No articles found. Please download articles first.")
        sys.exit(1)

    # Step 1: Initialize in-memory Qdrant
    print("Step 1: Initializing in-memory Qdrant...")
    qdrant = QdrantManager(use_memory=True)
    logger.info("In-memory Qdrant initialized")
    print()

    # Step 2: Index articles
    print("=" * 70)
    print(f"Step 2: Indexing {article_count} Articles")
    print("=" * 70)
    print()
    print("This will take approximately 5-10 minutes...")
    print("(Fast due to embedding cache)")
    print()

    indexer = DocumentIndexer(qdrant_manager=qdrant)

    stats = indexer.index_all_articles(
        collection_name=collection_name,
        articles_dir=articles_dir,
        recreate_collection=True,
    )

    if not stats["success"]:
        print(f"❌ Indexing failed: {stats.get('error')}")
        sys.exit(1)

    print(f"✓ Indexed {stats['chunks_indexed']} chunks from {stats['articles_loaded']} articles")
    print()

    # Print stats
    print(f"Indexing Statistics:")
    print(f"  Articles:     {stats['articles_loaded']:,}")
    print(f"  Chunks:       {stats['chunks_indexed']:,}")
    print(f"  Avg chunks/article: {stats['chunks_indexed'] / stats['articles_loaded']:.1f}")
    print()

    # Step 3: Generate questions
    print("=" * 70)
    print("Step 3: Generating Evaluation Questions")
    print("=" * 70)
    print()
    print(f"Target: {num_questions} questions")
    print("Note: This will take ~3-4 minutes with OpenRouter rate limits")
    print()

    generator = QuestionGenerator(qdrant_manager=qdrant)

    questions_data = generator.generate_evaluation_questions(
        collection_name=collection_name,
        num_chunks=num_questions,
        questions_per_chunk=1,
        sampling_strategy="stratified",
        ensure_taxonomy_diversity=False,
    )

    questions = questions_data.get("questions", [])
    metadata = questions_data.get("metadata", {})

    print()
    print("=" * 70)
    print("Generation Summary")
    print("=" * 70)
    print()
    print(f"✓ Generated {len(questions)} questions")
    print(f"  Chunks sampled:  {metadata.get('chunks_sampled')}")
    print(f"  Unique articles: {metadata.get('unique_articles')}")
    print()

    # Show distribution
    type_dist = metadata.get("taxonomy_distribution", {})
    if type_dist:
        print("Question Type Distribution:")
        for q_type in sorted(type_dist.keys()):
            count = type_dist[q_type]
            percentage = (count / len(questions)) * 100 if len(questions) > 0 else 0
            print(f"  {q_type:15s}: {count:2d} ({percentage:5.1f}%)")
        print()

    # Step 4: Save questions
    print("Saving questions...")
    generator.save_questions(questions_data, questions_file)

    print()
    print("=" * 70)
    print("✓ Complete!")
    print("=" * 70)
    print()
    print(f"Questions saved to: {questions_file}")
    print(f"Total questions: {len(questions)}")
    print()

    # Memory estimate
    estimated_vectors_mb = (stats['chunks_indexed'] * 384 * 4) / (1024**2)
    estimated_total_mb = estimated_vectors_mb * 2.5  # Account for metadata
    print(f"Memory Usage Estimate:")
    print(f"  Vectors: ~{estimated_vectors_mb:.0f} MB")
    print(f"  Total (with metadata): ~{estimated_total_mb:.0f} MB")
    print()

    print("Next steps:")
    print("  1. Run Phase 1 evaluation: python scripts/run_phase1_10k.py")
    print("  2. Compare results: python scripts/compare_dataset_sizes.py")
    print()


if __name__ == "__main__":
    main()
