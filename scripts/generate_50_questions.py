#!/usr/bin/env python3
"""
Generate 50 evaluation questions without taxonomy enforcement.

Let the LLM naturally generate whatever questions make sense from chunks.
"""

import sys
import time
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
    print("Generate 50 Evaluation Questions")
    print("=" * 70)
    print()

    collection_name = "ww2_historical_events"
    questions_file = settings.DATA_DIR / "evaluation" / "eval_50_questions.json"

    # Step 1: Index articles
    print("Step 1: Indexing articles...")
    qdrant = QdrantManager()
    indexer = DocumentIndexer(qdrant_manager=qdrant)

    stats = indexer.index_all_articles(
        collection_name=collection_name,
        articles_dir=None,
        recreate_collection=True,
    )

    if not stats["success"]:
        print(f"❌ Indexing failed: {stats.get('error')}")
        sys.exit(1)

    print(f"✓ Indexed {stats['chunks_indexed']} chunks from {stats['articles_loaded']} articles")
    print()

    # Step 2: Generate 50 questions
    print("=" * 70)
    print("Step 2: Generating 50 Questions")
    print("=" * 70)
    print()
    print("Note: OpenRouter free tier has 16 req/min limit")
    print("This will take ~3-4 minutes with automatic retries")
    print()

    generator = QuestionGenerator(qdrant_manager=qdrant)

    # Try to generate 50, expect some failures due to rate limit
    questions_data = generator.generate_evaluation_questions(
        collection_name=collection_name,
        num_chunks=50,  # Sample 50 chunks
        questions_per_chunk=1,
        sampling_strategy="stratified",
        ensure_taxonomy_diversity=False,  # No taxonomy enforcement
    )

    questions = questions_data.get("questions", [])
    metadata = questions_data.get("metadata", {})

    print()
    print(f"✓ Generated {len(questions)} questions")
    print(f"  Chunks sampled:  {metadata.get('chunks_sampled')}")
    print(f"  Unique articles: {metadata.get('unique_articles')}")
    print()

    # Show natural distribution
    type_dist = metadata.get("taxonomy_distribution", {})
    print("Natural Question Type Distribution:")
    for q_type in sorted(type_dist.keys()):
        count = type_dist[q_type]
        percentage = (count / len(questions)) * 100 if len(questions) > 0 else 0
        print(f"  {q_type:15s}: {count:2d} ({percentage:5.1f}%)")
    print()

    # Save
    generator.save_questions(questions_data, questions_file)
    print(f"✓ Questions saved to: {questions_file}")
    print()

    if len(questions) < 50:
        print(f"⚠ Only got {len(questions)}/50 questions due to rate limits")
        print(f"  Wait ~1 minute and run again to continue from {len(questions)}")
    else:
        print(f"✓ Successfully generated {len(questions)} questions!")
        print()
        print("Next steps:")
        print("  1. Run Phase 1 baseline: python scripts/run_phase1_baseline.py")
        print("  2. Implement Phase 2 (Hybrid + Temporal)")
        print("  3. Compare results")

    print()


if __name__ == "__main__":
    main()
