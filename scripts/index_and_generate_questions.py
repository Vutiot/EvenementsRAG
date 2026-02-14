#!/usr/bin/env python3
"""
Index articles and generate evaluation questions in the same process.

This script is needed when Docker isn't available and we need to use
in-memory Qdrant for both indexing and question generation.

Usage:
    python scripts/index_and_generate_questions.py
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
    print("Index Articles & Generate Questions (Combined)")
    print("=" * 70)
    print()

    # Configuration
    articles_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_1000"
    collection_name = "ww2_events_1000"
    questions_file = settings.DATA_DIR / "evaluation" / "eval_976_articles_50q.json"
    num_questions = 50

    # Step 1: Initialize in-memory Qdrant
    print("Step 1: Initializing in-memory Qdrant...")
    qdrant = QdrantManager(use_memory=True)
    logger.info("In-memory Qdrant initialized")
    print()

    # Step 2: Index articles
    print("=" * 70)
    print("Step 2: Indexing 976 Articles")
    print("=" * 70)
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

    # Step 3: Generate questions
    print("=" * 70)
    print("Step 3: Generating Evaluation Questions")
    print("=" * 70)
    print()
    print(f"Target: {num_questions} questions")
    print("Note: OpenRouter free tier has 16 req/min limit")
    print("This will take ~3-4 minutes with automatic retry handling")
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

    # Step 4: Merge with existing questions (if any)
    existing_questions = []
    if questions_file.exists():
        print(f"Found existing questions file, loading...")
        with open(questions_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_questions = existing_data.get("questions", [])
        print(f"  Existing: {len(existing_questions)} questions")

    # Merge questions, avoiding duplicates by source_chunk_id
    existing_chunk_ids = {q["source_chunk_id"] for q in existing_questions}
    new_questions = [q for q in questions if q["source_chunk_id"] not in existing_chunk_ids]

    all_questions = existing_questions + new_questions

    print(f"  New: {len(new_questions)} questions")
    print(f"  Total: {len(all_questions)} questions")
    print()

    # Update questions_data with merged questions
    questions_data["questions"] = all_questions
    questions_data["metadata"]["total_questions"] = len(all_questions)
    questions_data["metadata"]["new_questions"] = len(new_questions)
    questions_data["metadata"]["existing_questions"] = len(existing_questions)

    # Recompute taxonomy distribution for all questions
    taxonomy_dist = {}
    unique_articles = set()
    for q in all_questions:
        q_type = q.get("type", "unknown")
        taxonomy_dist[q_type] = taxonomy_dist.get(q_type, 0) + 1
        unique_articles.add(q.get("source_article"))

    questions_data["metadata"]["taxonomy_distribution"] = taxonomy_dist
    questions_data["metadata"]["unique_articles"] = len(unique_articles)

    # Step 5: Save questions
    print("Saving merged questions...")
    generator.save_questions(questions_data, questions_file)

    print()
    print("=" * 70)
    print("✓ Complete!")
    print("=" * 70)
    print()
    print(f"Questions saved to: {questions_file}")
    print(f"  Total questions: {len(all_questions)}")
    print(f"  New this run: {len(new_questions)}")
    print()

    if len(all_questions) < num_questions:
        print(f"⚠ Progress: {len(all_questions)}/{num_questions} questions")
        print("  Run this script again to generate more questions")
    else:
        print(f"✓ Target reached: {len(all_questions)}/{num_questions} questions!")

    print()
    print("Next steps:")
    print("  1. Review questions: cat data/evaluation/eval_976_articles_50q.json | jq")
    print("  2. Run Phase 1 evaluation: python scripts/run_phase1_baseline.py")
    print("  3. Run Phase 2 evaluation: python scripts/run_phase2_hybrid.py")
    print()


if __name__ == "__main__":
    main()
