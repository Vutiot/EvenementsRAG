#!/usr/bin/env python3
"""
Test the new chunk-based evaluation system end-to-end.

This script:
1. Indexes articles into in-memory Qdrant
2. Generates chunk-based questions
3. Runs evaluation with Recall@K metrics
4. Compares with old article-based approach
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager
from src.vector_store.indexer import DocumentIndexer
from src.evaluation.question_generator import QuestionGenerator
from src.evaluation.benchmark_runner import BenchmarkRunner

logger = get_logger(__name__)


def main():
    print("=" * 70)
    print("Chunk-Based Evaluation System Test")
    print("=" * 70)
    print()

    collection_name = "ww2_historical_events"
    questions_file = settings.DATA_DIR / "evaluation" / "chunk_based_questions.json"

    # Step 1: Initialize in-memory Qdrant and index
    print("Step 1: Indexing articles into in-memory Qdrant...")
    qdrant = QdrantManager(use_memory=True)
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

    # Step 2: Generate chunk-based questions
    print("Step 2: Generating chunk-based evaluation questions...")
    print()

    generator = QuestionGenerator(qdrant_manager=qdrant)

    questions_data = generator.generate_evaluation_questions(
        collection_name=collection_name,
        num_chunks=20,  # Sample 20 chunks
        questions_per_chunk=1,  # 1 question per chunk
        sampling_strategy="stratified",
        ensure_taxonomy_diversity=True,
    )

    questions = questions_data.get("questions", [])
    metadata = questions_data.get("metadata", {})

    if not questions:
        print("❌ No questions were generated")
        sys.exit(1)

    print(f"✓ Generated {len(questions)} questions from {metadata.get('chunks_sampled')} chunks")
    print(f"  Unique articles: {metadata.get('unique_articles')}")
    print(f"  Type distribution: {metadata.get('taxonomy_distribution')}")
    print()

    # Show sample questions
    print("Sample questions generated:")
    for i, q in enumerate(questions[:3], 1):
        print(f"\n  [{i}] ({q['type']}) {q['question']}")
        print(f"      Source: {q['source_article']}")
        print(f"      Chunk ID: {q.get('source_chunk_id', 'N/A')}")

    # Save questions
    generator.save_questions(questions_data, questions_file)
    print(f"\n✓ Questions saved to: {questions_file}")
    print()

    # Step 3: Run evaluation
    print("=" * 70)
    print("Step 3: Running evaluation with chunk-based ground truth...")
    print("=" * 70)
    print()

    runner = BenchmarkRunner(
        questions_file=questions_file,
        qdrant_manager=qdrant,
        k_values=[1, 3, 5, 10],
    )

    results = runner.run_benchmark(
        collection_name=collection_name,
        phase_name="phase1_vanilla_chunk_based",
        max_questions=None,
    )

    # Print summary
    runner.print_summary(results)

    # Export results
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "chunk_based_metrics.json"

    runner.export_results(results, output_path, format="json")
    print(f"\n✓ Results saved to: {output_path}")

    print()
    print("=" * 70)
    print("Chunk-Based Evaluation Complete")
    print("=" * 70)
    print()

    # Summary comparison
    print("Key Differences from Article-Based Approach:")
    print()
    print("1. Ground Truth:")
    print("   - Old: All chunks from the source article")
    print("   - New: Specific source chunk (+ optional neighbors)")
    print()
    print("2. Precision:")
    print("   - Old: Low precision (many chunks matched)")
    print("   - New: High precision (exact chunk matching)")
    print()
    print("3. Recall@K Interpretation:")
    print("   - Old: Did we find ANY chunk from the article?")
    print("   - New: Did we find THE EXACT chunk with the answer?")
    print()
    print("4. Benefits:")
    print("   - More granular evaluation")
    print("   - Tests retrieval precision, not just article-level recall")
    print("   - Better reflects real-world RAG performance")
    print()

    # Show specific metrics
    print("Metrics Summary:")
    print(f"  Recall@1:  {results.avg_recall_at_k.get(1, 0.0):.3f} (exact match in top result)")
    print(f"  Recall@5:  {results.avg_recall_at_k.get(5, 0.0):.3f} (exact match in top 5)")
    print(f"  Recall@10: {results.avg_recall_at_k.get(10, 0.0):.3f} (exact match in top 10)")
    print(f"  Avg MRR:   {results.avg_mrr:.3f} (how quickly we find the right chunk)")
    print()


if __name__ == "__main__":
    main()
