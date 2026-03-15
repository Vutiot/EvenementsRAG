#!/usr/bin/env python3
"""
Run Phase 1 baseline evaluation on 976-article dataset with 49 questions.

This script:
1. Indexes 976 articles into in-memory Qdrant
2. Loads the 49 evaluation questions
3. Runs comprehensive evaluation
4. Saves results for comparison with 49-article baseline
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
from src.evaluation.benchmark_runner import BenchmarkRunner

logger = get_logger(__name__)


def main():
    print("=" * 70)
    print("Phase 1 Baseline Evaluation - 976 Articles, 49 Questions")
    print("=" * 70)
    print()

    articles_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_1000"
    collection_name = "ww2_events_1000"
    questions_file = settings.DATA_DIR / "evaluation" / "eval_976_articles_50q.json"
    results_file = project_root / "results" / "phase1_baseline_976articles_49q.json"

    # Step 1: Index articles
    print("Step 1: Indexing 976 articles into in-memory Qdrant...")
    print()

    qdrant = QdrantManager()
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

    # Step 2: Run evaluation
    print("=" * 70)
    print("Step 2: Running Phase 1 Baseline Evaluation")
    print("=" * 70)
    print()

    runner = BenchmarkRunner(
        questions_file=questions_file,
        qdrant_manager=qdrant,
        k_values=[1, 3, 5, 10],
    )

    results = runner.run_benchmark(
        collection_name=collection_name,
        phase_name="phase1_baseline_976articles",
        max_questions=None,
    )

    # Print summary
    runner.print_summary(results)

    # Save results
    results_file.parent.mkdir(parents=True, exist_ok=True)
    runner.export_results(results, results_file, format="json")
    print(f"\n✓ Results saved to: {results_file}")

    # Step 3: Analysis
    print()
    print("=" * 70)
    print("Phase 1 Baseline Analysis")
    print("=" * 70)
    print()

    # Overall performance
    article_hit_5 = results.avg_article_hit_at_k.get(5, 0.0)
    chunk_hit_5 = results.avg_chunk_hit_at_k.get(5, 0.0)

    print("Overall Performance:")
    print(f"  Article Hit@5: {article_hit_5:.1%} - Found source article in top-5")
    print(f"  Chunk Hit@5:   {chunk_hit_5:.1%} - Found exact chunk in top-5")
    print(f"  Avg MRR:       {results.avg_mrr:.3f} - How quickly we find relevant chunks")
    print()

    # Performance by question type
    print("Performance by Question Type (Chunk Hit@5):")
    type_performance = []
    for q_type, metrics in results.metrics_by_type.items():
        chunk_hit = metrics.chunk_hit_at_5
        type_performance.append((q_type, chunk_hit))

    # Sort by performance
    type_performance.sort(key=lambda x: x[1], reverse=True)

    for q_type, hit_rate in type_performance:
        status = "🟢 Strong" if hit_rate >= 0.75 else "🟡 Moderate" if hit_rate >= 0.5 else "🔴 Weak"
        print(f"  {q_type:15s}: {hit_rate:.1%} {status}")
    print()

    # Precision gap
    gap = article_hit_5 - chunk_hit_5
    print("Precision Gap Analysis:")
    print(f"  Article Hit@5 - Chunk Hit@5 = {gap:.1%}")
    if gap < 0.15:
        print(f"  ✓ Good precision: When we find the article, we usually find the chunk")
    else:
        print(f"  ⚠ Precision gap: Often find article but not exact chunk → Reranking needed")
    print()

    print("=" * 70)
    print("Phase 1 Baseline Complete!")
    print("=" * 70)
    print()
    print("Next Step:")
    print("  python scripts/run_phase2_976articles.py")
    print()


if __name__ == "__main__":
    main()
