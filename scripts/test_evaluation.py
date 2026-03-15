#!/usr/bin/env python3
"""
Test evaluation with in-memory Qdrant.
Indexes data and runs evaluation in the same process.
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
    print("Test Evaluation with In-Memory Qdrant")
    print("=" * 70)
    print()

    collection_name = "ww2_historical_events"
    questions_file = settings.DATA_DIR / "evaluation" / "generated_questions.json"

    # Step 1: Initialize in-memory Qdrant
    print("Step 1: Initializing in-memory Qdrant...")
    qdrant = QdrantManager()
    print("✓ Qdrant initialized")
    print()

    # Step 2: Index articles
    print("Step 2: Indexing Wikipedia articles...")
    indexer = DocumentIndexer(qdrant_manager=qdrant)

    stats = indexer.index_all_articles(
        collection_name=collection_name,
        articles_dir=None,  # Use default
        recreate_collection=True,
    )

    if not stats["success"]:
        print(f"❌ Indexing failed: {stats.get('error')}")
        sys.exit(1)

    print(f"✓ Indexed {stats['chunks_indexed']} chunks from {stats['articles_loaded']} articles")
    print()

    # Step 3: Run evaluation
    print("Step 3: Running evaluation benchmark...")
    print()

    if not questions_file.exists():
        print(f"❌ Questions file not found: {questions_file}")
        print("Generate questions first: python scripts/generate_evaluation_questions.py")
        sys.exit(1)

    runner = BenchmarkRunner(
        questions_file=questions_file,
        qdrant_manager=qdrant,
        k_values=[1, 3, 5, 10],
    )

    results = runner.run_benchmark(
        collection_name=collection_name,
        phase_name="phase1_vanilla",
        max_questions=None,  # Evaluate all questions
    )

    # Print summary
    runner.print_summary(results)

    # Export results
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "phase1_vanilla_metrics.json"

    runner.export_results(results, output_path, format="json")
    print(f"\n✓ Results saved to: {output_path}")

    print()
    print("=" * 70)
    print("Evaluation Complete")
    print("=" * 70)

    # Show key metrics
    print()
    print("Key Metrics:")
    print(f"  Recall@1:  {results.avg_recall_at_k.get(1, 0.0):.3f}")
    print(f"  Recall@3:  {results.avg_recall_at_k.get(3, 0.0):.3f}")
    print(f"  Recall@5:  {results.avg_recall_at_k.get(5, 0.0):.3f}")
    print(f"  Recall@10: {results.avg_recall_at_k.get(10, 0.0):.3f}")
    print(f"  Avg MRR:   {results.avg_mrr:.3f}")
    print()


if __name__ == "__main__":
    main()
