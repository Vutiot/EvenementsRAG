#!/usr/bin/env python3
"""
Establish comprehensive Phase 1 baseline with 30 questions.

This script:
1. Indexes all articles into in-memory Qdrant
2. Generates 30 chunk-based questions with LLM
3. Runs comprehensive evaluation
4. Analyzes results and saves baseline report
"""

import json
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
    print("Phase 1 Baseline Evaluation - 30 Questions")
    print("=" * 70)
    print()

    collection_name = "ww2_historical_events"
    questions_file = settings.DATA_DIR / "evaluation" / "phase1_baseline_30q.json"
    results_file = project_root / "results" / "phase1_baseline_30q.json"

    # Step 1: Index articles
    print("Step 1: Indexing articles into in-memory Qdrant...")
    print()

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

    # Step 2: Generate 30 questions
    print("=" * 70)
    print("Step 2: Generating 30 evaluation questions")
    print("=" * 70)
    print()

    generator = QuestionGenerator(qdrant_manager=qdrant)

    questions_data = generator.generate_evaluation_questions(
        collection_name=collection_name,
        num_chunks=30,
        questions_per_chunk=1,
        sampling_strategy="stratified",
        ensure_taxonomy_diversity=True,
    )

    questions = questions_data.get("questions", [])
    metadata = questions_data.get("metadata", {})

    if not questions:
        print("❌ No questions were generated")
        sys.exit(1)

    print(f"✓ Generated {len(questions)} questions")
    print(f"  Chunks sampled:  {metadata.get('chunks_sampled')}")
    print(f"  Unique articles: {metadata.get('unique_articles')}")
    print()

    # Show type distribution
    type_dist = metadata.get("taxonomy_distribution", {})
    print("Question Type Distribution:")
    for q_type in sorted(type_dist.keys()):
        count = type_dist[q_type]
        percentage = (count / len(questions)) * 100
        print(f"  {q_type:15s}: {count:2d} ({percentage:5.1f}%)")
    print()

    # Save questions
    generator.save_questions(questions_data, questions_file)
    print(f"✓ Questions saved to: {questions_file}")
    print()

    # Step 3: Run evaluation
    print("=" * 70)
    print("Step 3: Running Phase 1 Baseline Evaluation")
    print("=" * 70)
    print()

    runner = BenchmarkRunner(
        questions_file=questions_file,
        qdrant_manager=qdrant,
        k_values=[1, 3, 5, 10],
    )

    results = runner.run_benchmark(
        collection_name=collection_name,
        phase_name="phase1_baseline",
        max_questions=None,
    )

    # Print summary
    runner.print_summary(results)

    # Save results
    results_file.parent.mkdir(parents=True, exist_ok=True)
    runner.export_results(results, results_file, format="json")
    print(f"\n✓ Results saved to: {results_file}")

    # Step 4: Detailed analysis
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

    # Identify strengths and weaknesses
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

    # Recommendations
    print("Key Insights:")
    print()

    weakest_type = type_performance[-1][0]
    weakest_score = type_performance[-1][1]
    strongest_type = type_performance[0][0]
    strongest_score = type_performance[0][1]

    print(f"✓ Strongest: {strongest_type} ({strongest_score:.1%})")
    print(f"⚠ Weakest:   {weakest_type} ({weakest_score:.1%})")
    print()

    if weakest_score < 0.6:
        print(f"📊 Recommendation: Phase 2 (Temporal RAG) should target {weakest_type} questions")
        if "temporal" in weakest_type.lower():
            print("   → Implement date-aware filtering and temporal metadata")
        elif "relationship" in weakest_type.lower():
            print("   → Phase 4 (Graph RAG) will address relationship questions")
        elif "comparative" in weakest_type.lower():
            print("   → Phase 3 (Hybrid + Reranking) should help with comparisons")
    else:
        print("📊 All question types performing reasonably well (>60%)")
        print("   → Vanilla RAG provides a solid baseline")

    print()

    # Article vs Chunk hit comparison
    gap = article_hit_5 - chunk_hit_5
    print("Precision Analysis:")
    if gap < 0.05:
        print(f"  ✓ High precision: When we find the right article, we almost always")
        print(f"    find the exact chunk (gap: {gap:.1%})")
    elif gap < 0.15:
        print(f"  ~ Moderate precision: Sometimes we find the article but miss the chunk")
        print(f"    (gap: {gap:.1%})")
    else:
        print(f"  ⚠ Low precision: Often find the article but not the exact chunk")
        print(f"    (gap: {gap:.1%}) - Reranking may help")
    print()

    print("=" * 70)
    print("Phase 1 Baseline Complete!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Review results in:", results_file)
    print("  2. Review questions in:", questions_file)
    print("  3. If temporal questions are weak → Implement Phase 2")
    print("  4. If all types strong → Focus on improving Chunk Hit rate")
    print()


if __name__ == "__main__":
    main()
