#!/usr/bin/env python3
"""
Run Phase 2 Evaluation on 976-Article Dataset: Hybrid Search + Temporal RAG

This script evaluates the combination of:
1. Hybrid Search (BM25 + Semantic with RRF)
2. Temporal Filtering (year-based query understanding)

Compares against Phase 1 baseline (976 articles) to measure improvement.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager
from src.vector_store.indexer import DocumentIndexer
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.temporal_filter import TemporalFilter
from src.evaluation.metrics import (
    RetrievalMetrics,
    compute_retrieval_metrics,
    article_hit_at_k,
    chunk_hit_at_k,
)

logger = get_logger(__name__)


class Phase2BenchmarkRunner:
    """Benchmark runner for Phase 2 (Hybrid + Temporal)."""

    def __init__(
        self,
        questions_file: Path,
        qdrant_manager: QdrantManager,
        k_values: List[int] = [1, 3, 5, 10],
        bm25_weight: float = 0.3,
    ):
        self.questions_file = questions_file
        self.qdrant = qdrant_manager
        self.k_values = k_values
        self.bm25_weight = bm25_weight

        # Initialize components
        self.embedding_gen = EmbeddingGenerator()
        self.hybrid_searcher = HybridSearcher(
            qdrant_manager=qdrant_manager,
            embedding_generator=self.embedding_gen,
            bm25_weight=bm25_weight,
        )
        self.temporal_filter = TemporalFilter()

        # Load questions
        with open(questions_file, "r") as f:
            data = json.load(f)
            self.questions = data["questions"]

        logger.info(
            f"Phase2BenchmarkRunner initialized: "
            f"{len(self.questions)} questions, bm25_weight={bm25_weight}"
        )

    def run_benchmark(self, collection_name: str) -> Dict:
        """
        Run Phase 2 benchmark with hybrid search + temporal filtering.

        Args:
            collection_name: Qdrant collection to search

        Returns:
            Results dictionary
        """
        print(f"Running Phase 2 evaluation on {len(self.questions)} questions...")
        print(f"Configuration: BM25 ({self.bm25_weight:.0%}) + Semantic ({1-self.bm25_weight:.0%})")
        print()

        results = {
            "phase": "phase2_hybrid_temporal",
            "timestamp": datetime.now().isoformat(),
            "bm25_weight": self.bm25_weight,
            "total_questions": len(self.questions),
            "temporal_queries_detected": 0,
            "questions": [],
        }

        all_metrics = []
        temporal_count = 0

        for i, question_data in enumerate(self.questions, 1):
            question = question_data["question"]
            source_chunk_id = question_data["source_chunk_id"]
            source_article = question_data["source_article"]

            # Detect temporal information
            temporal_info = self.temporal_filter.extract_temporal_info(question)
            is_temporal = temporal_info is not None
            if is_temporal:
                temporal_count += 1

            # Convert temporal info to Qdrant filter if needed
            filter_conditions = None
            if is_temporal:
                filter_conditions = self.temporal_filter.create_qdrant_filter(temporal_info)

            # Search with hybrid approach
            search_results = self.hybrid_searcher.search(
                query=question,
                collection_name=collection_name,
                top_k=max(self.k_values),
                filter_conditions=filter_conditions,
            )

            # Extract chunk IDs and payloads from SearchResult objects
            retrieved_chunk_ids = [r.chunk_id for r in search_results]
            retrieved_payloads = [r.payload for r in search_results]

            # Compute metrics
            metrics = compute_retrieval_metrics(
                retrieved_chunks=retrieved_chunk_ids,
                ground_truth_chunks=[source_chunk_id],
                k_values=self.k_values,
                retrieved_payloads=retrieved_payloads,
                source_article_id=source_article,
                source_chunk_id=source_chunk_id,
            )

            all_metrics.append(metrics)

            # Store question result
            results["questions"].append({
                "question": question,
                "source_article": source_article,
                "source_chunk_id": source_chunk_id,
                "is_temporal": is_temporal,
                "temporal_info": temporal_info if is_temporal else None,
                "metrics": metrics.to_dict(),
                "top_5_results": [
                    {
                        "chunk_id": r.chunk_id,
                        "article_title": r.payload.get("article_title", "Unknown"),
                        "score": r.score,
                    }
                    for r in search_results[:5]
                ],
            })

            # Progress
            if i % 10 == 0:
                print(f"  Processed {i}/{len(self.questions)} questions...")

        results["temporal_queries_detected"] = temporal_count

        # Aggregate metrics
        results["aggregated_metrics"] = self._aggregate_metrics(all_metrics)

        print()
        print("=" * 70)
        print("Phase 2 Results")
        print("=" * 70)
        print()

        agg = results["aggregated_metrics"]
        print(f"Total Questions: {len(self.questions)}")
        print(f"Temporal Queries: {temporal_count}/{len(self.questions)} ({temporal_count/len(self.questions):.1%})")
        print()

        print("Overall Performance:")
        print(f"  Article Hit@5: {agg['avg_article_hit_at_5']:.1%}")
        print(f"  Chunk Hit@5:   {agg['avg_chunk_hit_at_5']:.1%}")
        print(f"  MRR:           {agg['avg_mrr']:.3f}")
        print()

        return results

    def _aggregate_metrics(self, metrics_list: List[RetrievalMetrics]) -> Dict:
        """Aggregate metrics across all questions."""
        if not metrics_list:
            return {}

        n = len(metrics_list)
        aggregated = {}

        # MRR
        aggregated["avg_mrr"] = sum(m.mrr for m in metrics_list) / n

        # Hit rates at each K
        for k in self.k_values:
            # Article hit
            article_key = f"article_hit_at_{k}"
            values = [getattr(m, article_key) for m in metrics_list]
            aggregated[f"avg_{article_key}"] = sum(values) / n

            # Chunk hit
            chunk_key = f"chunk_hit_at_{k}"
            values = [getattr(m, chunk_key) for m in metrics_list]
            aggregated[f"avg_{chunk_key}"] = sum(values) / n

        return aggregated


def main():
    print("=" * 70)
    print("Phase 2 Evaluation: Hybrid Search + Temporal RAG")
    print("976 Articles, 49 Questions")
    print("=" * 70)
    print()

    articles_dir = settings.DATA_DIR / "raw" / "wikipedia_articles_1000"
    collection_name = "ww2_events_1000"
    questions_file = settings.DATA_DIR / "evaluation" / "eval_976_articles_50q.json"
    phase1_results_file = project_root / "results" / "phase1_baseline_976articles_49q.json"
    phase2_results_file = project_root / "results" / "phase2_hybrid_temporal_976articles_49q.json"

    # Step 1: Index articles
    print("Step 1: Indexing 976 articles...")
    print()

    qdrant = QdrantManager(use_memory=True)
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

    # Step 2: Run Phase 2 evaluation
    print("=" * 70)
    print("Step 2: Running Phase 2 Evaluation")
    print("=" * 70)
    print()

    runner = Phase2BenchmarkRunner(
        questions_file=questions_file,
        qdrant_manager=qdrant,
        k_values=[1, 3, 5, 10],
        bm25_weight=0.3,  # 30% BM25, 70% semantic
    )

    # Build BM25 index for hybrid search
    print("Building BM25 index for hybrid search...")
    runner.hybrid_searcher.index_collection(collection_name)
    print("✓ BM25 index ready")
    print()

    results = runner.run_benchmark(collection_name=collection_name)

    # Save results
    phase2_results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(phase2_results_file, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"✓ Results saved to: {phase2_results_file}")
    print()

    # Step 3: Compare with Phase 1
    print("=" * 70)
    print("Phase 1 vs Phase 2 Comparison")
    print("=" * 70)
    print()

    # Load Phase 1 results
    with open(phase1_results_file, "r") as f:
        phase1 = json.load(f)

    # Phase 1 has metrics directly in root, Phase 2 in aggregated_metrics
    phase1_agg = phase1
    phase2_agg = results["aggregated_metrics"]

    # Overall performance comparison
    print("Overall Performance Comparison:")
    print()

    metrics_to_compare = [
        ("Article Hit@5", "article_hit_at", "%", 5),
        ("Chunk Hit@5", "chunk_hit_at", "%", 5),
        ("MRR", "mrr", "", None),
    ]

    for label, metric_name, fmt, k_val in metrics_to_compare:
        # Handle different key formats between Phase 1 and Phase 2
        if k_val:
            # Phase 1: avg_article_hit_at_k['5']
            # Phase 2: avg_article_hit_at_5
            phase1_key = f"avg_{metric_name}_k"
            phase2_key = f"avg_{metric_name}_{k_val}"

            phase1_val = phase1_agg[phase1_key][str(k_val)]
            phase2_val = phase2_agg[phase2_key]
        else:
            # MRR is the same in both
            phase1_val = phase1_agg["avg_mrr"]
            phase2_val = phase2_agg["avg_mrr"]

        delta = phase2_val - phase1_val

        if fmt == "%":
            phase1_str = f"{phase1_val:.1%}"
            phase2_str = f"{phase2_val:.1%}"
            delta_str = f"{delta:+.1%}"
        else:
            phase1_str = f"{phase1_val:.3f}"
            phase2_str = f"{phase2_val:.3f}"
            delta_str = f"{delta:+.3f}"

        # Color code improvement
        if delta > 0.02:
            status = "🟢 Improved"
        elif delta > 0:
            status = "🟡 Slight improvement"
        elif delta > -0.02:
            status = "⚪ No change"
        else:
            status = "🔴 Decreased"

        print(f"  {label:15s}: {phase1_str:>8s} → {phase2_str:>8s} ({delta_str:>7s}) {status}")

    print()

    # Precision gap
    gap_phase1 = phase1_agg["avg_article_hit_at_k"]["5"] - phase1_agg["avg_chunk_hit_at_k"]["5"]
    gap_phase2 = phase2_agg["avg_article_hit_at_5"] - phase2_agg["avg_chunk_hit_at_5"]

    print(f"Precision Gap (Article Hit@5 - Chunk Hit@5):")
    print(f"  Phase 1: {gap_phase1:.1%}")
    print(f"  Phase 2: {gap_phase2:.1%}")
    print(f"  Change:  {gap_phase2 - gap_phase1:+.1%}")
    print()

    # Temporal queries
    print(f"Temporal Queries:")
    print(f"  Detected: {results['temporal_queries_detected']}/{results['total_questions']} ({results['temporal_queries_detected']/results['total_questions']:.1%})")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    chunk_improvement = phase2_agg["avg_chunk_hit_at_5"] - phase1_agg["avg_chunk_hit_at_k"]["5"]

    if chunk_improvement > 0.02:
        print("✓ Phase 2 (Hybrid + Temporal) shows meaningful improvement over Phase 1 baseline")
    elif chunk_improvement > 0:
        print("🟡 Phase 2 shows slight improvement over Phase 1")
    else:
        print("⚠ Phase 2 performance is similar to or below Phase 1")
        print("  Recommendation: Stick with Phase 1 (pure semantic search)")

    print()
    print(f"Phase 2 Configuration:")
    print(f"  BM25 Weight: {results['bm25_weight']:.0%}")
    print(f"  Semantic Weight: {1.0 - results['bm25_weight']:.0%}")
    print(f"  RRF Fusion: k=60")
    print()

    print("Next Steps:")
    print("  1. Review detailed results: cat results/phase2_hybrid_temporal_976articles_49q.json | jq")
    print("  2. Compare with 49-article Phase 2 results if available")
    print()


if __name__ == "__main__":
    main()
