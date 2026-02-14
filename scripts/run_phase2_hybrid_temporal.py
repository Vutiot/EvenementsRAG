#!/usr/bin/env python3
"""
Run Phase 2 Evaluation: Hybrid Search + Temporal RAG

This script evaluates the combination of:
1. Hybrid Search (BM25 + Semantic with RRF)
2. Temporal Filtering (year-based query understanding)

Compares against Phase 1 baseline to measure improvement.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

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
        print(f"Indexing BM25 for collection '{collection_name}'...")
        self.hybrid_searcher.index_collection(collection_name)
        print()

        print(f"Evaluating {len(self.questions)} questions...")
        print()

        all_metrics = []
        metrics_by_type = {}
        temporal_queries = 0
        temporal_improvements = 0

        for i, question_data in enumerate(self.questions, 1):
            question = question_data["question"]
            q_type = question_data["type"]
            source_chunk_id = question_data["source_chunk_id"]
            source_article_id = str(question_data["source_article_id"])

            # Extract temporal information
            query, temporal_filter_dict = self.temporal_filter.extract_and_filter(
                question
            )

            if temporal_filter_dict:
                temporal_queries += 1

            # Perform hybrid search with optional temporal filter
            results = self.hybrid_searcher.search(
                query=query,
                collection_name=collection_name,
                top_k=max(self.k_values),
                filter_conditions=temporal_filter_dict,
            )

            # Convert to format expected by metrics
            retrieved_chunks = [r.chunk_id for r in results]
            retrieved_payloads = [r.payload for r in results]

            # Compute metrics
            metrics = self._compute_metrics_for_question(
                retrieved_chunks=retrieved_chunks,
                retrieved_payloads=retrieved_payloads,
                source_chunk_id=source_chunk_id,
                source_article_id=source_article_id,
            )

            all_metrics.append(metrics)

            # Track by type
            if q_type not in metrics_by_type:
                metrics_by_type[q_type] = []
            metrics_by_type[q_type].append(metrics)

            # Progress
            if i % 5 == 0 or i == len(self.questions):
                print(f"  Evaluated {i}/{len(self.questions)} questions")

        # Aggregate results
        aggregated = self._aggregate_metrics(all_metrics)

        # Aggregate by type
        agg_by_type = {}
        for q_type, type_metrics in metrics_by_type.items():
            agg_by_type[q_type] = self._aggregate_metrics(type_metrics)

        return {
            "phase": "phase2_hybrid_temporal",
            "total_questions": len(self.questions),
            "temporal_queries_detected": temporal_queries,
            "bm25_weight": self.bm25_weight,
            "aggregated_metrics": aggregated,
            "metrics_by_type": agg_by_type,
            "all_metrics": [vars(m) for m in all_metrics],  # Convert to dicts
        }

    def _compute_metrics_for_question(
        self,
        retrieved_chunks: List[str],
        retrieved_payloads: List[Dict],
        source_chunk_id: str,
        source_article_id: str,
    ) -> RetrievalMetrics:
        """Compute all metrics for a single question."""
        metrics = RetrievalMetrics()

        # Traditional recall
        ground_truth = [source_chunk_id]
        recall_metrics = compute_retrieval_metrics(
            retrieved_chunks=retrieved_chunks,
            ground_truth_chunks=ground_truth,
            k_values=self.k_values,
        )

        # Copy recall values
        for k in self.k_values:
            recall_key = f"recall_at_{k}"
            setattr(metrics, recall_key, getattr(recall_metrics, recall_key))
        metrics.mrr = recall_metrics.mrr
        metrics.ndcg_at_5 = recall_metrics.ndcg_at_5
        metrics.ndcg_at_10 = recall_metrics.ndcg_at_10

        # Binary hit metrics
        for k in self.k_values:
            # Article hit
            article_hit = article_hit_at_k(
                retrieved_chunks=retrieved_chunks,
                retrieved_payloads=retrieved_payloads,
                source_article_id=source_article_id,
                k=k,
            )
            setattr(metrics, f"article_hit_at_{k}", article_hit)

            # Chunk hit
            chunk_hit = chunk_hit_at_k(
                retrieved_chunks=retrieved_chunks,
                source_chunk_id=source_chunk_id,
                k=k,
            )
            setattr(metrics, f"chunk_hit_at_{k}", chunk_hit)

        return metrics

    def _aggregate_metrics(self, metrics_list: List[RetrievalMetrics]) -> Dict:
        """Aggregate metrics across questions."""
        if not metrics_list:
            return {}

        n = len(metrics_list)

        aggregated = {
            "count": n,
            "avg_mrr": sum(m.mrr for m in metrics_list) / n,
            "avg_ndcg_at_5": sum(m.ndcg_at_5 for m in metrics_list) / n,
            "avg_ndcg_at_10": sum(m.ndcg_at_10 for m in metrics_list) / n,
        }

        # Average recall@k
        for k in self.k_values:
            recall_key = f"recall_at_{k}"
            values = [getattr(m, recall_key) for m in metrics_list]
            aggregated[f"avg_{recall_key}"] = sum(values) / n

        # Average hit rates
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
    print("=" * 70)
    print()

    collection_name = "ww2_historical_events"
    questions_file = settings.DATA_DIR / "evaluation" / "eval_50_questions.json"
    phase1_results_file = project_root / "results" / "phase1_baseline_35q.json"
    phase2_results_file = project_root / "results" / "phase2_hybrid_temporal_35q.json"

    # Step 1: Index articles
    print("Step 1: Indexing articles...")
    print()

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
        ("Recall@5", "recall_at", "", 5),
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
            status = "🟢"
        elif delta > 0:
            status = "🟡"
        elif delta > -0.02:
            status = "⚪"
        else:
            status = "🔴"

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
    print(f"  Detected: {results['temporal_queries_detected']}/{results['total_questions']}")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    if phase2_agg["avg_chunk_hit_at_5"] > phase1_agg["avg_chunk_hit_at_k"]["5"]:
        print("✓ Phase 2 (Hybrid + Temporal) shows improvement over Phase 1 baseline")
    else:
        print("⚠ Phase 2 performance is similar to or below Phase 1")

    print()
    print(f"Phase 2 Configuration:")
    print(f"  BM25 Weight: {results['bm25_weight']}")
    print(f"  Semantic Weight: {1.0 - results['bm25_weight']}")
    print()


if __name__ == "__main__":
    main()
