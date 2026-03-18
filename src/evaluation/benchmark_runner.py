"""
Benchmark runner for RAG system evaluation.

Orchestrates the complete evaluation pipeline:
1. Load generated questions
2. Query vector store for each question
3. Compute retrieval metrics
4. Aggregate results by question type
5. Generate evaluation reports

Usage:
    from src.evaluation.benchmark_runner import BenchmarkRunner
    from src.vector_store.qdrant_manager import QdrantManager

    runner = BenchmarkRunner(
        questions_file="data/evaluation/generated_questions.json",
        qdrant_manager=QdrantManager()
    )

    results = runner.run_benchmark(
        collection_name="ww2_events",
        phase_name="phase1_vanilla"
    )

    runner.export_results(results, "results/phase1_metrics.json")
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

from config.settings import settings
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.evaluation.metrics import (
    EvaluationResults,
    RetrievalMetrics,
    compute_metrics_by_type,
    compute_retrieval_metrics,
)
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager

logger = get_logger(__name__)


class BenchmarkRunner:
    """Runs benchmarks and evaluations for RAG system."""

    def __init__(
        self,
        questions_file: Optional[Path] = None,
        qdrant_manager: Optional[QdrantManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        k_values: Optional[List[int]] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            questions_file: Path to generated questions JSON file
            qdrant_manager: Qdrant manager instance
            embedding_generator: Embedding generator instance
            k_values: K values for Recall@K metrics
        """
        self.questions_file = questions_file or (
            settings.DATA_DIR / "evaluation" / "generated_questions.json"
        )
        self.qdrant = qdrant_manager or QdrantManager()
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
        self.k_values = k_values or settings.EVALUATION_K_VALUES

        logger.info(
            "BenchmarkRunner initialized",
            extra={
                "questions_file": str(self.questions_file),
                "k_values": self.k_values,
            },
        )

    def load_questions(self) -> Dict:
        """
        Load generated questions from JSON file.

        Returns:
            Dictionary with metadata and questions
        """
        logger.info(f"Loading questions from {self.questions_file}")

        if not self.questions_file.exists():
            raise FileNotFoundError(
                f"Questions file not found: {self.questions_file}. "
                "Generate questions first using QuestionGenerator."
            )

        with open(self.questions_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data.get("questions", [])
        metadata = data.get("metadata", {})

        logger.info(
            f"Loaded {len(questions)} questions",
            extra={
                "total": len(questions),
                "distribution": metadata.get("taxonomy_distribution", {}),
            },
        )

        return data

    def query_for_question(
        self,
        question: Dict,
        collection_name: str,
        top_k: int = 10,
    ) -> Dict:
        """
        Query vector store for a single question.

        Args:
            question: Question dictionary
            collection_name: Qdrant collection name
            top_k: Number of results to retrieve

        Returns:
            Dictionary with retrieved results and metadata
        """
        question_text = question.get("question", "")

        # Generate query embedding
        query_embedding = self.embedding_gen.generate_embedding(question_text)

        # Search Qdrant
        start_time = time.time()

        results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=None,  # No filtering
        )

        retrieval_time_ms = (time.time() - start_time) * 1000

        # Extract chunk IDs and payloads
        retrieved_chunks = []
        retrieved_payloads = []

        for result in results:
            # Use the Qdrant point ID (UUID) for chunk-based evaluation
            # For backward compatibility, fall back to payload chunk_id if needed
            chunk_id = result["id"]  # UUID format
            retrieved_chunks.append(chunk_id)
            retrieved_payloads.append(result["payload"])

        return {
            "question_id": question.get("id"),
            "question_text": question_text,
            "question_type": question.get("type"),
            "retrieved_chunks": retrieved_chunks,
            "retrieved_payloads": retrieved_payloads,
            "retrieval_time_ms": retrieval_time_ms,
            "num_retrieved": len(results),
        }

    def compute_ground_truth_chunks(
        self,
        question: Dict,
        include_neighbors: bool = True,
        neighbor_window: int = 2,
    ) -> List[str]:
        """
        Compute ground truth chunk IDs for a question.

        For chunk-based questions, ground truth = source chunk + optional neighbors.

        Args:
            question: Question dictionary
            include_neighbors: Include neighboring chunks in ground truth
            neighbor_window: Number of chunks before/after to include (±N)

        Returns:
            List of ground truth chunk IDs
        """
        # Check if pre-computed ground truth exists
        ground_truth = question.get("ground_truth_chunks", [])

        if ground_truth:
            # Use pre-computed ground truth
            return ground_truth

        # For new chunk-based structure: use source_chunk_id
        source_chunk_id = question.get("source_chunk_id")

        if source_chunk_id:
            # Primary ground truth is the exact source chunk
            ground_truth = [source_chunk_id]

            if include_neighbors:
                # Add neighboring chunks for context
                # This requires querying Qdrant to find chunks with adjacent indices
                source_article_id = question.get("source_article_id")

                if source_article_id:
                    # We need to find chunks from the same article with nearby indices
                    # For now, we'll generate expected chunk IDs based on pattern
                    # In production, this should be pre-computed during question generation
                    logger.debug(
                        f"Computing neighbors for chunk {source_chunk_id} "
                        f"(window=±{neighbor_window})"
                    )

            return ground_truth

        # Fallback: Old article-based structure
        # Ground truth = all chunks from the source article
        source_article_id = question.get("source_article_id", "unknown")

        logger.debug(
            f"Using article-based ground truth for question {question.get('id')} "
            f"(article_id={source_article_id})"
        )

        return ground_truth

    def evaluate_question(
        self,
        question: Dict,
        query_result: Dict,
    ) -> Dict:
        """
        Evaluate retrieval performance for a single question.

        Args:
            question: Question dictionary
            query_result: Query result from query_for_question

        Returns:
            Dictionary with question, results, and metrics
        """
        # Get ground truth
        ground_truth_chunks = self.compute_ground_truth_chunks(question)

        # If no pre-computed ground truth, match by source article
        if not ground_truth_chunks:
            source_article_id = str(question.get("source_article_id", "unknown"))
            # Match chunks from the same article
            ground_truth_chunks = []
            for chunk in query_result["retrieved_chunks"]:
                # Check if chunk is from the source article
                if str(chunk).startswith(f"{source_article_id}_"):
                    if chunk not in ground_truth_chunks:
                        ground_truth_chunks.append(chunk)

            # Also add chunks that might be in retrieved payloads
            for payload in query_result["retrieved_payloads"]:
                if str(payload.get("pageid")) == source_article_id:
                    chunk_id = payload.get("chunk_id")
                    if chunk_id and chunk_id not in ground_truth_chunks:
                        ground_truth_chunks.append(chunk_id)

        # Compute retrieval metrics
        metrics = compute_retrieval_metrics(
            retrieved_chunks=query_result["retrieved_chunks"],
            ground_truth_chunks=ground_truth_chunks,
            k_values=self.k_values,
            retrieved_payloads=query_result["retrieved_payloads"],
            source_article_id=question.get("source_article_id"),
            source_chunk_id=question.get("source_chunk_id"),
        )

        return {
            "question_id": question.get("id"),
            "question": question.get("question"),
            "type": question.get("type") or "unknown",
            "difficulty": question.get("difficulty"),
            "source_article": question.get("source_article"),
            "ground_truth_count": len(ground_truth_chunks),
            "retrieved_count": query_result["num_retrieved"],
            "retrieval_time_ms": query_result["retrieval_time_ms"],
            "metrics": metrics,
        }

    def run_benchmark(
        self,
        collection_name: str,
        phase_name: str = "default",
        max_questions: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, Dict], None]] = None,
    ) -> EvaluationResults:
        """
        Run complete benchmark evaluation.

        Args:
            collection_name: Qdrant collection name
            phase_name: Name of RAG phase being evaluated
            max_questions: Maximum questions to evaluate (None = all)

        Returns:
            EvaluationResults object with complete results
        """
        logger.info(
            f"Running benchmark for phase '{phase_name}' on collection '{collection_name}'"
        )

        # Load questions
        questions_data = self.load_questions()
        questions = questions_data.get("questions", [])

        if max_questions:
            questions = questions[:max_questions]
            logger.info(f"Limited to {max_questions} questions for testing")

        # Check collection exists
        if not self.qdrant.collection_exists(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' does not exist. "
                "Run indexing first using DocumentIndexer."
            )

        collection_info = self.qdrant.get_collection_info(collection_name)
        logger.info(
            f"Collection info: {collection_info['points_count']} chunks indexed"
        )

        # Evaluate each question
        per_question_results = []
        total_retrieval_time = 0

        for i, question in enumerate(tqdm(questions, desc=f"Evaluating {phase_name}")):
            # Query
            query_result = self.query_for_question(
                question,
                collection_name,
                top_k=max(self.k_values),
            )

            # Evaluate
            evaluation = self.evaluate_question(question, query_result)
            per_question_results.append(evaluation)

            total_retrieval_time += evaluation["retrieval_time_ms"]

            if progress_callback:
                progress_callback(i, len(questions), evaluation)

        # Aggregate metrics
        all_metrics = [r["metrics"] for r in per_question_results]

        avg_recall_at_k = {}
        for k in self.k_values:
            recalls = []
            for m in all_metrics:
                if k == 1:
                    recalls.append(m.recall_at_1)
                elif k == 3:
                    recalls.append(m.recall_at_3)
                elif k == 5:
                    recalls.append(m.recall_at_5)
                elif k == 10:
                    recalls.append(m.recall_at_10)
            avg_recall_at_k[k] = sum(recalls) / len(recalls) if recalls else 0.0

        avg_mrr = sum(m.mrr for m in all_metrics) / len(all_metrics)

        avg_ndcg = {
            5: sum(m.ndcg_at_5 for m in all_metrics) / len(all_metrics),
            10: sum(m.ndcg_at_10 for m in all_metrics) / len(all_metrics),
        }

        # Aggregate article-level hit rates (binary: found ANY chunk from source article)
        avg_article_hit_at_k = {}
        for k in self.k_values:
            hits = []
            for m in all_metrics:
                if k == 1:
                    hits.append(m.article_hit_at_1)
                elif k == 3:
                    hits.append(m.article_hit_at_3)
                elif k == 5:
                    hits.append(m.article_hit_at_5)
                elif k == 10:
                    hits.append(m.article_hit_at_10)
            avg_article_hit_at_k[k] = sum(hits) / len(hits) if hits else 0.0

        # Aggregate chunk-level hit rates (binary: found THE EXACT chunk)
        avg_chunk_hit_at_k = {}
        for k in self.k_values:
            hits = []
            for m in all_metrics:
                if k == 1:
                    hits.append(m.chunk_hit_at_1)
                elif k == 3:
                    hits.append(m.chunk_hit_at_3)
                elif k == 5:
                    hits.append(m.chunk_hit_at_5)
                elif k == 10:
                    hits.append(m.chunk_hit_at_10)
            avg_chunk_hit_at_k[k] = sum(hits) / len(hits) if hits else 0.0

        # Compute per-type metrics
        metrics_by_type = compute_metrics_by_type(per_question_results)

        # Count questions meeting quality threshold
        questions_with_good_recall = sum(
            1 for m in all_metrics if m.recall_at_5 >= settings.EVALUATION_MIN_RECALL_AT_5
        )

        # Create results
        results = EvaluationResults(
            avg_recall_at_k=avg_recall_at_k,
            avg_mrr=avg_mrr,
            avg_ndcg=avg_ndcg,
            avg_article_hit_at_k=avg_article_hit_at_k,
            avg_chunk_hit_at_k=avg_chunk_hit_at_k,
            metrics_by_type=metrics_by_type,
            per_question_metrics=per_question_results,
            total_questions=len(questions),
            questions_with_recall_at_5_gt_50=questions_with_good_recall,
            avg_retrieval_time_ms=total_retrieval_time / len(questions),
        )

        logger.info(
            "Benchmark completed",
            extra={
                "phase": phase_name,
                "questions": len(questions),
                "avg_recall@5": avg_recall_at_k.get(5, 0.0),
                "avg_mrr": avg_mrr,
            },
        )

        return results

    def export_results(
        self,
        results: EvaluationResults,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """
        Export evaluation results to file.

        Args:
            results: EvaluationResults object
            output_path: Output file path
            format: Export format ('json' or 'csv')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)

        elif format == "csv":
            # Export per-question metrics to CSV
            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = [
                    "question_id",
                    "type",
                    "difficulty",
                    "recall@1",
                    "recall@3",
                    "recall@5",
                    "recall@10",
                    "mrr",
                    "ndcg@5",
                    "retrieval_time_ms",
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in results.per_question_metrics:
                    metrics = result["metrics"]
                    writer.writerow({
                        "question_id": result["question_id"],
                        "type": result["type"],
                        "difficulty": result["difficulty"],
                        "recall@1": metrics.recall_at_1,
                        "recall@3": metrics.recall_at_3,
                        "recall@5": metrics.recall_at_5,
                        "recall@10": metrics.recall_at_10,
                        "mrr": metrics.mrr,
                        "ndcg@5": metrics.ndcg_at_5,
                        "retrieval_time_ms": result["retrieval_time_ms"],
                    })

        logger.info(f"Exported results to {output_path}")

    def print_summary(self, results: EvaluationResults) -> None:
        """
        Print a summary of evaluation results.

        Args:
            results: EvaluationResults object
        """
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        print(f"\nTotal Questions: {results.total_questions}")
        print(f"Avg Retrieval Time: {results.avg_retrieval_time_ms:.1f} ms")

        print("\n--- Traditional Recall@K Metrics ---")
        for k, recall in results.avg_recall_at_k.items():
            print(f"  Recall@{k}:  {recall:.3f}")
        print(f"  MRR:        {results.avg_mrr:.3f}")
        for k, ndcg in results.avg_ndcg.items():
            print(f"  NDCG@{k}:    {ndcg:.3f}")

        # Add new binary metrics if available
        if hasattr(results, 'avg_article_hit_at_k') and results.avg_article_hit_at_k:
            print("\n--- Article-Level Hit Rate (Binary: Found ANY chunk from source article?) ---")
            for k, hit_rate in results.avg_article_hit_at_k.items():
                print(f"  Article Hit@{k}:  {hit_rate:.1%} ({hit_rate * results.total_questions:.0f}/{results.total_questions})")

        if hasattr(results, 'avg_chunk_hit_at_k') and results.avg_chunk_hit_at_k:
            print("\n--- Chunk-Level Hit Rate (Binary: Found THE EXACT chunk?) ---")
            for k, hit_rate in results.avg_chunk_hit_at_k.items():
                print(f"  Chunk Hit@{k}:  {hit_rate:.1%} ({hit_rate * results.total_questions:.0f}/{results.total_questions})")

        print(f"\nQuestions with Recall@5 >= {settings.EVALUATION_MIN_RECALL_AT_5:.1%}: "
              f"{results.questions_with_recall_at_5_gt_50}/{results.total_questions}")

        print("\n--- Metrics by Question Type ---")
        for q_type, metrics in results.metrics_by_type.items():
            print(f"\n  {q_type.capitalize()}:")
            print(f"    Recall@5: {metrics.recall_at_5:.3f}")
            print(f"    MRR:      {metrics.mrr:.3f}")
            print(f"    NDCG@5:   {metrics.ndcg_at_5:.3f}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test benchmark runner
    import sys

    print("=" * 70)
    print("Benchmark Runner Test")
    print("=" * 70)

    # Check if questions file exists
    questions_file = settings.DATA_DIR / "evaluation" / "generated_questions.json"

    if not questions_file.exists():
        print(f"\n⚠ Questions file not found: {questions_file}")
        print("\nGenerate questions first:")
        print("  python -m src.evaluation.question_generator")
        print("  or run: python scripts/generate_evaluation_questions.py")
        sys.exit(1)

    # Check if vector store is ready
    print("\nChecking vector store...")
    qdrant = QdrantManager()

    collection_name = "test_benchmark"

    # We need to index some documents first
    print("⚠ This test requires indexed documents in Qdrant.")
    print("\nTo run a full benchmark:")
    print("  1. Start Qdrant: bash scripts/setup_qdrant.sh start")
    print("  2. Index documents: python -m src.vector_store.indexer")
    print("  3. Generate questions: python scripts/generate_evaluation_questions.py")
    print("  4. Run benchmark: python scripts/run_evaluation.py")

    print("\n✓ Benchmark runner module created successfully!")
    print("See scripts/run_evaluation.py for usage examples.")
    print("=" * 70)
