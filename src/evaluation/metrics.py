"""
Evaluation metrics for RAG system.

Implements both retrieval and generation metrics:
- Retrieval: Recall@K, MRR, NDCG
- Generation: ROUGE-L, BERTScore, RAGAS (faithfulness, answer relevance)

Usage:
    from src.evaluation.metrics import compute_retrieval_metrics, RetrievalMetrics

    metrics = compute_retrieval_metrics(
        retrieved_chunks=["chunk_1", "chunk_3", "chunk_5"],
        ground_truth_chunks=["chunk_1", "chunk_2", "chunk_3"],
        k_values=[1, 3, 5]
    )
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _hit_from_rank(rank: Optional[int], k: int) -> float:
    """Derive a binary hit value from a 1-indexed rank."""
    return 1.0 if rank is not None and rank <= k else 0.0


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics for a single question."""

    # Traditional recall: #relevant_in_topK / #total_relevant
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0

    # Ground truth rank: 1-indexed position (None = not found)
    ground_truth_article_rank: Optional[int] = None
    ground_truth_chunk_rank: Optional[int] = None

    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0

    # -- Derived binary hit properties (backward-compatible) ----------------

    @property
    def article_hit_at_1(self) -> float:
        return _hit_from_rank(self.ground_truth_article_rank, 1)

    @property
    def article_hit_at_3(self) -> float:
        return _hit_from_rank(self.ground_truth_article_rank, 3)

    @property
    def article_hit_at_5(self) -> float:
        return _hit_from_rank(self.ground_truth_article_rank, 5)

    @property
    def article_hit_at_10(self) -> float:
        return _hit_from_rank(self.ground_truth_article_rank, 10)

    @property
    def chunk_hit_at_1(self) -> float:
        return _hit_from_rank(self.ground_truth_chunk_rank, 1)

    @property
    def chunk_hit_at_3(self) -> float:
        return _hit_from_rank(self.ground_truth_chunk_rank, 3)

    @property
    def chunk_hit_at_5(self) -> float:
        return _hit_from_rank(self.ground_truth_chunk_rank, 5)

    @property
    def chunk_hit_at_10(self) -> float:
        return _hit_from_rank(self.ground_truth_chunk_rank, 10)

    def to_dict(self) -> Dict:
        """Convert to dictionary, including derived binary hit fields."""
        d = asdict(self)
        d["article_hit_at_1"] = self.article_hit_at_1
        d["article_hit_at_3"] = self.article_hit_at_3
        d["article_hit_at_5"] = self.article_hit_at_5
        d["article_hit_at_10"] = self.article_hit_at_10
        d["chunk_hit_at_1"] = self.chunk_hit_at_1
        d["chunk_hit_at_3"] = self.chunk_hit_at_3
        d["chunk_hit_at_5"] = self.chunk_hit_at_5
        d["chunk_hit_at_10"] = self.chunk_hit_at_10
        return d

    def __repr__(self) -> str:
        return (
            f"RetrievalMetrics(recall@5={self.recall_at_5:.3f}, "
            f"mrr={self.mrr:.3f}, ndcg@5={self.ndcg_at_5:.3f})"
        )


@dataclass
class EvaluationResults:
    """Complete evaluation results for a test set."""

    # Overall metrics
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)
    avg_mrr: float = 0.0
    avg_ndcg: Dict[int, float] = field(default_factory=dict)

    # Binary hit rates (averaged across questions)
    avg_article_hit_at_k: Dict[int, float] = field(default_factory=dict)
    avg_chunk_hit_at_k: Dict[int, float] = field(default_factory=dict)

    # Per-question-type breakdown
    metrics_by_type: Dict[str, RetrievalMetrics] = field(default_factory=dict)

    # Per-question results
    per_question_metrics: List[Dict] = field(default_factory=list)

    # Performance statistics
    total_questions: int = 0
    questions_with_recall_at_5_gt_50: int = 0
    avg_retrieval_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert RetrievalMetrics objects to dicts (with derived hit fields)
        result["metrics_by_type"] = {
            k: v.to_dict() if isinstance(v, RetrievalMetrics) else v
            for k, v in self.metrics_by_type.items()
        }
        # Ensure per_question_metrics uses RetrievalMetrics.to_dict() for
        # proper inclusion of derived binary hit fields
        serialized_pq = []
        for r in self.per_question_metrics:
            entry = dict(r)
            m = r.get("metrics")
            if isinstance(m, RetrievalMetrics):
                entry["metrics"] = m.to_dict()
            serialized_pq.append(entry)
        result["per_question_metrics"] = serialized_pq
        return result

    def __repr__(self) -> str:
        recall_5 = self.avg_recall_at_k.get(5, 0.0)
        return (
            f"EvaluationResults(questions={self.total_questions}, "
            f"avg_recall@5={recall_5:.3f}, avg_mrr={self.avg_mrr:.3f})"
        )


def recall_at_k(retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
    """
    Compute Recall@K: fraction of ground truth chunks found in top-K retrieved chunks.

    Args:
        retrieved_chunks: List of retrieved chunk IDs (ordered by relevance)
        ground_truth_chunks: List of ground truth chunk IDs
        k: Number of top results to consider

    Returns:
        Recall value between 0.0 and 1.0

    Examples:
        >>> recall_at_k(["a", "b", "c"], ["a", "b"], k=3)
        1.0
        >>> recall_at_k(["a", "b", "c"], ["a", "d"], k=3)
        0.5
    """
    if not ground_truth_chunks:
        logger.warning("Ground truth is empty for recall calculation")
        return 0.0

    top_k = set(retrieved_chunks[:k])
    relevant = set(ground_truth_chunks)

    recall = len(top_k & relevant) / len(relevant)
    return recall


def mrr(retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR): inverse of rank of first relevant result.

    Measures how quickly we find the first correct answer.
    Higher is better (1.0 = first result is relevant).

    Args:
        retrieved_chunks: List of retrieved chunk IDs (ordered by relevance)
        ground_truth_chunks: List of ground truth chunk IDs

    Returns:
        MRR value between 0.0 and 1.0

    Examples:
        >>> mrr(["a", "b", "c"], ["a"])
        1.0
        >>> mrr(["a", "b", "c"], ["b"])
        0.5
        >>> mrr(["a", "b", "c"], ["c"])
        0.333...
        >>> mrr(["a", "b", "c"], ["d"])
        0.0
    """
    if not ground_truth_chunks:
        logger.warning("Ground truth is empty for MRR calculation")
        return 0.0

    relevant = set(ground_truth_chunks)

    for i, chunk_id in enumerate(retrieved_chunks, 1):
        if chunk_id in relevant:
            return 1.0 / i

    return 0.0


def ndcg_at_k(retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).

    Considers both relevance and ranking position.
    Gives higher weight to relevant results at top positions.

    Args:
        retrieved_chunks: List of retrieved chunk IDs (ordered by relevance)
        ground_truth_chunks: List of ground truth chunk IDs
        k: Number of top results to consider

    Returns:
        NDCG value between 0.0 and 1.0

    Examples:
        >>> ndcg_at_k(["a", "b", "c"], ["a", "b"], k=3)
        1.0
        >>> ndcg_at_k(["c", "b", "a"], ["a", "b"], k=3)
        < 1.0  # Lower because relevant docs are not at top
    """
    if not ground_truth_chunks:
        logger.warning("Ground truth is empty for NDCG calculation")
        return 0.0

    relevant = set(ground_truth_chunks)

    # Binary relevance: 1 if in ground truth, 0 otherwise
    relevance_scores = [1 if chunk in relevant else 0 for chunk in retrieved_chunks[:k]]

    if sum(relevance_scores) == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i+1))
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores))

    # Ideal DCG (all relevant docs at top)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def precision_at_k(retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
    """
    Compute Precision@K: fraction of top-K retrieved chunks that are relevant.

    Args:
        retrieved_chunks: List of retrieved chunk IDs (ordered by relevance)
        ground_truth_chunks: List of ground truth chunk IDs
        k: Number of top results to consider

    Returns:
        Precision value between 0.0 and 1.0
    """
    if k == 0:
        return 0.0

    top_k = retrieved_chunks[:k]
    relevant = set(ground_truth_chunks)

    precision = len([c for c in top_k if c in relevant]) / k
    return precision


def find_article_rank(
    retrieved_chunks: List[str],
    retrieved_payloads: List[Dict],
    source_article_id: str,
) -> Optional[int]:
    """
    Find the 1-indexed position of the first chunk from the source article.

    Args:
        retrieved_chunks: List of retrieved chunk UUIDs
        retrieved_payloads: List of payload dicts with metadata
        source_article_id: The source article ID (pageid or article title)

    Returns:
        1-indexed rank, or None if not found
    """
    if not source_article_id:
        return None

    for i, payload in enumerate(retrieved_payloads, 1):
        if str(payload.get("pageid")) == str(source_article_id):
            return i
        if payload.get("article_title") == source_article_id:
            return i

    return None


def find_chunk_rank(
    retrieved_chunks: List[str],
    source_chunk_id: str,
) -> Optional[int]:
    """
    Find the 1-indexed position of the exact source chunk.

    Args:
        retrieved_chunks: List of retrieved chunk UUIDs (ordered by relevance)
        source_chunk_id: The exact source chunk UUID

    Returns:
        1-indexed rank, or None if not found
    """
    if not source_chunk_id:
        return None

    for i, chunk_id in enumerate(retrieved_chunks, 1):
        if chunk_id == source_chunk_id:
            return i

    return None


def article_hit_at_k(
    retrieved_chunks: List[str],
    retrieved_payloads: List[Dict],
    source_article_id: str,
    k: int
) -> float:
    """
    Binary metric: Did we retrieve ANY chunk from the source article in top-K?

    Delegates to find_article_rank for the actual search.
    """
    rank = find_article_rank(retrieved_chunks, retrieved_payloads, source_article_id)
    return 1.0 if rank is not None and rank <= k else 0.0


def chunk_hit_at_k(
    retrieved_chunks: List[str],
    source_chunk_id: str,
    k: int
) -> float:
    """
    Binary metric: Did we retrieve THE EXACT source chunk in top-K?

    Delegates to find_chunk_rank for the actual search.
    """
    rank = find_chunk_rank(retrieved_chunks, source_chunk_id)
    return 1.0 if rank is not None and rank <= k else 0.0


def compute_retrieval_metrics(
    retrieved_chunks: List[str],
    ground_truth_chunks: List[str],
    k_values: Optional[List[int]] = None,
    retrieved_payloads: Optional[List[Dict]] = None,
    source_article_id: Optional[str] = None,
    source_chunk_id: Optional[str] = None,
) -> RetrievalMetrics:
    """
    Compute all retrieval metrics for a single query.

    Args:
        retrieved_chunks: List of retrieved chunk IDs (ordered by relevance)
        ground_truth_chunks: List of ground truth chunk IDs
        k_values: List of K values to compute metrics for
        retrieved_payloads: Payload dicts for article-level metrics
        source_article_id: Source article ID for article_hit@K metric
        source_chunk_id: Source chunk ID for chunk_hit@K metric

    Returns:
        RetrievalMetrics object with all computed metrics
    """
    if k_values is None:
        k_values = settings.EVALUATION_K_VALUES

    metrics = RetrievalMetrics()

    # Compute Recall@K for each K
    for k in k_values:
        recall_k = recall_at_k(retrieved_chunks, ground_truth_chunks, k)
        if k == 1:
            metrics.recall_at_1 = recall_k
        elif k == 3:
            metrics.recall_at_3 = recall_k
        elif k == 5:
            metrics.recall_at_5 = recall_k
        elif k == 10:
            metrics.recall_at_10 = recall_k

    # Compute ground truth ranks (article_hit_at_K / chunk_hit_at_K are derived)
    if retrieved_payloads and source_article_id:
        metrics.ground_truth_article_rank = find_article_rank(
            retrieved_chunks, retrieved_payloads, source_article_id
        )

    if source_chunk_id:
        metrics.ground_truth_chunk_rank = find_chunk_rank(
            retrieved_chunks, source_chunk_id
        )

    # Compute MRR
    metrics.mrr = mrr(retrieved_chunks, ground_truth_chunks)

    # Compute NDCG@K
    metrics.ndcg_at_5 = ndcg_at_k(retrieved_chunks, ground_truth_chunks, k=5)
    metrics.ndcg_at_10 = ndcg_at_k(retrieved_chunks, ground_truth_chunks, k=10)

    return metrics


def aggregate_metrics(all_metrics: List[RetrievalMetrics]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries.

    Args:
        all_metrics: List of RetrievalMetrics for each query

    Returns:
        Dictionary of averaged metrics
    """
    if not all_metrics:
        logger.warning("No metrics to aggregate")
        return {}

    n = len(all_metrics)

    aggregated = {
        "avg_recall_at_1": sum(m.recall_at_1 for m in all_metrics) / n,
        "avg_recall_at_3": sum(m.recall_at_3 for m in all_metrics) / n,
        "avg_recall_at_5": sum(m.recall_at_5 for m in all_metrics) / n,
        "avg_recall_at_10": sum(m.recall_at_10 for m in all_metrics) / n,
        "avg_mrr": sum(m.mrr for m in all_metrics) / n,
        "avg_ndcg_at_5": sum(m.ndcg_at_5 for m in all_metrics) / n,
        "avg_ndcg_at_10": sum(m.ndcg_at_10 for m in all_metrics) / n,
    }

    # Average ranks (computed only over questions where the item was found)
    chunk_ranks = [m.ground_truth_chunk_rank for m in all_metrics if m.ground_truth_chunk_rank is not None]
    article_ranks = [m.ground_truth_article_rank for m in all_metrics if m.ground_truth_article_rank is not None]
    if chunk_ranks:
        aggregated["avg_ground_truth_chunk_rank"] = sum(chunk_ranks) / len(chunk_ranks)
    if article_ranks:
        aggregated["avg_ground_truth_article_rank"] = sum(article_ranks) / len(article_ranks)

    return aggregated


def compute_metrics_by_type(
    per_question_results: List[Dict],
) -> Dict[str, RetrievalMetrics]:
    """
    Group metrics by question type and compute averages.

    Args:
        per_question_results: List of dictionaries with 'type' and 'metrics' keys

    Returns:
        Dictionary mapping question type to averaged RetrievalMetrics
    """
    # Group by type
    by_type: Dict[str, List[RetrievalMetrics]] = {}

    for result in per_question_results:
        q_type = result.get("type", "unknown")
        metrics = result.get("metrics")

        if metrics:
            if q_type not in by_type:
                by_type[q_type] = []
            by_type[q_type].append(metrics)

    # Average each type
    averaged_by_type = {}

    for q_type, metrics_list in by_type.items():
        if not metrics_list:
            continue

        n = len(metrics_list)
        averaged = RetrievalMetrics(
            recall_at_1=sum(m.recall_at_1 for m in metrics_list) / n,
            recall_at_3=sum(m.recall_at_3 for m in metrics_list) / n,
            recall_at_5=sum(m.recall_at_5 for m in metrics_list) / n,
            recall_at_10=sum(m.recall_at_10 for m in metrics_list) / n,
            mrr=sum(m.mrr for m in metrics_list) / n,
            ndcg_at_5=sum(m.ndcg_at_5 for m in metrics_list) / n,
            ndcg_at_10=sum(m.ndcg_at_10 for m in metrics_list) / n,
        )
        averaged_by_type[q_type] = averaged

    return averaged_by_type


if __name__ == "__main__":
    # Test the metrics
    print("=" * 70)
    print("Testing Retrieval Metrics")
    print("=" * 70)

    # Test data
    ground_truth = ["chunk_1", "chunk_2", "chunk_5"]

    test_cases = [
        {
            "name": "Perfect retrieval",
            "retrieved": ["chunk_1", "chunk_2", "chunk_5", "chunk_8", "chunk_9"],
            "expected_recall_5": 1.0,
        },
        {
            "name": "Good retrieval (2/3 found)",
            "retrieved": ["chunk_1", "chunk_5", "chunk_3", "chunk_4", "chunk_6"],
            "expected_recall_5": 2 / 3,
        },
        {
            "name": "Poor retrieval (1/3 found)",
            "retrieved": ["chunk_3", "chunk_4", "chunk_2", "chunk_6", "chunk_7"],
            "expected_recall_5": 1 / 3,
        },
        {
            "name": "No relevant results",
            "retrieved": ["chunk_3", "chunk_4", "chunk_6", "chunk_7", "chunk_8"],
            "expected_recall_5": 0.0,
        },
    ]

    for test in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Test: {test['name']}")
        print(f"{'=' * 70}")

        retrieved = test["retrieved"]
        print(f"Ground truth: {ground_truth}")
        print(f"Retrieved:    {retrieved}")

        # Compute metrics
        metrics = compute_retrieval_metrics(retrieved, ground_truth, k_values=[1, 3, 5, 10])

        print(f"\nMetrics:")
        print(f"  Recall@1:  {metrics.recall_at_1:.3f}")
        print(f"  Recall@3:  {metrics.recall_at_3:.3f}")
        print(f"  Recall@5:  {metrics.recall_at_5:.3f}")
        print(f"  Recall@10: {metrics.recall_at_10:.3f}")
        print(f"  MRR:       {metrics.mrr:.3f}")
        print(f"  NDCG@5:    {metrics.ndcg_at_5:.3f}")
        print(f"  NDCG@10:   {metrics.ndcg_at_10:.3f}")

        # Verify expected recall@5
        assert abs(metrics.recall_at_5 - test["expected_recall_5"]) < 0.001, (
            f"Expected recall@5={test['expected_recall_5']:.3f}, "
            f"got {metrics.recall_at_5:.3f}"
        )
        print(f"\n✓ Recall@5 matches expected value: {test['expected_recall_5']:.3f}")

    # Test aggregation
    print(f"\n{'=' * 70}")
    print("Testing Metrics Aggregation")
    print(f"{'=' * 70}")

    all_metrics = [
        compute_retrieval_metrics(test["retrieved"], ground_truth, k_values=[1, 3, 5, 10])
        for test in test_cases
    ]

    aggregated = aggregate_metrics(all_metrics)

    print("\nAggregated Metrics (across all test cases):")
    for metric_name, value in aggregated.items():
        print(f"  {metric_name}: {value:.3f}")

    # Test per-question-type aggregation
    print(f"\n{'=' * 70}")
    print("Testing Per-Question-Type Aggregation")
    print(f"{'=' * 70}")

    per_question_results = [
        {"type": "factual", "metrics": all_metrics[0]},
        {"type": "factual", "metrics": all_metrics[1]},
        {"type": "temporal", "metrics": all_metrics[2]},
        {"type": "temporal", "metrics": all_metrics[3]},
    ]

    by_type = compute_metrics_by_type(per_question_results)

    print("\nMetrics by Question Type:")
    for q_type, metrics in by_type.items():
        print(f"\n  {q_type.capitalize()}:")
        print(f"    Recall@5: {metrics.recall_at_5:.3f}")
        print(f"    MRR:      {metrics.mrr:.3f}")
        print(f"    NDCG@5:   {metrics.ndcg_at_5:.3f}")

    print(f"\n{'=' * 70}")
    print("✓ All tests passed successfully!")
    print(f"{'=' * 70}")
