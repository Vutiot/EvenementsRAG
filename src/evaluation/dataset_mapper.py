"""Map eval dataset questions to target collection chunks via char-offset overlap.

When the eval dataset was generated from a different collection (e.g. different
chunk_size / chunk_overlap), the original ``source_chunk_id`` no longer points
to a valid chunk in the target collection. This module re-maps ground truth by
computing character-offset overlap ratios between the question's source span
and each chunk in the target collection that belongs to the same document.

Usage:
    from src.evaluation.dataset_mapper import map_dataset_to_chunks

    mapped = map_dataset_to_chunks(eval_questions, target_chunks)
    ground_truth = {m.question_id: m.relevant_chunk_ids for m in mapped}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MappedGroundTruth:
    """Ground truth for a single question mapped to a target collection."""

    question_id: str
    question: str
    expected_answer_hint: str
    source_doc_id: str
    char_start: int
    char_end: int
    relevant_chunk_ids: list[str] = field(default_factory=list)


def _overlap_ratio(q_start: int, q_end: int, c_start: int, c_end: int) -> float:
    """Fraction of the question's source span covered by a chunk."""
    q_len = q_end - q_start
    if q_len <= 0:
        return 0.0
    overlap = max(0, min(q_end, c_end) - max(q_start, c_start))
    return overlap / q_len


def map_dataset_to_chunks(
    eval_questions: Sequence[dict],
    target_chunks: Sequence[dict],
    overlap_threshold: float = 0.5,
) -> list[MappedGroundTruth]:
    """Map eval questions to target collection chunks via char-offset overlap.

    Args:
        eval_questions: List of question dicts with ``char_start``, ``char_end``,
            ``source_doc_id`` (or ``source_article_id``), ``id``, ``question``,
            ``expected_answer_hint``.
        target_chunks: List of chunk dicts with ``chunk_id``, ``char_start``,
            ``char_end``, and ``article_id`` (or ``pageid``).
        overlap_threshold: Minimum overlap ratio to consider a chunk relevant.

    Returns:
        List of :class:`MappedGroundTruth`, one per question that has valid
        char-offset data.
    """
    # Index target chunks by document ID for fast lookup
    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk in target_chunks:
        doc_id = str(chunk.get("article_id") or chunk.get("pageid") or "")
        if doc_id:
            chunks_by_doc.setdefault(doc_id, []).append(chunk)

    results: list[MappedGroundTruth] = []

    for q in eval_questions:
        q_start = q.get("char_start", 0)
        q_end = q.get("char_end", 0)

        # Skip questions without valid char offsets (legacy datasets)
        if q_start == 0 and q_end == 0:
            continue

        doc_id = str(
            q.get("source_doc_id")
            or q.get("source_article_id")
            or ""
        )

        mapped = MappedGroundTruth(
            question_id=q.get("id", ""),
            question=q.get("question", ""),
            expected_answer_hint=q.get("expected_answer_hint", ""),
            source_doc_id=doc_id,
            char_start=q_start,
            char_end=q_end,
        )

        doc_chunks = chunks_by_doc.get(doc_id, [])

        # Primary pass: overlap >= threshold
        for chunk in doc_chunks:
            c_start = chunk.get("char_start", 0)
            c_end = chunk.get("char_end", 0)
            ratio = _overlap_ratio(q_start, q_end, c_start, c_end)
            if ratio >= overlap_threshold:
                mapped.relevant_chunk_ids.append(chunk["chunk_id"])

        # Fallback 1: lower threshold (half) if no matches
        if not mapped.relevant_chunk_ids:
            fallback_threshold = overlap_threshold / 2
            for chunk in doc_chunks:
                c_start = chunk.get("char_start", 0)
                c_end = chunk.get("char_end", 0)
                ratio = _overlap_ratio(q_start, q_end, c_start, c_end)
                if ratio >= fallback_threshold:
                    mapped.relevant_chunk_ids.append(chunk["chunk_id"])

        # Fallback 2: use original source_chunk_id if still no matches
        if not mapped.relevant_chunk_ids:
            original_id = q.get("source_chunk_id")
            if original_id:
                mapped.relevant_chunk_ids.append(original_id)

        results.append(mapped)

    mapped_count = sum(1 for m in results if m.relevant_chunk_ids)
    logger.info(
        f"Mapped {mapped_count}/{len(results)} questions to target chunks "
        f"(threshold={overlap_threshold})"
    )

    return results


def load_chunks_from_collection(collection_name: str) -> list[dict]:
    """Load all chunks with char offsets from a Qdrant collection.

    Returns list of dicts with keys: chunk_id, article_id, char_start, char_end.
    """
    from src.vector_store.qdrant_manager import QdrantManager

    mgr = QdrantManager()
    chunks: list[dict] = []
    offset = None

    while True:
        points, next_offset = mgr.client.scroll(
            collection_name=collection_name,
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for pt in points:
            chunks.append({
                "chunk_id": str(pt.id),
                "article_id": str(pt.payload.get("pageid", "")),
                "char_start": pt.payload.get("char_start", 0),
                "char_end": pt.payload.get("char_end", 0),
            })
        if next_offset is None:
            break
        offset = next_offset

    return chunks
