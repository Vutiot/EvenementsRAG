"""Tests for src.evaluation.dataset_mapper."""

import pytest

from src.evaluation.dataset_mapper import (
    MappedGroundTruth,
    _overlap_ratio,
    load_chunks_from_collection,
    map_dataset_to_chunks,
)


# ---------------------------------------------------------------------------
# _overlap_ratio
# ---------------------------------------------------------------------------


class TestOverlapRatio:
    def test_full_overlap(self):
        assert _overlap_ratio(0, 100, 0, 100) == 1.0

    def test_no_overlap(self):
        assert _overlap_ratio(0, 50, 60, 100) == 0.0

    def test_partial_overlap(self):
        # question span [10, 60) = 50 chars; chunk [30, 80)
        # overlap [30, 60) = 30 chars → 30/50 = 0.6
        assert _overlap_ratio(10, 60, 30, 80) == pytest.approx(0.6)

    def test_chunk_inside_question(self):
        # chunk entirely inside question
        assert _overlap_ratio(0, 100, 20, 80) == pytest.approx(0.6)

    def test_question_inside_chunk(self):
        # question entirely inside chunk → full coverage
        assert _overlap_ratio(20, 80, 0, 100) == 1.0

    def test_zero_length_question(self):
        assert _overlap_ratio(50, 50, 0, 100) == 0.0

    def test_adjacent_no_overlap(self):
        assert _overlap_ratio(0, 50, 50, 100) == 0.0

    def test_single_char_overlap(self):
        # [0, 51) overlaps [50, 100) by 1 char → 1/51
        assert _overlap_ratio(0, 51, 50, 100) == pytest.approx(1 / 51)


# ---------------------------------------------------------------------------
# map_dataset_to_chunks
# ---------------------------------------------------------------------------


def _make_question(qid, doc_id, char_start, char_end, **extras):
    return {
        "id": qid,
        "question": f"Question {qid}",
        "expected_answer_hint": f"Hint {qid}",
        "source_doc_id": str(doc_id),
        "source_article_id": str(doc_id),
        "char_start": char_start,
        "char_end": char_end,
        **extras,
    }


def _make_chunk(chunk_id, doc_id, char_start, char_end):
    return {
        "chunk_id": chunk_id,
        "article_id": str(doc_id),
        "char_start": char_start,
        "char_end": char_end,
    }


class TestMapDatasetToChunks:
    def test_exact_match(self):
        questions = [_make_question("q1", "doc1", 0, 100)]
        chunks = [_make_chunk("c1", "doc1", 0, 100)]
        result = map_dataset_to_chunks(questions, chunks)
        assert len(result) == 1
        assert result[0].relevant_chunk_ids == ["c1"]

    def test_multiple_overlapping_chunks(self):
        questions = [_make_question("q1", "doc1", 50, 150)]
        chunks = [
            _make_chunk("c1", "doc1", 0, 100),    # overlap [50,100) = 50/100 = 0.5
            _make_chunk("c2", "doc1", 100, 200),   # overlap [100,150) = 50/100 = 0.5
            _make_chunk("c3", "doc1", 200, 300),   # no overlap
        ]
        result = map_dataset_to_chunks(questions, chunks, overlap_threshold=0.5)
        assert len(result) == 1
        assert set(result[0].relevant_chunk_ids) == {"c1", "c2"}

    def test_no_overlap_falls_back_to_lower_threshold(self):
        questions = [_make_question("q1", "doc1", 0, 100)]
        # Chunk covers 30% of question → below 0.5 threshold, above 0.25 fallback
        chunks = [_make_chunk("c1", "doc1", 0, 30)]
        result = map_dataset_to_chunks(questions, chunks, overlap_threshold=0.5)
        assert result[0].relevant_chunk_ids == ["c1"]

    def test_no_overlap_falls_back_to_source_chunk_id(self):
        questions = [_make_question("q1", "doc1", 100, 200, source_chunk_id="orig_c1")]
        # Chunk from a different range entirely, no overlap at all
        chunks = [_make_chunk("c1", "doc1", 500, 600)]
        result = map_dataset_to_chunks(questions, chunks, overlap_threshold=0.5)
        assert result[0].relevant_chunk_ids == ["orig_c1"]

    def test_different_doc_ids_not_matched(self):
        questions = [_make_question("q1", "doc1", 0, 100)]
        chunks = [_make_chunk("c1", "doc2", 0, 100)]  # different doc
        result = map_dataset_to_chunks(questions, chunks)
        # No overlap from same doc → falls back to source_chunk_id which is not set
        assert len(result) == 1
        assert result[0].relevant_chunk_ids == []

    def test_legacy_question_without_offsets_skipped(self):
        questions = [_make_question("q1", "doc1", 0, 0)]  # no offsets
        chunks = [_make_chunk("c1", "doc1", 0, 100)]
        result = map_dataset_to_chunks(questions, chunks)
        assert len(result) == 0

    def test_multiple_questions_multiple_docs(self):
        questions = [
            _make_question("q1", "doc1", 0, 100),
            _make_question("q2", "doc2", 50, 200),
        ]
        chunks = [
            _make_chunk("c1", "doc1", 0, 100),
            _make_chunk("c2", "doc2", 0, 100),
            _make_chunk("c3", "doc2", 100, 200),
        ]
        result = map_dataset_to_chunks(questions, chunks, overlap_threshold=0.3)
        assert len(result) == 2
        assert result[0].relevant_chunk_ids == ["c1"]
        # q2 [50,200): c2 [0,100) overlap [50,100)=50/150≈0.33; c3 [100,200) overlap [100,200)=100/150≈0.67
        assert set(result[1].relevant_chunk_ids) == {"c2", "c3"}

    def test_empty_questions(self):
        result = map_dataset_to_chunks([], [_make_chunk("c1", "doc1", 0, 100)])
        assert result == []

    def test_empty_chunks(self):
        questions = [_make_question("q1", "doc1", 0, 100)]
        result = map_dataset_to_chunks(questions, [])
        # Falls back to source_chunk_id (not set)
        assert len(result) == 1
        assert result[0].relevant_chunk_ids == []

    def test_mapped_ground_truth_fields(self):
        questions = [_make_question("q1", "doc1", 10, 90)]
        chunks = [_make_chunk("c1", "doc1", 0, 100)]
        result = map_dataset_to_chunks(questions, chunks)
        m = result[0]
        assert isinstance(m, MappedGroundTruth)
        assert m.question_id == "q1"
        assert m.question == "Question q1"
        assert m.expected_answer_hint == "Hint q1"
        assert m.source_doc_id == "doc1"
        assert m.char_start == 10
        assert m.char_end == 90

    def test_pageid_key_in_chunks(self):
        """Chunks may use 'pageid' instead of 'article_id'."""
        questions = [_make_question("q1", "42", 0, 100)]
        chunks = [{"chunk_id": "c1", "pageid": 42, "char_start": 0, "char_end": 100}]
        result = map_dataset_to_chunks(questions, chunks)
        assert result[0].relevant_chunk_ids == ["c1"]

    def test_custom_threshold(self):
        questions = [_make_question("q1", "doc1", 0, 100)]
        # Chunk covers 80% of question
        chunks = [_make_chunk("c1", "doc1", 0, 80)]
        # With threshold=0.9, 80% is below → should fallback to half (0.45)
        result = map_dataset_to_chunks(questions, chunks, overlap_threshold=0.9)
        assert result[0].relevant_chunk_ids == ["c1"]  # matched by fallback threshold


# ---------------------------------------------------------------------------
# load_chunks_from_collection (mocked)
# ---------------------------------------------------------------------------


class TestLoadChunksFromCollection:
    def test_loads_chunks_from_qdrant(self, monkeypatch):
        class FakePoint:
            def __init__(self, id_, payload):
                self.id = id_
                self.payload = payload

        class FakeClient:
            def scroll(self, **kwargs):
                if kwargs.get("offset") is None:
                    return (
                        [
                            FakePoint("uuid-1", {"pageid": 123, "char_start": 0, "char_end": 50}),
                            FakePoint("uuid-2", {"pageid": 123, "char_start": 50, "char_end": 100}),
                        ],
                        None,
                    )
                return ([], None)

        class FakeManager:
            client = FakeClient()

        monkeypatch.setattr(
            "src.vector_store.qdrant_manager.QdrantManager",
            lambda: FakeManager(),
        )

        chunks = load_chunks_from_collection("test_collection")
        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == "uuid-1"
        assert chunks[0]["article_id"] == "123"
        assert chunks[0]["char_start"] == 0
        assert chunks[0]["char_end"] == 50
        assert chunks[1]["chunk_id"] == "uuid-2"
