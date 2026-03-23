"""Tests for src.evaluation.offset_recovery."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.offset_recovery import (
    recover_chunk_offsets,
    recover_question_offsets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_article(tmp_path: Path, title: str, content: str) -> Path:
    """Write a mock article JSON to tmp_path and return its path."""
    fp = tmp_path / f"{title}.json"
    fp.write_text(json.dumps({"title": title, "content": content}), encoding="utf-8")
    return fp


ARTICLE_CONTENT = (
    "Alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango."
)


# ---------------------------------------------------------------------------
# recover_chunk_offsets
# ---------------------------------------------------------------------------


class TestRecoverChunkOffsets:
    def test_basic_single_chunk(self, tmp_path):
        _write_article(tmp_path, "TestArticle", ARTICLE_CONTENT)
        chunk_text = "charlie delta echo"
        expected_start = ARTICLE_CONTENT.find(chunk_text)

        chunks = [{
            "content": chunk_text,
            "article_title": "TestArticle",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 0,
        }]

        recovered = recover_chunk_offsets(chunks, tmp_path)

        assert recovered == 1
        assert chunks[0]["char_start"] == expected_start
        assert chunks[0]["char_end"] == expected_start + len(chunk_text)

    def test_multiple_chunks_advancing_search(self, tmp_path):
        _write_article(tmp_path, "TestArticle", ARTICLE_CONTENT)
        chunk_a = "Alpha bravo charlie"
        chunk_b = "delta echo foxtrot"

        chunks = [
            {"content": chunk_b, "article_title": "TestArticle", "chunk_index": 1, "char_start": 0, "char_end": 0},
            {"content": chunk_a, "article_title": "TestArticle", "chunk_index": 0, "char_start": 0, "char_end": 0},
        ]

        recovered = recover_chunk_offsets(chunks, tmp_path)

        assert recovered == 2
        # chunk_a (index 0) should be processed first
        assert chunks[1]["char_start"] == 0
        assert chunks[1]["char_end"] == len(chunk_a)
        # chunk_b (index 1) should start after chunk_a
        pos_b = ARTICLE_CONTENT.find(chunk_b)
        assert chunks[0]["char_start"] == pos_b
        assert chunks[0]["char_end"] == pos_b + len(chunk_b)

    def test_article_file_missing(self, tmp_path):
        chunks = [{
            "content": "some text",
            "article_title": "NonExistent",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 0,
        }]

        recovered = recover_chunk_offsets(chunks, tmp_path)

        assert recovered == 0
        assert chunks[0]["char_start"] == 0
        assert chunks[0]["char_end"] == 0

    def test_chunk_not_found_in_article(self, tmp_path):
        _write_article(tmp_path, "TestArticle", "Short article content.")
        chunks = [{
            "content": "this text does not exist in article",
            "article_title": "TestArticle",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 0,
        }]

        recovered = recover_chunk_offsets(chunks, tmp_path)

        # Falls back to search_from-based offset
        assert recovered == 1
        assert chunks[0]["char_start"] == 0  # search_from starts at 0
        assert chunks[0]["char_end"] == len(chunks[0]["content"])

    def test_already_present_offsets_not_modified(self, tmp_path):
        _write_article(tmp_path, "TestArticle", ARTICLE_CONTENT)
        chunks = [{
            "content": "Alpha bravo",
            "article_title": "TestArticle",
            "chunk_index": 0,
            "char_start": 999,
            "char_end": 1010,
        }]

        recovered = recover_chunk_offsets(chunks, tmp_path)

        assert recovered == 0
        assert chunks[0]["char_start"] == 999
        assert chunks[0]["char_end"] == 1010

    def test_articles_dir_not_found(self):
        chunks = [{
            "content": "text",
            "article_title": "Art",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 0,
        }]

        recovered = recover_chunk_offsets(chunks, "/nonexistent/path")

        assert recovered == 0

    def test_empty_content_skipped(self, tmp_path):
        _write_article(tmp_path, "TestArticle", ARTICLE_CONTENT)
        chunks = [{
            "content": "",
            "article_title": "TestArticle",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 0,
        }]

        recovered = recover_chunk_offsets(chunks, tmp_path)

        assert recovered == 0

    def test_none_offsets_treated_as_zero(self, tmp_path):
        _write_article(tmp_path, "TestArticle", ARTICLE_CONTENT)
        chunk_text = "Alpha bravo"
        chunks = [{
            "content": chunk_text,
            "article_title": "TestArticle",
            "chunk_index": 0,
            "char_start": None,
            "char_end": None,
        }]

        recovered = recover_chunk_offsets(chunks, tmp_path)

        assert recovered == 1
        assert chunks[0]["char_start"] == 0
        assert chunks[0]["char_end"] == len(chunk_text)

    def test_multiple_articles(self, tmp_path):
        _write_article(tmp_path, "Art1", "First article content here.")
        _write_article(tmp_path, "Art2", "Second article content here.")

        chunks = [
            {"content": "First article", "article_title": "Art1", "chunk_index": 0, "char_start": 0, "char_end": 0},
            {"content": "Second article", "article_title": "Art2", "chunk_index": 0, "char_start": 0, "char_end": 0},
        ]

        recovered = recover_chunk_offsets(chunks, tmp_path)

        assert recovered == 2
        assert chunks[0]["char_start"] == 0
        assert chunks[1]["char_start"] == 0


# ---------------------------------------------------------------------------
# recover_question_offsets
# ---------------------------------------------------------------------------


def _make_mock_point(pt_id: str, content: str, article_title: str):
    pt = MagicMock()
    pt.id = pt_id
    pt.payload = {
        "content": content,
        "article_title": article_title,
    }
    return pt


class TestRecoverQuestionOffsets:
    def test_basic_recovery(self, tmp_path):
        _write_article(tmp_path, "TestArticle", ARTICLE_CONTENT)
        chunk_text = "charlie delta echo"
        expected_start = ARTICLE_CONTENT.find(chunk_text)

        questions = [{
            "id": "q1",
            "source_chunk_id": "uuid-1",
            "char_start": 0,
            "char_end": 0,
        }]

        mock_points = [_make_mock_point("uuid-1", chunk_text, "TestArticle")]

        with patch("src.vector_store.qdrant_manager.QdrantManager") as MockMgr:
            mock_client = MockMgr.return_value.client
            mock_client.scroll.return_value = (mock_points, None)

            recovered = recover_question_offsets(questions, "test_collection", tmp_path)

        assert recovered == 1
        assert questions[0]["char_start"] == expected_start
        assert questions[0]["char_end"] == expected_start + len(chunk_text)

    def test_no_recovery_needed(self, tmp_path):
        questions = [{
            "id": "q1",
            "source_chunk_id": "uuid-1",
            "char_start": 100,
            "char_end": 200,
        }]

        # Should not even touch Qdrant
        recovered = recover_question_offsets(questions, "test_collection", tmp_path)

        assert recovered == 0
        assert questions[0]["char_start"] == 100

    def test_chunk_not_in_collection(self, tmp_path):
        _write_article(tmp_path, "TestArticle", ARTICLE_CONTENT)

        questions = [{
            "id": "q1",
            "source_chunk_id": "uuid-missing",
            "char_start": 0,
            "char_end": 0,
        }]

        with patch("src.vector_store.qdrant_manager.QdrantManager") as MockMgr:
            mock_client = MockMgr.return_value.client
            mock_client.scroll.return_value = ([], None)

            recovered = recover_question_offsets(questions, "test_collection", tmp_path)

        assert recovered == 0
        assert questions[0]["char_start"] == 0

    def test_collection_load_error(self, tmp_path):
        questions = [{
            "id": "q1",
            "source_chunk_id": "uuid-1",
            "char_start": 0,
            "char_end": 0,
        }]

        with patch("src.vector_store.qdrant_manager.QdrantManager") as MockMgr:
            MockMgr.return_value.client.scroll.side_effect = RuntimeError("connection failed")

            recovered = recover_question_offsets(questions, "bad_collection", tmp_path)

        assert recovered == 0

    def test_multiple_questions(self, tmp_path):
        _write_article(tmp_path, "Art1", "First article with some content.")
        _write_article(tmp_path, "Art2", "Second article with other content.")

        questions = [
            {"id": "q1", "source_chunk_id": "u1", "char_start": 0, "char_end": 0},
            {"id": "q2", "source_chunk_id": "u2", "char_start": 0, "char_end": 0},
        ]

        mock_points = [
            _make_mock_point("u1", "First article", "Art1"),
            _make_mock_point("u2", "Second article", "Art2"),
        ]

        with patch("src.vector_store.qdrant_manager.QdrantManager") as MockMgr:
            mock_client = MockMgr.return_value.client
            mock_client.scroll.return_value = (mock_points, None)

            recovered = recover_question_offsets(questions, "test_col", tmp_path)

        assert recovered == 2
        assert questions[0]["char_start"] == 0
        assert questions[0]["char_end"] == len("First article")
        assert questions[1]["char_start"] == 0
        assert questions[1]["char_end"] == len("Second article")
