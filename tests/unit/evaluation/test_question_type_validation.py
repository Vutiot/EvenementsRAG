"""Tests for question-type validation at generation time."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.question_generator import (
    VALID_QUESTION_TYPES,
    QuestionGenerator,
    validate_question_type,
)


# ---------------------------------------------------------------------------
# validate_question_type — unit tests
# ---------------------------------------------------------------------------


class TestValidateQuestionType:
    """Unit tests for the validate_question_type helper."""

    @pytest.mark.parametrize("qtype", sorted(VALID_QUESTION_TYPES))
    def test_valid_matching_type_passes(self, qtype: str):
        q = {"question": "What happened?", "type": qtype}
        assert validate_question_type(q, qtype) is True
        assert q["type"] == qtype  # unchanged

    def test_valid_mismatched_type_corrected(self):
        q = {"question": "When?", "type": "temporal"}
        assert validate_question_type(q, "factual") is True
        assert q["type"] == "factual"  # force-corrected

    @pytest.mark.parametrize(
        "bad_type",
        [None, "", "  ", "unknown", "trivia", "multiple_choice", 123, True],
        ids=["None", "empty", "whitespace", "unknown", "trivia", "mc", "int", "bool"],
    )
    def test_invalid_type_rejected(self, bad_type):
        q = {"question": "What?", "type": bad_type}
        assert validate_question_type(q, "factual") is False

    def test_missing_type_key(self):
        q = {"question": "What?"}
        assert validate_question_type(q, "factual") is False

    def test_whitespace_around_valid_type(self):
        q = {"question": "What?", "type": "  temporal  "}
        assert validate_question_type(q, "temporal") is True

    @pytest.mark.parametrize("qtype", sorted(VALID_QUESTION_TYPES))
    def test_all_taxonomy_types_accepted(self, qtype: str):
        q = {"question": "Q?", "type": qtype}
        assert validate_question_type(q, qtype) is True


# ---------------------------------------------------------------------------
# generate_question_for_chunk — integration tests (mocked LLM)
# ---------------------------------------------------------------------------


def _make_chunk(**overrides):
    base = {
        "chunk_id": "c1",
        "content": "Some WW2 content " * 30,
        "article_title": "Battle of Normandy",
        "article_id": "12345",
        "chunk_index": 0,
    }
    base.update(overrides)
    return base


class TestGenerateQuestionForChunkValidation:
    """Integration tests: LLM returns questions → validation filters them."""

    def _make_generator(self, llm_response_text: str) -> QuestionGenerator:
        gen = QuestionGenerator(skip_api_init=True)
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = llm_response_text
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        gen.client = mock_client
        return gen

    def test_invalid_type_returns_empty(self):
        payload = json.dumps([{
            "question": "What happened?",
            "type": "unknown",
            "difficulty": "easy",
            "expected_answer_hint": "something",
        }])
        gen = self._make_generator(payload)
        result = gen.generate_question_for_chunk(_make_chunk(), 1, "factual")
        assert result == []

    def test_missing_type_returns_empty(self):
        payload = json.dumps([{
            "question": "What happened?",
            "difficulty": "easy",
            "expected_answer_hint": "something",
        }])
        gen = self._make_generator(payload)
        result = gen.generate_question_for_chunk(_make_chunk(), 1, "factual")
        assert result == []

    def test_wrong_but_valid_type_corrected(self):
        payload = json.dumps([{
            "question": "When did it happen?",
            "type": "temporal",
            "difficulty": "medium",
            "expected_answer_hint": "1944",
        }])
        gen = self._make_generator(payload)
        result = gen.generate_question_for_chunk(_make_chunk(), 1, "factual")
        assert len(result) == 1
        assert result[0]["type"] == "factual"

    def test_valid_matching_type_unchanged(self):
        payload = json.dumps([{
            "question": "Who led the attack?",
            "type": "entity_centric",
            "difficulty": "hard",
            "expected_answer_hint": "Eisenhower",
        }])
        gen = self._make_generator(payload)
        result = gen.generate_question_for_chunk(
            _make_chunk(), 1, "entity_centric"
        )
        assert len(result) == 1
        assert result[0]["type"] == "entity_centric"

    def test_mix_valid_and_invalid_keeps_valid(self):
        payload = json.dumps([
            {
                "question": "Good question?",
                "type": "factual",
                "difficulty": "easy",
                "expected_answer_hint": "hint",
            },
            {
                "question": "Bad question?",
                "type": "garbage_type",
                "difficulty": "easy",
                "expected_answer_hint": "hint",
            },
        ])
        gen = self._make_generator(payload)
        result = gen.generate_question_for_chunk(_make_chunk(), 2, "factual")
        assert len(result) == 1
        assert result[0]["question"] == "Good question?"


# ---------------------------------------------------------------------------
# DatasetCategoryConfig Pydantic validator
# ---------------------------------------------------------------------------


class TestDatasetCategoryConfigValidator:
    def test_valid_type_accepted(self):
        from src.api.schemas import DatasetCategoryConfig

        cfg = DatasetCategoryConfig(
            type="factual", prompt="p", model="m", count=1
        )
        assert cfg.type == "factual"

    def test_any_type_accepted(self):
        from src.api.schemas import DatasetCategoryConfig

        cfg = DatasetCategoryConfig(
            type="unknown", prompt="p", model="m", count=1
        )
        assert cfg.type == "unknown"

    @pytest.mark.parametrize("qtype", sorted(VALID_QUESTION_TYPES))
    def test_all_taxonomy_types_accepted_in_schema(self, qtype: str):
        from src.api.schemas import DatasetCategoryConfig

        cfg = DatasetCategoryConfig(
            type=qtype, prompt="p", model="m", count=1
        )
        assert cfg.type == qtype
