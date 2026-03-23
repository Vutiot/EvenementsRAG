"""Tests for EntityExtractor — all mocked, no LLM calls."""

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.entity_extractor import EntityExtractor, _ENTITY_EXTRACTION_PROMPT


@pytest.fixture
def mock_client():
    """Patch openai.OpenAI so no real client is created."""
    with patch("src.evaluation.entity_extractor.openai.OpenAI") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


def _make_response(content: str):
    """Build a mock ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestEntityExtractor:
    def test_basic_extraction(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response(
            '["Winston Churchill", "Battle of Britain", "London"]'
        )
        extractor = EntityExtractor(rate_limit_seconds=0)
        entities = extractor.extract_entities("Winston Churchill led Britain during the Battle of Britain in London.")

        assert entities == {"winston churchill", "battle of britain", "london"}
        mock_client.chat.completions.create.assert_called_once()

    def test_empty_text_returns_empty(self, mock_client):
        extractor = EntityExtractor(rate_limit_seconds=0)
        assert extractor.extract_entities("") == set()
        assert extractor.extract_entities("   ") == set()
        mock_client.chat.completions.create.assert_not_called()

    def test_caching(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response(
            '["Berlin"]'
        )
        extractor = EntityExtractor(rate_limit_seconds=0)

        result1 = extractor.extract_entities("The fall of Berlin.")
        result2 = extractor.extract_entities("The fall of Berlin.")

        assert result1 == result2 == {"berlin"}
        assert mock_client.chat.completions.create.call_count == 1

    def test_cache_different_texts(self, mock_client):
        mock_client.chat.completions.create.side_effect = [
            _make_response('["Berlin"]'),
            _make_response('["Paris"]'),
        ]
        extractor = EntityExtractor(rate_limit_seconds=0)

        r1 = extractor.extract_entities("The fall of Berlin.")
        r2 = extractor.extract_entities("The liberation of Paris.")

        assert r1 == {"berlin"}
        assert r2 == {"paris"}
        assert mock_client.chat.completions.create.call_count == 2

    def test_clear_cache(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response('["Berlin"]')
        extractor = EntityExtractor(rate_limit_seconds=0)

        extractor.extract_entities("The fall of Berlin.")
        extractor.clear_cache()
        extractor.extract_entities("The fall of Berlin.")

        assert mock_client.chat.completions.create.call_count == 2

    def test_parse_markdown_fences(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response(
            '```json\n["Churchill", "London"]\n```'
        )
        extractor = EntityExtractor(rate_limit_seconds=0)
        entities = extractor.extract_entities("Churchill was in London.")

        assert entities == {"churchill", "london"}

    def test_parse_invalid_json_returns_empty(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response(
            "not valid json"
        )
        extractor = EntityExtractor(rate_limit_seconds=0)
        entities = extractor.extract_entities("Some text.")

        assert entities == set()

    def test_parse_non_list_returns_empty(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response(
            '{"entities": ["a", "b"]}'
        )
        extractor = EntityExtractor(rate_limit_seconds=0)
        entities = extractor.extract_entities("Some text.")

        assert entities == set()

    def test_filters_non_string_entries(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response(
            '["Berlin", 42, null, "", "London"]'
        )
        extractor = EntityExtractor(rate_limit_seconds=0)
        entities = extractor.extract_entities("Berlin and London.")

        assert entities == {"berlin", "london"}

    def test_retry_on_failure(self, mock_client):
        mock_client.chat.completions.create.side_effect = [
            Exception("API error"),
            _make_response('["Berlin"]'),
        ]
        extractor = EntityExtractor(rate_limit_seconds=0, max_retries=3)

        with patch("src.evaluation.entity_extractor.time.sleep"):
            entities = extractor.extract_entities("The fall of Berlin.")

        assert entities == {"berlin"}
        assert mock_client.chat.completions.create.call_count == 2

    def test_all_retries_exhausted(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("API error")
        extractor = EntityExtractor(rate_limit_seconds=0, max_retries=2)

        with patch("src.evaluation.entity_extractor.time.sleep"):
            entities = extractor.extract_entities("Some text.")

        assert entities == set()
        assert mock_client.chat.completions.create.call_count == 2

    def test_batch_extraction(self, mock_client):
        mock_client.chat.completions.create.side_effect = [
            _make_response('["Berlin"]'),
            _make_response('["Paris"]'),
        ]
        extractor = EntityExtractor(rate_limit_seconds=0)

        results = extractor.extract_entities_batch(["Berlin fell.", "Paris was liberated."])

        assert results == [{"berlin"}, {"paris"}]

    def test_batch_reuses_cache(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response('["Berlin"]')
        extractor = EntityExtractor(rate_limit_seconds=0)

        results = extractor.extract_entities_batch(["Berlin fell.", "Berlin fell."])

        assert results == [{"berlin"}, {"berlin"}]
        assert mock_client.chat.completions.create.call_count == 1

    def test_null_content_returns_empty(self, mock_client):
        msg = MagicMock()
        msg.content = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        extractor = EntityExtractor(rate_limit_seconds=0)
        entities = extractor.extract_entities("Some text.")

        assert entities == set()

    def test_normalization_lowercase_strip(self, mock_client):
        mock_client.chat.completions.create.return_value = _make_response(
            '["  Winston Churchill  ", "LONDON", "battle of britain"]'
        )
        extractor = EntityExtractor(rate_limit_seconds=0)
        entities = extractor.extract_entities("Text.")

        assert entities == {"winston churchill", "london", "battle of britain"}

    def test_prompt_template_has_text_placeholder(self):
        assert "{text}" in _ENTITY_EXTRACTION_PROMPT

    def test_custom_model_and_url(self, mock_client):
        extractor = EntityExtractor(
            model="custom/model",
            base_url="https://custom.api.com/v1",
            api_key="test-key",
            rate_limit_seconds=0,
        )
        mock_client.chat.completions.create.return_value = _make_response('["Entity"]')

        extractor.extract_entities("Test text.")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "custom/model"
