"""Tests for entity-level metric functions (entity_recall_at_k, entity_precision_at_k, entity_mrr_score)."""

from unittest.mock import MagicMock

import pytest

from src.evaluation.metrics import (
    RetrievalMetrics,
    EvaluationResults,
    aggregate_metrics,
    entity_mrr_score,
    entity_precision_at_k,
    entity_recall_at_k,
)


@pytest.fixture
def mock_extractor():
    """Mock EntityExtractor that returns pre-defined entity sets per text."""
    extractor = MagicMock()
    # Default: return empty set
    extractor.extract_entities.return_value = set()
    return extractor


class TestEntityRecallAtK:
    def test_perfect_recall(self, mock_extractor):
        def side_effect(text):
            if text == "ground truth text":
                return {"berlin", "london"}
            return {"berlin", "london", "paris"}

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_recall_at_k(
            ["chunk with berlin london paris"],
            "ground truth text",
            k=5,
            extractor=mock_extractor,
        )
        assert result == 1.0

    def test_partial_recall(self, mock_extractor):
        def side_effect(text):
            if text == "ground truth text":
                return {"berlin", "london"}
            return {"berlin"}

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_recall_at_k(
            ["chunk with berlin"],
            "ground truth text",
            k=5,
            extractor=mock_extractor,
        )
        assert result == 0.5

    def test_no_recall(self, mock_extractor):
        def side_effect(text):
            if text == "ground truth text":
                return {"berlin", "london"}
            return {"paris"}

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_recall_at_k(
            ["chunk with paris"],
            "ground truth text",
            k=5,
            extractor=mock_extractor,
        )
        assert result == 0.0

    def test_empty_ground_truth(self, mock_extractor):
        mock_extractor.extract_entities.return_value = set()

        result = entity_recall_at_k(
            ["some chunk"],
            "ground truth text",
            k=5,
            extractor=mock_extractor,
        )
        assert result == 0.0

    def test_respects_k(self, mock_extractor):
        """Only top-k chunks should be concatenated."""
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"berlin", "london"}  # ground truth
            return {"berlin"}  # only first chunk's entities

        mock_extractor.extract_entities.side_effect = side_effect

        # k=1 means only the first chunk is considered
        result = entity_recall_at_k(
            ["chunk1 with berlin", "chunk2 with london"],
            "ground truth text",
            k=1,
            extractor=mock_extractor,
        )
        assert result == 0.5


class TestEntityPrecisionAtK:
    def test_perfect_precision(self, mock_extractor):
        def side_effect(text):
            if text == "ground truth text":
                return {"berlin", "london"}
            return {"berlin", "london"}

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_precision_at_k(
            ["chunk"],
            "ground truth text",
            k=5,
            extractor=mock_extractor,
        )
        assert result == 1.0

    def test_partial_precision(self, mock_extractor):
        def side_effect(text):
            if text == "ground truth text":
                return {"berlin"}
            return {"berlin", "london"}

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_precision_at_k(
            ["chunk"],
            "ground truth text",
            k=5,
            extractor=mock_extractor,
        )
        assert result == 0.5

    def test_no_retrieved_entities(self, mock_extractor):
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"berlin"}
            return set()

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_precision_at_k(
            ["chunk"],
            "ground truth text",
            k=5,
            extractor=mock_extractor,
        )
        assert result == 0.0


class TestEntityMRR:
    def test_first_chunk_matches(self, mock_extractor):
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"berlin"}  # ground truth
            if call_count[0] == 2:
                return {"berlin"}  # first chunk
            return set()

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_mrr_score(
            ["chunk1", "chunk2"],
            "ground truth text",
            extractor=mock_extractor,
        )
        assert result == 1.0

    def test_second_chunk_matches(self, mock_extractor):
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"berlin"}  # ground truth
            if call_count[0] == 2:
                return {"paris"}  # first chunk, no match
            if call_count[0] == 3:
                return {"berlin"}  # second chunk, match
            return set()

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_mrr_score(
            ["chunk1", "chunk2"],
            "ground truth text",
            extractor=mock_extractor,
        )
        assert result == 0.5

    def test_no_match(self, mock_extractor):
        call_count = [0]

        def side_effect(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"berlin"}  # ground truth
            return {"paris"}  # no match in any chunk

        mock_extractor.extract_entities.side_effect = side_effect

        result = entity_mrr_score(
            ["chunk1", "chunk2"],
            "ground truth text",
            extractor=mock_extractor,
        )
        assert result == 0.0

    def test_empty_ground_truth(self, mock_extractor):
        mock_extractor.extract_entities.return_value = set()

        result = entity_mrr_score(
            ["chunk1"],
            "ground truth text",
            extractor=mock_extractor,
        )
        assert result == 0.0


class TestRetrievalMetricsEntityFields:
    def test_default_values(self):
        m = RetrievalMetrics()
        assert m.entity_precision_at_5 == 0.0
        assert m.entity_recall_at_5 == 0.0
        assert m.entity_mrr == 0.0

    def test_to_dict_includes_entity_fields(self):
        m = RetrievalMetrics(entity_precision_at_5=0.7, entity_recall_at_5=0.8, entity_mrr=0.5)
        d = m.to_dict()
        assert d["entity_precision_at_5"] == 0.7
        assert d["entity_recall_at_5"] == 0.8
        assert d["entity_mrr"] == 0.5


class TestEvaluationResultsEntityFields:
    def test_default_values(self):
        r = EvaluationResults()
        assert r.avg_entity_precision_at_5 == 0.0
        assert r.avg_entity_recall_at_5 == 0.0
        assert r.avg_entity_mrr == 0.0

    def test_to_dict_includes_entity_fields(self):
        r = EvaluationResults(
            avg_entity_precision_at_5=0.6,
            avg_entity_recall_at_5=0.7,
            avg_entity_mrr=0.4,
        )
        d = r.to_dict()
        assert d["avg_entity_precision_at_5"] == 0.6
        assert d["avg_entity_recall_at_5"] == 0.7
        assert d["avg_entity_mrr"] == 0.4


class TestAggregateMetricsEntity:
    def test_aggregation_includes_entity_fields(self):
        m1 = RetrievalMetrics(entity_precision_at_5=0.8, entity_recall_at_5=0.6, entity_mrr=1.0)
        m2 = RetrievalMetrics(entity_precision_at_5=0.4, entity_recall_at_5=0.2, entity_mrr=0.5)

        agg = aggregate_metrics([m1, m2])

        assert agg["avg_entity_precision_at_5"] == pytest.approx(0.6)
        assert agg["avg_entity_recall_at_5"] == pytest.approx(0.4)
        assert agg["avg_entity_mrr"] == pytest.approx(0.75)
