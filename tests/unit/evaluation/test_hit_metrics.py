"""Tests for the core retrieval metrics: hit@K and token-budget hit@C (E10-F1)."""

from src.evaluation.metrics import (
    RetrievalMetrics,
    budget_key,
    budget_prefix_length,
    chunk_token_count,
    compute_hit_metrics,
    estimate_tokens,
    hit_at_budget,
    hit_at_k,
)


class TestHitAtK:
    def test_gold_inside_top_k(self):
        assert hit_at_k(["a", "b", "c"], ["c"], 3) == 1.0

    def test_gold_outside_top_k(self):
        assert hit_at_k(["a", "b", "c"], ["c"], 2) == 0.0

    def test_any_gold_counts(self):
        assert hit_at_k(["a", "b"], ["x", "b"], 2) == 1.0

    def test_no_gold_is_zero(self):
        assert hit_at_k(["a"], [], 5) == 0.0

    def test_ids_compared_as_strings(self):
        assert hit_at_k([12, 7], ["7"], 2) == 1.0


class TestBudgetPrefix:
    def test_prefix_stops_before_overflow(self):
        # 1000 + 1500 = 2500 fits in 3000; adding 1000 more overflows
        assert budget_prefix_length([1000, 1500, 1000, 10], 3000) == 2

    def test_exact_fit_is_included(self):
        assert budget_prefix_length([1000, 2000], 3000) == 2

    def test_first_chunk_too_big(self):
        assert budget_prefix_length([5000, 10], 4000) == 0

    def test_whole_list_fits(self):
        assert budget_prefix_length([10, 20], 4000) == 2


class TestHitAtBudget:
    def test_gold_in_budget_prefix(self):
        assert hit_at_budget(["a", "b", "c"], ["b"], [2000, 1500, 900], 4000) == 1.0

    def test_gold_after_budget(self):
        assert hit_at_budget(["a", "b", "c"], ["c"], [2000, 1500, 900], 4000) == 0.0

    def test_gold_after_chunk_that_overflows(self):
        # the prefix ends at the first chunk that does not fit, even if later ones would
        assert hit_at_budget(["a", "b", "c"], ["c"], [3000, 2000, 100], 4000) == 0.0

    def test_larger_budget_reaches_further(self):
        tokens = [2000, 1500, 900, 3000]
        ids = ["a", "b", "c", "d"]
        assert hit_at_budget(ids, ["d"], tokens, 4000) == 0.0
        assert hit_at_budget(ids, ["d"], tokens, 8000) == 1.0


class TestComputeHitMetrics:
    def test_keys_and_values(self):
        ids = [f"c{i}" for i in range(12)]
        tokens = [500] * 12
        hits = compute_hit_metrics(ids, ["c7"], tokens, k_values=[5, 10], budgets=[4000, 8000])
        assert hits == {
            "hit_at_5": 0.0,
            "hit_at_10": 1.0,
            "hit_at_c4k": 1.0,  # 8 chunks x 500 = 4000 tokens, c7 is the 8th
            "hit_at_c8k": 1.0,
        }

    def test_budget_key(self):
        assert budget_key(4000) == "hit_at_c4k"
        assert budget_key(1500) == "hit_at_c1500"


class TestTokenCounting:
    def test_payload_count_preferred(self):
        assert chunk_token_count({"token_count": 321, "content": "x" * 10}) == 321

    def test_estimated_from_content(self):
        assert chunk_token_count({"content": "word " * 40}) == estimate_tokens("word " * 40)
        assert chunk_token_count({"content": "word " * 40}) > 0

    def test_empty_payload(self):
        assert chunk_token_count(None) == 0
        assert chunk_token_count({}) == 0


def test_retrieval_metrics_to_dict_flattens_hits():
    m = RetrievalMetrics(hits={"hit_at_5": 1.0, "hit_at_c4k": 0.0})
    d = m.to_dict()
    assert d["hit_at_5"] == 1.0
    assert d["hit_at_c4k"] == 0.0
    assert "hits" not in d


def test_benchmark_runner_evaluate_question_hits_on_deep_list():
    from unittest.mock import MagicMock

    from src.evaluation.benchmark_runner import BenchmarkRunner

    runner = BenchmarkRunner(
        qdrant_manager=MagicMock(), embedding_generator=MagicMock(),
        k_values=[1, 3, 5, 10], hit_token_budgets=[4000, 8000], budget_retrieval_depth=50,
    )
    assert runner._retrieval_depth() == 50
    ids = [f"c{i}" for i in range(20)]
    payloads = [{"content": "x", "token_count": 500, "pageid": 1} for _ in ids]
    query_result = {
        "retrieved_chunks": ids, "retrieved_payloads": payloads,
        "retrieval_time_ms": 5.0, "num_retrieved": len(ids),
    }
    question = {"id": "q", "question": "?", "source_chunk_id": "c12", "source_article_id": "1"}
    out = runner.evaluate_question(question, query_result)
    m = out["metrics"].to_dict()
    assert out["retrieved_chunk_tokens"] == [500] * 20
    assert (m["hit_at_5"], m["hit_at_10"], m["hit_at_c4k"], m["hit_at_c8k"]) == (0.0, 0.0, 0.0, 1.0)
    # legacy metrics only look at the top max(k) = 10 results
    assert m["mrr"] == 0.0 and m["ground_truth_chunk_rank"] is None
