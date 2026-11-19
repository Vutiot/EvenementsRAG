"""Tests for per-stage token and cost tracking (E9-F1-T2)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.automl.pricing import ModelPrice, PricingCatalog
from src.benchmarks.config import EvaluationConfig
from src.evaluation.metrics_collector import MetricsCollector
from src.llm.usage import UsageTracker, instrument_client

PRICING = PricingCatalog(
    models={
        "priced": ModelPrice(input_per_1m=1.0, output_per_1m=2.0),
        "gpt-5-nano": ModelPrice(input_per_1m=0.05, output_per_1m=0.4),
    }
)


def test_provider_cost_wins_and_catalog_fills_the_rest():
    t = UsageTracker()
    t.record("generation", "priced", 1000, 500, cost_usd=0.01)  # reported
    t.record("generation", "priced", 1_000_000, 0)  # catalog: $1
    s = t.summary(PRICING)
    m = s["by_stage"]["generation"]["models"]["priced"]
    assert m["calls"] == 2
    assert m["input_tokens"] == 1_001_000
    assert m["cost_usd"] == pytest.approx(1.01)
    assert m["cost_source"] == "mixed"
    assert s["total_cost_usd"] == pytest.approx(1.01)
    assert s["complete"] is True


def test_unpriced_model_is_flagged_not_counted_as_free():
    t = UsageTracker()
    t.record("ragas", "mystery", 100, 10)
    t.record("judge_llm", "priced", 1_000_000, 0)
    s = t.summary(PRICING)
    assert s["unpriced_models"] == ["mystery"]
    assert s["complete"] is False
    assert s["by_stage"]["ragas"]["models"]["mystery"]["cost_usd"] is None
    assert s["by_stage"]["ragas"]["models"]["mystery"]["cost_source"] == "unpriced"
    assert s["total_cost_usd"] == pytest.approx(1.0)


def test_merge_and_record_usage():
    a, b = UsageTracker(), UsageTracker()
    a.record_usage("generation", "priced", {"prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.1})
    b.record_usage("generation", "priced", {"prompt_tokens": 20, "completion_tokens": 5, "cost_usd": 0.2})
    a.merge(b)
    m = a.summary(PRICING)["by_stage"]["generation"]["models"]["priced"]
    assert (m["calls"], m["input_tokens"], m["cost_usd"]) == (2, 30, pytest.approx(0.3))


def test_instrument_client_records_every_call():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, model_extra={}))
    client = MagicMock()
    client.chat.completions.create.return_value = response
    t = UsageTracker()
    instrument_client(client, t, "ragas", "gpt-5-nano")
    assert client.chat.completions.create(model="gpt-5-nano", messages=[]) is response
    client.chat.completions.create(messages=[])  # model taken from the default
    m = t.summary(PRICING)["by_stage"]["ragas"]["models"]["gpt-5-nano"]
    assert (m["calls"], m["input_tokens"], m["output_tokens"]) == (2, 14, 6)


def test_collector_cost_summary_covers_all_stages():
    cfg = EvaluationConfig()
    collector = MetricsCollector(cfg)
    collector.usage.record("ragas", "gpt-5-nano", 1_000_000, 0)  # $0.05 from catalog
    collector.usage.record("entity_extraction", "priced", 0, 0, cost_usd=0.0)
    per_q = [
        {
            "generation_usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.002},
            "correctness": {
                "llm": {"correct": 1.0, "input_tokens": 300, "output_tokens": 40, "cost_usd": 0.003},
                "jev": {"correct": 1.0, "input_tokens": 200, "cost_usd": 0.00001},
                "agree": True,
            },
        },
        {"generation_usage": None, "correctness": {"llm": {"correct": 0.0, "error": "no answer generated"}}},
    ]
    s = collector.compute_cost_summary(per_q, generation_model="z-ai/glm-5.3-flash", pricing=PRICING)
    stages = s["by_stage"]
    assert set(stages) == {"generation", "judge_llm", "judge_jev", "ragas", "entity_extraction"}
    assert stages["judge_llm"]["models"][cfg.correctness_judge_model]["calls"] == 1  # skipped one not counted
    assert stages["judge_jev"]["models"][cfg.jev_snapshot]["cost_usd"] == pytest.approx(0.00001)
    assert s["total_cost_usd"] == pytest.approx(0.002 + 0.003 + 0.00001 + 0.05)
    assert collector.get_summary()["cost_summary"] == s


def test_result_file_info_exposes_run_cost(tmp_path, monkeypatch):
    from src.api.routers import results as results_router

    f = tmp_path / "vanilla" / "run.json"
    f.parent.mkdir()
    f.write_text(
        json.dumps(
            {
                "config": {"name": "run"},
                "evaluation": {"avg_mrr": 0.5, "avg_recall_at_k": {}, "total_questions": 1},
                "metrics_summary": {"cost_summary": {"total_cost_usd": 0.42, "complete": False}},
                "cached_from": "vanilla/old.json",
            }
        )
    )
    monkeypatch.setattr(results_router, "RESULTS_DIR", tmp_path)
    info = results_router._parse_file_info(f)
    assert info.total_cost_usd == 0.42
    assert info.cost_complete is False
    assert info.cached_from == "vanilla/old.json"
