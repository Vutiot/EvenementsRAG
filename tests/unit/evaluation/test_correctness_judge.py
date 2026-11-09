"""Tests for binary answer correctness: LLM judge, Jev judge, agreement, calibration (E10-F2)."""

import json
from types import SimpleNamespace

import pytest

from src.benchmarks.config import EvaluationConfig
from src.evaluation.correctness_judge import (
    JevError,
    JevJudge,
    JudgeVerdict,
    LLMCorrectnessJudge,
    build_jev_payload,
    build_judge_prompt,
    calibrate_threshold,
    cohen_kappa,
    criteria_for,
    is_failed_generation,
    judge_agreement,
    parse_jev_response,
    score_verdicts,
)
from src.evaluation.metrics_collector import MetricsCollector, core_metrics_summary

QUESTION = {
    "id": "q1",
    "question": "When did the Normandy landings start?",
    "type": "temporal",
    "expected_answer_hint": "6 June 1944",
}
MULTI_HOP = {**QUESTION, "id": "q2", "type": "comparative"}


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


class TestCriteria:
    def test_single_hop_has_three_criteria(self):
        assert criteria_for(QUESTION) == ["facts_vs_gold", "numbers_units_correct", "addresses_question"]

    def test_multi_hop_adds_combines_all_hops(self):
        assert criteria_for(MULTI_HOP)[-1] == "combines_all_hops"
        assert criteria_for({**QUESTION, "hops": 2})[-1] == "combines_all_hops"

    def test_no_unsupported_claims_criterion(self):
        assert "no_unsupported_claims" not in criteria_for(MULTI_HOP)
        assert "no_unsupported_claims" not in build_judge_prompt(MULTI_HOP, "x")

    def test_prompt_is_english_and_mentions_abstention(self):
        prompt = build_judge_prompt(QUESTION, "June 6, 1944")
        assert "6 June 1944" in prompt and "June 6, 1944" in prompt
        assert "abstention" in prompt
        assert "emit_verdicts" in prompt


class TestScoreVerdicts:
    crits = ["facts_vs_gold", "numbers_units_correct", "addresses_question"]

    def _v(self, facts="equivalent", numbers="yes", addresses="yes"):
        return [
            {"criterion": "facts_vs_gold", "verdict": facts},
            {"criterion": "numbers_units_correct", "verdict": numbers},
            {"criterion": "addresses_question", "verdict": addresses},
        ]

    @pytest.mark.parametrize("facts", ["equivalent", "superset", "style_only"])
    def test_passing_fact_verdicts(self, facts):
        assert score_verdicts(self.crits, self._v(facts=facts))["correct"] == 1.0

    @pytest.mark.parametrize("facts", ["subset", "disagree"])
    def test_failing_fact_verdicts(self, facts):
        assert score_verdicts(self.crits, self._v(facts=facts))["correct"] == 0.0

    def test_any_no_fails(self):
        assert score_verdicts(self.crits, self._v(numbers="no"))["correct"] == 0.0

    def test_missing_criterion_fails(self):
        out = score_verdicts(self.crits, self._v()[:2])
        assert out["correct"] == 0.0 and out["complete"] is False

    def test_extra_criterion_ignored(self):
        verdicts = self._v() + [{"criterion": "no_unsupported_claims", "verdict": "no"}]
        assert score_verdicts(self.crits, verdicts)["correct"] == 1.0


def _tool_response(verdicts, cost=0.0003):
    call = SimpleNamespace(function=SimpleNamespace(name="emit_verdicts", arguments=json.dumps({"verdicts": verdicts})))
    message = SimpleNamespace(tool_calls=[call], content=None)
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=120, cost=cost, model_extra={})
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    def __init__(self, response=None, exc=None):
        self.calls = []
        self._response, self._exc = response, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc:
            raise self._exc
        return self._response


class TestLLMCorrectnessJudge:
    def test_correct_answer(self):
        verdicts = [
            {"criterion": c, "verdict": "equivalent" if c == "facts_vs_gold" else "yes", "rationale": "", "evidence_quote": ""}
            for c in criteria_for(QUESTION)
        ]
        client = FakeClient(_tool_response(verdicts))
        judge = LLMCorrectnessJudge(model="openai/gpt-5.4-mini", client=client)
        v = judge.judge(QUESTION, "It started on 6 June 1944.")
        assert v.correct == 1.0
        assert v.cost_usd == pytest.approx(0.0003)
        call = client.calls[0]
        assert call["model"] == "openai/gpt-5.4-mini"
        assert call["temperature"] == 0.0 and call["max_tokens"] == 700
        assert call["tool_choice"]["function"]["name"] == "emit_verdicts"
        assert call["extra_body"] == {"usage": {"include": True}}

    def test_failure_returns_none(self):
        judge = LLMCorrectnessJudge(model="m", client=FakeClient(exc=RuntimeError("boom")))
        v = judge.judge(QUESTION, "x")
        assert v.correct is None and "boom" in v.error


def test_failed_generation_detection():
    assert is_failed_generation(None)
    assert is_failed_generation("  ")
    assert is_failed_generation("[Generation failed: timeout]\n\nRetrieved 3 passages")
    assert not is_failed_generation("6 June 1944")


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------

JEV_OK = {
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "answers": {"same": {"type": "noul", "noul": 0.83}},
    "usage": {"input_tokens": 1234, "output_tokens": 0, "cost": 0.0000518},
}


class FakeResp:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code, self._data, self.headers, self.text = status, data, headers or {}, ""

    def json(self):
        return self._data


class TestJevPayload:
    def test_payload_has_only_three_keys(self):
        payload = build_jev_payload(QUESTION, "6 June 1944")
        assert set(payload) == {"model", "state", "questions"}
        q = payload["questions"]["same"]
        assert q["type"] == "noul"
        assert set(q["criteria"]) == {"true", "false"}
        assert "6 June 1944" in payload["state"]


class TestParseJev:
    def test_ok(self):
        out = parse_jev_response(JEV_OK)
        assert out["prob_yes"] == 0.83
        assert out["cost_usd"] == pytest.approx(0.0000518)

    def test_cost_fallback_from_input_tokens(self):
        data = {**JEV_OK, "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}
        assert parse_jev_response(data, price_per_mtok=0.042)["cost_usd"] == pytest.approx(0.042)

    def test_error_in_200_body(self):
        with pytest.raises(JevError):
            parse_jev_response({"error": {"code": 502, "message": "upstream"}})

    def test_unexpected_snapshot(self):
        with pytest.raises(JevError):
            parse_jev_response({**JEV_OK, "model": "typesafe/jev-2.0"})


class TestJevJudge:
    def _judge(self, responses, threshold=0.5):
        it = iter(responses)
        posted = []

        def post(url, headers, json_body, timeout):
            posted.append((url, headers, json_body))
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r

        sleeps = []
        judge = JevJudge(threshold=threshold, api_key="test-key", http_post=post, sleep=sleeps.append)
        return judge, posted, sleeps

    def test_threshold_applied(self):
        judge, posted, _ = self._judge([FakeResp(data=JEV_OK)], threshold=0.9)
        v = judge.judge(QUESTION, "6 June 1944")
        assert v.prob_yes == 0.83 and v.correct == 0.0
        url, headers, body = posted[0]
        assert url.endswith("/api/alpha/decisions")
        assert headers["Authorization"] == "Bearer test-key"
        assert body["model"] == "typesafe/jev-1.13"

    def test_retries_on_429_with_retry_after(self):
        judge, _, sleeps = self._judge([FakeResp(429, headers={"Retry-After": "120"}), FakeResp(data=JEV_OK)])
        v = judge.judge(QUESTION, "x")
        assert v.correct == 1.0
        assert sleeps == [60.0]  # Retry-After capped at 60 s

    def test_retries_on_timeout_with_backoff(self):
        judge, _, sleeps = self._judge([TimeoutError(), TimeoutError(), FakeResp(data=JEV_OK)])
        assert judge.judge(QUESTION, "x").correct == 1.0
        assert sleeps == [2.0, 4.0]

    def test_gives_up_after_four_attempts(self):
        judge, posted, _ = self._judge([FakeResp(503)] * 4)
        v = judge.judge(QUESTION, "x")
        assert v.correct is None and len(posted) == 4

    def test_no_retry_on_400(self):
        judge, posted, _ = self._judge([FakeResp(400)])
        assert judge.judge(QUESTION, "x").correct is None
        assert len(posted) == 1


# ---------------------------------------------------------------------------
# Agreement and calibration
# ---------------------------------------------------------------------------


class TestAgreement:
    def test_perfect_agreement(self):
        out = judge_agreement([1, 0, 1], [1, 0, 1])
        assert out["agreement"] == 1.0 and out["kappa"] == 1.0

    def test_counts_and_none_skipped(self):
        out = judge_agreement([1, 1, 0, 0, None], [1, 0, 1, 0, 1])
        assert out["n"] == 4
        assert (out["both_correct"], out["llm_only"], out["jev_only"], out["both_wrong"]) == (1, 1, 1, 1)
        assert out["agreement"] == 0.5
        assert out["kappa"] == pytest.approx(0.0)

    def test_kappa_undefined_on_single_class_disagreement(self):
        assert cohen_kappa([], []) is None


class TestCalibration:
    def test_finds_threshold_between_classes(self):
        probs = [0.30, 0.45, 0.52, 0.55, 0.60, 0.70, 0.80]
        labels = [0, 0, 0, 0, 1, 1, 1]
        out = calibrate_threshold(probs, labels)
        assert 0.55 < out["threshold"] <= 0.60
        assert out["balanced_accuracy"] == 1.0
        assert out["at_0_5"]["balanced_accuracy"] < 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            calibrate_threshold([], [])


# ---------------------------------------------------------------------------
# MetricsCollector integration
# ---------------------------------------------------------------------------


class StubJudge:
    def __init__(self, name, correct, cost=0.001, prob=None):
        self.name, self._correct, self._cost, self._prob = name, correct, cost, prob
        self.threshold = 0.5
        self.calls = 0

    def judge(self, question, answer):
        self.calls += 1
        return JudgeVerdict(judge=self.name, correct=self._correct[question["id"]], cost_usd=self._cost, prob_yes=self._prob)


def test_collector_scores_both_judges_and_agreement():
    questions = {
        "q1": QUESTION,
        "q2": {**MULTI_HOP},
        "q3": {**QUESTION, "id": "q3"},
        "q4": {"id": "q4", "question": "no gold"},
    }
    per_q = [
        {"question_id": "q1", "generated_answer": "6 June 1944", "metrics": {"hit_at_5": 1.0},
         "retrieval_time_ms": 10.0, "generation_time_ms": 90.0, "generation_usage": {"cost_usd": 0.002}},
        {"question_id": "q2", "generated_answer": "answer", "metrics": {"hit_at_5": 0.0},
         "retrieval_time_ms": 20.0, "generation_time_ms": 180.0, "generation_usage": {"cost_usd": 0.004}},
        {"question_id": "q3", "generated_answer": "[Generation failed: boom]", "metrics": {}},
        {"question_id": "q4", "generated_answer": "whatever", "metrics": {}},
    ]
    llm = StubJudge("llm", {"q1": 1.0, "q2": 1.0})
    jev = StubJudge("jev", {"q1": 1.0, "q2": 0.0}, cost=0.00005, prob=0.6)
    mc = MetricsCollector(EvaluationConfig())
    mc.compute_answer_correctness(per_q, questions, llm_judge=llm, jev_judge=jev)

    assert llm.calls == 2 and jev.calls == 2  # failed generation and no-gold question cost nothing
    assert per_q[0]["metrics"]["answer_correctness"] == 1.0
    assert per_q[1]["metrics"]["answer_correctness_jev"] == 0.0
    assert per_q[1]["correctness"]["agree"] is False
    assert per_q[2]["metrics"]["answer_correctness"] == 0.0  # failed generation scores 0
    assert "correctness" not in per_q[3]

    mc.compute_cost_metrics(per_q)
    summary = mc.get_summary()
    corr = summary["correctness"]
    assert corr["num_questions_scored"] == 3
    assert corr["answer_correctness"] == pytest.approx(2 / 3)
    assert corr["answer_correctness_jev"] == pytest.approx(1 / 3)
    assert corr["judge_agreement"] == pytest.approx(2 / 3)
    assert per_q[0]["cost_usd"] == pytest.approx(0.002 + 0.001 + 0.00005)
    assert per_q[0]["latency_ms"] == 100.0
    assert summary["cost"]["latency_per_question_ms"] == pytest.approx((100 + 200) / 4)

    flat = core_metrics_summary({**summary, "hits": MetricsCollector.summarize_hits(per_q)})
    assert flat["hit_at_5"] == 0.5
    assert flat["answer_correctness"] == pytest.approx(0.6667, abs=1e-4)
    assert "cost_per_question_usd" in flat


def test_collector_skips_when_disabled():
    per_q = [{"question_id": "q1", "generated_answer": "x"}]
    MetricsCollector(EvaluationConfig(compute_answer_correctness=False)).compute_answer_correctness(per_q, {"q1": QUESTION})
    assert "correctness" not in per_q[0]


def test_generation_config_routes_known_models():
    from src.benchmarks.config import GenerationConfig

    assert GenerationConfig().model == "z-ai/glm-5.3-flash"
    assert GenerationConfig().llm_provider == "openrouter"
    assert GenerationConfig(model="google/gemma-3-27b-it").llm_provider == "nvidia"
    assert GenerationConfig(model="some/other", llm_provider="openai").llm_provider == "openai"
