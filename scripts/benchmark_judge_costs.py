"""Benchmark RAGAS judge LLMs — cost and latency comparison.

Replays RAGAS on the most recent benchmark run (or `--run PATH`) against
multiple judge configurations. Primary (default) path uses RAGAS v0.4
``llm_factory`` + Instructor. Legacy 0.1.x-style conditions (LangchainLLMWrapper
+ manual temperature stripping) are kept for A/B comparison and opt-in via
``--legacy``.

Default conditions:
  * gpt5_nano_minimal          — gpt-5-nano, reasoning=minimal, v0.4 Instructor
  * gemma3_novita_bf16         — google/gemma-3-27b-it via OpenRouter→Novita
                                 (bf16), plain text (legacy path — Instructor
                                 would fail since Novita rejects response_format)

Legacy comparison conditions (opt-in with --legacy):
  * gpt5_nano_minimal_raw      — gpt-5-nano minimal, LangchainLLMWrapper, no json
  * gpt5_nano_minimal_json     — gpt-5-nano minimal, LangchainLLMWrapper + json_object
  * gpt5_nano_low_raw          — gpt-5-nano low,     LangchainLLMWrapper, no json
  * gpt5_nano_low_json         — gpt-5-nano low,     LangchainLLMWrapper + json_object

Usage:
    .venv/bin/python scripts/benchmark_judge_costs.py \\
        [--run PATH] [--sample N] [--metrics a,b,c] [--out PATH] [--legacy] [--only name]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

from config.settings import settings  # noqa: E402
# Import benchmarks first so its __init__.py completes before we touch
# src.evaluation.ragas_evaluator directly — avoids a circular import with
# src.benchmarks.__init__ that re-imports RagasEvaluator.
from src.benchmarks.config import EvaluationConfig  # noqa: E402
from src.evaluation.ragas_evaluator import (  # noqa: E402
    _build_evaluator_embeddings,
    _build_run_config,
    _load_ragas_metrics,
)
DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]

# USD per 1M tokens. Update as prices change; Novita/Gemma figure is sourced
# from OpenRouter's endpoints API (Apr 2026).
PRICING_USD_PER_1M = {
    "gpt-5-nano":            {"in": 0.05,  "out": 0.40},  # VERIFY before relying
    "google/gemma-3-27b-it": {"in": 0.119, "out": 0.20},  # Novita bf16 via OpenRouter
}


@dataclass
class ConditionSpec:
    name: str
    model: str
    reasoning_effort: str | None
    structured_output: bool
    provider: str                  # "openai" | "openrouter"
    v04_native: bool = False       # Use ragas v0.4 llm_factory + Instructor path


@dataclass
class Tracker(BaseCallbackHandler):
    """Captures per-call latency and token usage during RAGAS eval."""
    latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    api_failures: int = 0
    _starts: dict = field(default_factory=dict)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        t0 = self._starts.pop(run_id, None)
        if t0 is not None:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._starts.pop(run_id, None)
        self.api_failures += 1
        if self.api_failures <= 3:  # first few only, to avoid log spam
            msg = str(error)
            if len(msg) > 400:
                msg = msg[:400] + "..."
            print(f"  [llm_error #{self.api_failures}] {type(error).__name__}: {msg}", flush=True)


def _conditions(include_legacy: bool = False) -> list[ConditionSpec]:
    """Return conditions to benchmark.

    By default: the v0.4 Instructor path (our production RAGAS path) and the
    Gemma/Novita baseline (must stay on the legacy LangchainLLMWrapper path
    because Instructor forces response_format which Novita rejects).

    ``--legacy`` adds the 0.1.x-style LangchainLLMWrapper conditions for
    documenting cost/latency deltas against the v0.4 path.
    """
    primary = [
        # Production path: RAGAS v0.4 llm_factory + Instructor (Pydantic-validated json_object).
        ConditionSpec("gpt5_nano_minimal", "gpt-5-nano", "minimal", True, "openai", v04_native=True),
        # Novita rejects response_format at request time, so this stays on the
        # legacy LangchainLLMWrapper path running plain text.
        ConditionSpec("gemma3_novita_bf16", "google/gemma-3-27b-it", None, False, "openrouter"),
    ]
    legacy = [
        ConditionSpec("gpt5_nano_minimal_raw",  "gpt-5-nano", "minimal", False, "openai"),
        ConditionSpec("gpt5_nano_minimal_json", "gpt-5-nano", "minimal", True,  "openai"),
        ConditionSpec("gpt5_nano_low_raw",      "gpt-5-nano", "low",     False, "openai"),
        ConditionSpec("gpt5_nano_low_json",     "gpt-5-nano", "low",     True,  "openai"),
    ]
    return primary + legacy if include_legacy else primary


def _build_chat_llm(spec: ConditionSpec, tracker: Tracker):
    from langchain_openai import ChatOpenAI

    is_reasoning = spec.model.startswith("gpt-5") or spec.model.startswith("o")

    # Reasoning models (GPT-5, o1, o3) only accept temperature=1. RAGAS metrics
    # inject per-call temperature overrides (e.g. 0.01, 0.3) which OpenAI rejects.
    # Override the final-payload hook so temperature never reaches the API.
    class _ReasoningChat(ChatOpenAI):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            kwargs.pop("temperature", None)
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            payload.pop("temperature", None)
            return payload

    chat_cls = _ReasoningChat if is_reasoning else ChatOpenAI

    model_kwargs: dict[str, Any] = {}
    if spec.reasoning_effort:
        model_kwargs["reasoning_effort"] = spec.reasoning_effort
    if spec.structured_output:
        model_kwargs["response_format"] = {"type": "json_object"}

    common = dict(
        model=spec.model,
        max_retries=2,
        model_kwargs=model_kwargs,
        callbacks=[tracker],
        # RAGAS guidance: reasoning models with structured output need ≥4096
        # to avoid truncated JSON. 4096 is safe for non-reasoning too.
        max_tokens=4096,
    )
    # For reasoning models, only temperature=1 is accepted. For others, use 0.
    common["temperature"] = 1 if is_reasoning else 0

    if spec.provider == "openai":
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is empty — set it in .env.")
        return chat_cls(
            openai_api_key=api_key,
            openai_api_base="https://api.openai.com/v1",
            **common,
        )

    if spec.provider == "openrouter":
        api_key = settings.OPENROUTER_API_KEY or settings.GEMMA_API_KEY
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY / GEMMA_API_KEY is empty — set it in .env.")
        return chat_cls(
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            extra_body={
                "provider": {
                    "order": ["Novita"],
                    "allow_fallbacks": False,
                    "quantizations": ["bf16"],
                }
            },
            **common,
        )

    raise ValueError(f"Unknown provider: {spec.provider}")


def _find_latest_run() -> Path:
    candidates: list[Path] = []
    for sub in ("vanilla", "hybrid"):
        d = PROJECT_ROOT / "results" / sub
        if d.exists():
            candidates.extend(p for p in d.glob("*.json") if p.is_file())
    if not candidates:
        raise SystemExit(
            "No benchmark runs found under results/{vanilla,hybrid}/. "
            "Pass --run PATH explicitly."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_questions(run_data: dict) -> dict[str, dict]:
    qpath = (run_data.get("config", {}).get("dataset") or {}).get("questions_file")
    if qpath and Path(qpath).exists():
        payload = json.loads(Path(qpath).read_text())
        qlist = payload.get("questions") if isinstance(payload, dict) else payload
        if isinstance(qlist, list):
            return {q["id"]: q for q in qlist if "id" in q}
    # Fallback — questions_by_id empty means we use the question text from per_question_full.
    print(f"[warn] questions_file not found at {qpath!r}; falling back to inline question text")
    return {}


def _build_samples(per_question: list[dict], questions_by_id: dict[str, dict]):
    from ragas.dataset_schema import SingleTurnSample

    kept: list[tuple[int, SingleTurnSample]] = []
    for i, entry in enumerate(per_question):
        answer = entry.get("generated_answer")
        contexts = entry.get("retrieved_contexts") or []
        if not answer or not contexts:
            continue
        q_id = entry.get("question_id")
        q_data = questions_by_id.get(q_id, {})
        q_text = q_data.get("question") or entry.get("question", "")
        if not q_text:
            continue
        reference = q_data.get("expected_answer_hint") or answer
        kept.append((i, SingleTurnSample(
            user_input=q_text,
            response=answer,
            retrieved_contexts=list(contexts),
            reference=reference,
        )))
    return kept


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]


def _aggregate_scores(result_df, metric_names: list[str]) -> tuple[dict[str, float], int]:
    """Return (metric_name → mean, nan_count_total)."""
    avgs: dict[str, float] = {}
    nan_count = 0
    for m in metric_names:
        if m not in result_df.columns:
            continue
        col = result_df[m]
        vals = [float(v) for v in col if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
        nans = sum(1 for v in col if isinstance(v, float) and math.isnan(v))
        nan_count += nans
        avgs[m] = round(mean(vals), 4) if vals else float("nan")
    return avgs, nan_count


def _build_v04_judge(spec: ConditionSpec, tracker: Tracker):
    """Build a RAGAS v0.4 InstructorLLM, with an httpx transport that records
    per-call latency + token usage into `tracker`."""
    import httpx
    from openai import OpenAI
    from ragas.llms import llm_factory

    class _TrackingTransport(httpx.HTTPTransport):
        def handle_request(self, request):
            t0 = time.perf_counter()
            response = super().handle_request(request)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            tracker.latencies_ms.append(dt_ms)
            try:
                response.read()
                body = json.loads(response.content.decode("utf-8"))
                usage = body.get("usage") or {}
                tracker.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                tracker.completion_tokens += int(usage.get("completion_tokens") or 0)
                details = usage.get("completion_tokens_details") or {}
                if isinstance(details, dict):
                    tracker.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
            except Exception:
                pass
            if response.status_code >= 400:
                tracker.api_failures += 1
            return response

    http_client = httpx.Client(transport=_TrackingTransport())
    client = OpenAI(api_key=settings.OPENAI_API_KEY, http_client=http_client)
    model_args: dict[str, Any] = {"max_tokens": 4096}
    if spec.reasoning_effort:
        model_args["reasoning_effort"] = spec.reasoning_effort
    return llm_factory(spec.model, client=client, **model_args)


def _run_condition(spec: ConditionSpec, samples, metric_names: list[str]) -> dict:
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper

    tracker = Tracker()
    if spec.v04_native:
        judge = _build_v04_judge(spec, tracker)
    else:
        chat = _build_chat_llm(spec, tracker)
        judge = LangchainLLMWrapper(chat)

    embeddings = _build_evaluator_embeddings()
    run_config = _build_run_config(EvaluationConfig(ragas_timeout=180, ragas_max_workers=1))
    registry = _load_ragas_metrics()
    metrics = [registry[m] for m in metric_names if m in registry]

    dataset = EvaluationDataset(samples=[s for _, s in samples])

    print(f"\n[condition] {spec.name} — model={spec.model}, reasoning={spec.reasoning_effort}, json={spec.structured_output}")
    t0 = time.perf_counter()
    parse_error: str | None = None
    try:
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )
        result_df = result.to_pandas()
        avgs, nan_count = _aggregate_scores(result_df, metric_names)
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"
        avgs, nan_count = {}, len(samples) * len(metric_names)
    wall_s = time.perf_counter() - t0

    total_tok = tracker.prompt_tokens + tracker.completion_tokens
    model_price = PRICING_USD_PER_1M.get(spec.model, {"in": 0.0, "out": 0.0})
    cost_usd = (
        tracker.prompt_tokens * model_price["in"] / 1_000_000.0
        + tracker.completion_tokens * model_price["out"] / 1_000_000.0
    )

    return {
        "name": spec.name,
        "model": spec.model,
        "reasoning_effort": spec.reasoning_effort,
        "structured_output": spec.structured_output,
        "wall_time_s": round(wall_s, 2),
        "llm_calls": len(tracker.latencies_ms),
        "latency_ms": {
            "mean": round(mean(tracker.latencies_ms), 1) if tracker.latencies_ms else 0.0,
            "p50": round(_percentile(tracker.latencies_ms, 50), 1),
            "p95": round(_percentile(tracker.latencies_ms, 95), 1),
        },
        "tokens": {
            "prompt": tracker.prompt_tokens,
            "completion": tracker.completion_tokens,
            "reasoning": tracker.reasoning_tokens,
            "total": total_tok,
        },
        "cost_usd": round(cost_usd, 6),
        "failures": {
            "api": tracker.api_failures,
            "json_parse": nan_count,
            "fatal": parse_error,
        },
        "ragas_averages": avgs,
    }


def _print_table(conditions: list[dict], metric_names: list[str]) -> None:
    hdr = [
        "condition", "calls", "wall_s", "p50_ms", "p95_ms",
        "prompt_tok", "compl_tok", "reason_tok", "cost_usd",
        "api_err", "nan", *metric_names,
    ]
    rows = [hdr]
    for c in conditions:
        row = [
            c["name"],
            str(c["llm_calls"]),
            f"{c['wall_time_s']:.1f}",
            f"{c['latency_ms']['p50']:.0f}",
            f"{c['latency_ms']['p95']:.0f}",
            str(c["tokens"]["prompt"]),
            str(c["tokens"]["completion"]),
            str(c["tokens"]["reasoning"]),
            f"{c['cost_usd']:.5f}",
            str(c["failures"]["api"]),
            str(c["failures"]["json_parse"]),
            *(
                f"{c['ragas_averages'].get(m, float('nan')):.3f}"
                if not math.isnan(c["ragas_averages"].get(m, float("nan")))
                else "—"
                for m in metric_names
            ),
        ]
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(hdr))]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    print()
    for idx, r in enumerate(rows):
        print("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)) + " |")
        if idx == 0:
            print(sep)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=None, help="Benchmark run JSON to replay (default: newest)")
    ap.add_argument("--sample", type=int, default=20, help="Questions to sample (0 = all); default 20")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated RAGAS metric names")
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None, help="Comma-separated condition name prefixes (e.g. 'gpt5')")
    ap.add_argument("--legacy", action="store_true", help="Also run the 0.1.x LangchainLLMWrapper conditions")
    args = ap.parse_args()

    run_path = args.run or _find_latest_run()
    run_data = json.loads(run_path.read_text())
    per_q = run_data.get("per_question_full") or []
    questions_by_id = _load_questions(run_data)

    all_samples = _build_samples(per_q, questions_by_id)
    if not all_samples:
        raise SystemExit(f"No eligible samples in {run_path} (need generated_answer + retrieved_contexts).")

    if args.sample and args.sample < len(all_samples):
        rng = random.Random(args.seed)
        samples = rng.sample(all_samples, args.sample)
    else:
        samples = all_samples

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print(f"source run:   {run_path.relative_to(PROJECT_ROOT)}")
    print(f"samples:      {len(samples)} / {len(all_samples)} eligible (seed={args.seed})")
    print(f"metrics:      {metric_names}")

    only_prefixes = [p.strip() for p in args.only.split(",")] if args.only else None
    results = []
    for spec in _conditions(include_legacy=args.legacy):
        if only_prefixes and not any(spec.name.startswith(p) for p in only_prefixes):
            print(f"[skip] {spec.name} (filtered by --only)")
            continue
        results.append(_run_condition(spec, samples, metric_names))

    _print_table(results, metric_names)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_hash = run_data.get("config_hash", "unknown")[:8]
    out_path = args.out or (PROJECT_ROOT / "results" / "judge_benchmarks" / f"judge_bench_{run_hash}_{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "source_run": str(run_path.relative_to(PROJECT_ROOT)),
        "sample_size": len(samples),
        "eligible_size": len(all_samples),
        "seed": args.seed,
        "metrics": metric_names,
        "pricing_usd_per_1m": PRICING_USD_PER_1M,
        "timestamp": ts,
        "conditions": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nReport written to {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
