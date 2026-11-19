"""Token and cost tracking per pipeline stage (E9-F1-T2).

A ``UsageTracker`` collects input/output tokens per (stage, model) for every
LLM call of a run. The cost of a call is the one the provider reported
(OpenRouter ``usage.cost``, Jev) when there is one, otherwise it is computed
from ``config/pricing.yaml``; a model with neither is listed as unpriced and
its cost is left out of the total rather than counted as zero.

    tracker = UsageTracker()
    tracker.record("ragas", "gpt-5-nano", input_tokens=900, output_tokens=120)
    tracker.summary()          # stored as "cost_summary" in the result JSON

``instrument_client(client, tracker, stage, model)`` wraps a *private*
``openai.OpenAI`` client so every ``chat.completions.create`` call is
recorded, including calls made by libraries (RAGAS / Instructor) that receive
the client. Never instrument the shared clients from ``get_llm_client``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

STAGES = (
    "generation",
    "judge_llm",
    "judge_jev",
    "ragas",
    "entity_extraction",
    "dataset_generation",
    "assistant",
)


@dataclass
class _Bucket:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reported_cost_usd: float = 0.0
    calls_with_reported_cost: int = 0
    # Token counts of calls without a reported cost (priced from the catalog)
    unreported_input_tokens: int = 0
    unreported_output_tokens: int = 0


class UsageTracker:
    """Thread-safe accumulator of LLM usage, keyed by (stage, model)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[tuple, _Bucket] = {}

    def record(
        self,
        stage: str,
        model: Optional[str],
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        key = (stage, model or "unknown")
        inp, out = int(input_tokens or 0), int(output_tokens or 0)
        with self._lock:
            b = self._buckets.setdefault(key, _Bucket())
            b.calls += 1
            b.input_tokens += inp
            b.output_tokens += out
            if cost_usd is not None:
                b.reported_cost_usd += float(cost_usd)
                b.calls_with_reported_cost += 1
            else:
                b.unreported_input_tokens += inp
                b.unreported_output_tokens += out

    def record_usage(self, stage: str, model: Optional[str], usage: Optional[dict]) -> None:
        """Record an ``extract_usage()`` dict (``prompt_tokens``/``completion_tokens``/``cost_usd``)."""
        if not usage:
            self.record(stage, model)
            return
        self.record(
            stage,
            model,
            usage.get("prompt_tokens", usage.get("input_tokens")),
            usage.get("completion_tokens", usage.get("output_tokens")),
            usage.get("cost_usd"),
        )

    def merge(self, other: "UsageTracker") -> None:
        with other._lock:
            items = list(other._buckets.items())
        with self._lock:
            for key, ob in items:
                b = self._buckets.setdefault(key, _Bucket())
                for f in _Bucket.__dataclass_fields__:
                    setattr(b, f, getattr(b, f) + getattr(ob, f))

    @property
    def empty(self) -> bool:
        return not self._buckets

    def summary(self, pricing=None) -> Dict[str, Any]:
        """Tokens and cost by stage and model, plus run totals.

        ``pricing`` is a ``PricingCatalog``; loaded from ``config/pricing.yaml``
        when omitted.
        """
        if pricing is None:
            from src.automl.pricing import PricingCatalog

            try:
                pricing = PricingCatalog.load()
            except Exception as exc:  # invalid YAML must not break a run
                logger.warning(f"pricing catalog unreadable, costs from providers only: {exc}")
                pricing = PricingCatalog()

        with self._lock:
            items = sorted(self._buckets.items())

        stages: Dict[str, Dict[str, Any]] = {}
        unpriced: set = set()
        total_cost = 0.0
        total_in = total_out = 0
        for (stage, model), b in items:
            cost: Optional[float] = b.reported_cost_usd
            if b.calls > b.calls_with_reported_cost:
                estimated = pricing.llm_cost(
                    model, b.unreported_input_tokens, b.unreported_output_tokens
                )
                if estimated is None:
                    unpriced.add(model)
                    cost = b.reported_cost_usd if b.calls_with_reported_cost else None
                else:
                    cost = b.reported_cost_usd + estimated
            entry = {
                "calls": b.calls,
                "input_tokens": b.input_tokens,
                "output_tokens": b.output_tokens,
                "cost_usd": round(cost, 6) if cost is not None else None,
                "cost_source": _cost_source(b, model in unpriced),
            }
            s = stages.setdefault(
                stage, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "models": {}}
            )
            s["models"][model] = entry
            s["input_tokens"] += b.input_tokens
            s["output_tokens"] += b.output_tokens
            s["cost_usd"] = round(s["cost_usd"] + (cost or 0.0), 6)
            total_cost += cost or 0.0
            total_in += b.input_tokens
            total_out += b.output_tokens

        return {
            "currency": getattr(pricing, "currency", "USD"),
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "by_stage": stages,
            "unpriced_models": sorted(unpriced),
            "complete": not unpriced,
        }


def _cost_source(b: _Bucket, unpriced: bool) -> str:
    if b.calls_with_reported_cost == b.calls:
        return "provider"
    if unpriced:
        return "partial" if b.calls_with_reported_cost else "unpriced"
    return "mixed" if b.calls_with_reported_cost else "catalog"


def instrument_client(client, tracker: UsageTracker, stage: str, model: Optional[str] = None):
    """Record usage of every ``chat.completions.create`` call made with *client*.

    Mutates and returns *client*: use only on a client built for this purpose.
    """
    from src.llm.clients import extract_usage

    completions = client.chat.completions
    original = completions.create

    def create(*args, **kwargs):
        response = original(*args, **kwargs)
        try:
            tracker.record_usage(stage, kwargs.get("model") or model, extract_usage(response))
        except Exception as exc:  # never break the call for accounting
            logger.debug(f"usage tracking failed: {exc}")
        return response

    completions.create = create
    return client
