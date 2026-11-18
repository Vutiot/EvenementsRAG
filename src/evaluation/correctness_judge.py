"""
Binary answer correctness, judged by two independent judges (E10-F2).

Judge 1: a reference-guided LLM judge (default ``openai/gpt-5.4-mini`` via
OpenRouter) that returns one verdict per criterion through an
``emit_verdicts`` tool call. The answer is correct only if every criterion
passes:

- ``facts_vs_gold``: equivalent / superset / style_only pass; subset / disagree fail
- ``numbers_units_correct``: numbers, units, signs and qualifiers the question asks for
- ``addresses_question``: answers this question
- ``combines_all_hops``: multi-hop questions only

There is deliberately no ``no_unsupported_claims`` criterion. An abstention on
a question that has a reference answer scores 0.

Judge 2: TypeSafe Jev through OpenRouter's decisions endpoint
(``POST /api/alpha/decisions``, question type ``noul``). Jev returns P(yes),
no rationale. It is under-confident at 0.5, so the decision threshold is a
config value that :func:`calibrate_threshold` fits on labelled verdicts.

Both judges use English prompts. :func:`judge_agreement` reports how often
they agree (raw agreement and Cohen's kappa).
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Question types of the EvenementsRAG taxonomy that need several facts combined.
MULTI_HOP_TYPES = frozenset({"comparative", "relationship", "analytical", "multi_hop"})

_GENERATION_FAILURE_RE = re.compile(r"^\s*\[(generation failed|model returned empty response)", re.I)


def is_multi_hop(question: dict) -> bool:
    """True when the question needs several hops (explicit flag or taxonomy type)."""
    if question.get("multi_hop") is True:
        return True
    hops = question.get("hops")
    if isinstance(hops, (int, float)) and hops > 1:
        return True
    if isinstance(hops, list) and len(hops) > 1:
        return True
    return (question.get("type") or "").strip() in MULTI_HOP_TYPES


def gold_answer(question: dict) -> Optional[str]:
    """Reference answer of a dataset question (EvenementsRAG stores it as a hint)."""
    for key in ("gold_answer", "reference_answer", "answer", "expected_answer_hint"):
        val = question.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def is_failed_generation(answer: Optional[str]) -> bool:
    """Generation errors and empty answers are scored 0 without calling a judge."""
    return not answer or not answer.strip() or bool(_GENERATION_FAILURE_RE.match(answer))


# ---------------------------------------------------------------------------
# Judge 1 — criterion-based LLM judge
# ---------------------------------------------------------------------------

FACTS_PASS = frozenset({"equivalent", "superset", "style_only"})
FACTS_VERDICTS = ("equivalent", "superset", "subset", "disagree", "style_only")

CRITERIA_DESCRIPTIONS: Dict[str, str] = {
    "facts_vs_gold": (
        "Does the answer state the same facts as the reference? Use the factuality "
        "taxonomy below."
    ),
    "numbers_units_correct": (
        "Every number, date, unit, sign and qualifier that the QUESTION asks for matches "
        "the reference in value. Formatting differences do not count. A detail the "
        "reference carries but the question does not ask for is not a missing number. "
        "A correct number added beyond the reference is not an error unless it "
        "contradicts the reference. If the question asks for no number or date, answer yes."
    ),
    "addresses_question": "The answer actually answers THIS question (not a related one).",
    "combines_all_hops": (
        "The question needs several facts combined (multi-hop). The answer uses and "
        "combines every hop needed to reach the reference answer."
    ),
}

TAXONOMY_BLOCK = """For the criterion `facts_vs_gold` the verdict is one of (OpenAI-evals factuality taxonomy):
- "equivalent": the answer states the same facts as the reference
- "superset": the answer contains all reference facts plus additional consistent facts
- "subset": the answer contains only part of the reference facts
- "disagree": the answer contradicts the reference on any fact, number, date or unit
- "style_only": same facts, different wording or format only
For every other criterion the verdict is "yes" or "no"."""

JUDGE_PROMPT_TEMPLATE = """You are a strict answer evaluator for a question-answering evaluation over a corpus of Wikipedia articles about historical events (mostly World War II). Judge the assistant answer against the reference answer on the criteria listed below — binary, reference-guided, pointwise. Length and style are NEVER criteria: a short correct answer passes, a long fluent wrong answer fails.

Question:
{question}

Reference (gold) answer:
{gold}
Accepted variants: {variants}

Assistant answer:
{answer}

Formatting differences that lose no meaning are NEVER a mismatch: a decimal comma versus a decimal point (2,69 = 2.69), date formats (6 June 1944 = 1944-06-06), the spelling or case of a unit or a name, thousands separators, spacing, accents. A number or detail the reference carries but the question does not ask for is not a missing number. A correct number the answer adds beyond the reference is not a wrong number unless it contradicts the reference.

The question has a reference answer, so an abstention ("I don't know", "the context does not say", a refusal) fails every criterion.

Criteria to judge, in order:
{criteria}

{taxonomy}

Call the emit_verdicts tool with one verdict object per listed criterion, in the listed order: {{criterion, verdict, rationale (one or two sentences), evidence_quote (the answer fragment the verdict rests on, verbatim, or "")}}.

Judge ONLY the listed criteria. Emit exactly as many verdict objects as there are listed criteria, and use the listed criterion names verbatim. Do NOT add a criterion that is not listed, even when it looks relevant to this answer — the list above is complete and final for this question's category, and an extra criterion invalidates the whole verdict."""

EMIT_VERDICTS_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_verdicts",
        "description": "Emit one verdict per listed criterion, in order.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "verdict": {
                                "type": "string",
                                "enum": ["yes", "no", *FACTS_VERDICTS],
                            },
                            "rationale": {"type": "string"},
                            "evidence_quote": {"type": "string"},
                        },
                        "required": ["criterion", "verdict", "rationale", "evidence_quote"],
                    },
                }
            },
            "required": ["verdicts"],
        },
    },
}


def criteria_for(question: dict) -> List[str]:
    """Criteria judged for *question* (``combines_all_hops`` only when multi-hop)."""
    crits = ["facts_vs_gold", "numbers_units_correct", "addresses_question"]
    if is_multi_hop(question):
        crits.append("combines_all_hops")
    return crits


def build_judge_prompt(question: dict, answer: str) -> str:
    crits = criteria_for(question)
    variants = question.get("accepted_variants") or question.get("gold_variants") or []
    if isinstance(variants, str):
        variants = [variants]
    criteria_text = "\n".join(
        f"{i}. `{c}`: {CRITERIA_DESCRIPTIONS[c]}" for i, c in enumerate(crits, 1)
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question.get("question", ""),
        gold=gold_answer(question) or "",
        variants="; ".join(variants) if variants else "none",
        answer=answer,
        criteria=criteria_text,
        taxonomy=TAXONOMY_BLOCK,
    )


def criterion_passes(criterion: str, verdict: str) -> bool:
    v = (verdict or "").strip().lower()
    if criterion == "facts_vs_gold":
        return v in FACTS_PASS
    return v == "yes"


def score_verdicts(criteria: Sequence[str], verdicts: List[dict]) -> Dict:
    """AND of the listed criteria. Missing or unknown criteria count as failures."""
    by_name = {}
    for item in verdicts or []:
        name = (item or {}).get("criterion")
        if name in criteria and name not in by_name:
            by_name[name] = item
    passed = {c: criterion_passes(c, (by_name.get(c) or {}).get("verdict", "")) for c in criteria}
    complete = all(c in by_name for c in criteria)
    return {
        "correct": 1.0 if complete and all(passed.values()) else 0.0,
        "criteria": passed,
        "complete": complete,
    }


def _parse_tool_verdicts(response) -> List[dict]:
    """Read the emit_verdicts arguments from a chat completion (tool call or JSON text)."""
    msg = response.choices[0].message
    raw = None
    for call in getattr(msg, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        if fn is not None and getattr(fn, "name", "") == "emit_verdicts":
            raw = fn.arguments
            break
    if raw is None:
        raw = getattr(msg, "content", None) or ""
        match = re.search(r"\{.*\}|\[.*\]", raw, re.S)
        raw = match.group(0) if match else "[]"
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        data = data.get("verdicts", [])
    return data if isinstance(data, list) else []


@dataclass
class JudgeVerdict:
    """One judge's verdict on one answer."""

    judge: str
    correct: Optional[float]                 # 1.0 / 0.0, None when the judge failed
    criteria: Dict[str, bool] = field(default_factory=dict)
    details: List[dict] = field(default_factory=list)
    prob_yes: Optional[float] = None         # Jev only
    cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {})}


class LLMCorrectnessJudge:
    """Criterion-based binary judge over an OpenAI-compatible chat endpoint."""

    name = "llm"

    def __init__(self, model: str, provider: str = "openrouter", client=None, max_tokens: int = 700):
        from src.llm import get_llm_client, usage_request_kwargs

        self.model = model
        self.provider = provider
        self.client = client or get_llm_client(provider)
        self.max_tokens = max_tokens
        self._extra = usage_request_kwargs(provider)

    def judge(self, question: dict, answer: str) -> JudgeVerdict:
        from src.llm import extract_usage

        criteria = criteria_for(question)
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": build_judge_prompt(question, answer)}],
                tools=[EMIT_VERDICTS_TOOL],
                tool_choice={"type": "function", "function": {"name": "emit_verdicts"}},
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.max_tokens,
                **self._extra,
            )
            verdicts = _parse_tool_verdicts(response)
        except Exception as exc:  # network, provider or parsing failure
            logger.warning(f"LLM judge failed for {question.get('id')}: {exc}")
            return JudgeVerdict(judge=self.name, correct=None, error=str(exc),
                                latency_ms=(time.time() - start) * 1000)

        usage = extract_usage(response)
        scored = score_verdicts(criteria, verdicts)
        return JudgeVerdict(
            judge=self.name,
            correct=scored["correct"],
            criteria=scored["criteria"],
            details=verdicts,
            cost_usd=usage["cost_usd"],
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            latency_ms=(time.time() - start) * 1000,
            error=None if scored["complete"] else "incomplete verdict list",
        )


# ---------------------------------------------------------------------------
# Judge 2 — TypeSafe Jev (OpenRouter decisions endpoint)
# ---------------------------------------------------------------------------

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_SNAPSHOT_RE = re.compile(r"^typesafe/jev-1\.13(-\d{8})?$")

JEV_INSTRUCTIONS = (
    "You are checking an answer from a question-answering system over Wikipedia articles "
    "about historical events. The state gives the question, the reference (gold) answer "
    "and the assistant answer. Does the assistant answer correctly answer the question, "
    "in agreement with the reference? Length, style and formatting differences "
    "(date formats, decimal separators, spelling of names or units) do not matter. "
    "The question has a reference answer, so an abstention is not a correct answer."
)
JEV_CRITERIA = {
    "true": (
        "The answer states the same facts as the reference, every number, date and unit "
        "the question asks for matches the reference, and it answers this question"
        " (combining every fact needed when the question needs several)."
    ),
    "false": (
        "The answer contradicts the reference, misses a fact the question asks for, "
        "answers a different question, or abstains."
    ),
}


def build_jev_state(question: dict, answer: str) -> str:
    return (
        f"Question:\n{question.get('question', '')}\n\n"
        f"Reference answer:\n{gold_answer(question) or ''}\n\n"
        f"Assistant answer:\n{answer}"
    )


def build_jev_payload(question: dict, answer: str, model: str = "typesafe/jev-1.13") -> dict:
    """Request body: exactly ``model``, ``state`` and ``questions`` (no sampling params)."""
    return {
        "model": model,
        "state": build_jev_state(question, answer),
        "questions": {
            "same": {
                "type": "noul",
                "instructions": JEV_INSTRUCTIONS,
                "criteria": dict(JEV_CRITERIA),
            }
        },
    }


class JevError(RuntimeError):
    pass


def parse_jev_response(data: dict, price_per_mtok: float = 0.042, expected_snapshot: Optional[str] = None) -> dict:
    """Extract ``prob_yes``, tokens and cost from a decisions response.

    An upstream failure can come back as HTTP 200 with ``{"error": ...}`` and no
    ``answers``: that is an error, not a verdict.
    """
    if not isinstance(data, dict):
        raise JevError("non-JSON response")
    if "error" in data and "answers" not in data:
        err = data["error"] or {}
        raise JevError(f"upstream error {err.get('code')}: {err.get('message')}")
    try:
        prob = float(data["answers"]["same"]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JevError(f"missing answers.same.noul in response: {exc}") from exc
    if not 0.0 <= prob <= 1.0:
        raise JevError(f"noul out of range: {prob}")

    snapshot = data.get("model")
    if snapshot and not JEV_SNAPSHOT_RE.match(snapshot):
        raise JevError(f"unexpected Jev snapshot '{snapshot}'")
    if expected_snapshot and snapshot and snapshot != expected_snapshot:
        logger.warning(f"Jev answered with snapshot {snapshot}, config pins {expected_snapshot}")

    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)):
        cost = (input_tokens or 0) * price_per_mtok / 1_000_000 if input_tokens else None
    return {
        "prob_yes": prob,
        "snapshot": snapshot,
        "input_tokens": input_tokens,
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": float(cost) if cost is not None else None,
    }


class JevJudge:
    """Second correctness judge: TypeSafe Jev, P(yes) thresholded."""

    name = "jev"
    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        threshold: float = 0.5,
        snapshot: str = "typesafe/jev-1.13-20260917",
        price_per_mtok: float = 0.042,
        api_key: Optional[str] = None,
        http_post: Optional[Callable] = None,
        max_attempts: int = 4,
        timeout_s: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        from config.settings import settings

        self.threshold = threshold
        self.snapshot = snapshot
        self.price_per_mtok = price_per_mtok
        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._post = http_post or self._default_post
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY is not set: Jev calls will fail.")

    def _default_post(self, url: str, headers: dict, json_body: dict, timeout: float):
        import httpx

        return httpx.post(url, headers=headers, json=json_body, timeout=timeout)

    def _request(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        delay = 2.0
        last_error = "no attempt"
        for attempt in range(1, self.max_attempts + 1):
            retry_after = None
            try:
                resp = self._post(JEV_URL, headers, payload, self.timeout_s)
                status = getattr(resp, "status_code", 200)
                if status == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "error" in data and "answers" not in data:
                        code = (data.get("error") or {}).get("code")
                        if isinstance(code, int) and code >= 500 and attempt < self.max_attempts:
                            last_error = f"upstream error {code}"
                            self._sleep(delay)
                            delay *= 2
                            continue
                    return data
                last_error = f"HTTP {status}"
                if status not in self.RETRY_STATUS:
                    raise JevError(f"{last_error}: {getattr(resp, 'text', '')[:200]}")
                ra = (getattr(resp, "headers", None) or {}).get("Retry-After")
                if ra is not None:
                    try:
                        retry_after = min(float(ra), 60.0)
                    except ValueError:
                        retry_after = None
            except JevError:
                raise
            except Exception as exc:  # timeout / connection error
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_attempts:
                self._sleep(retry_after if retry_after is not None else delay)
                delay *= 2
        raise JevError(f"Jev request failed after {self.max_attempts} attempts ({last_error})")

    def judge(self, question: dict, answer: str) -> JudgeVerdict:
        start = time.time()
        try:
            data = self._request(build_jev_payload(question, answer))
            parsed = parse_jev_response(data, self.price_per_mtok, self.snapshot)
        except Exception as exc:
            logger.warning(f"Jev judge failed for {question.get('id')}: {exc}")
            return JudgeVerdict(judge=self.name, correct=None, error=str(exc),
                                latency_ms=(time.time() - start) * 1000)
        prob = parsed["prob_yes"]
        return JudgeVerdict(
            judge=self.name,
            correct=1.0 if prob >= self.threshold else 0.0,
            prob_yes=prob,
            cost_usd=parsed["cost_usd"],
            input_tokens=parsed["input_tokens"],
            output_tokens=parsed["output_tokens"],
            latency_ms=(time.time() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# Agreement and threshold calibration
# ---------------------------------------------------------------------------


def cohen_kappa(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Cohen's kappa for two binary label sequences (None when undefined)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else None
    return (po - pe) / (1 - pe)


def judge_agreement(llm: Sequence[Optional[float]], jev: Sequence[Optional[float]]) -> Dict:
    """Agreement between the two judges over questions both of them scored."""
    pairs = [(x, y) for x, y in zip(llm, jev) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    return {
        "n": n,
        "agreement": sum(1 for x, y in pairs if x == y) / n,
        "kappa": cohen_kappa(a, b),
        "both_correct": sum(1 for x, y in pairs if x == 1 and y == 1),
        "both_wrong": sum(1 for x, y in pairs if x == 0 and y == 0),
        "llm_only": sum(1 for x, y in pairs if x == 1 and y == 0),
        "jev_only": sum(1 for x, y in pairs if x == 0 and y == 1),
    }


def calibrate_threshold(probs: Sequence[float], labels: Sequence[float]) -> Dict:
    """Pick the P(yes) threshold that best separates labelled verdicts.

    Candidates are the midpoints between consecutive distinct probabilities;
    the threshold maximising balanced accuracy wins (ties: closest to 0.5).
    """
    pairs = [(float(p), 1.0 if l else 0.0) for p, l in zip(probs, labels) if p is not None and l is not None]
    if not pairs:
        raise ValueError("no labelled verdicts to calibrate on")
    values = sorted({p for p, _ in pairs})
    candidates = [0.5] + [(x + y) / 2 for x, y in zip(values, values[1:])]
    candidates += [values[0] - 1e-6, values[-1] + 1e-6]
    n_pos = sum(1 for _, l in pairs if l == 1.0)
    n_neg = len(pairs) - n_pos

    def evaluate(t: float) -> Dict:
        preds = [1.0 if p >= t else 0.0 for p, _ in pairs]
        tp = sum(1 for pr, (_, l) in zip(preds, pairs) if pr == 1 and l == 1)
        tn = sum(1 for pr, (_, l) in zip(preds, pairs) if pr == 0 and l == 0)
        tpr = tp / n_pos if n_pos else 1.0
        tnr = tn / n_neg if n_neg else 1.0
        return {
            "threshold": t,
            "balanced_accuracy": (tpr + tnr) / 2,
            "accuracy": (tp + tn) / len(pairs),
            "kappa": cohen_kappa(preds, [l for _, l in pairs]),
        }

    scored = [evaluate(t) for t in candidates]
    best = max(scored, key=lambda r: (r["balanced_accuracy"], -abs(r["threshold"] - 0.5)))
    best = dict(best, threshold=round(min(max(best["threshold"], 0.0), 1.0), 4))
    best["at_0_5"] = evaluate(0.5)
    best["n"] = len(pairs)
    return best
