# RAGAS Judge LLM Benchmark

## Context

This doc captures a head-to-head comparison of six judge-LLM configurations used by our RAGAS evaluation pipeline. The goal was to pick a sensible default for `EvaluationConfig.ragas_evaluator_model` and to understand the cost/quality/latency tradeoffs of:

- OpenAI reasoning-effort modes (`minimal` vs `low`)
- OpenAI JSON mode (`response_format={"type":"json_object"}`) on vs off
- Open-weight alternative (Gemma-3 27B via OpenRouter → Novita, bf16)
- RAGAS 0.1.x `LangchainLLMWrapper` path vs RAGAS 0.4 `llm_factory` + Instructor path

All six conditions were evaluated on the **same 20 questions** (seed=42) replayed from a stored benchmark run — each judge saw identical `generated_answer` + `retrieved_contexts`, so quality differences are attributable to the judge, not retrieval/generation.

Prior default: `mistralai/mistral-small-4-119b-2603` (NVIDIA API).
New default: **`gpt-5-nano`**.

---

## Conditions

| # | Name | Provider | reasoning_effort | response_format | RAGAS path |
|---|---|---|---|---|---|
| 1 | `gpt5_nano_minimal_raw`  | OpenAI     | minimal | — (free text) | 0.1.x LangchainLLMWrapper |
| 2 | `gpt5_nano_minimal_json` | OpenAI     | minimal | json_object   | 0.1.x LangchainLLMWrapper |
| 3 | `gpt5_nano_low_raw`      | OpenAI     | low     | — (free text) | 0.1.x LangchainLLMWrapper |
| 4 | `gpt5_nano_low_json`     | OpenAI     | low     | json_object   | 0.1.x LangchainLLMWrapper |
| 5 | `gemma3_novita_bf16`     | OpenRouter → Novita bf16 | n/a | — (Novita rejects) | 0.1.x LangchainLLMWrapper |
| 6 | `gpt5_nano_minimal_v04`  | OpenAI     | minimal | json_object (via Instructor) | 0.4 `llm_factory` |

Source run replayed: `results/vanilla/ChunkSize_Sweep_co0_reranker__cs256_901173af_20260405T085525Z.json` (99 eligible questions, 20 sampled with seed=42).

Metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `answer_correctness`.

---

## Headline results (20 samples, 4 metrics)

| condition | calls | wall_s | p50 ms | p95 ms | prompt tok | compl tok | reason tok | **cost $** | api err | NaN | faithful | ans_rel | ctx_prec | ans_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt5_nano_minimal_raw   | 520 | 1 365 | 1 592 | 4 035  | 607 620 | 70 087  | 0       | **0.058** | 0 | 0 | 0.918 | 0.840 | 0.516 | 0.372 |
| gpt5_nano_minimal_json  | 519 | 1 604 | 1 674 | 4 741  | 606 087 | 70 314  | 0       | **0.058** | 0 | 1 | 0.823 | 0.791 | 0.551 | 0.401 |
| gpt5_nano_low_raw       | 520 | 2 751 | 3 320 | 11 637 | 607 166 | 199 608 | 137 536 | **0.110** | 0 | 0 | 0.874 | 0.808 | 0.573 | 0.326 |
| gpt5_nano_low_json      | 520 | 2 267 | 2 822 | 8 812  | 607 200 | 201 492 | 139 456 | **0.111** | 0 | 0 | 0.820 | 0.802 | 0.592 | 0.302 |
| gemma3_novita_bf16      | 441 | 3 444 | 3 075 | 17 660 | 560 236 | 71 904  | 0       | **0.081** | 0 | 6 | 0.906 | 0.779 | 0.667 | 0.403 |
| gpt5_nano_minimal_v04   | 521 | 1 309 | 1 407 | 4 214  | 694 628 | 68 007  | 4 096   | **0.062** | 0 | 0 | 0.865 | 0.749 | 0.557 | 0.406 |

Raw JSON: `results/judge_benchmarks/judge_bench_901173af_20260423T004831Z.json` (conditions 1-5) and `.../judge_bench_901173af_20260423T094842Z.json` (condition 6).

---

## Comparison axes

### minimal vs low reasoning

`low` roughly **doubles cost** ($0.058 → $0.110) and **doubles p50 latency** (~1.6 s → ~3 s). The reason is ~138k reasoning tokens per condition that `minimal` doesn't emit. Despite the extra thinking, `low` does not improve — and in some metrics slightly hurts — the RAGAS scores (faithfulness drops 0.918 → 0.874).

**Verdict**: for judge use-cases, `minimal` beats `low`. Use `low` only if a future metric is demonstrably gated by multi-step reasoning.

### JSON mode on vs off (0.1.x path)

Flipping `response_format={"type":"json_object"}` costs +5-15% wall time and can introduce surprising score drift: faithfulness falls ~0.1 on both minimal (0.918 → 0.823) and low (0.874 → 0.820). Hypothesis: the JSON-prompt scaffolding trades off some of the model's free-form reasoning quality.

**Verdict**: if you're running the 0.1.x path, `raw` (no `response_format`) is the better default on GPT-5-nano. The 0.4 path forces JSON anyway, so this knob disappears.

### GPT-5-nano (OpenAI) vs Gemma-3 27B (OpenRouter / Novita bf16)

Gemma is cheaper per token but **2.5× slower wall time** on 20 samples because Novita throttles — p95 per-call latency is 17.7 s (vs 4–11 s for GPT-5-nano). It posts the **highest `context_precision` (0.667)** and competitive `answer_correctness` (0.403), but 6 NaN cells because Novita rejects JSON mode outright (see gotchas below).

**Verdict**: Gemma is a valid cost-optimized fallback for asynchronous/offline eval, but the throttling makes it painful for tight iteration. Not the default.

### RAGAS 0.1.x `LangchainLLMWrapper` vs 0.4 `llm_factory` + Instructor

The 0.4 native path:

- **~5% faster wall time, ~12% lower p50** on the same 20 samples vs `minimal_raw`.
- **0 NaN vs 1 NaN** on `minimal_json` — Instructor's Pydantic validation eliminates parse failures.
- **~7% higher cost** because Instructor injects the Pydantic schema into the prompt (prompt tokens 694k vs 607k, ~14% inflation).
- **Forces json_object mode** on OpenAI; you can't turn it off without bypassing Instructor.
- Reports `reasoning_tokens=4096` at `minimal` where the 0.1.x tracker reported 0 — our v0.4 tracker reads the raw OpenAI `usage.completion_tokens_details.reasoning_tokens`, which OpenAI populates even at minimal effort.

**Verdict**: the v0.4 path is a better default if you value reliability (zero-NaN) and future-proofing (temperature/n/max_tokens handled natively). A formal migration is deferred because RAGAS 0.4's API diverges from 0.1.x — `LangchainLLMWrapper` is deprecated, metric classes moved, etc. We already have `ragas==0.4.3` installed in the venv; pyproject still pins `^0.1.1`.

---

## Gotchas encountered

### GPT-5 / o-series reject non-default temperature

**Symptom**: every RAGAS call silently returns `NaN`. Root cause (only visible if you force-print from an `on_llm_error` callback): OpenAI returns `400 "temperature does not support 0.01 with this model. Only the default (1) value is supported."`

RAGAS injects per-call `temperature=0.01` (deterministic prompts) or `0.3` (diverse generation) at the `agenerate_prompt` layer. These reach `_get_invocation_params` and from there the final OpenAI request. A single-layer override of `_generate` / `_agenerate` **does not** work because the override happens at a higher layer.

**Fix** (applied in `src/evaluation/ragas_evaluator.py` and `scripts/benchmark_judge_costs.py`):

```python
class _ReasoningChat(ChatOpenAI):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        kwargs.pop("temperature", None)
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("temperature", None)
        return payload
```

`_get_request_payload` is the final-assembly hook — hit once per request, catches both inbound kwargs and defaults from `self._default_params`. Initialize with `temperature=1` (the only accepted value).

RAGAS v0.4's `llm_factory()` handles this natively via `bypass_temperature=True`.

### Novita/Gemma rejects `response_format` despite OpenRouter advertising it

OpenRouter's endpoints API for `google/gemma-3-27b-it` lists `response_format: ✅` for Novita, yet live requests return:

```
400 {"code":400,"reason":"INVALID_REQUEST_BODY",
     "message":"model: google/gemma-3-27b-it does not support feature: structured-outputs"}
```

Workaround: run the Gemma condition **without** `response_format`; RAGAS 0.1.x's prompt-based parser degrades gracefully (some NaNs, but most metrics still score).

Per-provider support matrix (OpenRouter endpoints API, Apr 2026):

| Provider | Quant | Context | Prompt $/1M | Completion $/1M | `response_format` | `structured_outputs` | `tools` |
|---|---|---|---|---|---|---|---|
| DeepInfra | fp8     | 131 072 | 0.080 | 0.160 | advertised | ❌ | ❌ |
| Parasail  | fp8     | 131 072 | 0.080 | 0.450 | advertised | advertised | ❌ |
| Nebius    | fp8     | 110 000 | 0.100 | 0.300 | advertised | advertised | ❌ |
| Phala     | ?       | 53 920  | 0.110 | 0.400 | advertised | advertised | ❌ |
| **Novita**| **bf16**| **98 304** | **0.119** | **0.200** | advertised (rejected at request time) | ❌ | ❌ |

If reliable structured output with Gemma is required, route through Parasail or Nebius instead of Novita; bf16 specifically is Novita-only.

### RAGAS metrics request `n=3`; GPT-5 rejects `n>1`

Not yet hit in our runs — RAGAS 0.1.x's default path issues `n=1` serial calls for the metrics we ran. But if a future metric calls `llm.generate(..., n=3)` directly against a reasoning model, OpenAI will 400. The same `_get_request_payload` trick (pop `n` when `n>1`) applies. RAGAS v0.4 ships `bypass_n=True`.

### `max_tokens=4096` recommended for GPT-5 + structured output

Default `max_tokens=1024` can truncate JSON payloads, especially when the model emits reasoning tokens. RAGAS v0.4 release notes call this out explicitly. Applied in:

- `scripts/benchmark_judge_costs.py` — `_build_chat_llm` and `_build_v04_judge`
- `src/evaluation/ragas_evaluator.py` — `_build_evaluator_llm`

### Instructor uses `json_object`, not `json_schema`

Even on OpenAI (which supports strict `json_schema` mode), Instructor pins `Mode.JSON` because function-calling mode has a documented bug with Pydantic `Dict` annotations — returns empty `{}` instead of structured data. So our "structured vs unstructured" comparison is specifically about `response_format={"type":"json_object"}` + prompted schema, not strict schema enforcement. If you switch to strict `json_schema` later, expect different numbers.

---

## Decisions landed in the repo

1. `EvaluationConfig.ragas_evaluator_model` default flipped from `mistralai/mistral-small-4-119b-2603` to **`gpt-5-nano`** (`src/benchmarks/config.py`).
2. `_build_evaluator_llm()` now routes by model prefix (`src/evaluation/ragas_evaluator.py`):
   - `gpt-*` → OpenAI (`https://api.openai.com/v1`, `OPENAI_API_KEY`)
   - model in `NVIDIA_MODELS` allowlist → NVIDIA (preserves reproducibility of older RAGAS runs)
   - `vendor/model` → OpenRouter (`OPENROUTER_API_KEY` or `GEMMA_API_KEY` fallback)
3. New settings fields added in `config/settings.py`: `OPENROUTER_API_KEY`, `GEMMA_API_KEY`.
4. `_ReasoningChat` subclass for temperature-stripping + `max_tokens=4096` applied in both `src/evaluation/ragas_evaluator.py` and the benchmark script.
5. Test `tests/unit/benchmarks/test_config.py::TestEvaluationConfigRagas::test_ragas_defaults` updated to assert the new default.

---

## Reproducing / extending

```bash
# Full 6-condition comparison on the newest result file under results/{vanilla,hybrid}/
.venv/bin/python scripts/benchmark_judge_costs.py --sample 20

# Only GPT-5 conditions (skip slow Gemma)
.venv/bin/python scripts/benchmark_judge_costs.py --sample 20 --only gpt5

# Pin a specific source run
.venv/bin/python scripts/benchmark_judge_costs.py --run results/vanilla/SOMEFILE.json --sample 20

# Just the v0.4 native condition
.venv/bin/python scripts/benchmark_judge_costs.py --sample 20 --only gpt5_nano_minimal_v04
```

Artifacts:

- **Script**: `scripts/benchmark_judge_costs.py`
- **Results**: `results/judge_benchmarks/judge_bench_<run-hash8>_<iso-ts>.json`
- **Pricing table**: hardcoded in the script under `PRICING_USD_PER_1M`; update when OpenAI/OpenRouter prices change.

A single new condition is a new `ConditionSpec(...)` entry in `_conditions()`. For the v0.4 path, set `v04_native=True`.

---

## Open follow-ups

- Proper RAGAS 0.4 migration: unpin `ragas==0.1.1` in `pyproject.toml`, drop `LangchainLLMWrapper`, use `llm_factory()` everywhere. That makes `_ReasoningChat` and the temperature workaround unnecessary. Scope: replace 0.1.x metric classes (lowercase) with 0.4 ones (uppercase), adjust `_load_ragas_metrics` registry, retest.
- Extend the comparison with `gpt-5-mini` / `claude-haiku-4-5` / `gemini-2.5-flash` once we want a second stronger-judge class for spot-checking.
- If we need strict schema enforcement, wire `response_format={"type":"json_schema", ...}` per metric (requires forking RAGAS or upgrading further and using its schema-based path).
