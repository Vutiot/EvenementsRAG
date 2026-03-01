# Branch Comparison: `main` vs `pre-merge-backup` (E2-F1, E2-F2, E2-F4)

## Context

Both branches independently implement features E2-F1 (dataset switching + chunk sweeps), E2-F2 (vector DB abstraction + distance/embedding sweeps), and E2-F4 (generation parameters). This document compares them side-by-side and recommends the best implementation for each area.

---

## E2-F1: Dataset Switching + Chunk Sweeps

| Aspect | `main` | `pre-merge-backup` | Winner |
|--------|--------|---------------------|--------|
| **DatasetManager API** | Config bound at construction; requires both `qdrant_manager` and `embedding_generator` | Stateless; accepts config per-method; lazy-creates dependencies | **pre-merge-backup** (more flexible) |
| **Chunk sweep methods** | Hardcoded YAML file loading (`cls.from_yaml(...)`) | Parametric: `chunk_size_sweep(base, sizes=(256,512,1024))` with custom args | **pre-merge-backup** (programmable) |
| **Collection naming** | `ww2_cs256_co50` (no dataset prefix) | `wiki_10k_cs256_co50` (includes dataset name) | **pre-merge-backup** (avoids collisions across datasets) |
| **Octank support** | Registry entry only, no PDF converter | Full: `convert_octank_pdf_to_json.py` script included | **pre-merge-backup** |
| **DATASET_REGISTRY export** | Not exported | Exported for external use | **pre-merge-backup** |
| **Test coverage** | 152 lines config + 182 lines dataset manager (split) | 377 lines config (comprehensive sweep tests) | **pre-merge-backup** (more sweep coverage) |

**Verdict: `pre-merge-backup` wins E2-F1.** Parametric sweeps are fundamentally more useful than hardcoded YAML loading. Collection naming with dataset prefix prevents cross-dataset collisions.

---

## E2-F2: Vector DB Abstraction + Distance Metrics + Embedding Sweeps

| Aspect | `main` | `pre-merge-backup` | Winner |
|--------|--------|---------------------|--------|
| **Base design** | `Protocol` (loose, runtime-checkable) | `ABC` with `@abstractmethod` (strict contracts) | **pre-merge-backup** |
| **DistanceMetric** | Plain strings (`"cosine"`) | Proper `Enum` class (4 values) | **pre-merge-backup** |
| **Methods on ABC** | 7 methods (no scroll, no delete_vectors, no get_collection_info) | 8+ methods (scroll, delete_vectors, get_collection_info, get_statistics) | **pre-merge-backup** |
| **Factory pattern** | `if/elif` dispatch (hardcoded) | Registry + `importlib` lazy loading (extensible) | **pre-merge-backup** |
| **QdrantAdapter** | None; uses QdrantManager directly | Thin adapter wrapping QdrantManager to BaseVectorStore | **pre-merge-backup** |
| **FAISSStore** | In-memory only, no persistence, no filtering, no pagination | Full: persistence (.faiss + .meta.pkl), filtering, pagination, updates | **pre-merge-backup** |
| **PgVectorStore** | Basic (7 methods, no filtering, no batch optimization) | Full (8+ methods, JSONB filtering, `execute_values()` batch, pagination) | **pre-merge-backup** |
| **VectorDBConfig** | Backend-specific fields (`host`, `port`, `index_path`, `connection_string`) + validator | Generic `connection_params: dict` (extensible) | Tie (tradeoff) |
| **Distance metric sweep** | Not implemented | `distance_metric_sweep()` classmethod, 3 configs + YAML presets | **pre-merge-backup** |
| **Embedding model sweep** | Not implemented | `embedding_model_sweep()` classmethod, 4 models + YAML presets | **pre-merge-backup** |
| **Embedding cache fix** | Bug: `_hash_text(text)` shared across models | Fixed: `_hash_text(f"{model_name}::{text}")` | **pre-merge-backup** (critical fix) |
| **Runner wiring** | `EmbeddingGenerator()` without model_name | `EmbeddingGenerator(model_name=config.embedding.model_name)` | **pre-merge-backup** |
| **Test coverage** | ~4 test files, ~50-100 tests | 7 test files, 152 tests (5 pgvector skipped) | **pre-merge-backup** (3x more) |
| **Extra: metrics_collector** | Present (LatencyMetrics + MetricsCollector) | Not present | **main** (unique addition) |
| **Extra: ragas_evaluator** | Present (RAGASScore + RAGASEvaluator with caching) | Not present | **main** (unique addition) |
| **Extra: embedding_factory** | Present (convenience wrapper) | Not present | **main** (minor) |

**Verdict: `pre-merge-backup` wins E2-F2 decisively.** The ABC-based design, registry factory, production-grade store implementations, sweep methods, and the critical embedding cache fix make it far superior. However, `main` has `metrics_collector.py` and `ragas_evaluator.py` which should be cherry-picked.

---

## E2-F4: Generation Parameters

| Aspect | `main` | `pre-merge-backup` | Winner |
|--------|--------|---------------------|--------|
| **GenerationConfig flexibility** | `top_k_articles` required (default=3); `prompt_template` is enum of 3 values | Both optional; `prompt_template` is free-form string | **pre-merge-backup** |
| **Sweep methods** | Same 3 methods (temperature, top_k_chunks, model) | Same 3 methods (identical code) | Tie |
| **Temperature/model pass-through** | NOT passed to LLM (hardcoded `settings.CURRENT_LLM_MODEL`) | Passed via `**gen_kwargs` to `generate()` | **pre-merge-backup** (critical) |
| **Retrieval/generation separation** | Mixed: `pipeline.query()` does both | Clean: `retrieve()` then `generate()` separately | **pre-merge-backup** |
| **Model override in retriever** | `settings.CURRENT_LLM_MODEL` hardcoded | `kwargs.pop("model", None) or settings.CURRENT_LLM_MODEL` | **pre-merge-backup** |
| **Article filtering** | None | `_filter_top_k_articles()` helper | **pre-merge-backup** |
| **Result auto-save** | `result.save()` instance method + manifest | `_save_result()` module function + `output_dir` param on `run()` | Tie (design choice) |
| **YAML presets** | 1 base file (`generation_params.yaml`) | 9 sweep variants (temp x3, topk x3, model x3) | **pre-merge-backup** |
| **Settings additions** | None | `BENCHMARK_RESULTS_DIR` field | **pre-merge-backup** |

**Verdict: `pre-merge-backup` wins E2-F4.** The core purpose of E2-F4 is to parametrize generation settings for benchmarking. `main` defines sweep methods but **doesn't actually pass temperature/model to the LLM** -- making the sweeps non-functional. `pre-merge-backup` properly threads all generation params through to the LLM call.

---

## Overall Recommendation

**`pre-merge-backup` is the superior implementation across all three features.**

### Key reasons:
1. **Parametric sweep methods** (programmable) vs hardcoded YAML loading
2. **ABC + registry factory** vs Protocol + if/elif dispatch
3. **Production-grade store implementations** (FAISS persistence, PgVector filtering/pagination) vs minimal/toy
4. **Critical bug fix**: embedding cache isolation (`model_name::text` hash)
5. **Generation params actually work**: temperature, model, max_tokens passed to LLM
6. **3x test coverage**: 152+ tests vs ~50-100

### Cherry-pick from `main`:
- `src/benchmarks/metrics_collector.py` -- LatencyMetrics + MetricsCollector
- `src/benchmarks/ragas_evaluator.py` -- RAGASScore + RAGASEvaluator
- `src/embeddings/embedding_factory.py` -- convenience wrapper (minor)
- Associated tests: `test_metrics_collector.py`, `test_ragas_evaluator.py`, `test_embedding_factory.py`

These modules are orthogonal to the core features and add value without conflicting with `pre-merge-backup`'s architecture.

---

## Merge Plan

### Step 1: Cherry-pick unique files from `main`
Extract these files from the `main` branch and add them to `pre-merge-backup`:
- `src/benchmarks/metrics_collector.py` (147 lines)
- `src/benchmarks/ragas_evaluator.py` (283 lines)
- `src/embeddings/embedding_factory.py` (71 lines)
- `tests/unit/benchmarks/test_metrics_collector.py` (87 lines)
- `tests/unit/benchmarks/test_ragas_evaluator.py` (128 lines)
- `tests/unit/embeddings/__init__.py` (1 line)
- `tests/unit/embeddings/test_embedding_factory.py` (69 lines)

### Step 2: Verify cherry-picked files integrate cleanly
- Check imports resolve against `pre-merge-backup`'s module structure
- Run full test suite to confirm no conflicts

### Step 3: Update `main` to match merged result
- Fast-forward or reset `main` to the merged `pre-merge-backup` + cherry-picked files

### Step 4: Update ROADMAP.md
- Record architecture decisions from the merge
- Update task statuses as needed
