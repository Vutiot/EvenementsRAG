# EvenementsRAG — Complete Feature Description

> Blueprint reference capturing every capability of the codebase, suitable for recreating the project with new prototype features.

---

## 1. Configuration Management

- **Centralized Pydantic Settings** (`config/settings.py`): All settings loaded from `.env` with type validation, defaults, and computed properties
- **Period Configuration** (`config/periods/world_war_2.yaml`): YAML-based domain config with seed articles (75+), Wikipedia categories (6 major + 30 subcategories), entity types (30+), geographic regions (5 theaters), crawl parameters (depth, rate limits, parallelism), and validation rules
- **Multi-provider LLM support**: OpenRouter (default, free Mistral models), Anthropic (Claude), OpenAI (GPT) — switchable via `LLM_PROVIDER` env var
- **All parameters environment-injectable**: API keys, Qdrant connection, embedding model, chunk sizes, retrieval weights, evaluation thresholds

---

## 2. Data Ingestion Pipeline

- **Wikipedia Fetcher** (`src/data_ingestion/wikipedia_fetcher.py`):
  - Fetch individual articles or batch from priority list with rate limiting (1s delay, 30 req/min)
  - Exponential backoff retry (3 attempts)
  - Skip-if-exists for resumable downloads
  - Validation: min/max article length, disambiguation page exclusion
  - Metadata tracking in `data/raw/metadata/{period}_fetched.json`
  - Returns: title, url, content, summary, categories, links, references, images, pageid, word_count, language, fetched_at
  - Scales tested: 50 → 976 → 10,000 articles

---

## 3. Text Preprocessing

- **Token-aware Chunker** (`src/preprocessing/text_chunker.py`):
  - Token-based sizing via `tiktoken` (cl100k_base tokenizer), default 512 tokens/chunk
  - Configurable overlap (default 50 tokens)
  - Structure-preserving: respects paragraph and sentence boundaries
  - Max chunks per document cap (100)
  - Fallback to word-based tokenization if tiktoken unavailable
  - Each chunk enriched with: chunk_index, total_chunks, article_title, source_url, categories (max 10), pageid, token_count, char_count
  - Statistics: avg/min/max tokens and chars per chunk

---

## 4. Embedding Generation

- **Embedding Generator** (`src/embeddings/embedding_generator.py`):
  - Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - Batch processing (batch_size=32) with auto GPU/CPU detection
  - MD5 hash-based `.npy` file caching in `.cache/embeddings/`
  - Single text and batch embedding methods
  - `embed_chunks()` enriches chunk dicts with embedding, embedding_model, embedding_dimension

---

## 5. Vector Store Management

- **Qdrant Manager** (`src/vector_store/qdrant_manager.py`):
  - Collection CRUD: create (with configurable vector_size, distance=COSINE), exists check, delete, info
  - Batch upsert with configurable batch_size (default 100), auto-generated UUIDs
  - Similarity search with optional score_threshold and filter_conditions
  - Filter builder: supports range queries (gte/lte/gt/lt), exact match, list match
  - Vector count and deletion (by IDs or filter)
  - Supports: localhost Docker, Qdrant Cloud (API key), in-memory mode

- **Document Indexer** (`src/vector_store/indexer.py`):
  - End-to-end pipeline: load articles → chunk → embed → upsert
  - Deterministic UUID generation from chunk_id hash
  - Payload schema: content, chunk_id, chunk_index, total_chunks, article_title, title, source_url, categories, pageid, token_count, char_count, embedding_model

---

## 6. RAG Architecture (4-Phase Progressive Design)

### Abstract Base (`src/rag/base_rag.py`)
- Dataclasses: `RetrievedChunk` (chunk_id, content, score, metadata), `RAGResponse` (query, answer, retrieved_chunks, retrieval_time_ms, generation_time_ms)
- Abstract interface: `retrieve(query, top_k, filters)`, `generate(query, context_chunks)`
- Concrete: `query()` (retrieve + generate with timing), `format_context()` (numbered source headers), `get_statistics()`

### Phase 1 — Vanilla RAG (`src/rag/phase1_vanilla/retriever.py`) — **IMPLEMENTED**
- Pure semantic search via Qdrant cosine similarity
- LLM generation via OpenAI-compatible API (OpenRouter default)
- Historian-persona prompt template with context injection
- Temperature=0.0 for deterministic output, max_tokens=2000
- Fallback to context summary on LLM error

### Phase 2 — Temporal RAG (`src/rag/phase2_temporal/`) — **STUB**

### Phase 3 — Hybrid RAG (`src/rag/phase3_hybrid/`) — **STUB** (logic in `src/retrieval/`)

### Phase 4 — Graph RAG (`src/rag/phase4_graph/`) — **STUB**

---

## 7. Retrieval Strategies

### Hybrid Search (`src/retrieval/hybrid_search.py`)
- **BM25 implementation**: k1=1.5, b=0.75, tokenization, IDF computation, document length normalization
- **HybridSearcher**: Runs semantic (Qdrant) + keyword (BM25) searches in parallel, each at 2×top_k
- **Reciprocal Rank Fusion (RRF)**: `score = sum(weight_i / (k + rank_i))` with k=60
- Configurable weights: semantic=0.7, BM25=0.3
- BM25 index built from Qdrant collection via scrolling

### Temporal Filter (`src/retrieval/temporal_filter.py`)
- Regex extraction of: explicit years ("in 1944"), year ranges ("between 1941 and 1945"), early/late war
- Key event mapping: Pearl Harbor→1941, Stalingrad→1942, D-Day/Normandy→1944, Berlin/Hiroshima→1945
- Creates Qdrant year-range filters from extracted temporal info
- Temporal boost: multiplies scores by configurable factor (default 1.5) for temporally relevant chunks
- Returns cleaned query (temporal terms removed) + filter

---

## 8. Evaluation Framework

### 8.1 Data Structures

**`RetrievalMetrics` dataclass** (`src/evaluation/metrics.py:28-62`):
- `recall_at_1`, `recall_at_3`, `recall_at_5`, `recall_at_10` (float, default 0.0)
- `article_hit_at_1`, `article_hit_at_3`, `article_hit_at_5`, `article_hit_at_10` (float, binary 0/1)
- `chunk_hit_at_1`, `chunk_hit_at_3`, `chunk_hit_at_5`, `chunk_hit_at_10` (float, binary 0/1)
- `mrr` (float, default 0.0)
- `ndcg_at_5`, `ndcg_at_10` (float, default 0.0)
- Methods: `to_dict()`, `__repr__()` showing recall@5, mrr, ndcg@5

**`EvaluationResults` dataclass** (`src/evaluation/metrics.py:65-104`):
- `avg_recall_at_k`: Dict[int, float] — averaged recall at each K
- `avg_mrr`: float
- `avg_ndcg`: Dict[int, float] — averaged NDCG at K=5 and K=10
- `avg_article_hit_at_k`: Dict[int, float] — binary hit rates averaged across questions
- `avg_chunk_hit_at_k`: Dict[int, float] — binary hit rates averaged across questions
- `metrics_by_type`: Dict[str, RetrievalMetrics] — per question-type averages
- `per_question_metrics`: List[Dict] — full per-question evaluation details
- `total_questions`: int
- `questions_with_recall_at_5_gt_50`: int — count meeting quality threshold
- `avg_retrieval_time_ms`: float

### 8.2 Retrieval Metrics (`src/evaluation/metrics.py`)

**`recall_at_k(retrieved_chunks, ground_truth_chunks, k)` → float**:
- Formula: `|top_k ∩ ground_truth| / |ground_truth|`
- Returns 0.0 if ground truth is empty (with warning)
- Operates on chunk ID strings

**`mrr(retrieved_chunks, ground_truth_chunks)` → float**:
- Returns `1/rank` of the first relevant result found
- Returns 0.0 if no relevant result found
- MRR=1.0 means first result is relevant; MRR=0.5 means second result

**`ndcg_at_k(retrieved_chunks, ground_truth_chunks, k)` → float**:
- Binary relevance: 1 if chunk in ground truth, 0 otherwise
- DCG = `Σ(rel_i / log2(i+2))` for i=0..k-1
- NDCG = DCG / IDCG where IDCG uses ideal ordering (all relevant at top)
- Rewards relevant results appearing earlier in ranking

**`precision_at_k(retrieved_chunks, ground_truth_chunks, k)` → float**:
- Formula: `|{c ∈ top_k : c ∈ ground_truth}| / k`
- Fraction of top-K results that are relevant

**`article_hit_at_k(retrieved_chunks, retrieved_payloads, source_article_id, k)` → float**:
- Binary: 1.0 if ANY chunk from the source article appears in top-K, else 0.0
- Matching: tries `pageid` first (string comparison), then `article_title` exact match
- Measures topic-level retrieval quality

**`chunk_hit_at_k(retrieved_chunks, source_chunk_id, k)` → float**:
- Binary: 1.0 if THE EXACT source chunk UUID appears in top-K, else 0.0
- Measures passage-level precision — can we find the specific passage with the answer?
- Hardest metric: only 1 correct target out of entire collection

**`compute_retrieval_metrics(...)` → RetrievalMetrics**:
- Orchestrates all metric computations for a single query
- Accepts: retrieved_chunks, ground_truth_chunks, k_values, retrieved_payloads, source_article_id, source_chunk_id
- Default k_values from `settings.EVALUATION_K_VALUES` = [1, 3, 5, 10]

**`aggregate_metrics(all_metrics: List[RetrievalMetrics])` → Dict[str, float]**:
- Simple averaging across all questions for each metric field

**`compute_metrics_by_type(per_question_results)` → Dict[str, RetrievalMetrics]**:
- Groups results by `question["type"]` field
- Averages each metric within each type group
- Returns map: type_name → averaged RetrievalMetrics

### 8.3 Two-Level Evaluation Strategy

The system supports two evaluation granularities documented in `docs/chunk_vs_article_evaluation.md`:

**Article-Level Evaluation** (coarse):
- Ground truth = ALL chunks from the source article (e.g., 20–50 chunks)
- Easier to achieve high recall (multiple targets)
- Measures: "Can we find content related to this topic?"
- Risk: false sense of success — high recall doesn't mean we found the answer passage

**Chunk-Level Evaluation** (fine, preferred):
- Ground truth = THE EXACT chunk the question was generated from (1 target)
- Optional neighbor chunks (±N window) for context
- Much harder — must find 1 specific chunk out of entire collection
- Measures: "Can we find the exact passage containing the answer?"
- More representative of real RAG performance since LLMs need precise passages

**Key insight**: When ground truth is a single chunk, Recall@K = Chunk Hit@K (they're equivalent)

**Observed results comparison**:
- Article-based Recall@10: 100% (misleading — just found *any* chunk from article)
- Chunk-based Recall@5: 76.5% (realistic — found the actual answer passage)

### 8.4 Question Generator (`src/evaluation/question_generator.py`)

**Class: QuestionGenerator**

**Initialization**:
- `api_key`: OpenRouter API key (from settings)
- `model`: defaults to `settings.QUESTION_GEN_MODEL` (mistralai/mistral-small-3.1-24b-instruct:free)
- `base_url`: OpenRouter base URL
- `qdrant_manager`: optional QdrantManager instance
- `skip_api_init`: for testing without API
- Uses OpenAI SDK client pointed at OpenRouter endpoint

**6-Type Question Taxonomy** with target distribution:
```
factual:        25% — "When?", "What?", "Who?" (simple/complex fact retrieval)
temporal:       20% — "Before/after?", chronological ordering, duration
comparative:    15% — "How different?", similarities, superlatives
entity_centric: 15% — "What did X do?", role of Y, person/location/org-focused
relationship:   15% — "How influenced?", cause-effect, network connections
analytical:     10% — "Summarize", "Why significant?", synthesis
```

**`load_chunks_from_qdrant(collection_name, max_chunks=None)` → List[Dict]**:
- Scrolls entire Qdrant collection with pagination (limit=100 per batch)
- Extracts: chunk_id (point UUID), content, article_title, article_id (pageid), chunk_index, full metadata payload

**`sample_chunks(chunks, num_samples, strategy='stratified', min_length=200)` → List[Dict]**:
- Filters by minimum character length (200)
- **random**: uniform random sampling
- **stratified**: round-robin across articles — shuffles article list, takes 1 chunk per article per round until target reached — ensures diverse article coverage
- **diverse**: from each article, picks shortest + median + longest chunk — captures content variety by length

**`generate_question_for_chunk(chunk, num_questions, target_type)` → List[Dict]**:
- Builds prompt from `QUESTION_GENERATION_PROMPT` template with: article_title, chunk_text (truncated to 2000 chars), target question type, num_questions
- System prompt: "You are an expert question generator for historical content evaluation"
- LLM call: temperature from `settings.QUESTION_GEN_TEMPERATURE`, max_tokens from `settings.QUESTION_GEN_MAX_TOKENS`
- Parses JSON array response (handles markdown code block wrapping)
- Each question enriched with: source_chunk_id, source_article, source_article_id, generated_at (ISO timestamp), model name
- Returns empty list on JSON parse failure or API error

**Generated question structure**:
```json
{
  "id": "gen_q001",
  "question": "...",
  "type": "factual|temporal|comparative|entity_centric|relationship|analytical",
  "difficulty": "easy|medium|hard",
  "expected_answer_hint": "brief hint about what the answer should contain",
  "source_chunk_id": "<qdrant point UUID>",
  "source_article": "<article title>",
  "source_article_id": "<pageid>",
  "generated_at": "ISO timestamp",
  "model": "model name"
}
```

**`generate_evaluation_questions(collection_name, num_chunks=30, questions_per_chunk=1, chunks=None, sampling_strategy='stratified', ensure_taxonomy_diversity=True)` → Dict**:
- If `ensure_taxonomy_diversity`: distributes question types according to taxonomy proportions, fills remainder randomly, shuffles order
- Iterates sampled chunks with tqdm progress bar, assigns target type per chunk
- Counts actual type distribution via Counter
- Returns metadata (generated_at, model, total_questions, chunks_sampled, questions_per_chunk, sampling_strategy, taxonomy_distribution, unique_articles) + questions list

**`save_questions(questions_data, output_path)`** / **`load_questions(questions_path)`**:
- JSON serialization with indent=2, ensure_ascii=False
- Creates parent directories as needed

### 8.5 Benchmark Runner (`src/evaluation/benchmark_runner.py`)

**Class: BenchmarkRunner**

**Initialization**:
- `questions_file`: path to JSON (default: `data/evaluation/generated_questions.json`)
- `qdrant_manager`: QdrantManager instance
- `embedding_generator`: EmbeddingGenerator instance (for query embedding)
- `k_values`: list of K values (default: `settings.EVALUATION_K_VALUES` = [1, 3, 5, 10])

**`load_questions()` → Dict**:
- Loads from JSON file, validates existence
- Logs total count and taxonomy distribution

**`query_for_question(question, collection_name, top_k=10)` → Dict**:
- Generates query embedding from question text via `EmbeddingGenerator.generate_embedding()`
- Searches Qdrant with: query_vector, limit=top_k, score_threshold=None (no filtering), NO filter_conditions (searches ALL chunks in collection)
- Measures retrieval_time_ms via `time.time()` delta
- Extracts: point UUID as chunk_id, full payload
- Returns: question_id, question_text, question_type, retrieved_chunks (UUIDs), retrieved_payloads, retrieval_time_ms, num_retrieved

**`compute_ground_truth_chunks(question, include_neighbors=True, neighbor_window=2)` → List[str]**:
- Priority 1: Pre-computed `ground_truth_chunks` field in question JSON
- Priority 2: `source_chunk_id` field — returns [source_chunk_id] (optionally with neighbors ±N, but neighbor computation is a stub/log-only currently)
- Priority 3 (fallback): Old article-based structure — returns empty list and matches by article ID at evaluation time

**`evaluate_question(question, query_result)` → Dict**:
- Gets ground truth chunks
- If no ground truth: dynamically matches retrieved chunks against source_article_id (by pageid prefix or payload pageid)
- Calls `compute_retrieval_metrics()` with all available parameters
- Returns: question_id, question text, type, difficulty, source_article, ground_truth_count, retrieved_count, retrieval_time_ms, RetrievalMetrics object

**`run_benchmark(collection_name, phase_name='default', max_questions=None)` → EvaluationResults**:
- Full pipeline: load questions → verify collection exists → evaluate each question with tqdm progress
- Queries at `top_k = max(k_values)` (default 10) to compute all K levels from one search
- Aggregates: avg_recall_at_k, avg_mrr, avg_ndcg (K=5,10), avg_article_hit_at_k, avg_chunk_hit_at_k
- Computes `metrics_by_type` via `compute_metrics_by_type()`
- Counts questions meeting quality threshold: `recall_at_5 >= settings.EVALUATION_MIN_RECALL_AT_5` (default 0.5)
- Returns complete `EvaluationResults`

**`export_results(results, output_path, format='json'|'csv')`**:
- JSON: full EvaluationResults serialized with `to_dict()`
- CSV: per-question rows with columns: question_id, type, difficulty, recall@1/3/5/10, mrr, ndcg@5, retrieval_time_ms

**`print_summary(results)`**:
- Formatted console output with sections:
  1. Total questions and avg retrieval time
  2. Traditional Recall@K metrics + MRR + NDCG
  3. Article-Level Hit Rate (with percentage and count, e.g., "91.4% (32/35)")
  4. Chunk-Level Hit Rate (same format)
  5. Quality threshold count
  6. Metrics by Question Type breakdown (recall@5, mrr, ndcg@5 per type)

### 8.6 Question Types Taxonomy (`docs/question_types_taxonomy.md`)

175+ example questions across 6 categories, 13 subcategories, 3 difficulty levels:

**1. Factual** (14% of recommended test set):
- 1.1 Simple Fact Retrieval: single-hop, low ambiguity — best Phase 1
- 1.2 Complex Fact Retrieval: multi-faceted, aggregation — best Phase 3

**2. Temporal** (23% of test set):
- 2.1 Chronological Ordering: "before/after", sequence — best Phase 2
- 2.2 Duration and Timing: "how long", date arithmetic — best Phase 2
- 2.3 Causal Chains: cause-effect with temporal dependency — best Phase 4

**3. Comparative** (11% of test set):
- 3.1 Event Comparison: parallel fact extraction — best Phase 3
- 3.2 Superlative: ranking, "most/largest/deadliest" — best Phase 3

**4. Entity-Centric** (17% of test set):
- 4.1 Person-Focused: roles, actions of individuals — best Phase 3
- 4.2 Location-Focused: events at geographic locations — best Phase 3
- 4.3 Organization-Focused: military units, alliances — best Phase 3

**5. Relationship** (20% of test set):
- 5.1 Influence and Impact: "how did X affect Y" — best Phase 4
- 5.2 Network and Alliance: collaborations, partnerships — best Phase 4
- 5.3 Multi-Hop Reasoning: 3+ inference steps, path finding — best Phase 4

**6. Analytical** (14% of test set):
- 6.1 Synthesis: summarization, key patterns — best Phase 4
- 6.2 Interpretive: "why significant", explanation — best Phase 4
- 6.3 Counterfactual: "what if" hypotheticals — best Phase 4 (stretch goal)

**Difficulty Levels**: Simple (67q, single-hop), Medium (70q, 2–3 hop), Hard (38q, 3+ hop)

**Ground Truth Annotation Format per Question**:
```json
{
  "id": "q042",
  "question": "...",
  "category": "temporal",
  "subcategory": "causal_chain",
  "complexity": "hard",
  "expected_best_phase": "phase4_graph",
  "ground_truth": {
    "answer": "...",
    "relevant_documents": ["..."],
    "relevant_events": ["..."],
    "entities": { "people": [], "locations": [], "organizations": [] },
    "dates": ["YYYY-MM-DD"],
    "reasoning_hops": 3,
    "evaluation_notes": "..."
  }
}
```

### 8.7 Expected Performance by Phase (from taxonomy doc)

| Question Type | Phase 1 (Vanilla) | Phase 2 (Temporal) | Phase 3 (Hybrid) | Phase 4 (Graph) |
|---|---|---|---|---|
| Factual | 80–90% | 80–90% | 85–90% | 80%+ |
| Temporal | 40–50% | 75–85% | 75–85% | 80%+ |
| Entity-centric | 70–80% | — | 85–95% | 80%+ |
| Comparative | — | — | 80–90% | 80%+ |
| Relationship | 30–40% | — | 45–55% | 80–90% |
| Causal/Multi-hop | — | 50–60% | 45–55% | 75–90% |

### 8.8 Production Evaluation Plan (`docs/evaluation_plan.md`)

**System Constraints**: CPU-only (Intel i5 vPro), ≤32 GB RAM, local embedding inference, API-based LLM

**Embeddings to Test** (all 384-dim, CPU-friendly):
- `all-MiniLM-L6-v2` (current — fastest, lowest memory)
- `e5-small` (better semantic recall)
- `bge-small-en` (better query-doc alignment, reranker-friendly)

**Vector Stores to Test**:
- FAISS: Flat (accuracy ceiling), HNSW (production ANN), IVFFlat (memory tradeoff)
- PostgreSQL + pgvector: IVFFlat, HNSW
- Qdrant: HNSW with payload filtering (current)

**RAG Strategy Tiers**:
- Tier 0: Dense ANN only, BM25 only
- Tier 1: Hybrid (dense+sparse w/ RRF), Dense → cross-encoder rerank
- Tier 2: Hybrid → rerank, Parent-child retrieval, Filtered retrieval
- Tier X: GraphRAG (Neo4j expansion)

**Test Matrix**: 9 core configs (3 embeddings × 3 vector stores) + RAG strategy variations

**Latency Budgets**: Retrieval p50 ≤80ms, p95 ≤150ms, End-to-end (no LLM) ≤200ms, GraphRAG p95 ≤250ms
**Indexing Budgets**: Build time ≤2h, Peak RAM ≤24GB, Disk ≤10GB

**End-to-End QA Metrics**: Answer correctness, Faithfulness (grounded in docs), Abstention rate ("I don't know" — abstaining > hallucinating)

**Specialized Metrics by Type**:
- Temporal: date extraction accuracy, chronological ordering correctness, temporal span accuracy
- Comparative: coverage completeness (both entities), contrast clarity, factual accuracy per subject
- Relationship: path correctness, relationship accuracy (edge types), causal chain completeness

### 8.9 Evaluation Scripts & Automation

- **`scripts/generate_evaluation_questions.py`**: Standalone CLI to generate questions from indexed Qdrant collection
- **`scripts/run_phase1_baseline.py`** / `run_phase1_976articles.py` / `run_phase1_10k.py`: Phase 1 evaluation at different scales
- **`scripts/run_phase2_hybrid_temporal.py`** / `run_phase2_976articles.py`: Phase 2 evaluation
- **`scripts/run_evaluation.py`**: Master evaluation runner
- **`scripts/establish_phase1_baseline.py`**: Establish baseline metrics for regression tracking
- **`scripts/compare_dataset_sizes.py`**: Compare metrics across 49, 976, 10,000 article datasets
- **`scripts/auto_run_10k_pipeline.py`**: Monitors download directory, auto-triggers full pipeline (index → generate questions → Phase 1 eval → comparison) when target reached (with progress bar, ETA, download rate tracking)
- **`scripts/generate_analysis_report.py`**: Generates markdown reports from JSON results
- **`scripts/test_evaluation.py`** / `test_chunk_based_evaluation.py`: In-memory Qdrant evaluation tests

### 8.10 Actual Results Observed

**Phase 1 Vanilla (49 articles, 35 questions)**:
- Article Hit@5: 91.4%, Chunk Hit@5: 71.4%
- Recall@5: 0.714, MRR: 0.654, NDCG@5: 0.659
- Avg retrieval time: 9.3ms

**Phase 2 Hybrid+Temporal (49 articles, 35 questions)**:
- Same hit rates as Phase 1 (no improvement)
- MRR: 0.553 (−15.4% vs Phase 1)
- Temporal detection: 22.9% of questions

**Phase 1 at 10,000 articles**: 50 evaluation questions, results stored in `results/phase1_baseline_10000articles_50q.json`

**Scaling insight**: The `run_phase1_10k.py` script computes precision gap analysis (Article Hit − Chunk Hit) and classifies: <15% gap = "good precision", ≥15% = "reranking needed"

---

## 9. Scripts & Automation

- **Data pipeline scripts**: download (50/976/10k articles), index, generate questions — each as standalone CLI
- **Evaluation scripts**: per-phase baselines, cross-phase comparison, dataset size comparison
- **Automated pipeline** (`scripts/auto_run_10k_pipeline.py`): end-to-end from download to evaluation
- **Infrastructure** (`scripts/setup_qdrant.sh`): Docker Qdrant lifecycle (start/stop/restart/status/logs/clean) with health checks
- **Analysis reporting** (`scripts/generate_analysis_report.py`): markdown reports from JSON results

---

## 10. Results & Documentation

- **Evaluation results** in `/results/`: JSON per-phase metrics, CSV comparison, markdown reports (baseline, changelog, cross-phase summary, dataset size comparison)
- **Documentation** in `/docs/`: question types taxonomy (175+ examples), evaluation metrics explained, evaluation plan with budgets, chunk vs article evaluation analysis
- **Logging**: loguru-based with configurable level, file output to `logs/evenementsrag.log`

---

## 11. Technology Stack

- **Runtime**: Python 3.10+, Poetry dependency management
- **Vector DB**: Qdrant (Docker, cloud, or in-memory)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **LLM**: OpenRouter (free Mistral), Anthropic Claude, OpenAI GPT via OpenAI SDK
- **NLP**: spaCy (NER), dateparser/datefinder, rank-bm25, tiktoken, Wikipedia API, BeautifulSoup4
- **Evaluation**: RAGAS, BERTScore, ROUGE
- **Graph (planned)**: NetworkX (dev), Neo4j (prod), pyvis (visualization)
- **Dev tools**: pytest, black, ruff, isort, jupyter, loguru

---

## 12. Key Design Patterns

- **Progressive phases**: Each RAG phase builds on the previous, sharing base interface
- **Modular components**: Ingestion, chunking, embedding, indexing, retrieval, generation, evaluation are independently testable
- **Configuration-driven**: All behavior adjustable via environment variables without code changes
- **Caching**: Embeddings cached by content hash to avoid recomputation
- **Deterministic IDs**: Chunk IDs from pageid+chunk_index, vector IDs from UUID5 hash
- **Resumable pipelines**: Skip-if-exists on downloads, cache-aware embedding
- **Multi-scale testing**: Same evaluation framework across 50, 976, 10,000 article datasets
