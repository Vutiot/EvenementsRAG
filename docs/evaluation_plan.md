# Recap: RAG Benchmark Project Plan

## Project Goal
Compare embedding storage techniques and RAG strategies on 10k Wikipedia articles, measuring **accuracy, speed, and memory** under realistic CPU-only constraints.

---

## System Constraints

| Dimension | Constraint |
|-----------|------------|
| Compute | CPU-only (Intel i5 vPro) |
| RAM | ≤ 32 GB |
| Dataset | 10k Wikipedia articles (JSON) |
| Embedding | Local inference only |
| LLM | Mistral Small (API) |
| Index build time | ≤ 2 hours |

---

## Embeddings to Test

All 384-dim, CPU-friendly models:

| Model | Why |
|-------|-----|
| `all-MiniLM-L6-v2` | Fastest, lowest memory baseline |
| `e5-small` | Better semantic recall |
| `bge-small-en` | Strong query-doc alignment, reranker-friendly |

---

## Vector Stores to Test

### 1. FAISS (local ANN gold standard)
- **Indexes:** Flat (accuracy ceiling), HNSW (production ANN), IVFFlat (memory tradeoff)
- **Purpose:** Speed/recall upper bound

### 2. PostgreSQL + pgvector
- **Indexes:** IVFFlat, HNSW (if v0.5+)
- **Purpose:** "Already have a DB" scenario, test filtering

### 3. Qdrant
- **Index:** HNSW with payload filtering
- **Purpose:** Dedicated vector DB representative

---

## RAG Strategies to Compare

| Tier | Strategy |
|------|----------|
| **Tier 0** | Dense ANN only, BM25 only |
| **Tier 1** | Hybrid (dense + sparse w/ RRF), Dense → cross-encoder rerank |
| **Tier 2** | Hybrid → rerank, Parent-child retrieval, Filtered retrieval |
| **Tier X** | GraphRAG (Neo4j expansion) |

---

## Evaluation Metrics

### Retrieval Metrics (Single Ground-Truth per Query)

Since each evaluation query has exactly **one relevant document** (the source it was generated from), your metrics simplify:

| Metric | Definition | Per Query | Aggregation | Notes |
|--------|------------|-----------|-------------|-------|
| **Hit@K** | Did retriever include exact doc in top K? | 1 if found, 0 otherwise | Average = Recall@K | Binary; insensitive to rank within K |
| **MRR** | 1 / rank of exact document | 1/rank if found, 0 if not | Average over queries | Rank-sensitive; rewards docs near top |
| **Rank Distribution** | Position of exact doc in results | Integer rank | Distribution/histogram | Helps diagnose reranker potential |

**Key insight:** Hit@K is equivalent to Recall@K when you have single ground-truth. MRR adds rank sensitivity—if Hit@5 is decent but MRR is low, your docs are being retrieved but buried.

### Recommended Evaluation Approach

1. **Report multiple K values:** Hit@5, Hit@10, Hit@20
   - If Hit@5 = 60% but Hit@10 = 95%, retriever is close—reranking can help
   - If Hit@20 is still low, retriever needs tuning first

2. **Log exact rank distribution** to see where reranking could help:
   - Example: 40% at rank 1, 30% at ranks 2–5, 30% beyond 5

3. **MRR as primary quality metric** since it captures both coverage AND ranking quality in one number

### Why This Matters for Reranking

- Rerankers can only reorder what the retriever gives them
- If Hit@K is low, reranking can't fix missing docs
- High Hit@K + low MRR = reranker has high potential to help
- Typical pattern: retrieve K=50–200, rerank to top 5–10

---

## Production-Style Evaluation Protocol

### Latency Budgets
| Metric | Budget |
|--------|--------|
| Retrieval p50 | ≤ 80 ms |
| Retrieval p95 | ≤ 150 ms |
| End-to-end (no LLM) | ≤ 200 ms |
| GraphRAG p95 | ≤ 250 ms |

### Indexing Budgets
| Metric | Budget |
|--------|--------|
| Index build time | ≤ 2 hours |
| Peak RAM (indexing) | ≤ 24 GB |
| Disk footprint | ≤ 10 GB |

### Query Set
~200 queries, manually tagged:
- Factoid (40%), Entity lookup (25%), Multi-hop (20%), Ambiguous (10%), Long-form (5%)

### Retrieval Rules
- **Adaptive k:** Start at k=5, expand until score < threshold or max k=30
- **No fixed k** — mimics real production

### Performance Metrics
- **Latency:** p50/p90/p95, cold vs warm cache
- **Throughput:** 1× and 10× concurrency
- **Memory:** Peak RSS (not just index size)

### End-to-End QA Metrics
| Metric | What it measures |
|--------|------------------|
| Answer correctness | Is it factually correct? |
| Faithfulness | Is answer grounded in retrieved docs? |
| Abstention rate | % of "I don't know" responses |

**Principle:** Abstaining is better than hallucinating.

### Reporting
- Recall vs latency curves
- Recall vs memory
- Faithfulness vs abstention
- "Viable on CPU" vs "Not viable" tables

---

## Test Matrix

**9 core retrieval configs:**

| Embedding | FAISS | pgvector | Qdrant |
|-----------|-------|----------|--------|
| MiniLM-L6 | ✅ | ✅ | ✅ |
| e5-small | ✅ | ✅ | ✅ |
| bge-small | ✅ | ✅ | ✅ |

Plus RAG strategy variations layered on top.

---

## Explicitly Excluded
- GPU-dependent methods
- SPLADE / ColBERT (late interaction)
- Large embeddings (>768 dim)
- Distributed/cloud indexes
- End-to-end retriever fine-tuning

---

This gives you a clean, defensible benchmark answering real questions like: *"Is pgvector good enough?"*, *"When does GraphRAG actually help?"*, and *"How much does reranking cost vs. gain?"*