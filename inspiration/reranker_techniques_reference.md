# Reranker Techniques — Production Reference

> **Purpose:** Concise spec of reranker techniques to implement in a multi-vector-store benchmark project (Qdrant, FAISS, pgvector). Same embedding model across stores. Raw Python, no frameworks.

---

## How it works

```
Vector store returns top-100 candidates (fast, approximate)
        ↓
Reranker rescores/reorders the top-20 to top-100 (slow, precise)
        ↓
Top-5 or top-10 go to the LLM for generation
```

---

## Technique 1: Cross-Encoder

**The workhorse. Used in the vast majority of production RAG systems.**

A transformer model takes the concatenated (query + document) as one input and outputs a relevance score. Query and document tokens attend to each other through all layers.

- **Input:** list of (query, document) string pairs
- **Output:** one float score per pair, sort descending
- **No storable embeddings** — every score requires the full pair through the model

### Models (pick based on latency budget)

| Model | HuggingFace ID | Params | Use when |
|-------|---------------|--------|----------|
| MiniLM L12 | `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | Need speed, CPU-friendly |
| BGE v2 M3 | `BAAI/bge-reranker-v2-m3` | 278M | Best open-source quality/speed tradeoff |
| MXBai v2 Large | `mixedbread-ai/mxbai-rerank-large-v2` | 1.5B | Maximum accuracy, need GPU |

### Libraries
- `sentence-transformers` → `CrossEncoder` class
- `FlagEmbedding` → `FlagReranker` (for BGE models)

### Production notes
- Rerank top 20–100 candidates, not more
- Use fp16 and batched inference
- Cache scores for repeated queries: key = (query_hash, doc_id)
- Latency: ~150ms for 100 docs × 256 tokens on GPU

---

## Technique 2: ColBERT (Late Interaction)

**Used when you need reranker-level quality with retriever-level speed.**

Each token gets its own embedding. Relevance = MaxSim: for each query token, find max cosine similarity with any document token, sum all maxima.

- **Input:** query token embeddings, document token embeddings
- **Output:** one float score per document
- **Document embeddings CAN be precomputed and stored** — only query needs encoding at search time

### Models

| Model | ID | Notes |
|-------|----|-------|
| ColBERTv2 | `colbert-ir/colbertv2.0` | Standard, 128-dim per token |

### Libraries
- `colbert-ai` (official)
- `rerankers` with `model_type='colbert'`

### Production notes
- Storage: ~100KB per 200-token doc (vs ~3KB for single-vector bi-encoder)
- Qdrant has native multi-vector + MaxSim support — cleanest storage option
- 180× fewer FLOPs than cross-encoders at k=10
- Best when corpus is large and you need sub-100ms latency

---

## Technique 3: BM25 Rescoring (Sparse Signal)

**Used alongside dense retrieval in every serious hybrid search setup.**

Classical term-frequency scoring. Captures exact keyword matches that dense models miss.

- **Input:** tokenized query, candidate documents
- **Output:** BM25 score per document
- **No GPU needed, very fast**

### Libraries
- `rank_bm25` (simplest)
- `pyserini` (full-featured, Lucene-based)

### Production notes
- Best used as a complementary signal: dense top-100 → rescore with BM25 → combine scores
- Parameters: k1=1.2–2.0, b=0.75
- Catches queries with specific terms, names, codes that dense embeddings fumble
- No model to load, no GPU, runs anywhere

---

## Technique 4: FlashRank (Lightweight/CPU Reranker)

**Used when you need reranking without GPU or with strict latency constraints.**

Small ONNX-optimized cross-encoder models. Fast on CPU.

- **Input:** query + list of passage dicts with "text" key
- **Output:** scored passages
- **CPU-only, minimal dependencies**

### Libraries
- `flashrank` (pip install FlashRank)

### Models available (built-in)

| Model | Notes |
|-------|-------|
| `ms-marco-MiniLM-L-12-v2` | Default, English |
| `ms-marco-MultiBERT-L-12` | Multilingual |
| `rank-T5-flan` | T5-based variant |

### Production notes
- ONNX runtime under the hood — fast even without GPU
- Not the most accurate but excellent latency/quality ratio
- Good for high-throughput, cost-sensitive deployments
- Often used as a cheap first reranker before a more expensive one (cascading)

---

## Technique 5: Jina Reranker (API)

**Growing in production adoption, especially for multilingual and code search.**

API documentation: **https://jina.ai/reranker**

### Models

| Model | Notes |
|-------|-------|
| `jina-reranker-v2-base-multilingual` | 100+ languages, 1024 token context |
| `jina-reranker-v3` | 0.6B params, highest BEIR score (61.94 nDCG-10), processes up to 64 docs at once |

### API call

```
POST https://api.jina.ai/v1/rerank
Headers:
  Authorization: Bearer <API_KEY>
  Content-Type: application/json
Body:
  {
    "query": "...",
    "documents": ["doc1", "doc2", ...],
    "top_n": 10
  }
```

### Production notes
- v2: sliding window for long docs, function-calling-aware, code search capable
- v3: 131K context window, listwise architecture, 10× smaller than generative rerankers
- Also deployable self-hosted on AWS SageMaker / Azure Marketplace
- API key required, token-based pricing

---

## Recommended Benchmark Pipeline

```
                    ┌──────────┐
                    │  Query   │
                    └────┬─────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │  Qdrant  │  │  FAISS   │  │ pgvector │
    │ top-100  │  │ top-100  │  │ top-100  │
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │              │              │
         │   (same embeddings, same    │
         │    results — benchmark      │
         │    each store separately)   │
         │              │              │
         ▼              ▼              ▼
    For each store, apply each reranker independently:
    
    ┌─────────────────────────────────────────────┐
    │  1. Cross-Encoder  (BGE v2 M3 or MXBai)    │
    │  2. ColBERT        (colbertv2.0)            │
    │  3. BM25           (rank_bm25)              │
    │  4. FlashRank      (ONNX MiniLM)            │
    │  5. Jina API       (jina-reranker-v3)       │
    └─────────────────┬───────────────────────────┘
                      ▼
              ┌──────────────────┐
              │  Compare results │
              │  per store ×     │
              │  per reranker    │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Return top-5    │
              │  to LLM / user   │
              └──────────────────┘
```

## Evaluation Metrics

| Metric | What it measures | Use for |
|--------|-----------------|---------|
| NDCG@10 | Ranking quality | Primary metric |
| MRR | First relevant result position | QA / chatbot |
| Recall@k | Coverage in top-k | Completeness |
| Latency p50/p95 | Response time | Production readiness |

## Storage Summary

| Technique | Precomputable? | What to cache |
|-----------|---------------|---------------|
| Cross-Encoder | No | (query_hash, doc_id) → score |
| ColBERT | Yes — per-token matrices | (num_tokens, 128) per doc |
| BM25 | Yes — inverted index | Term frequencies per doc |
| FlashRank | No | (query_hash, doc_id) → score |
| Jina API | No | (query_hash, doc_id) → score |
