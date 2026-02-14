# Evaluation Summary: Phase 1 vs Phase 2

**Date**: 2025-12-31
**Evaluation Set**: 35 questions (chunk-based generation)
**Dataset**: 49 WW2 articles, 1849 chunks

---

## Overview

This evaluation compares two RAG approaches:
- **Phase 1**: Pure semantic search (baseline)
- **Phase 2**: Hybrid search (BM25 + Semantic) + Temporal filtering

---

## Phase 1: Baseline (Pure Semantic Search)

### Configuration
- **Search Method**: Semantic vector search only
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
- **Distance Metric**: Cosine similarity
- **Vector Store**: Qdrant (in-memory)

### Results

| Metric | @1 | @3 | @5 | @10 |
|--------|-----|-----|-----|-----|
| **Article Hit** | 74.3% | 91.4% | 91.4% | 97.1% |
| **Chunk Hit** | 60.0% | 65.7% | 71.4% | 80.0% |
| **Recall** | 0.600 | 0.657 | 0.714 | 0.800 |

**Additional Metrics**:
- **MRR**: 0.654
- **NDCG@5**: 0.659
- **NDCG@10**: 0.688
- **Avg Retrieval Time**: 9.3 ms

### Performance by Question Type (Recall@5)

| Question Type | Recall@5 | Performance |
|---------------|----------|-------------|
| Relationship | 100.0% | 🟢 Excellent |
| Comparative | 80.0% | 🟢 Strong |
| Entity-centric | 71.4% | 🟡 Good |
| Factual | 66.7% | 🟡 Moderate |
| Temporal | 60.0% | 🟡 Moderate |
| Analytical | 33.3% | 🔴 Weak |

### Key Insights

1. **Strong Article-Level Retrieval**: 91.4% article hit@5 shows excellent broad retrieval
2. **Moderate Chunk Precision**: 71.4% chunk hit@5 indicates room for improvement
3. **20% Precision Gap**: Often find the right article but not the exact chunk
   - Suggests need for reranking or better chunk matching
4. **Question Type Variance**:
   - Relationship queries perform best (100%)
   - Analytical queries struggle most (33%)

---

## Phase 2: Hybrid Search + Temporal RAG

### Configuration
- **Search Method**: Hybrid (RRF fusion)
  - BM25 weight: 30%
  - Semantic weight: 70%
  - RRF constant k=60
- **Temporal Filtering**: Year/timeframe extraction and filtering
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
- **BM25 Parameters**: k1=1.5, b=0.75

### Results

| Metric | @1 | @3 | @5 | @10 |
|--------|-----|-----|-----|-----|
| **Article Hit** | 74.3% | 91.4% | 91.4% | 97.1% |
| **Chunk Hit** | 60.0% | 65.7% | 71.4% | 80.0% |
| **Recall** | 0.600 | 0.657 | 0.714 | 0.800 |

**Additional Metrics**:
- **MRR**: 0.553 (**-15.4%** vs Phase 1 🔴)
- **NDCG@5**: 0.659
- **NDCG@10**: 0.688

### Temporal Query Detection
- **Detected**: 8/35 questions (22.9%)
- **Example patterns**: "in 1939", "during 1944", "early war"

---

## Phase 1 vs Phase 2 Comparison

### Overall Performance

| Metric | Phase 1 | Phase 2 | Change | Status |
|--------|---------|---------|--------|--------|
| **Article Hit@5** | 91.4% | 91.4% | +0.0% | ⚪ Unchanged |
| **Chunk Hit@5** | 71.4% | 71.4% | +0.0% | ⚪ Unchanged |
| **Recall@5** | 0.714 | 0.714 | +0.000 | ⚪ Unchanged |
| **MRR** | 0.654 | 0.553 | **-0.101** | 🔴 Worse |

### Precision Gap
- **Phase 1**: 20.0% (Article Hit@5 - Chunk Hit@5)
- **Phase 2**: 20.0%
- **Change**: +0.0% (no improvement)

### Key Findings

**Phase 2 performed slightly worse than Phase 1:**

1. **No Improvement in Hit Rates**: Both article and chunk hit rates remained identical
2. **MRR Decreased by 15.4%**: Hybrid approach reduced ranking quality
3. **Precision Gap Unchanged**: Reranking still needed
4. **Limited Temporal Benefit**: Only 8/35 questions had temporal info

---

## Analysis

### Why Phase 2 Underperformed

1. **Question Generation Bias**:
   - Questions were generated FROM specific chunks
   - Optimized for semantic similarity to source chunk
   - BM25 keyword matching adds noise rather than signal

2. **Small Dataset**:
   - 49 articles, 1849 chunks
   - Limited vocabulary diversity
   - Keyword matching less effective at this scale

3. **BM25 Weight Too High**:
   - 30% BM25 may dilute high-quality semantic search
   - Pure semantic (Phase 1) already performing well

4. **Temporal Filtering Too Restrictive**:
   - Only 22.9% of questions had temporal information
   - May have filtered out relevant chunks for non-temporal queries
   - Need more temporal questions to validate this approach

### Strengths of Phase 1 (Pure Semantic)

1. **Consistent Performance**: 71.4% chunk hit@5
2. **Fast**: 9.3 ms avg retrieval time
3. **Simple**: No complexity overhead
4. **Question-Optimized**: Matches how questions were generated

---

## Recommendations

### Immediate Next Steps

1. **Generate More Questions**:
   - Currently: 35/50 (hit OpenRouter rate limits)
   - Target: 50+ questions for more robust evaluation
   - Include more temporal questions specifically

2. **Optimize BM25 Weight**:
   - Try: 10%, 15%, 20% (lower weights)
   - Or: Use BM25 for reranking instead of fusion
   - Benchmark each configuration

3. **Address Precision Gap (20%)**:
   - Implement cross-encoder reranking
   - Try contextual chunk retrieval (window expansion)
   - Test query expansion techniques

### Future Enhancements

1. **Better Temporal Approach**:
   - Generate more temporal-specific questions
   - Use temporal boost instead of hard filtering
   - Consider time-aware embeddings

2. **Query Enhancement**:
   - Query expansion (add synonyms, related terms)
   - Multi-query approach (generate variations)
   - Use LLM to reformulate queries

3. **Chunk Strategy**:
   - Test different chunk sizes (current: 512 tokens)
   - Overlap variation (current: 50 tokens)
   - Hierarchical chunking (section → subsection → chunk)

4. **Hybrid Reranking**:
   - Use semantic for retrieval (Phase 1)
   - Rerank top-K with BM25 + cross-encoder
   - May preserve semantic quality while adding keyword precision

---

## Conclusion

**Recommendation: Use Phase 1 (Pure Semantic Search) as the current best approach.**

Phase 2's hybrid + temporal approach did not improve performance on this dataset. The pure semantic search in Phase 1 achieves:
- **91.4% Article Hit@5** (excellent broad retrieval)
- **71.4% Chunk Hit@5** (good but improvable)
- **0.654 MRR** (solid ranking quality)

The 20% precision gap between article and chunk hit rates indicates the main opportunity for improvement is **reranking**, not retrieval strategy changes.

---

## Files Generated

- `phase1_baseline_35q.json` - Full Phase 1 results
- `phase2_hybrid_temporal_35q.json` - Full Phase 2 results
- `EVALUATION_SUMMARY.md` - This summary document
- `eval_50_questions.json` - Evaluation question set (35 questions generated)

---

## Experimental Configuration

### Hardware/Environment
- **Platform**: Linux 6.14.0-36-generic
- **Python**: 3.13
- **Vector Store**: Qdrant (in-memory)
- **Embedding Cache**: Enabled (1849 embeddings cached)

### Dataset Characteristics
- **Articles**: 49 WW2 Wikipedia articles
- **Total Chunks**: 1849
- **Avg Chunks per Article**: 37.7
- **Chunk Size**: 512 tokens
- **Chunk Overlap**: 50 tokens
- **Embedding Dimension**: 384

### Question Generation
- **Model**: mistralai/mistral-small-3.1-24b-instruct:free (via OpenRouter)
- **Sampling**: Stratified (across all articles)
- **Questions per Chunk**: 1
- **Taxonomy Enforcement**: Disabled (natural distribution)
- **Rate Limit**: 16 requests/min (free tier)
- **Generation Time**: ~2-3 minutes with retries
