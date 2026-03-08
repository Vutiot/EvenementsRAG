# Phase 1 Baseline Evaluation Report

**Date**: 2025-12-31
**Model**: Vanilla RAG (Semantic Search Only)
**Questions**: 21 (generated from 21 unique chunks/articles)
**Search Space**: 1,849 chunks across 49 WW2 articles
**Avg Retrieval Time**: 16.3ms

---

## Executive Summary

Phase 1 Vanilla RAG achieves:
- ✅ **95.2% Article Hit@5** - Excellent topic-level retrieval
- ⚠️ **61.9% Chunk Hit@5** - Moderate precision in finding exact chunks
- 📊 **33.3% precision gap** - Often finds right article but wrong chunk → **Reranking needed**

**Key Finding**: Analytical and Comparative questions perform excellently (100%), while Factual, Temporal, Entity, and Relationship questions struggle (50%).

---

## Overall Performance Metrics

| Metric | @K=1 | @K=3 | @K=5 | @K=10 | Interpretation |
|--------|------|------|------|-------|----------------|
| **Article Hit** | 95.2% | 95.2% | 95.2% | 100% | ✅ Excellent topic retrieval |
| **Chunk Hit** | 47.6% | 61.9% | 61.9% | 71.4% | ⚠️ Moderate precision |
| **Traditional Recall** | 0.476 | 0.619 | 0.619 | 0.714 | Same as Chunk Hit (1 ground truth) |
| **MRR** | - | - | 0.553 | - | Fair ranking quality |
| **NDCG@5** | - | - | 0.560 | 0.591 | Fair ranking quality |

### Key Observations

1. **Perfect Topic Retrieval @ K=10**: All 21 questions found at least one chunk from the source article
2. **Chunk Hit plateaus at K=5**: Increasing K from 5→10 only improves Chunk Hit by 9.5%, suggesting missing chunks are ranked very low
3. **High Article Hit, Lower Chunk Hit**: 95.2% vs 61.9% indicates we find the right topic but not always the exact passage

---

## Performance by Question Type

### Chunk Hit@5 (Finding THE EXACT chunk with answer)

| Question Type | Chunk Hit@5 | Count | Status | Phase 2 Priority |
|---------------|-------------|-------|--------|------------------|
| **Analytical** | **100%** (2/2) | n=2 | 🟢 Strong | Low |
| **Comparative** | **100%** (3/3) | n=3 | 🟢 Strong | Low |
| **Entity-Centric** | 50% (1/2) | n=2 | 🟡 Moderate | Medium |
| **Factual** | 50% (4/8) | n=8 | 🟡 Moderate | **High** |
| **Relationship** | 50% (2/4) | n=4 | 🟡 Moderate | Medium |
| **Temporal** | 50% (1/2) | n=2 | 🟡 Moderate | **High** |

### Article Hit@5 (Finding ANY chunk from source article)

All question types: **95.2%** (20/21)
Only 1 question failed across all types.

---

## Detailed Analysis

### ✅ Strengths

1. **Comparative Questions (100% Chunk Hit)**
   - Questions asking "How did X differ from Y?" perform perfectly
   - Vanilla semantic search is excellent at finding contrasting information
   - Example success: Comparative questions about historical events

2. **Analytical Questions (100% Chunk Hit)**
   - High-level synthesis questions retrieve exact passages
   - Broader semantic queries match well with embeddings
   - Example success: Questions about impacts, influences, strategic considerations

3. **Topic-Level Retrieval (95.2%)**
   - Nearly perfect at identifying the right article
   - Validates embedding model quality for topic matching

### ⚠️ Weaknesses

1. **Factual Questions (50% Chunk Hit)** 🔴 **Critical**
   - Only half of "who/what/when/where" questions find the exact chunk
   - **Root cause**: Factual questions need precise chunks with specific facts
   - **Example failures**: Questions about specific names, dates, locations
   - **Phase 2 fix**: Hybrid search (semantic + BM25) for keyword matching

2. **Temporal Questions (50% Chunk Hit)** 🔴 **Critical**
   - Date-based questions struggle to find exact chunks
   - **Root cause**: Semantic embeddings don't capture temporal relationships well
   - **Example failures**: "When did X happen?", "What happened before Y?"
   - **Phase 2 fix**: Temporal metadata filtering + date-aware ranking

3. **Relationship Questions (50% Chunk Hit)**
   - Causal chain questions ("What led to X?") only 50% successful
   - **Root cause**: Multi-hop reasoning requires graph structure
   - **Phase 4 fix**: Knowledge graph traversal

4. **Precision Gap (33.3%)**
   - We find the right article 95.2% of the time
   - But only find the exact chunk 61.9% of the time
   - **Implication**: Retrieving ~20-50 chunks from right article but ranking the wrong ones highly
   - **Phase 3 fix**: Cross-encoder reranking to improve chunk-level precision

---

## Question Distribution

```
Total: 21 questions

Factual:        8 (38.1%) ← Largest group, 50% success rate
Relationship:   4 (19.0%) ← 50% success rate
Comparative:    3 (14.3%) ← 100% success rate ✓
Analytical:     2 ( 9.5%) ← 100% success rate ✓
Entity-Centric: 2 ( 9.5%) ← 50% success rate
Temporal:       2 ( 9.5%) ← 50% success rate
```

**Observation**: Most questions (8/21 = 38%) are factual, and they perform poorly. This is a critical gap to address.

---

## Failed Questions Analysis

**6 questions failed to find exact chunk in top-5:**

By type:
- 4 Factual questions failed
- 1 Temporal question failed
- 1 Entity-Centric question failed

**Common failure patterns**:
1. **Keyword mismatch**: Question uses different wording than chunk
2. **Temporal filtering needed**: Date-based questions retrieve chronologically distant chunks
3. **Entity ambiguity**: Multiple chunks mention the entity, wrong one ranked higher

---

## Recommendations for Phase 2

### Priority 1: Fix Factual Question Performance 🔥

**Problem**: 50% Chunk Hit on factual questions (largest question category)

**Solution**: Hybrid Search (Semantic + BM25)
```python
# Phase 2: Combine semantic and keyword search
semantic_results = embed_and_search(query)
bm25_results = keyword_search(query)
combined = reciprocal_rank_fusion([semantic_results, bm25_results])
```

**Expected improvement**: 50% → 75-80% Chunk Hit on factual questions

### Priority 2: Add Temporal Filtering 🔥

**Problem**: 50% Chunk Hit on temporal questions

**Solution**: Extract dates from chunks, filter by temporal relevance
```python
# Phase 2: Temporal-aware retrieval
dates_in_query = extract_dates(query)  # "June 1944"
if dates_in_query:
    results = search_with_temporal_filter(query, date_range=dates_in_query)
```

**Expected improvement**: 50% → 80-90% Chunk Hit on temporal questions

### Priority 3: Cross-Encoder Reranking

**Problem**: 33.3% precision gap (find article but not exact chunk)

**Solution**: Phase 3 reranking
```python
# Phase 3: Rerank top-K with cross-encoder
top_20 = vanilla_search(query, k=20)
reranked_5 = cross_encoder_rerank(query, top_20, output_k=5)
```

**Expected improvement**: Close the 33% gap by 15-20%

---

## Success Metrics for Phase 2

**Baseline (Phase 1)**:
- Article Hit@5: 95.2%
- Chunk Hit@5: 61.9%
- Factual Chunk Hit@5: 50%
- Temporal Chunk Hit@5: 50%

**Phase 2 Targets**:
- Article Hit@5: 95%+ (maintain)
- **Chunk Hit@5: 75%+** (↑14pp)
- **Factual Chunk Hit@5: 75%+** (↑25pp)
- **Temporal Chunk Hit@5: 85%+** (↑35pp)

**Validation Strategy**:
- Use same 21-question test set
- Run Phase 1 vs Phase 2 comparison
- Prove Phase 2 improvements are statistically significant

---

## Implementation Roadmap

### Week 1-2: Phase 2 Implementation
1. **Hybrid Search**: Implement BM25 + Semantic fusion
2. **Temporal Extraction**: Add date metadata to chunks
3. **Temporal Filtering**: Date-aware query processing
4. **Evaluation**: Compare Phase 1 vs Phase 2 on 21-question baseline

### Week 3: Phase 3 Planning (if needed)
- If Phase 2 doesn't close precision gap → Implement reranking
- If Phase 2 succeeds → Move to Phase 4 (Graph RAG)

---

## Data Files

- **Questions**: `data/evaluation/phase1_baseline_30q.json` (21 questions)
- **Results**: `results/phase1_baseline_30q.json` (full metrics)
- **Generated by**: OpenRouter Mistral Small (hit 16req/min rate limit)

---

## Conclusion

Phase 1 Vanilla RAG provides a **solid but insufficient baseline**:

✅ **What works well**:
- Topic-level retrieval (95.2% Article Hit)
- Comparative questions (100%)
- Analytical questions (100%)

❌ **What needs improvement**:
- Factual question precision (50% → need keyword matching)
- Temporal question accuracy (50% → need date filtering)
- Overall chunk precision (62% → need reranking)

**Next Step**: Implement Phase 2 (Hybrid + Temporal RAG) to target the 50% success rate on Factual and Temporal questions.

---

*Generated: 2025-12-31*
*Evaluation Framework: Chunk-based ground truth with binary hit rates*
*Model: sentence-transformers/all-MiniLM-L6-v2*
