# Quick Comparison: Phase 1 vs Phase 2

**Date**: 2025-12-31 | **Questions**: 35 | **Dataset**: 49 articles, 1849 chunks

---

## Bottom Line

**Winner: Phase 1 (Pure Semantic Search)**

Phase 2's hybrid approach did not improve performance and decreased MRR by 15.4%.

---

## Key Metrics Comparison

```
┌─────────────────────┬──────────┬──────────┬─────────┬────────┐
│ Metric              │ Phase 1  │ Phase 2  │  Delta  │ Status │
├─────────────────────┼──────────┼──────────┼─────────┼────────┤
│ Article Hit@5       │  91.4%   │  91.4%   │  +0.0%  │   ⚪   │
│ Chunk Hit@5         │  71.4%   │  71.4%   │  +0.0%  │   ⚪   │
│ MRR                 │  0.654   │  0.553   │  -0.101 │   🔴   │
│ Recall@5            │  0.714   │  0.714   │  +0.000 │   ⚪   │
│ Precision Gap       │  20.0%   │  20.0%   │  +0.0%  │   ⚪   │
└─────────────────────┴──────────┴──────────┴─────────┴────────┘
```

---

## Configuration Summary

### Phase 1: Pure Semantic
- ✅ Sentence-transformers embeddings (384d)
- ✅ Cosine similarity search
- ✅ Simple and fast (9.3ms avg)

### Phase 2: Hybrid + Temporal
- 🔄 BM25 (30%) + Semantic (70%)
- 🔄 Reciprocal Rank Fusion (k=60)
- 🔄 Temporal filtering (8/35 queries)
- ❌ Slower, more complex
- ❌ Lower MRR

---

## Performance by K

```
Article Hit Rate:
  @1:  74.3% → 74.3%  (unchanged)
  @3:  91.4% → 91.4%  (unchanged)
  @5:  91.4% → 91.4%  (unchanged)
  @10: 97.1% → 97.1%  (unchanged)

Chunk Hit Rate:
  @1:  60.0% → 60.0%  (unchanged)
  @3:  65.7% → 65.7%  (unchanged)
  @5:  71.4% → 71.4%  (unchanged)
  @10: 80.0% → 80.0%  (unchanged)
```

---

## Main Takeaway

**The 20% precision gap is the real problem to solve:**
- We find the right article 91.4% of the time
- But only find the exact chunk 71.4% of the time
- This 20% gap suggests **reranking is needed**, not retrieval changes

---

## Recommendation

**Use Phase 1 + Add Reranking**

Instead of hybrid search, try:
1. Keep pure semantic for retrieval (Phase 1)
2. Add cross-encoder reranking on top-K results
3. This preserves semantic quality while adding precision

---

## Files

- `phase1_baseline_35q.json` - Detailed Phase 1 results
- `phase2_hybrid_temporal_35q.json` - Detailed Phase 2 results
- `EVALUATION_SUMMARY.md` - Full analysis
- `QUICK_COMPARISON.md` - This file
