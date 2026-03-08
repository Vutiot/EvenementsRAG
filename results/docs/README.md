# Evaluation Results

This directory contains all evaluation results, comparisons, and analysis for the WW2 RAG system.

---

## Quick Start

**Want a quick summary?** → Read `QUICK_COMPARISON.md`

**Want detailed analysis?** → Read `EVALUATION_SUMMARY.md`

**Want to track metrics over time?** → See `CHANGELOG.md`

**Want raw data?** → Import `metrics_comparison.csv` into Excel/Sheets

---

## File Organization

### Summary Documents
- **`QUICK_COMPARISON.md`** - TL;DR comparison of Phase 1 vs Phase 2
- **`EVALUATION_SUMMARY.md`** - Comprehensive analysis with recommendations
- **`CHANGELOG.md`** - Experiment history and lessons learned

### Raw Results (JSON)
- **`phase1_baseline_35q.json`** - Phase 1 detailed results (pure semantic search)
- **`phase2_hybrid_temporal_35q.json`** - Phase 2 detailed results (hybrid + temporal)

### Data Exports
- **`metrics_comparison.csv`** - All metrics in CSV format for spreadsheet analysis

### Archive
- **`PHASE1_BASELINE_REPORT.md`** - Earlier 21-question baseline (superseded by 35q)

---

## Current Best Approach

**✅ Use Phase 1: Pure Semantic Search**

- **Article Hit@5**: 91.4%
- **Chunk Hit@5**: 71.4%
- **MRR**: 0.654

Phase 2 (Hybrid + Temporal) did not improve performance and decreased MRR by 15.4%.

---

## Key Metrics Explained

### Binary Hit Metrics (Recommended)

**Article Hit@K**: Did we retrieve ANY chunk from the source article in top-K?
- Measures broad retrieval quality
- Current: **91.4%** @5

**Chunk Hit@K**: Did we retrieve THE EXACT source chunk in top-K?
- Measures precise retrieval quality
- Current: **71.4%** @5

**Precision Gap**: Article Hit@K - Chunk Hit@K
- Shows how often we find article but miss exact chunk
- Current: **20%**
- **Interpretation**: Reranking opportunity

### Traditional Metrics

**Recall@K**: Proportion of relevant chunks retrieved in top-K
- Same as Chunk Hit@K for single-chunk ground truth
- Current: **71.4%** @5

**MRR** (Mean Reciprocal Rank): Average of 1/rank for first relevant result
- Measures ranking quality
- Current: **0.654** (Phase 1)
- Range: 0-1, higher is better

**NDCG@K**: Normalized Discounted Cumulative Gain
- Weighted ranking metric (higher ranks matter more)
- Current: **0.659** @5

---

## Evaluation Setup

### Dataset
- **49 articles** (WW2 Wikipedia)
- **1,849 chunks** (512 tokens, 50 overlap)
- **35 questions** (chunk-based generation)

### Question Generation
- Model: mistralai/mistral-small-3.1-24b-instruct:free
- Sampling: Stratified across all articles
- Distribution: Natural (no taxonomy enforcement)
- Generated: 35/50 (rate limited at 16 req/min)

### Search Configuration
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (384d)
- Vector store: Qdrant (in-memory)
- Distance: Cosine similarity
- Top-K: [1, 3, 5, 10]

---

## Results by Phase

### Phase 1: Pure Semantic Search

```
Article Hit:  74.3% @1, 91.4% @3, 91.4% @5, 97.1% @10
Chunk Hit:    60.0% @1, 65.7% @3, 71.4% @5, 80.0% @10
MRR:          0.654
Avg Time:     9.3 ms
```

**Best for**: Relationship (100%), Comparative (80%), Factual (67%)
**Weak for**: Analytical (33%)

### Phase 2: Hybrid + Temporal

```
Article Hit:  74.3% @1, 91.4% @3, 91.4% @5, 97.1% @10 (unchanged)
Chunk Hit:    60.0% @1, 65.7% @3, 71.4% @5, 80.0% @10 (unchanged)
MRR:          0.553 (-15.4% ❌)
Temporal:     8/35 queries detected
```

**Configuration**: BM25 (30%) + Semantic (70%), RRF k=60

---

## Main Findings

1. **Pure semantic search is sufficient** for this dataset
2. **20% precision gap** is the main issue to address
3. **Hybrid search did not help** - questions are semantically biased
4. **Temporal filtering underutilized** - only 22.9% of questions temporal
5. **Reranking > Hybrid retrieval** for improving precision gap

---

## Recommendations

### Immediate
1. ✅ **Use Phase 1** as production approach
2. 🔄 **Add cross-encoder reranking** to address 20% precision gap
3. 📝 **Generate 50 questions** (currently 35/50)

### Next Experiments
1. Test reranking with cross-encoder
2. Try lower BM25 weights (10-15%) if hybrid needed
3. Generate temporal-specific questions
4. Test query expansion techniques

---

## Files Quick Reference

| File | Purpose | Format |
|------|---------|--------|
| `QUICK_COMPARISON.md` | Quick summary | Markdown |
| `EVALUATION_SUMMARY.md` | Detailed analysis | Markdown |
| `CHANGELOG.md` | Experiment history | Markdown |
| `metrics_comparison.csv` | All metrics | CSV |
| `phase1_baseline_35q.json` | Phase 1 raw results | JSON |
| `phase2_hybrid_temporal_35q.json` | Phase 2 raw results | JSON |

---

## Questions?

- See `../docs/evaluation_metrics_explained.md` for metric definitions
- See `../docs/chunk_vs_article_evaluation.md` for evaluation approach
- See `CHANGELOG.md` for experiment history

---

**Last Updated**: 2025-12-31
**Dataset Version**: 49 articles, 1849 chunks
**Best Approach**: Phase 1 (Pure Semantic Search)
