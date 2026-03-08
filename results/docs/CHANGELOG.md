# Evaluation Changelog

Track all evaluation experiments, results, and learnings.

---

## 2025-12-31: Phase 1 vs Phase 2 Comparison

### Experiment Goal
Compare pure semantic search (baseline) against hybrid search + temporal filtering.

### Configuration

**Phase 1: Pure Semantic Search**
- Embedding: sentence-transformers/all-MiniLM-L6-v2 (384d)
- Vector store: Qdrant (in-memory)
- Distance: Cosine similarity
- No filtering or hybrid approaches

**Phase 2: Hybrid Search + Temporal RAG**
- Hybrid: BM25 (30%) + Semantic (70%)
- Fusion: Reciprocal Rank Fusion (k=60)
- Temporal: Year/timeframe extraction and filtering
- BM25 params: k1=1.5, b=0.75

### Dataset
- **Articles**: 49 WW2 Wikipedia articles
- **Chunks**: 1849 (512 tokens, 50 overlap)
- **Questions**: 35 (chunk-based generation, natural distribution)
- **Generation model**: mistralai/mistral-small-3.1-24b-instruct:free

### Results Summary

| Metric | Phase 1 | Phase 2 | Winner |
|--------|---------|---------|--------|
| Article Hit@5 | 91.4% | 91.4% | TIE |
| Chunk Hit@5 | 71.4% | 71.4% | TIE |
| MRR | 0.654 | 0.553 | **Phase 1** ✓ |

**Winner**: Phase 1 (Pure Semantic Search)

### Key Findings

1. **Hybrid search did not improve hit rates** - Both phases achieved identical article/chunk hit rates
2. **MRR decreased 15.4%** - Phase 2's hybrid approach reduced ranking quality
3. **20% precision gap** - Main issue is finding exact chunk when article is found (needs reranking)
4. **Limited temporal benefit** - Only 8/35 (22.9%) questions had temporal information
5. **Question generation bias** - Questions optimized for semantic similarity, not keyword matching

### Lessons Learned

**Don't**:
- ❌ Use hybrid search when questions are semantically generated
- ❌ Add BM25 at 30% weight - dilutes semantic quality
- ❌ Use hard temporal filtering without sufficient temporal queries

**Do**:
- ✅ Start with pure semantic as strong baseline
- ✅ Generate more temporal-specific questions before testing temporal RAG
- ✅ Try reranking instead of hybrid retrieval
- ✅ Test lower BM25 weights (10-15%) if hybrid is needed

### Files Generated
- `phase1_baseline_35q.json`
- `phase2_hybrid_temporal_35q.json`
- `EVALUATION_SUMMARY.md`
- `QUICK_COMPARISON.md`
- `metrics_comparison.csv`

### Next Steps
1. Generate full 50 questions (currently 35/50 due to rate limits)
2. Implement cross-encoder reranking on Phase 1
3. Test lower BM25 weights (10%, 15%, 20%)
4. Address 20% precision gap with better chunk matching

---

## 2025-12-31 (Earlier): Evaluation System Rewrite

### Changes
- **Rewrote question generation** from article-based to chunk-based
- **Implemented binary metrics**: Article Hit@K and Chunk Hit@K
- **Fixed ground truth computation**: Now uses exact chunk UUID
- **Added temporal filtering**: Extract years/timeframes from queries
- **Added hybrid search**: BM25 + Semantic with RRF

### Validation
- Verified search across all 1849 chunks (no pre-filtering by document ID)
- Confirmed UUID-based chunk matching works correctly
- Tested temporal pattern extraction (years, ranges, relative terms)

### Issues Resolved
- Fixed 0% recall bug (UUID vs old chunk_id format mismatch)
- Fixed Qdrant filter format for range queries
- Added proper JSON serialization for RetrievalMetrics

---

## Future Experiments to Track

### Planned
- [ ] Cross-encoder reranking on Phase 1 baseline
- [ ] Lower BM25 weights (10%, 15%, 20%)
- [ ] Query expansion techniques
- [ ] Different chunk sizes (256, 512, 1024 tokens)
- [ ] Hierarchical chunking strategies
- [ ] Time-aware embeddings

### Ideas
- [ ] Multi-query approach (generate question variations)
- [ ] LLM query reformulation
- [ ] Contextual chunk retrieval (window expansion)
- [ ] Hybrid reranking (semantic retrieval → BM25 + cross-encoder rerank)

---

## Metric Tracking Template

For future experiments, track:

```
Experiment: [Name]
Date: [YYYY-MM-DD]
Goal: [What we're testing]

Configuration:
- Method: [Approach]
- Parameters: [Key settings]
- Dataset: [Size, characteristics]

Results:
- Article Hit@5: [%]
- Chunk Hit@5: [%]
- MRR: [value]
- Precision Gap: [%]

Winner: [Baseline | Experiment | Tie]
Key Finding: [Main insight]
Next Step: [What to try next]
```
