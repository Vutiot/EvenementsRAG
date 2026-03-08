# Dataset Size Impact Analysis: 49 vs 976 Articles

**Date:** 2025-12-31
**Comparison:** Phase 1 Pure Semantic Search

---

## Executive Summary

Testing the same RAG system on datasets of vastly different sizes (49 articles vs 976 articles - **20x increase**) reveals surprising insights about retrieval behavior at scale.

**Key Finding:** Larger datasets can actually IMPROVE precision metrics while predictably decreasing recall metrics.

---

## Dataset Specifications

| Metric | 49-Article Dataset | 976-Article Dataset | Change |
|--------|-------------------|---------------------|--------|
| **Articles** | 49 | 976 | **+1894%** (20x) |
| **Chunks** | ~1,849 | 12,927 | **+599%** (7x) |
| **Questions** | 35 | 49 | +40% |
| **Question Source** | 48 articles | 48 articles | Similar coverage |
| **Embedding Model** | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | Same |
| **Chunk Size** | 512 tokens, 50 overlap | 512 tokens, 50 overlap | Same |

---

## Performance Comparison

### Overall Metrics

| Metric | 49 Articles | 976 Articles | Change | Interpretation |
|--------|-------------|--------------|--------|----------------|
| **Article Hit@5** | 91.4% | 81.6% | **-9.8%** | Expected decrease with 20x more articles |
| **Chunk Hit@5** | 71.4% | 77.6% | **+6.2%** | ✅ Improved precision! |
| **MRR** | 0.654 | 0.592 | **-9.5%** | Expected with larger search space |
| **Precision Gap** | 20.0% | 4.1% | **-15.9%** | ✅ Dramatic improvement! |
| **Avg Retrieval Time** | 9.3 ms | 80.8 ms | +77 ms | Expected with 7x more vectors |

### Hit Rates Across K Values

**Article Hit@K (Found ANY chunk from source article)**

| K | 49 Articles | 976 Articles | Change |
|---|-------------|--------------|--------|
| @1 | 74.3% | 57.1% | -17.2% |
| @3 | 91.4% | 77.6% | -13.8% |
| @5 | 91.4% | 81.6% | -9.8% |
| @10 | 97.1% | 91.8% | -5.3% |

**Chunk Hit@K (Found THE EXACT chunk)**

| K | 49 Articles | 976 Articles | Change |
|---|-------------|--------------|--------|
| @1 | 60.0% | 44.9% | -15.1% |
| @3 | 65.7% | 73.5% | **+7.8%** ✅ |
| @5 | 71.4% | 77.6% | **+6.2%** ✅ |
| @10 | 80.0% | 81.6% | **+1.6%** ✅ |

---

## Key Insights

### 1. The Precision Gap Paradox

**Finding:** Adding 20x more articles dramatically REDUCED the precision gap from 20% to 4.1%.

**What is Precision Gap?**
- `Precision Gap = Article Hit@5 - Chunk Hit@5`
- Measures: "We found the right article, but returned the wrong chunk"

**49-Article Dataset:**
- Article Hit@5: 91.4%
- Chunk Hit@5: 71.4%
- Gap: **20%** (found article 20% more often than exact chunk)

**976-Article Dataset:**
- Article Hit@5: 81.6%
- Chunk Hit@5: 77.6%
- Gap: **4.1%** (almost always find chunk when we find article)

**Why This Happens:**
1. **Smaller dataset:** High article recall (91.4%) because there are few articles to choose from, but chunks within articles can be diverse, leading to wrong chunk selection
2. **Larger dataset:** Lower article recall (81.6%) due to more competition, BUT when we DO find the right article, we're much more likely to get the exact chunk (77.6%)
3. **Implication:** The larger dataset forces the semantic search to be more precise - it can't rely on "close enough" matches

### 2. Improved Chunk Precision at K>1

**Finding:** Chunk Hit@3 and Chunk Hit@5 actually IMPROVED with the larger dataset.

| Metric | 49 Articles | 976 Articles | Improvement |
|--------|-------------|--------------|-------------|
| Chunk Hit@3 | 65.7% | 73.5% | **+7.8%** |
| Chunk Hit@5 | 71.4% | 77.6% | **+6.2%** |

**Why This Happens:**
- With more articles, the embedding model is forced to make finer semantic distinctions
- Irrelevant but semantically-similar content from other articles helps "push" the truly relevant chunks higher in rankings
- The larger context creates better contrast for the embedding model

### 3. Expected MRR Decrease

**Finding:** MRR decreased from 0.654 to 0.592 (-9.5%).

**What is MRR?**
- Mean Reciprocal Rank = Average of (1 / rank of first relevant result)
- Measures how quickly we find relevant chunks

**Interpretation:**
- With 7x more chunks (1,849 → 12,927), relevant chunks appear slightly later in rankings
- -9.5% decrease is actually quite modest given the 7x increase in search space
- The semantic model handles scale reasonably well

### 4. Retrieval Speed vs Scale

**Finding:** Retrieval time increased from 9.3ms to 80.8ms (+77ms, 8.7x slower).

**Analysis:**
- Linear relationship with vector count: 7x more vectors → ~8.7x slower
- Still well within acceptable range for production use (<100ms)
- Qdrant's HNSW index scales well (not exponential degradation)

---

## Performance by Question Type

### 49-Article Dataset (35 questions)

| Question Type | Chunk Hit@5 | Performance |
|---------------|-------------|-------------|
| Relationship | 100% | 🟢 Excellent |
| Comparative | 80% | 🟢 Strong |
| Factual | 67% | 🟡 Moderate |
| Temporal | 67% | 🟡 Moderate |
| Analytical | 33% | 🔴 Weak |

### 976-Article Dataset (49 questions)

| Question Type | Recall@5 | Performance |
|---------------|----------|-------------|
| Analytical | 87.5% | 🟢 Excellent |
| Comparative | 85.7% | 🟢 Excellent |
| Relationship | 83.3% | 🟢 Strong |
| Temporal | 80.0% | 🟢 Strong |
| Factual | 71.4% | 🟡 Moderate |
| Entity-centric | 60.0% | 🟡 Moderate |

**Key Changes:**
- Analytical questions improved from 33% → 87.5% (+54.5%)
- Comparative questions improved from 80% → 85.7% (+5.7%)
- All question types except Entity-centric achieved >70% recall

**Note:** Direct type-to-type comparison is limited due to different question sets and possible taxonomy differences.

---

## Implications for RAG System Design

### 1. Dataset Size is Not Always the Enemy

**Conventional Wisdom:** "More documents = worse retrieval"

**Reality:**
- More documents can improve precision when the right chunk is found
- Larger datasets force semantic models to make finer distinctions
- Precision gap dramatically improves with scale

### 2. Optimize for Different Scales

**Small Datasets (< 100 articles):**
- Focus on reducing precision gap through reranking
- May need query expansion to find correct chunks within articles
- Consider cross-encoder reranking as high-priority optimization

**Large Datasets (> 500 articles):**
- Precision gap naturally smaller
- Focus on improving Article Hit@K (finding the right article)
- May benefit from hierarchical retrieval (article → chunk)

### 3. Acceptable Performance Thresholds

**976-Article Dataset Performance:**
- 81.6% Article Hit@5: Finding correct article 4 out of 5 times
- 77.6% Chunk Hit@5: Finding exact chunk 3.9 out of 5 times
- 4.1% Precision Gap: When we find article, we almost always find chunk

**Assessment:** These are solid production-ready metrics for a dataset of this size with pure semantic search.

### 4. When to Invest in Optimizations

**High Priority** (based on this analysis):
- Improve Article Hit@1 (currently only 57.1%)
- Consider hybrid search primarily to improve article-level recall
- Test query reformulation for difficult question types

**Lower Priority** (based on this analysis):
- Reranking for precision gap (already at 4.1%)
- Complex multi-stage retrieval (precision already strong)

---

## Recommendations

### For Production Deployment

**Current System (Pure Semantic Search) is suitable for:**
- Datasets up to ~1000 articles
- Applications where 75-80% chunk retrieval accuracy is acceptable
- Use cases with <100ms latency requirements

**Consider Enhancements When:**
- Need >85% chunk retrieval accuracy → Add cross-encoder reranking
- Need >90% article recall → Implement hybrid search (BM25 + Semantic)
- Dataset grows beyond 5000 articles → Test hierarchical retrieval

### For Further Experimentation

1. **Test Intermediate Scales**
   - Try 200, 500 datasets to find the "precision gap inflection point"
   - Map the curve of precision gap vs dataset size

2. **Optimize Article-Level Recall**
   - Current bottleneck: Finding the right article (81.6%)
   - Test query expansion, hybrid search, or BM25 for title matching

3. **Question Type Analysis**
   - Generate more questions per type for statistical significance
   - Understand why entity-centric questions perform worse (60%)

---

## Conclusion

Scaling from 49 to 976 articles (20x increase) revealed counter-intuitive benefits:

✅ **Chunk Hit@5 improved** from 71.4% → 77.6% (+6.2%)
✅ **Precision Gap improved** from 20% → 4.1% (-15.9%)
❌ **Article Hit@5 decreased** from 91.4% → 81.6% (-9.8%)
❌ **MRR decreased** from 0.654 → 0.592 (-9.5%)

**Bottom Line:** Pure semantic search scales surprisingly well. The larger dataset forces the embedding model to make finer semantic distinctions, which actually improves precision metrics. The main trade-off is in finding the correct article in the first place, but once found, chunk precision is excellent.

**Next Steps:** Focus optimization efforts on improving Article Hit@K rather than chunk-level precision.

---

**Dataset Files:**
- 49 articles: `data/raw/wikipedia_articles/`
- 976 articles: `data/raw/wikipedia_articles_1000/`

**Results Files:**
- 49 articles: `results/phase1_baseline_35q.json`
- 976 articles: `results/phase1_baseline_976articles_49q.json`

**Questions Files:**
- 49 articles: `data/evaluation/eval_50_questions.json` (35 generated)
- 976 articles: `data/evaluation/eval_976_articles_50q.json` (49 generated)
