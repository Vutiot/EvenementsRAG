# Chunk-Based vs Article-Based Evaluation Comparison

## Overview

This document compares two approaches to RAG evaluation:
1. **Article-Based**: Ground truth = all chunks from the source article
2. **Chunk-Based**: Ground truth = the specific chunk containing the answer

## Key Differences

### 1. Ground Truth Definition

**Article-Based (Old)**:
- Question generated from an entire article
- Ground truth = ALL chunks from that article (e.g., 20-50 chunks)
- Example: Question about D-Day → All 28 chunks from "Normandy landings" article

**Chunk-Based (New)**:
- Question generated from a specific chunk
- Ground truth = THAT EXACT chunk (optionally ±2 neighbors for context)
- Example: Question about D-Day timing → Specific chunk #12 containing the date

### 2. Precision vs Recall Trade-off

**Article-Based**:
- **High Recall** (100% in previous test): Easy to find *some* chunk from the article
- **Low Precision**: Finding the *right* chunk is not tested
- **Interpretation**: "Can we find anything related to this topic?"

**Chunk-Based**:
- **Lower Recall** (76.5% @ K=5): Harder to find the exact chunk
- **Higher Precision**: Must retrieve the specific chunk with the answer
- **Interpretation**: "Can we find the exact passage containing the answer?"

### 3. Evaluation Results Comparison

| Metric | Article-Based | Chunk-Based | Interpretation |
|--------|---------------|-------------|----------------|
| **Recall@1** | 21.9% | 70.6% | 🟢 Better at finding exact chunk first |
| **Recall@5** | 57.0% | 76.5% | 🟢 Better exact chunk retrieval |
| **Recall@10** | 100% | 94.1% | 🔴 Slightly lower but more meaningful |
| **MRR** | 0.850 | 0.752 | 🔴 Slightly lower ranking |
| **NDCG@5** | 0.899 | 0.735 | 🔴 Lower ranking quality |

**Note**: The article-based "Recall@10 = 100%" is misleading - it just means we found *any* chunk from the article, not necessarily the one with the answer!

### 4. Performance by Question Type

**Article-Based Recall@5**:
- Factual: 75%
- Temporal: 50%
- Comparative: 65%
- Entity-centric: 50%
- Relationship: 45%

**Chunk-Based Recall@5**:
- Factual: 25% ⚠️ (much harder!)
- Temporal: 100% ✅ (excellent!)
- Comparative: 67% ✅
- Entity-centric: 100% ✅
- Analytical: 100% ✅
- Relationship: 100% ✅

### 5. Real-World Implications

**Article-Based Pros**:
- Easier baseline to achieve
- Good for measuring topic-level retrieval
- Useful for evaluating if the RAG system can find *related* content

**Article-Based Cons**:
- **False sense of success**: High recall doesn't mean we retrieved the right chunk
- Doesn't test answer extraction precision
- Not representative of real RAG performance (LLM needs the exact passage, not just any passage from the article)

**Chunk-Based Pros**:
- **More realistic**: Tests if we can find the specific passage with the answer
- **Better reflects production RAG**: LLMs perform better with precise, focused context
- **Granular evaluation**: Identifies which types of questions are actually hard
- **Reveals retrieval weaknesses**: Factual questions (25% recall) clearly need improvement

**Chunk-Based Cons**:
- Stricter evaluation (lower scores)
- Requires chunk-level question generation (more complex)
- May need to include neighboring chunks for contextual questions

## Implementation Details

### Data Structure Changes

**Article-Based Question**:
```json
{
  "id": "gen_q001",
  "question": "Who was the Prime Minister during Casablanca Conference?",
  "type": "factual",
  "source_article": "Casablanca Conference",
  "source_article_id": "1030859",
  "ground_truth_chunks": []  // Computed on-the-fly from all chunks with pageid=1030859
}
```

**Chunk-Based Question**:
```json
{
  "id": "gen_q001",
  "question": "Who was the Prime Minister during Casablanca Conference?",
  "type": "factual",
  "source_chunk_id": "9a37818e-6758-b35e-36b2-3463bbcca6fc",  // Exact chunk UUID
  "source_article": "Casablanca Conference",
  "source_article_id": "1030859"
}
```

### Retrieval Comparison

**Example Query**: "Who commanded the Allied forces during D-Day?"

**Article-Based Results**:
```
Top-5 Retrieved:
1. normandy_landings_chunk_05 ✅ (from source article)
2. normandy_landings_chunk_12 ✅ (from source article)
3. eisenhower_chunk_23 ❌ (different article)
4. normandy_landings_chunk_03 ✅ (from source article)
5. operation_overlord_chunk_08 ❌ (different article)

Recall@5 = 60% (3 out of 5 from source article)
```

**Chunk-Based Results**:
```
Ground Truth: normandy_landings_chunk_12 (contains "General Dwight D. Eisenhower")

Top-5 Retrieved:
1. normandy_landings_chunk_05 ❌ (wrong chunk from same article)
2. normandy_landings_chunk_12 ✅ (EXACT chunk with answer!)
3. eisenhower_chunk_23 ❌
4. normandy_landings_chunk_03 ❌
5. operation_overlord_chunk_08 ❌

Recall@5 = 100% (exact chunk found in top-5)
MRR = 0.50 (found at position 2)
```

## Recommendations

### When to Use Article-Based Evaluation

1. **Initial baseline testing**: Quick smoke test of retrieval
2. **Topic-level coverage**: Testing if RAG can find related articles
3. **Broad knowledge base evaluation**: Ensuring all topics are indexed

### When to Use Chunk-Based Evaluation

1. **Production RAG systems**: Real-world performance evaluation ✅
2. **Comparing retrieval methods**: Phase 1 vs Phase 2 vs Phase 3 RAG
3. **Debugging retrieval issues**: Identifying which question types fail
4. **Optimizing chunk size**: Testing different chunking strategies
5. **Measuring real precision**: Beyond topic matching

## Conclusion

**Chunk-based evaluation is significantly more meaningful** for production RAG systems because:

1. **It tests what matters**: Can we retrieve the exact passage with the answer?
2. **It reveals real weaknesses**: Factual questions (25% recall) need Phase 2/3 improvements
3. **It aligns with LLM behavior**: LLMs perform better with precise, focused context
4. **It prevents overconfidence**: Article-based 100% recall masks poor chunk-level precision

**Example from our results**:
- Article-based: "Great! 100% recall at K=10!"
- Reality check: Only 57% recall at K=5 for finding relevant chunks
- Chunk-based: "76.5% recall at K=5 for exact chunks"
- Actionable insight: Need to improve factual question retrieval (only 25%)

## Next Steps

1. ✅ Migrate all evaluation to chunk-based approach
2. ✅ Regenerate evaluation questions from chunks (not articles)
3. 🔄 Add neighboring chunk support for contextual questions
4. 🔄 Compare Phase 1 (vanilla) vs Phase 2 (temporal) vs Phase 3 (hybrid) using chunk-based metrics
5. 🔄 Optimize chunking strategy based on chunk-level recall

---

*Generated: 2025-12-31*
*System: EvenementsRAG Phase 1 Vanilla*
