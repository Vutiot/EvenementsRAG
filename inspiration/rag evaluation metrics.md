# RAG evaluation metrics

## Retrieval metrics

Evaluate whether the retriever surfaces the right information before anything is sent to the LLM.

|  | Precision (signal-to-noise) | Recall (coverage) |
|---|---|---|
| **Chunk-level** (ID-based matching) | **Context precision** — % of retrieved chunks that are relevant to the query. Biased by chunk size: larger chunks are easier to "hit" but contain more irrelevant content. | **Context recall** — % of relevant chunks in the knowledge base that were actually retrieved. Misses show up as incomplete answers. |
| **Text-level** (LLM-judged content) | **Context relevancy** — % of the retrieved text that is actually useful for answering the question. Fair across chunk sizes because it measures content, not chunk IDs. | **Context completeness** — % of the information needed to answer the question that is present somewhere in the retrieved context. Checks if the retriever found everything, regardless of how it was chunked. |
| **Entity-level** (named entity matching) | **Entity precision** — % of named entities found in retrieved chunks that also appear in the ground truth answer. | **Entity recall** — % of named entities in the ground truth answer that appear somewhere in the retrieved chunks. Useful for entity-heavy domains (legal, medical, financial). |

## Generation metrics

Evaluate whether the LLM produces a good answer given what was retrieved.

| Metric | What it measures | Inputs used |
|---|---|---|
| **Faithfulness** | Is every claim in the generated answer grounded in the retrieved context? Detects hallucination. A score of 1.0 means no hallucinated claims. | response + contexts |
| **Answer relevancy** | Does the generated answer actually address the user's question? Checks subject matter match and focus. Does not look at retrieved context at all. | query + response |
| **Answer correctness** | Does the generated answer match the ground truth? The most important end-to-end metric. Measured via LLM-as-judge or semantic similarity. Requires labeled data. | query + response + expected answer |

## Diagnostic reasoning

These three combinations help pinpoint where failures occur:

- **Low context recall + low correctness** → retrieval problem (chunking, embedding model, top-K)
- **High context recall + low faithfulness** → generation problem (the LLM ignored or misused the context)
- **High context recall + high faithfulness + low correctness** → ground truth mismatch or question ambiguity

## Notes on chunk size comparison

- Chunk-level metrics (context precision, context recall) are biased by chunk size and should not be directly compared across different chunking configurations.
- Text-level metrics (context relevancy, context completeness) are fair across chunk sizes because they evaluate content rather than chunk IDs.
- Answer correctness is the safest single metric for cross-configuration comparison since it measures end-to-end output quality regardless of how chunks are structured.
- When comparing across chunk sizes, also track average input tokens per query to capture the cost tradeoff.
