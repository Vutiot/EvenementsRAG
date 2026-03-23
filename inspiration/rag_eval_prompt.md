# Prompt: Implement a Portable RAG Evaluation System

## Context

I am building a RAG (Retrieval-Augmented Generation) evaluation pipeline. I need to generate an evaluation dataset **once** from a specific chunking configuration, then **reuse the same questions** across different chunking settings (varying chunk size, overlap, etc.) without regenerating anything.

## Goal

Build a Python module that implements the following three-stage pipeline:

---

### Stage 1 — Generate Evaluation Dataset (run once)

- Take as input a list of chunks, where each chunk is a dict with at least:
  ```python
  {
      "chunk_id": str,
      "doc_id": str,
      "text": str,
      "char_start": int,  # start offset in original document
      "char_end": int      # end offset in original document
  }
  ```
- Sample N chunks (default 100) using a heuristic: prefer chunks with more substantive content (e.g., filter out very short or boilerplate chunks), ensure diversity across documents.
- For each sampled chunk, call an LLM to generate a question and expected answer grounded in that chunk's content.
- Store the evaluation dataset as a list of records:
  ```python
  {
      "question_id": str,
      "question": str,
      "expected_answer": str,
      "source_doc_id": str,
      "char_start": int,   # passage start offset in original document
      "char_end": int       # passage end offset in original document
  }
  ```
- Save this dataset to a JSON file. This is the **portable** artifact — it never changes.

---

### Stage 2 — Map Evaluation Dataset to a New Chunking Configuration

- Take as input:
  - The saved evaluation dataset (JSON from Stage 1).
  - A new set of chunks (from a different chunking config), each with `chunk_id`, `doc_id`, `text`, `char_start`, `char_end`.
- For each question in the evaluation dataset, find all new chunks that overlap with the original passage offsets (`char_start`, `char_end`), filtered by a **minimum overlap threshold** (default: at least 50% of the original passage must be covered by the chunk).
- Produce a mapped ground truth file:
  ```python
  {
      "question_id": str,
      "question": str,
      "expected_answer": str,
      "relevant_chunk_ids": [str]  # chunk IDs in the NEW config that cover the passage
  }
  ```

The overlap matching logic should be:
```
overlap_start = max(question.char_start, chunk.char_start)
overlap_end = min(question.char_end, chunk.char_end)
overlap_length = max(0, overlap_end - overlap_start)
passage_length = question.char_end - question.char_start
overlap_ratio = overlap_length / passage_length
→ include chunk if overlap_ratio >= threshold
```

---

### Stage 3 — Run Evaluation and Compute Metrics

- Take as input:
  - The mapped ground truth from Stage 2.
  - A RAG retrieval function: `query(question: str) -> list[str]` that returns a ranked list of chunk IDs.
- For each question, call the RAG retrieval function and compare retrieved chunk IDs against `relevant_chunk_ids`.
- Compute the following **retrieval metrics**:
  - **Precision@K**: fraction of top-K retrieved chunks that are relevant.
  - **Recall@K**: fraction of relevant chunks that appear in top-K retrieved results.
  - **MRR (Mean Reciprocal Rank)**: average of 1/rank of the first relevant chunk.
  - **Hit Rate@K**: fraction of questions where at least one relevant chunk appears in top-K.
- Aggregate metrics across all questions (mean + std).
- Output a summary report as a dict and optionally print a formatted table.

---

## Requirements

- Use Python 3.10+.
- Use dataclasses or Pydantic models for all data structures.
- The LLM call in Stage 1 should be abstracted behind a callable interface so the user can plug in any LLM (OpenAI, Anthropic, local model, etc.).
- Include a `__main__` example showing a full workflow:
  1. Generate eval dataset from an initial chunking config.
  2. Re-chunk with different parameters.
  3. Map the eval dataset to the new chunks.
  4. Run retrieval and compute metrics.
- Add docstrings and type hints to all public functions.
- Save/load evaluation datasets as JSON files.
