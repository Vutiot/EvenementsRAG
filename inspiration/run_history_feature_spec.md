# Run history table — feature specification

## Overview

The "Run history" section displays a single chronological table (most recent first) of all past runs in a RAG hyperparameter tuning application. There are two types of runs: **benchmarks** and **sweeps**. Both share the same table and the same columns.

---

## Table columns

The table has the following columns, in order:

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| Status    | Success/failure indicator                        |
| Timestamp | Date and time of the run                         |
| Name      | Run name (user-defined or "default")             |
| Type      | Badge indicating "bench" or "sweep"              |
| Technique | Retrieval technique (e.g. vanilla, hybrid)       |
| Chunk     | Chunk size (e.g. 256, 512, 1024)                 |
| Top K     | Number of retrieved documents                    |
| LLM       | Language model used, or `__none__`               |
| MRR       | Mean Reciprocal Rank metric                      |
| R@5       | Recall at 5 metric                               |
| R@10      | Recall at 10 metric                              |
| Time      | Execution duration                               |

---

## Benchmark rows

A benchmark is a single-configuration run. It occupies one flat row in the table. Every column is filled with a single value. Nothing special.

---

## Sweep rows

A sweep is a run where the user selected **multiple values for one or more parameters**. A sweep produces N individual benchmark executions — one per combination of the selected parameter values.

### Collapsed state (default)

A sweep occupies a single row in the table, using the exact same columns as a benchmark. The key difference:

- **Parameter columns with a single value** display that value normally, exactly like a benchmark row.
- **Parameter columns with multiple swept values** display all values **stacked vertically within the same cell**, each value on its own line. For example, if chunk sizes 256, 512, and 1024 were swept, the Chunk cell shows three lines:
  ```
  256
  512
  1024
  ```

- **Metric columns** (MRR, R@5, R@10, Time) show a dash (`-`) since there is no single aggregated result for the sweep.
- The **Name cell** includes a clickable expand/collapse chevron before the name, and a subtitle below showing the number of configurations (e.g. "6 configs").
- The **Type cell** shows a "sweep" badge, visually distinct from the "bench" badge.

This design means you can scan the table and immediately see which parameters were varied in a sweep (they have stacked values) versus which were held constant (single value), without needing any special layout or merged columns.

### Expanded state

Clicking the sweep row expands it, revealing child rows directly beneath the parent. Each child row represents one specific parameter combination from the sweep:

- Child rows use the **same column structure** as the parent table — every child value aligns under the correct column header.
- Each child row shows its specific parameter values (technique, chunk, top K, LLM) and its individual metric results (MRR, R@5, R@10, Time).
- The **Name cell** of each child row shows a sequential index (#1, #2, #3…) and optionally any extra sweep-specific parameters that don't have their own column (e.g. overlap, embedding model, reranker) as secondary metadata text.
- Child rows have a **subtly different background** to visually group them under their parent sweep.
- The **last child row** has a slightly stronger bottom border to clearly separate the sweep group from the next run below.
- The **best-performing child** (highest MRR) has its metric values highlighted in a success/green color so the winner is instantly visible.
- Clicking the parent row again collapses all child rows.

---

## Visual design notes

- **Type badges**: "bench" and "sweep" use distinct background colors to be easily scannable.
- **Technique badges**: Values like "vanilla" and "hybrid" are displayed as small colored badges.
- **Monospace font** for all numeric and parameter values.
- **Muted text** for timestamps, secondary info, and the time column.
- Sweep rows are clickable with a chevron indicator that rotates when expanded.
- The table is sorted by timestamp descending. Benchmarks and sweeps are interleaved chronologically.

---

## Example

Given a sweep named `chunk_overlap_sweep` that varies chunk size (256, 512, 1024) while holding technique (vanilla), top K (100), and LLM (__none__) fixed:

**Collapsed row:**

| Status | Timestamp        | Name                  | Type  | Tech.   | Chunk              | Top K | LLM      | MRR | R@5 | R@10 | Time |
|--------|------------------|-----------------------|-------|---------|--------------------|-------|----------|-----|-----|------|------|
| ✓      | 19 mars, 14:20   | ▶ chunk_overlap_sweep | sweep | vanilla | 256 / 512 / 1024   | 100   | __none__ | -   | -   | -    | -    |
|        |                  | *6 configs*           |       |         | *(each on own line)*|       |          |     |     |      |      |

The Chunk cell displays `256`, `512`, `1024` stacked vertically (one per line). Technique, Top K, and LLM each have a single value so they display normally.

**Expanded child rows (appear beneath the parent):**

| # | Extra params            | Tech.   | Chunk | Top K | LLM      | MRR       | R@5       | R@10      | Time |
|---|-------------------------|---------|-------|-------|----------|-----------|-----------|-----------|------|
| 1 | ovlp:0 · MiniLM-L6     | vanilla | 256   | 100   | __none__ | 0.482     | 0.540     | 0.620     | 1.8s |
| 2 | ovlp:128 · MiniLM-L6   | vanilla | 256   | 100   | __none__ | 0.510     | 0.570     | 0.650     | 1.9s |
| 3 | ovlp:256 · BGE-Small    | vanilla | 512   | 100   | __none__ | **0.640** | **0.780** | **0.860** | 3.0s |
| … | …                       | …       | …     | …     | …        | …         | …         | …         | …    |

Row #3 has its metrics highlighted in green as the best performer by MRR.
