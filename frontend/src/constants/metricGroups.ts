import type { NormalizedBenchmarkResult } from "../api/types";

// ─── Types ───────────────────────────────────────────────────────────

export interface MetricDef {
  key: string;
  label: string;
  higherIsBetter: boolean;
}

export interface MetricSubGroup {
  name: string;
  metrics: MetricDef[];
}

export interface MetricGroup {
  name: string;
  subGroups: MetricSubGroup[];
}

// ─── Retrieval Metrics ───────────────────────────────────────────────

export const RETRIEVAL_METRICS: MetricGroup = {
  name: "Retrieval",
  subGroups: [
    {
      name: "Document-Level",
      metrics: [
        { key: "doc_precision_at_5", label: "Doc P@5", higherIsBetter: true },
        { key: "doc_mrr", label: "Doc MRR", higherIsBetter: true },
        { key: "doc_recall_at_5", label: "Doc R@5", higherIsBetter: true },
      ],
    },
    {
      name: "Chunk-Level",
      metrics: [
        { key: "mrr", label: "MRR", higherIsBetter: true },
        { key: "recall_at_5", label: "R@5", higherIsBetter: true },
        { key: "recall_at_10", label: "R@10", higherIsBetter: true },
        { key: "chunk_precision_at_5", label: "Chunk P@5", higherIsBetter: true },
      ],
    },
    {
      name: "Context / LLM",
      metrics: [
        { key: "context_precision", label: "Context Precision", higherIsBetter: true },
        { key: "context_recall", label: "Context Recall", higherIsBetter: true },
        { key: "context_relevance", label: "Context Relevance", higherIsBetter: true },
      ],
    },
    {
      name: "Entity / NER",
      metrics: [
        { key: "entity_precision_at_5", label: "Ent P@5", higherIsBetter: true },
        { key: "entity_recall_at_5", label: "Ent R@5", higherIsBetter: true },
        { key: "entity_mrr", label: "Ent MRR", higherIsBetter: true },
      ],
    },
  ],
};

// ─── Generation Metrics ──────────────────────────────────────────────

export const GENERATION_METRICS: MetricGroup = {
  name: "Generation",
  subGroups: [
    {
      name: "Faithfulness",
      metrics: [
        { key: "faithfulness", label: "Faithfulness", higherIsBetter: true },
      ],
    },
    {
      name: "Answer Relevancy",
      metrics: [
        { key: "answer_relevancy", label: "Answer Relevancy", higherIsBetter: true },
      ],
    },
    {
      name: "Factual Correctness",
      metrics: [
        { key: "answer_correctness", label: "Answer Correctness", higherIsBetter: true },
        { key: "answer_similarity", label: "Answer Similarity", higherIsBetter: true },
      ],
    },
    {
      name: "Quality",
      metrics: [
        { key: "coherence", label: "Coherence", higherIsBetter: true },
        { key: "correctness", label: "Correctness", higherIsBetter: true },
        { key: "conciseness", label: "Conciseness", higherIsBetter: true },
      ],
    },
    {
      name: "Safety",
      metrics: [
        { key: "harmfulness", label: "Harmfulness", higherIsBetter: false },
        { key: "maliciousness", label: "Maliciousness", higherIsBetter: false },
      ],
    },
  ],
};

// ─── Derived lookups ─────────────────────────────────────────────────

function flatMetrics(group: MetricGroup): MetricDef[] {
  return group.subGroups.flatMap((sg) => sg.metrics);
}

const ALL_RETRIEVAL = flatMetrics(RETRIEVAL_METRICS);
const ALL_GENERATION = flatMetrics(GENERATION_METRICS);

/** Every known metric key. */
export const ALL_METRIC_KEYS: string[] = [
  ...ALL_RETRIEVAL.map((m) => m.key),
  ...ALL_GENERATION.map((m) => m.key),
];

/** Maps metric key → whether higher values are better. */
export const HIGHER_IS_BETTER: Record<string, boolean> = Object.fromEntries(
  [...ALL_RETRIEVAL, ...ALL_GENERATION].map((m) => [m.key, m.higherIsBetter]),
);

// ─── Value extraction ────────────────────────────────────────────────

/**
 * Extract a metric's aggregate value from a NormalizedBenchmarkResult.
 *
 * Handles the heterogeneous storage of metrics across the result object:
 * - Direct scalars: avg_mrr, avg_doc_mrr
 * - Keyed records: avg_recall_at_k["5"], avg_ndcg["5"], avg_doc_precision_at_k["5"]
 * - RAGAS summary: metrics_summary.ragas.avg_<key>
 * - Entity metrics: metrics_summary.entity.avg_<key>
 * - Per-question fallback: average q.metrics[key] or q.ragas_metrics[key]
 */
export function getMetricValue(
  result: NormalizedBenchmarkResult,
  key: string,
): number | null {
  // Direct scalar fields
  if (key === "mrr") return result.avg_mrr;
  if (key === "doc_mrr") return result.avg_doc_mrr ?? null;

  // Keyed record fields: recall_at_N, ndcg_at_N
  const atMatch = key.match(/^(.+?)_at_(\d+)$/);
  if (atMatch) {
    const base = atMatch[1]!;
    const k = atMatch[2]!;

    if (base === "recall") return result.avg_recall_at_k[k] ?? null;
    if (base === "ndcg") return result.avg_ndcg[k] ?? null;
    if (base === "doc_precision") return result.avg_doc_precision_at_k?.[k] ?? null;
    if (base === "doc_recall") return result.avg_doc_recall_at_k?.[k] ?? null;
    if (base === "chunk_precision") return result.avg_chunk_precision_at_k?.[k] ?? null;
    if (base === "article_hit") return result.avg_article_hit_at_k?.[k] ?? null;
    if (base === "chunk_hit") return result.avg_chunk_hit_at_k?.[k] ?? null;

    // Entity metrics from metrics_summary
    if (base === "entity_precision" || base === "entity_recall") {
      const entity = result.metrics_summary?.entity as Record<string, number> | undefined;
      return entity?.["avg_" + key] ?? null;
    }
  }

  // Entity MRR
  if (key === "entity_mrr") {
    const entity = result.metrics_summary?.entity as Record<string, number> | undefined;
    return entity?.["avg_entity_mrr"] ?? null;
  }

  // RAGAS summary: metrics_summary.ragas.avg_<key>
  const ragas = result.metrics_summary?.ragas as Record<string, number> | undefined;
  if (ragas) {
    const val = ragas["avg_" + key];
    if (val !== undefined) return val;
  }

  // Context metrics from RAGAS
  if (key.startsWith("context_")) {
    if (ragas) {
      const val = ragas["avg_" + key];
      if (val !== undefined) return val;
    }
  }

  // Per-question average fallback
  const questions = result.per_question;
  if (questions.length === 0) return null;

  // Try retrieval metrics first
  const retValues = questions
    .map((q) => q.metrics[key])
    .filter((v): v is number => v !== undefined);
  if (retValues.length > 0) {
    return retValues.reduce((a, b) => a + b, 0) / retValues.length;
  }

  // Try RAGAS per-question metrics
  const ragasValues = questions
    .map((q) => q.ragas_metrics?.[key])
    .filter((v): v is number => v !== undefined && v !== null);
  if (ragasValues.length > 0) {
    return ragasValues.reduce((a, b) => a + b, 0) / ragasValues.length;
  }

  return null;
}
