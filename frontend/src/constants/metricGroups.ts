import type { NormalizedBenchmarkResult, NormalizedQuestion } from "../api/types";

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

// ─── Core metric set (E10) ──────────────────────────────────────────
// Retrieval: hit@5, hit@10 and hit@C (token budget). Answer: binary
// correctness by the LLM judge and by Jev, plus their agreement. Cost and
// latency per question are kept as axes (lower is better, not 0–1).

export const RETRIEVAL_METRICS: MetricGroup = {
  name: "Retrieval",
  subGroups: [
    {
      name: "Rank cut-off",
      metrics: [
        { key: "hit_at_5", label: "Hit@5", higherIsBetter: true },
        { key: "hit_at_10", label: "Hit@10", higherIsBetter: true },
      ],
    },
    {
      name: "Token budget",
      metrics: [
        { key: "hit_at_c4k", label: "Hit@C 4k", higherIsBetter: true },
        { key: "hit_at_c8k", label: "Hit@C 8k", higherIsBetter: true },
      ],
    },
  ],
};

export const GENERATION_METRICS: MetricGroup = {
  name: "Answer",
  subGroups: [
    {
      name: "Answer Correctness",
      metrics: [
        { key: "answer_correctness", label: "Correct (LLM judge)", higherIsBetter: true },
        { key: "answer_correctness_jev", label: "Correct (Jev)", higherIsBetter: true },
      ],
    },
    {
      name: "Judge Agreement",
      metrics: [
        { key: "judge_agreement", label: "Judge Agreement", higherIsBetter: true },
      ],
    },
  ],
};

export const EFFICIENCY_METRICS: MetricGroup = {
  name: "Cost & Latency",
  subGroups: [
    {
      name: "Per Question",
      metrics: [
        { key: "cost_per_question_usd", label: "Cost / question ($)", higherIsBetter: false },
        { key: "latency_per_question_ms", label: "Latency / question (ms)", higherIsBetter: false },
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
const ALL_EFFICIENCY = flatMetrics(EFFICIENCY_METRICS);

/** Quality metrics: all in [0, 1], safe for 0–1 heatmaps and radars. */
export const QUALITY_METRIC_KEYS: string[] = [
  ...ALL_RETRIEVAL.map((m) => m.key),
  ...ALL_GENERATION.map((m) => m.key),
];

/** Cost and latency axes (absolute units, lower is better). */
export const EFFICIENCY_METRIC_KEYS: string[] = ALL_EFFICIENCY.map((m) => m.key);

/** Every known metric key. */
export const ALL_METRIC_KEYS: string[] = [...QUALITY_METRIC_KEYS, ...EFFICIENCY_METRIC_KEYS];

/** Maps metric key → whether higher values are better. */
export const HIGHER_IS_BETTER: Record<string, boolean> = Object.fromEntries(
  [...ALL_RETRIEVAL, ...ALL_GENERATION, ...ALL_EFFICIENCY].map((m) => [m.key, m.higherIsBetter]),
);

/** Display label of a metric key. */
export function metricLabel(key: string): string {
  return (
    [...ALL_RETRIEVAL, ...ALL_GENERATION, ...ALL_EFFICIENCY].find((m) => m.key === key)?.label ?? key
  );
}

// ─── Value extraction ────────────────────────────────────────────────

/** Summary blocks written by the backend MetricsCollector (E10). */
const SUMMARY_BLOCKS = ["hits", "correctness", "cost"] as const;

function mean(values: number[]): number | null {
  return values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

/**
 * Extract a metric's aggregate value from a NormalizedBenchmarkResult.
 *
 * Looks in metrics_summary.{hits,correctness,cost} first (written by the
 * backend), then falls back to averaging the per-question values
 * (q.metrics[key], q.cost_usd / q.latency_ms, q.correctness.agree).
 * Runs recorded before E10 have none of these keys and return null.
 */
export function getMetricValue(
  result: NormalizedBenchmarkResult,
  key: string,
): number | null {
  const summary = result.metrics_summary as Record<string, Record<string, unknown> | undefined> | null;
  for (const block of SUMMARY_BLOCKS) {
    const val = summary?.[block]?.[key];
    if (typeof val === "number") return val;
  }

  const questions = result.per_question;
  if (questions.length === 0) return null;

  if (key === "cost_per_question_usd") {
    return mean(questions.map((q) => q.cost_usd).filter((v): v is number => typeof v === "number"));
  }
  if (key === "latency_per_question_ms") {
    return mean(questions.map((q) => q.latency_ms).filter((v): v is number => typeof v === "number"));
  }
  if (key === "judge_agreement") {
    return mean(
      questions
        .map((q) => q.correctness?.agree)
        .filter((v): v is boolean => typeof v === "boolean")
        .map((v) => (v ? 1 : 0)),
    );
  }

  return mean(
    questions
      .map((q) => q.metrics[key])
      .filter((v): v is number => typeof v === "number"),
  );
}

/** Per-question value of a metric (0/1 for hits and correctness, absolute for cost/latency). */
export function getQuestionMetricValue(q: NormalizedQuestion, key: string): number | null {
  if (key === "cost_per_question_usd") return q.cost_usd ?? null;
  if (key === "latency_per_question_ms") return q.latency_ms ?? null;
  if (key === "judge_agreement") {
    const agree = q.correctness?.agree;
    return typeof agree === "boolean" ? (agree ? 1 : 0) : null;
  }
  const v = q.metrics[key];
  return typeof v === "number" ? v : null;
}
