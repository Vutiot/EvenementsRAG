import { useMemo } from "react";
import type { NormalizedBenchmarkResult } from "../../api/types";
import { EChartWrapper, buildBarChart, formatMetricValue } from "../charts";
import {
  RETRIEVAL_METRICS,
  getMetricValue,
} from "../../constants/metricGroups";
import type { MetricDef } from "../../constants/metricGroups";

interface Props {
  result: NormalizedBenchmarkResult;
  activeMetrics: Set<string>;
}

/* ── helpers ───────────────────────────────────────────────────────── */

function flatActive(active: Set<string>): MetricDef[] {
  return RETRIEVAL_METRICS.subGroups
    .flatMap((sg) => sg.metrics)
    .filter((m) => active.has(m.key));
}

/* ── Precision Tier Cards ──────────────────────────────────────────── */

interface TierDef {
  label: string;
  keys: string[];
  accent: string;
}

const TIERS: TierDef[] = [
  { label: "Document", keys: ["doc_precision_at_5", "doc_mrr", "doc_recall_at_5"], accent: "border-l-purple-500" },
  { label: "Chunk", keys: ["mrr", "recall_at_5", "recall_at_10", "chunk_precision_at_5"], accent: "border-l-blue-500" },
  { label: "Context", keys: ["context_precision", "context_recall", "context_relevance"], accent: "border-l-teal-500" },
  { label: "Entity", keys: ["entity_precision_at_5", "entity_recall_at_5", "entity_mrr"], accent: "border-l-amber-500" },
];

function PrecisionTierCards({ result, activeMetrics }: Props) {
  const visibleTiers = TIERS.filter((t) =>
    t.keys.some((k) => activeMetrics.has(k)),
  );

  if (visibleTiers.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {visibleTiers.map((tier) => {
        const entries = tier.keys
          .filter((k) => activeMetrics.has(k))
          .map((k) => {
            const def = RETRIEVAL_METRICS.subGroups
              .flatMap((sg) => sg.metrics)
              .find((m) => m.key === k);
            const val = getMetricValue(result, k);
            return { label: def?.label ?? k, val };
          })
          .filter((e) => e.val !== null);

        return (
          <div
            key={tier.label}
            className={`rounded border border-slate-200 bg-white px-4 py-3 border-l-3 ${tier.accent}`}
          >
            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              {tier.label}
            </p>
            <div className="mt-1 space-y-0.5">
              {entries.map((e) => (
                <div key={e.label} className="flex items-baseline justify-between gap-2">
                  <span className="text-xs text-slate-500">{e.label}</span>
                  <span className="font-mono text-sm font-medium text-slate-900">
                    {formatMetricValue(
                      RETRIEVAL_METRICS.subGroups
                        .flatMap((sg) => sg.metrics)
                        .find((m) => m.label === e.label)?.key ?? "",
                      e.val!,
                    )}
                  </span>
                </div>
              ))}
              {entries.length === 0 && (
                <p className="text-xs text-slate-300 italic">No data</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Main component ────────────────────────────────────────────────── */

export default function RetrievalBarCharts({ result, activeMetrics }: Props) {
  const metrics = useMemo(() => flatActive(activeMetrics), [activeMetrics]);

  /* Overview bar chart — one bar per metric */
  const overviewOption = useMemo(() => {
    const categories: string[] = [];
    const values: number[] = [];
    for (const m of metrics) {
      const v = getMetricValue(result, m.key);
      if (v !== null) {
        categories.push(m.label);
        values.push(v);
      }
    }
    return buildBarChart({
      categories,
      series: [{ name: "Average", data: values }],
      title: "Retrieval Metrics Overview",
    });
  }, [result, metrics]);

  /* By question-type grouped bar chart */
  const byTypeOption = useMemo(() => {
    const types = Object.keys(result.metrics_by_type).sort();
    if (types.length === 0) return null;

    const series = metrics
      .map((m) => {
        const data = types.map((t) => result.metrics_by_type[t]?.[m.key] ?? 0);
        const hasData = data.some((v) => v > 0);
        return hasData ? { name: m.label, data } : null;
      })
      .filter((s): s is { name: string; data: number[] } => s !== null);

    if (series.length === 0) return null;

    return buildBarChart({
      categories: types,
      series,
      title: "Metrics by Question Type",
    });
  }, [result, metrics]);

  if (metrics.length === 0) {
    return (
      <p className="text-sm text-slate-400 italic">No retrieval metrics selected.</p>
    );
  }

  return (
    <div className="space-y-6">
      <PrecisionTierCards result={result} activeMetrics={activeMetrics} />
      <EChartWrapper option={overviewOption} height={350} />
      {byTypeOption && <EChartWrapper option={byTypeOption} height={400} />}
    </div>
  );
}
