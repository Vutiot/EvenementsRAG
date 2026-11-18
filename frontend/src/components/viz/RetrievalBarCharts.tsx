import { useMemo } from "react";
import type { NormalizedBenchmarkResult } from "../../api/types";
import { EChartWrapper, buildBarChart, formatMetricValue, SERIES_COLORS } from "../charts";
import {
  RETRIEVAL_METRICS,
  getMetricValue,
  getQuestionMetricValue,
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
  { label: "Rank cut-off", keys: ["hit_at_5", "hit_at_10"], accent: "border-l-blue-500" },
  { label: "Token budget", keys: ["hit_at_c4k", "hit_at_c8k"], accent: "border-l-teal-500" },
];

function PrecisionTierCards({ result, activeMetrics }: Props) {
  const visibleTiers = TIERS.filter((t) =>
    t.keys.some((k) => activeMetrics.has(k)),
  );

  if (visibleTiers.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-3">
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

  /* Per-metric color map: index in `metrics` → SERIES_COLORS slot.
     Both charts use this map so the same metric gets the same color. */
  const metricColor = useMemo(() => {
    const map: Record<string, string> = {};
    metrics.forEach((m, i) => {
      map[m.key] = SERIES_COLORS[i % SERIES_COLORS.length]!;
    });
    return map;
  }, [metrics]);

  /* Overview bar chart — one bar per metric, each bar colored to match
     the corresponding series in the by-type chart */
  const overviewOption = useMemo(() => {
    const categories: string[] = [];
    const data: { value: number; itemStyle: { color: string } }[] = [];
    for (const m of metrics) {
      const v = getMetricValue(result, m.key);
      if (v !== null) {
        categories.push(m.label);
        data.push({ value: v, itemStyle: { color: metricColor[m.key]! } });
      }
    }
    return buildBarChart({
      categories,
      series: [{ name: "Average", data: data as unknown as number[] }],
      title: "Retrieval Metrics Overview",
    });
  }, [result, metrics, metricColor]);

  /* By question-type grouped bar chart */
  const byTypeOption = useMemo(() => {
    const types = Array.from(new Set(result.per_question.map((q) => q.type))).sort();
    if (types.length === 0) return null;

    const series = metrics
      .map((m) => {
        const data = types.map((t) => {
          const vals = result.per_question
            .filter((q) => q.type === t)
            .map((q) => getQuestionMetricValue(q, m.key))
            .filter((v): v is number => v !== null);
          return vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
        });
        const hasData = data.some((v) => v > 0);
        return hasData
          ? { name: m.label, data, color: metricColor[m.key]! }
          : null;
      })
      .filter((s): s is { name: string; data: number[]; color: string } => s !== null);

    if (series.length === 0) return null;

    return buildBarChart({
      categories: types,
      series,
      title: "Metrics by Question Type",
    });
  }, [result, metrics, metricColor]);

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
