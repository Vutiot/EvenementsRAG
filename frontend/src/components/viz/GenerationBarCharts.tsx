import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import type { NormalizedBenchmarkResult, NormalizedQuestion } from "../../api/types";
import {
  EChartWrapper,
  buildBarChart,
  buildRadarChart,
  SERIES_COLORS,
} from "../charts";
import {
  GENERATION_METRICS,
  getMetricValue,
} from "../../constants/metricGroups";
import type { MetricDef } from "../../constants/metricGroups";
import { downloadCSV } from "../../utils/csvExport";

interface Props {
  result: NormalizedBenchmarkResult;
  activeMetrics: Set<string>;
}

/* ── helpers ───────────────────────────────────────────────────────── */

function flatActive(active: Set<string>): MetricDef[] {
  return GENERATION_METRICS.subGroups
    .flatMap((sg) => sg.metrics)
    .filter((m) => active.has(m.key));
}

function valueColor(v: number): string {
  if (v >= 0.7) return "#10b981";
  if (v >= 0.4) return "#f59e0b";
  return "#ef4444";
}

function cellColorClass(v: number): string {
  if (v >= 0.7) return "bg-green-50 text-green-700";
  if (v >= 0.4) return "bg-yellow-50 text-yellow-700";
  return "bg-red-50 text-red-700";
}

/** Check if any question has ragas_metrics_std data */
function hasStdData(questions: NormalizedQuestion[]): boolean {
  return questions.some(
    (q) => q.ragas_metrics_std != null && Object.keys(q.ragas_metrics_std).length > 0,
  );
}

/** Compute average std for a metric across questions */
function avgStd(
  result: NormalizedBenchmarkResult,
  metricKey: string,
): number | null {
  const stds = result.per_question
    .map((q) => q.ragas_metrics_std?.[metricKey])
    .filter((v): v is number => v !== undefined && v !== null);
  if (stds.length === 0) return null;
  return stds.reduce((a, b) => a + b, 0) / stds.length;
}

/* ── Per-question table ────────────────────────────────────────────── */

function RagasTable({
  questions,
  metrics,
  phaseName,
  showStd,
}: {
  questions: NormalizedQuestion[];
  metrics: MetricDef[];
  phaseName: string;
  showStd: boolean;
}) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [open, setOpen] = useState(false);

  const sorted = useMemo(() => {
    if (!sortKey) return questions;
    return [...questions].sort((a, b) => {
      const av = a.ragas_metrics?.[sortKey] ?? -1;
      const bv = b.ragas_metrics?.[sortKey] ?? -1;
      return sortAsc ? av - bv : bv - av;
    });
  }, [questions, sortKey, sortAsc]);

  const handleSort = (key: string) => {
    if (sortKey === key) setSortAsc((p) => !p);
    else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const handleExport = () => {
    const rows = questions.map((q) => {
      const row: Record<string, string | number | null> = {
        question_id: q.question_id,
        question: q.question,
        type: q.type,
      };
      for (const m of metrics) {
        row[m.label] = q.ragas_metrics?.[m.key] ?? null;
        if (showStd) {
          row[`${m.label} Std`] = q.ragas_metrics_std?.[m.key] ?? null;
        }
      }
      return row;
    });
    downloadCSV(rows, `${phaseName}_ragas.csv`);
  };

  return (
    <details open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="cursor-pointer text-sm font-medium text-slate-600 hover:text-slate-800">
        Per-Question RAGAS Detail ({questions.length} questions)
      </summary>

      <div className="mt-3 space-y-2">
        <button
          type="button"
          onClick={handleExport}
          className="rounded border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
        >
          Export CSV
        </button>

        <div className="overflow-x-auto rounded border border-slate-200">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-50 text-left">
                <th className="px-3 py-2 font-medium text-slate-500">Question</th>
                <th className="px-2 py-2 font-medium text-slate-500">Type</th>
                {metrics.map((m) => (
                  <th
                    key={m.key}
                    className="cursor-pointer px-2 py-2 font-medium text-slate-500 hover:text-slate-800 whitespace-nowrap"
                    onClick={() => handleSort(m.key)}
                    colSpan={showStd ? 2 : 1}
                  >
                    {m.label}
                    {sortKey === m.key && (sortAsc ? " \u25B2" : " \u25BC")}
                  </th>
                ))}
              </tr>
              {showStd && (
                <tr className="bg-slate-50/50">
                  <th className="px-3 py-1" />
                  <th className="px-2 py-1" />
                  {metrics.map((m) => (
                    <th key={m.key} className="px-2 py-1" colSpan={2}>
                      <div className="flex text-[10px] text-slate-400">
                        <span className="flex-1 text-center">Mean</span>
                        <span className="flex-1 text-center">&sigma;</span>
                      </div>
                    </th>
                  ))}
                </tr>
              )}
            </thead>
            <tbody>
              {sorted.map((q) => (
                <tr key={q.question_id} className="border-t border-slate-100">
                  <td className="max-w-[300px] truncate px-3 py-1.5 text-slate-700">
                    {q.question}
                  </td>
                  <td className="px-2 py-1.5 text-slate-500">{q.type}</td>
                  {metrics.map((m) => {
                    const v = q.ragas_metrics?.[m.key];
                    const std = q.ragas_metrics_std?.[m.key];
                    return showStd ? (
                      <td key={m.key} className="px-0 py-1.5" colSpan={2}>
                        <div className="flex font-mono text-center">
                          <span
                            className={`flex-1 px-1 ${v !== undefined ? cellColorClass(v) : "text-slate-300"}`}
                          >
                            {v !== undefined ? v.toFixed(3) : "\u2014"}
                          </span>
                          <span className="flex-1 px-1 text-slate-400">
                            {std !== undefined ? std.toFixed(3) : "\u2014"}
                          </span>
                        </div>
                      </td>
                    ) : (
                      <td
                        key={m.key}
                        className={`px-2 py-1.5 font-mono text-center ${v !== undefined ? cellColorClass(v) : "text-slate-300"}`}
                      >
                        {v !== undefined ? v.toFixed(3) : "\u2014"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  );
}

/* ── Main component ────────────────────────────────────────────────── */

export default function GenerationBarCharts({ result, activeMetrics }: Props) {
  const metrics = useMemo(() => flatActive(activeMetrics), [activeMetrics]);

  const hasRagas = result.metrics_summary?.ragas != null;
  const showStd = useMemo(
    () => hasStdData(result.per_question),
    [result],
  );

  /* Overview bar chart with color-coded bars + error bars when std available */
  const overviewOption = useMemo((): EChartsOption | null => {
    const categories: string[] = [];
    const data: { value: number; itemStyle: { color: string } }[] = [];
    const errorUpper: number[] = [];
    const errorLower: number[] = [];
    let hasError = false;

    for (const m of metrics) {
      const v = getMetricValue(result, m.key);
      if (v !== null) {
        categories.push(m.label);
        data.push({ value: v, itemStyle: { color: valueColor(v) } });
        const std = avgStd(result, m.key);
        if (std !== null) {
          hasError = true;
          errorUpper.push(v + std);
          errorLower.push(v - std);
        } else {
          errorUpper.push(v);
          errorLower.push(v);
        }
      }
    }
    if (categories.length === 0) return null;

    const baseOption = buildBarChart({
      categories,
      series: [{ name: "Average", data: data as unknown as number[] }],
      title: "Generation Metrics Overview",
    });

    // Add error bar series if std data exists
    if (hasError) {
      const seriesArr = Array.isArray(baseOption.series) ? baseOption.series : [];
      // Use markPoint on the bar series instead of custom series for type safety
      // We'll add error bars via a line series overlay
      const whiskerData: { value: [number, number]; symbol: string }[] = [];
      const capData: [number, number][] = [];
      for (let i = 0; i < categories.length; i++) {
        whiskerData.push({ value: [i, errorUpper[i]!], symbol: "none" });
        whiskerData.push({ value: [i, errorLower[i]!], symbol: "none" });
        capData.push([i, errorUpper[i]!]);
        capData.push([i, errorLower[i]!]);
      }
      return {
        ...baseOption,
        series: [
          ...seriesArr,
          // Upper whisker caps as scatter
          {
            type: "scatter" as const,
            name: "\u00b1\u03c3",
            data: capData,
            symbol: "rect",
            symbolSize: [12, 2],
            itemStyle: { color: "#64748b" },
            z: 10,
          },
          // Vertical lines via markLine on each bar
          ...categories.map((_, i) => ({
            type: "line" as const,
            data: [
              [i, errorLower[i]],
              [i, errorUpper[i]],
            ],
            lineStyle: { color: "#64748b", width: 1.5 },
            symbol: "none",
            silent: true,
            z: 10,
          })),
        ] as EChartsOption["series"],
      };
    }

    return baseOption;
  }, [result, metrics]);

  /* Radar chart with optional std shaded band */
  const radarOption = useMemo((): EChartsOption | null => {
    const indicators: { name: string; max: number }[] = [];
    const values: number[] = [];
    const upperVals: number[] = [];
    const lowerVals: number[] = [];
    let hasStdBand = false;

    for (const m of metrics) {
      const v = getMetricValue(result, m.key);
      if (v !== null) {
        indicators.push({ name: m.label, max: 1 });
        values.push(v);
        const std = avgStd(result, m.key);
        if (std !== null) {
          hasStdBand = true;
          upperVals.push(Math.min(1, v + std));
          lowerVals.push(Math.max(0, v - std));
        } else {
          upperVals.push(v);
          lowerVals.push(v);
        }
      }
    }
    if (indicators.length < 3) return null;

    const baseOption = buildRadarChart({
      indicators,
      series: [{ name: result.phase_name, values }],
    });

    if (hasStdBand) {
      const radarSeries = Array.isArray(baseOption.series) ? [...baseOption.series] : [];
      // Add upper bound as a shaded area
      const existingData = radarSeries[0] as { type: string; data: unknown[] } | undefined;
      if (existingData && Array.isArray(existingData.data)) {
        radarSeries.push({
          type: "radar" as const,
          data: [
            {
              name: "\u00b1\u03c3 band",
              value: upperVals,
              lineStyle: { opacity: 0 },
              areaStyle: {
                color: SERIES_COLORS[0],
                opacity: 0.08,
              },
              itemStyle: { opacity: 0 },
              symbol: "none",
            },
          ],
        } as never);
      }
      return { ...baseOption, series: radarSeries };
    }

    return baseOption;
  }, [result, metrics]);

  /* Per-question data for table */
  const questionsWithRagas = useMemo(
    () => result.per_question.filter((q) => q.ragas_metrics != null),
    [result],
  );

  if (metrics.length === 0) {
    return (
      <p className="text-sm text-slate-400 italic">No generation metrics selected.</p>
    );
  }

  if (!hasRagas) {
    return (
      <div className="rounded border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm text-slate-400">
        No RAGAS evaluation data available for this run.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {overviewOption && <EChartWrapper option={overviewOption} height={350} />}
      {radarOption && (
        <div>
          <h3 className="mb-2 text-sm font-medium text-slate-600">Quality Fingerprint</h3>
          <EChartWrapper option={radarOption} height={380} />
        </div>
      )}
      {questionsWithRagas.length > 0 && (
        <RagasTable
          questions={questionsWithRagas}
          metrics={metrics}
          phaseName={result.phase_name}
          showStd={showStd}
        />
      )}
    </div>
  );
}
