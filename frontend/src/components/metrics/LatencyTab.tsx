import { useState, useMemo } from "react";
import Plot from "react-plotly.js";
import type { NormalizedBenchmarkResult } from "../../api/types";
import { downloadCSV } from "../../utils/csvExport";

interface Props {
  result: NormalizedBenchmarkResult;
}

type SortDir = "asc" | "desc";

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (idx - lo);
}

function MetricCard({ label, value, accent, delay }: {
  label: string; value: string; accent: string; delay: number;
}) {
  return (
    <div
      className={`rounded border border-slate-200 bg-white px-4 py-3 border-l-3 ${accent} opacity-0 animate-fade-in-up`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-1 font-mono text-lg font-medium text-slate-900">{value}</p>
    </div>
  );
}

export default function LatencyTab({ result }: Props) {
  const [sortKey, setSortKey] = useState("question_id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const withLatency = useMemo(
    () => result.per_question.filter((q) => q.retrieval_time_ms != null),
    [result.per_question],
  );

  if (withLatency.length === 0) {
    return (
      <div className="rounded border border-slate-200 bg-white p-12 text-center text-sm text-slate-400">
        Latency data not available for this result.
      </div>
    );
  }

  // Compute percentiles from per-question data
  const retrievalTimes = withLatency.map((q) => q.retrieval_time_ms!).sort((a, b) => a - b);
  const genTimes = withLatency.map((q) => q.generation_time_ms).filter((v) => v != null).sort((a, b) => a - b) as number[];

  const rP50 = percentile(retrievalTimes, 50);
  const rP95 = percentile(retrievalTimes, 95);
  const rP99 = percentile(retrievalTimes, 99);
  const gP50 = genTimes.length ? percentile(genTimes, 50) : null;
  const gP95 = genTimes.length ? percentile(genTimes, 95) : null;
  const gP99 = genTimes.length ? percentile(genTimes, 99) : null;

  // Try to use metrics_summary.latency if available
  const lat = result.metrics_summary?.latency as Record<string, number> | undefined;
  const cards = [
    { label: "Retrieval p50", value: `${(lat?.retrieval_p50_ms ?? rP50).toFixed(1)} ms`, accent: "border-l-blue-500" },
    { label: "Retrieval p95", value: `${(lat?.retrieval_p95_ms ?? rP95).toFixed(1)} ms`, accent: "border-l-blue-500" },
    { label: "Retrieval p99", value: `${(lat?.retrieval_p99_ms ?? rP99).toFixed(1)} ms`, accent: "border-l-blue-500" },
    { label: "Generation p50", value: gP50 != null ? `${(lat?.generation_p50_ms ?? gP50).toFixed(1)} ms` : "—", accent: "border-l-amber-500" },
    { label: "Generation p95", value: gP95 != null ? `${(lat?.generation_p95_ms ?? gP95).toFixed(1)} ms` : "—", accent: "border-l-amber-500" },
    { label: "Generation p99", value: gP99 != null ? `${(lat?.generation_p99_ms ?? gP99).toFixed(1)} ms` : "—", accent: "border-l-amber-500" },
  ];

  const handleSort = (key: string) => {
    if (key === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("asc"); }
  };

  const sorted = [...withLatency].sort((a, b) => {
    let va: string | number | null;
    let vb: string | number | null;
    if (sortKey === "question_id") { va = a.question_id; vb = b.question_id; }
    else if (sortKey === "type") { va = a.type; vb = b.type; }
    else if (sortKey === "retrieval_time_ms") { va = a.retrieval_time_ms; vb = b.retrieval_time_ms; }
    else if (sortKey === "generation_time_ms") { va = a.generation_time_ms; vb = b.generation_time_ms; }
    else if (sortKey === "total_ms") {
      va = (a.retrieval_time_ms ?? 0) + (a.generation_time_ms ?? 0);
      vb = (b.retrieval_time_ms ?? 0) + (b.generation_time_ms ?? 0);
    } else { va = null; vb = null; }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    const cmp = typeof va === "string" ? va.localeCompare(vb as string) : (va as number) - (vb as number);
    return sortDir === "asc" ? cmp : -cmp;
  });

  const handleExport = () => {
    const rows = withLatency.map((q) => ({
      question_id: q.question_id,
      type: q.type,
      retrieval_time_ms: q.retrieval_time_ms,
      generation_time_ms: q.generation_time_ms,
      total_ms: (q.retrieval_time_ms ?? 0) + (q.generation_time_ms ?? 0),
    }));
    downloadCSV(rows, `latency_metrics_${result.filename}`);
  };

  const SortHeader = ({ label, sk }: { label: string; sk: string }) => {
    const active = sk === sortKey;
    return (
      <th
        className="px-2 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-400 cursor-pointer hover:text-slate-600 select-none whitespace-nowrap"
        onClick={() => handleSort(sk)}
      >
        {label} {active ? (sortDir === "asc" ? "\u25B2" : "\u25BC") : ""}
      </th>
    );
  };

  // Box plot data
  const boxTraces = [
    {
      y: retrievalTimes,
      name: "Retrieval",
      type: "box" as const,
      boxpoints: "outliers" as const,
      marker: { color: "#3b82f6" },
    },
  ];
  if (genTimes.length > 0) {
    boxTraces.push({
      y: genTimes,
      name: "Generation",
      type: "box" as const,
      boxpoints: "outliers" as const,
      marker: { color: "#f59e0b" },
    });
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Latency Overview
        </h3>
        <button
          onClick={handleExport}
          className="rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          Export CSV
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
        {cards.map((c, i) => (
          <MetricCard key={c.label} {...c} delay={i * 60} />
        ))}
      </div>

      {/* Sortable table */}
      <div className="rounded border border-slate-200 bg-white overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead className="bg-slate-50/80 border-b border-slate-100">
            <tr>
              <SortHeader label="Question ID" sk="question_id" />
              <SortHeader label="Type" sk="type" />
              <SortHeader label="Retrieval ms" sk="retrieval_time_ms" />
              <SortHeader label="Generation ms" sk="generation_time_ms" />
              <SortHeader label="Total ms" sk="total_ms" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {sorted.map((q) => (
              <tr key={q.question_id} className="hover:bg-slate-50/50">
                <td className="px-2 py-1.5 text-slate-700 max-w-[120px] truncate" title={q.question_id}>{q.question_id}</td>
                <td className="px-2 py-1.5 text-slate-500">{q.type}</td>
                <td className="px-2 py-1.5 text-right text-slate-700">
                  {q.retrieval_time_ms != null ? q.retrieval_time_ms.toFixed(1) : "—"}
                </td>
                <td className="px-2 py-1.5 text-right text-slate-700">
                  {q.generation_time_ms != null ? q.generation_time_ms.toFixed(1) : "—"}
                </td>
                <td className="px-2 py-1.5 text-right text-slate-700">
                  {((q.retrieval_time_ms ?? 0) + (q.generation_time_ms ?? 0)).toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Box plot */}
      <div className="rounded border border-slate-200 bg-white p-4">
        <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
          Latency Distribution
        </h3>
        <Plot
          data={boxTraces}
          layout={{
            height: 320,
            margin: { t: 10, b: 40, l: 60, r: 20 },
            font: { family: "DM Sans, sans-serif", size: 11 },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            yaxis: { title: { text: "ms" }, gridcolor: "#f1f5f9" },
            showlegend: false,
          }}
          config={{ displayModeBar: false, responsive: true }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </div>
    </div>
  );
}
