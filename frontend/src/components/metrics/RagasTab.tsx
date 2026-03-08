import { useState, useMemo } from "react";
import Plot from "react-plotly.js";
import type { NormalizedBenchmarkResult } from "../../api/types";
import RagasMetricsGrid from "../benchmarks/RagasMetricsGrid";
import { downloadCSV } from "../../utils/csvExport";

interface Props {
  result: NormalizedBenchmarkResult;
}

type SortDir = "asc" | "desc";

function colorClass(v: number): string {
  if (v >= 0.7) return "text-green-700";
  if (v >= 0.4) return "text-yellow-700";
  return "text-red-700";
}

export default function RagasTab({ result }: Props) {
  const [sortKey, setSortKey] = useState("question_id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const withRagas = useMemo(
    () => result.per_question.filter((q) => q.ragas_metrics != null && Object.keys(q.ragas_metrics).length > 0),
    [result.per_question],
  );

  // Detect all RAGAS metric keys across questions
  const metricKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const q of withRagas) {
      for (const k of Object.keys(q.ragas_metrics!)) keys.add(k);
    }
    return Array.from(keys).sort();
  }, [withRagas]);

  if (withRagas.length === 0) {
    return (
      <div className="rounded border border-slate-200 bg-white p-12 text-center text-sm text-slate-400">
        RAGAS metrics not available for this result. Enable RAGAS evaluation in the benchmark config.
      </div>
    );
  }

  // Aggregated averages
  const avgMetrics: Record<string, number> = {};
  for (const key of metricKeys) {
    const vals = withRagas.map((q) => q.ragas_metrics![key]).filter((v) => v != null);
    if (vals.length > 0) {
      avgMetrics[key] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
  }

  // Try to use metrics_summary.ragas if available
  const ragasSummary = result.metrics_summary?.ragas as Record<string, unknown> | undefined;
  const summaryAvgs: Record<string, number> = ragasSummary
    ? Object.fromEntries(
        Object.entries(ragasSummary)
          .filter(([k, v]) => k.startsWith("avg_") && typeof v === "number")
          .map(([k, v]) => [k, v as number]),
      )
    : Object.fromEntries(Object.entries(avgMetrics).map(([k, v]) => [`avg_${k}`, v]));

  const handleSort = (key: string) => {
    if (key === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("asc"); }
  };

  const sorted = [...withRagas].sort((a, b) => {
    let va: string | number | null;
    let vb: string | number | null;
    if (sortKey === "question_id") { va = a.question_id; vb = b.question_id; }
    else if (sortKey === "type") { va = a.type; vb = b.type; }
    else { va = a.ragas_metrics?.[sortKey] ?? null; vb = b.ragas_metrics?.[sortKey] ?? null; }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    const cmp = typeof va === "string" ? va.localeCompare(vb as string) : (va as number) - (vb as number);
    return sortDir === "asc" ? cmp : -cmp;
  });

  const handleExport = () => {
    const rows = withRagas.map((q) => ({
      question_id: q.question_id,
      type: q.type,
      ...Object.fromEntries(metricKeys.map((k) => [k, q.ragas_metrics?.[k] ?? null])),
    }));
    downloadCSV(rows, `ragas_metrics_${result.filename}`);
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

  // Radar chart data
  const radarKeys = Object.keys(avgMetrics);
  const radarLabels: string[] = radarKeys.map((k) =>
    k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
  );
  const radarValues: number[] = radarKeys.map((k) => avgMetrics[k]!);

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          RAGAS Overview
        </h3>
        <button
          onClick={handleExport}
          className="rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          Export CSV
        </button>
      </div>

      {/* Aggregated RAGAS grid */}
      <div className="rounded border border-slate-200 bg-white p-4">
        <RagasMetricsGrid metrics={summaryAvgs} title="Aggregated RAGAS Averages" />
      </div>

      {/* Sortable table */}
      <div className="rounded border border-slate-200 bg-white overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead className="bg-slate-50/80 border-b border-slate-100">
            <tr>
              <SortHeader label="Question ID" sk="question_id" />
              <SortHeader label="Type" sk="type" />
              {metricKeys.map((k) => (
                <SortHeader
                  key={k}
                  label={k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                  sk={k}
                />
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {sorted.map((q) => (
              <tr key={q.question_id} className="hover:bg-slate-50/50">
                <td className="px-2 py-1.5 text-slate-700 max-w-[120px] truncate" title={q.question_id}>{q.question_id}</td>
                <td className="px-2 py-1.5 text-slate-500">{q.type}</td>
                {metricKeys.map((k) => {
                  const v = q.ragas_metrics?.[k];
                  return (
                    <td key={k} className={`px-2 py-1.5 text-right ${v != null ? colorClass(v) : "text-slate-400"}`}>
                      {v != null ? v.toFixed(3) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Radar chart */}
      {radarKeys.length >= 3 && (
        <div className="rounded border border-slate-200 bg-white p-4">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Quality Fingerprint
          </h3>
          <Plot
            data={[
              {
                type: "scatterpolar",
                r: [...radarValues, radarValues[0]!] as number[],
                theta: [...radarLabels, radarLabels[0]!] as string[],
                fill: "toself",
                fillcolor: "rgba(59, 130, 246, 0.15)",
                line: { color: "#3b82f6" },
                marker: { size: 5 },
              },
            ]}
            layout={{
              height: 380,
              margin: { t: 30, b: 30, l: 60, r: 60 },
              font: { family: "DM Sans, sans-serif", size: 10 },
              paper_bgcolor: "transparent",
              polar: {
                radialaxis: { range: [0, 1], tickformat: ".1f", gridcolor: "#e2e8f0" },
                angularaxis: { gridcolor: "#e2e8f0" },
              },
              showlegend: false,
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </div>
      )}
    </div>
  );
}
