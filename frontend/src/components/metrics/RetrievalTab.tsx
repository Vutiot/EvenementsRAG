import { useState, useMemo } from "react";
import Plot from "react-plotly.js";
import type { NormalizedBenchmarkResult } from "../../api/types";
import { downloadCSV } from "../../utils/csvExport";

interface Props {
  result: NormalizedBenchmarkResult;
}

type SortDir = "asc" | "desc";

const METRIC_KEYS = [
  { key: "recall_at_1", label: "Recall@1" },
  { key: "recall_at_3", label: "Recall@3" },
  { key: "recall_at_5", label: "Recall@5" },
  { key: "recall_at_10", label: "Recall@10" },
  { key: "mrr", label: "MRR" },
  { key: "ndcg_at_5", label: "NDCG@5" },
];

const CHART_COLORS = ["#3b82f6", "#6366f1", "#10b981", "#14b8a6", "#f59e0b", "#8b5cf6"];

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

function SortHeader({ label, sortKey, currentKey, currentDir, onSort }: {
  label: string; sortKey: string; currentKey: string; currentDir: SortDir;
  onSort: (key: string) => void;
}) {
  const active = sortKey === currentKey;
  return (
    <th
      className="px-2 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-400 cursor-pointer hover:text-slate-600 select-none whitespace-nowrap"
      onClick={() => onSort(sortKey)}
    >
      {label} {active ? (currentDir === "asc" ? "\u25B2" : "\u25BC") : ""}
    </th>
  );
}

export default function RetrievalTab({ result }: Props) {
  const [sortKey, setSortKey] = useState("question_id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const handleSort = (key: string) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const sorted = useMemo(() => {
    const rows = [...result.per_question];
    rows.sort((a, b) => {
      let va: string | number | null;
      let vb: string | number | null;
      if (sortKey === "question_id") {
        va = a.question_id; vb = b.question_id;
      } else if (sortKey === "type") {
        va = a.type; vb = b.type;
      } else {
        va = a.metrics[sortKey] ?? null;
        vb = b.metrics[sortKey] ?? null;
      }
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      const cmp = typeof va === "string" ? va.localeCompare(vb as string) : (va as number) - (vb as number);
      return sortDir === "asc" ? cmp : -cmp;
    });
    return rows;
  }, [result.per_question, sortKey, sortDir]);

  const summaryCards = [
    { label: "MRR", value: result.avg_mrr.toFixed(4), accent: "border-l-blue-500" },
    { label: "Recall@1", value: result.avg_recall_at_k["1"]?.toFixed(4) ?? "—", accent: "border-l-blue-500" },
    { label: "Recall@5", value: result.avg_recall_at_k["5"]?.toFixed(4) ?? "—", accent: "border-l-indigo-500" },
    { label: "Recall@10", value: result.avg_recall_at_k["10"]?.toFixed(4) ?? "—", accent: "border-l-teal-500" },
    { label: "NDCG@5", value: result.avg_ndcg["5"]?.toFixed(4) ?? "—", accent: "border-l-green-500" },
  ];

  // Chart data: avg metrics by question type
  const types = Object.keys(result.metrics_by_type);
  const traces = METRIC_KEYS.map(({ key, label }, i) => ({
    x: types,
    y: types.map((t) => result.metrics_by_type[t]?.[key] ?? 0),
    name: label,
    type: "bar" as const,
    marker: { color: CHART_COLORS[i] },
  }));

  const handleExport = () => {
    const rows = result.per_question.map((q) => ({
      question_id: q.question_id,
      type: q.type,
      ...Object.fromEntries(METRIC_KEYS.map(({ key }) => [key, q.metrics[key] ?? null])),
    }));
    downloadCSV(rows, `retrieval_metrics_${result.filename}`);
  };

  return (
    <div className="space-y-5">
      {/* Summary cards */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Retrieval Overview
        </h3>
        <button
          onClick={handleExport}
          className="rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          Export CSV
        </button>
      </div>
      <div className="grid grid-cols-3 gap-3 sm:grid-cols-5">
        {summaryCards.map((c, i) => (
          <MetricCard key={c.label} {...c} delay={i * 60} />
        ))}
      </div>

      {/* Sortable table */}
      <div className="rounded border border-slate-200 bg-white overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead className="bg-slate-50/80 border-b border-slate-100">
            <tr>
              <SortHeader label="Question ID" sortKey="question_id" currentKey={sortKey} currentDir={sortDir} onSort={handleSort} />
              <SortHeader label="Type" sortKey="type" currentKey={sortKey} currentDir={sortDir} onSort={handleSort} />
              {METRIC_KEYS.map(({ key, label }) => (
                <SortHeader key={key} label={label} sortKey={key} currentKey={sortKey} currentDir={sortDir} onSort={handleSort} />
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {sorted.map((q) => (
              <tr key={q.question_id} className="hover:bg-slate-50/50">
                <td className="px-2 py-1.5 text-slate-700 max-w-[120px] truncate" title={q.question_id}>{q.question_id}</td>
                <td className="px-2 py-1.5 text-slate-500">{q.type}</td>
                {METRIC_KEYS.map(({ key }) => (
                  <td key={key} className="px-2 py-1.5 text-right text-slate-700">
                    {q.metrics[key] != null ? q.metrics[key].toFixed(4) : "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Chart */}
      {types.length > 0 && (
        <div className="rounded border border-slate-200 bg-white p-4">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Retrieval Metrics by Question Type
          </h3>
          <Plot
            data={traces}
            layout={{
              barmode: "group",
              height: 320,
              margin: { t: 10, b: 60, l: 50, r: 20 },
              font: { family: "DM Sans, sans-serif", size: 11 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              xaxis: { tickangle: -30, gridcolor: "#f1f5f9" },
              yaxis: { range: [0, 1.05], gridcolor: "#f1f5f9", tickformat: ".0%" },
              legend: { orientation: "h", y: 1.18, x: 0.5, xanchor: "center", font: { size: 10 } },
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
