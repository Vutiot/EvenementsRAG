import { useState, useMemo } from "react";
import Plot from "react-plotly.js";
import type { NormalizedBenchmarkResult } from "../../api/types";
import { downloadCSV } from "../../utils/csvExport";

interface Props {
  result: NormalizedBenchmarkResult;
}

type SortDir = "asc" | "desc";

const GEN_COLS = [
  { key: "rouge_l_f1", label: "ROUGE-L F1" },
  { key: "rouge_l_precision", label: "ROUGE-L P" },
  { key: "rouge_l_recall", label: "ROUGE-L R" },
  { key: "bert_score_f1", label: "BERT F1" },
  { key: "bert_score_precision", label: "BERT P" },
  { key: "bert_score_recall", label: "BERT R" },
];

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

export default function GenerationTab({ result }: Props) {
  const [sortKey, setSortKey] = useState("question_id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const withGen = useMemo(
    () => result.per_question.filter((q) => q.generation_metrics != null),
    [result.per_question],
  );

  if (withGen.length === 0) {
    return (
      <div className="rounded border border-slate-200 bg-white p-12 text-center text-sm text-slate-400">
        Generation metrics not available for this result. Enable generation and ROUGE/BERTScore in the benchmark config.
      </div>
    );
  }

  const handleSort = (key: string) => {
    if (key === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("asc"); }
  };

  const sorted = [...withGen].sort((a, b) => {
    let va: string | number | null;
    let vb: string | number | null;
    if (sortKey === "question_id") { va = a.question_id; vb = b.question_id; }
    else if (sortKey === "type") { va = a.type; vb = b.type; }
    else if (sortKey === "generation_time_ms") { va = a.generation_time_ms; vb = b.generation_time_ms; }
    else { va = a.generation_metrics?.[sortKey] ?? null; vb = b.generation_metrics?.[sortKey] ?? null; }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    const cmp = typeof va === "string" ? va.localeCompare(vb as string) : (va as number) - (vb as number);
    return sortDir === "asc" ? cmp : -cmp;
  });

  // Averages
  const avgOf = (key: string) => {
    const vals = withGen.map((q) => q.generation_metrics![key]).filter((v) => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };
  const avgRouge = avgOf("rouge_l_f1");
  const avgBert = avgOf("bert_score_f1");

  // Histogram data
  const rougeVals = withGen
    .map((q) => q.generation_metrics!["rouge_l_f1"])
    .filter((v) => v != null);

  const handleExport = () => {
    const rows = withGen.map((q) => ({
      question_id: q.question_id,
      type: q.type,
      ...Object.fromEntries(GEN_COLS.map(({ key }) => [key, q.generation_metrics?.[key] ?? null])),
      generation_time_ms: q.generation_time_ms,
    }));
    downloadCSV(rows, `generation_metrics_${result.filename}`);
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

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Generation Overview
        </h3>
        <button
          onClick={handleExport}
          className="rounded border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          Export CSV
        </button>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Avg ROUGE-L F1" value={avgRouge != null ? avgRouge.toFixed(4) : "—"} accent="border-l-blue-500" delay={0} />
        <MetricCard label="Avg BERTScore F1" value={avgBert != null ? avgBert.toFixed(4) : "—"} accent="border-l-green-500" delay={60} />
        <MetricCard label="Questions Scored" value={String(withGen.length)} accent="border-l-slate-400" delay={120} />
      </div>

      {/* Sortable table */}
      <div className="rounded border border-slate-200 bg-white overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead className="bg-slate-50/80 border-b border-slate-100">
            <tr>
              <SortHeader label="Question ID" sk="question_id" />
              <SortHeader label="Type" sk="type" />
              {GEN_COLS.map(({ key, label }) => (
                <SortHeader key={key} label={label} sk={key} />
              ))}
              <SortHeader label="Gen ms" sk="generation_time_ms" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {sorted.map((q) => (
              <tr key={q.question_id} className="hover:bg-slate-50/50">
                <td className="px-2 py-1.5 text-slate-700 max-w-[120px] truncate" title={q.question_id}>{q.question_id}</td>
                <td className="px-2 py-1.5 text-slate-500">{q.type}</td>
                {GEN_COLS.map(({ key }) => (
                  <td key={key} className="px-2 py-1.5 text-right text-slate-700">
                    {q.generation_metrics?.[key] != null ? q.generation_metrics[key].toFixed(4) : "—"}
                  </td>
                ))}
                <td className="px-2 py-1.5 text-right text-slate-500">
                  {q.generation_time_ms != null ? q.generation_time_ms.toFixed(1) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ROUGE-L F1 histogram */}
      {rougeVals.length > 0 && (
        <div className="rounded border border-slate-200 bg-white p-4">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            ROUGE-L F1 Distribution
          </h3>
          <Plot
            data={[
              {
                x: rougeVals,
                type: "histogram",
                marker: { color: "#3b82f6" },
              },
            ]}
            layout={{
              height: 280,
              margin: { t: 10, b: 40, l: 50, r: 20 },
              font: { family: "DM Sans, sans-serif", size: 11 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              xaxis: { title: { text: "ROUGE-L F1" }, gridcolor: "#f1f5f9" },
              yaxis: { title: { text: "Count" }, gridcolor: "#f1f5f9" },
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
