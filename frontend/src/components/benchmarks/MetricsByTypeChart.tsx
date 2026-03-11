import Plot from "react-plotly.js";

interface Props {
  metricsByType: Record<string, Record<string, number>>;
}

const METRIC_KEYS = [
  { key: "recall_at_1", label: "Recall@1" },
  { key: "recall_at_5", label: "Recall@5" },
  { key: "mrr", label: "MRR" },
  { key: "ndcg_at_5", label: "NDCG@5" },
];

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"];

export default function MetricsByTypeChart({ metricsByType }: Props) {
  const types = Object.keys(metricsByType);
  if (types.length === 0) return null;

  const traces = METRIC_KEYS.map(({ key, label }, i) => ({
    x: types,
    y: types.map((t) => metricsByType[t]?.[key] ?? 0),
    name: label,
    type: "bar" as const,
    marker: { color: COLORS[i] },
  }));

  return (
    <div className="rounded border border-slate-200 bg-white p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Metrics by Question Type
      </h3>
      <Plot
        data={traces}
        layout={{
          barmode: "group",
          height: 300,
          margin: { t: 10, b: 60, l: 50, r: 20 },
          font: { family: "DM Sans, sans-serif", size: 11 },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
          xaxis: {
            tickangle: -30,
            gridcolor: "#f1f5f9",
          },
          yaxis: {
            range: [0, 1.05],
            gridcolor: "#f1f5f9",
            tickformat: ".0%",
          },
          legend: {
            orientation: "h",
            y: 1.15,
            x: 0.5,
            xanchor: "center",
            font: { size: 10 },
          },
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%" }}
      />
    </div>
  );
}
