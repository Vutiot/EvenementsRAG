import Plot from "react-plotly.js";
import type { RetrievedChunk } from "../../api/types";

interface Props {
  chunks: RetrievedChunk[];
}

// Color palette for distinct articles
const COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

export default function ChunkScoresChart({ chunks }: Props) {
  // Assign colors by article title
  const articles = [...new Set(chunks.map((c) => c.article_title))];
  const colorMap = new Map(articles.map((a, i) => [a, COLORS[i % COLORS.length]!]));

  // Sort by score ascending for horizontal bar (top = highest score)
  const sorted = [...chunks].sort((a, b) => a.score - b.score);

  const labels = sorted.map(
    (c) => `${c.article_title} #${c.chunk_index}`,
  );

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Chunk Scores</h3>
      <div className="rounded border border-gray-200 bg-white p-2">
        <Plot
          data={[
            {
              type: "bar",
              orientation: "h",
              y: labels,
              x: sorted.map((c) => c.score),
              marker: {
                color: sorted.map((c) => colorMap.get(c.article_title) ?? "#6b7280"),
              },
              hovertemplate: "%{y}<br>Score: %{x:.3f}<extra></extra>",
            },
          ]}
          layout={{
            margin: { l: 200, r: 20, t: 10, b: 40 },
            height: Math.max(200, sorted.length * 40),
            xaxis: { title: { text: "Similarity Score" }, range: [0, 1] },
            yaxis: { automargin: true },
            font: { size: 11 },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
      </div>
    </div>
  );
}
