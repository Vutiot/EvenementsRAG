import Plot from "react-plotly.js";
import type { RetrievedChunk } from "../../api/types";
import { getRelevanceTier, RELEVANCE_HEX, type RelevanceTier } from "../../utils/chunkRelevance";

interface Props {
  chunks: RetrievedChunk[];
  sourceChunkId?: string | null;
  highlightedChunkIds?: string[];
  highlightedContent?: Record<string, string>;
  relevanceMap?: Record<string, string>;
}

// Color palette for distinct articles
const COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

const LEGEND_ITEMS: { tier: RelevanceTier; label: string }[] = [
  { tier: "source", label: "Source" },
  { tier: "exact_answer", label: "Exact answer" },
  { tier: "related", label: "Related" },
  { tier: "not_relevant", label: "Not relevant" },
];

export default function ChunkScoresChart({
  chunks,
  sourceChunkId,
  highlightedChunkIds,
  highlightedContent,
  relevanceMap,
}: Props) {
  // Assign colors by article title (default mode)
  const articles = [...new Set(chunks.map((c) => c.article_title))];
  const colorMap = new Map(articles.map((a, i) => [a, COLORS[i % COLORS.length]!]));

  const useRelevanceColors = highlightedContent && Object.keys(highlightedContent).length > 0;

  // Sort by score ascending for horizontal bar (top = highest score)
  const sorted = [...chunks].sort((a, b) => a.score - b.score);

  const labels = sorted.map(
    (c) => `${c.article_title} #${c.chunk_index}`,
  );

  // Bar fill: relevance-based when highlighting data is available, else article-based
  const barColors = sorted.map((c) => {
    if (useRelevanceColors) {
      const tier = getRelevanceTier(
        c.chunk_id,
        sourceChunkId ?? null,
        relevanceMap?.[c.chunk_id],
        highlightedContent?.[c.chunk_id],
      );
      return RELEVANCE_HEX[tier];
    }
    return colorMap.get(c.article_title) ?? "#6b7280";
  });

  // Border highlights only in article-color mode (redundant with relevance fills)
  const highlightedSet = new Set(highlightedChunkIds ?? []);
  const borderColors = sorted.map((c) => {
    if (useRelevanceColors) return "rgba(0,0,0,0)";
    if (sourceChunkId && c.chunk_id === sourceChunkId) return "#ef4444";
    if (highlightedSet.size && highlightedSet.has(c.chunk_id)) return "#eab308";
    return "rgba(0,0,0,0)";
  });
  const borderWidths = sorted.map((c) => {
    if (useRelevanceColors) return 0;
    if (sourceChunkId && c.chunk_id === sourceChunkId) return 3;
    if (highlightedSet.size && highlightedSet.has(c.chunk_id)) return 3;
    return 0;
  });

  // Compute x-axis range: center origin at 0 when negative scores exist
  const scores = sorted.map((c) => c.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const hasNegative = minScore < 0;

  let xRange: [number, number];
  let xTitle: string;
  if (hasNegative) {
    const absMax = Math.max(Math.abs(minScore), Math.abs(maxScore));
    const pad = absMax * 0.1 || 0.1;
    xRange = [-(absMax + pad), absMax + pad];
    xTitle = "Reranker Score";
  } else {
    xRange = [0, Math.max(maxScore * 1.05, 1)];
    xTitle = "Similarity Score";
  }

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
              x: scores,
              marker: {
                color: barColors,
                line: { color: borderColors, width: borderWidths },
              },
              hovertemplate: "%{y}<br>Score: %{x:.3f}<extra></extra>",
            },
          ]}
          layout={{
            margin: { l: 200, r: 20, t: 10, b: 40 },
            height: Math.max(200, sorted.length * 40),
            xaxis: {
              title: { text: xTitle },
              range: xRange,
              zeroline: hasNegative,
              zerolinecolor: "#9ca3af",
              zerolinewidth: hasNegative ? 2 : 1,
            },
            yaxis: { automargin: true },
            font: { size: 11 },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
        {useRelevanceColors && (
          <div className="flex items-center gap-4 px-2 pt-1 pb-1 text-xs text-gray-500">
            {LEGEND_ITEMS.map(({ tier, label }) => (
              <span key={tier} className="flex items-center gap-1">
                <span
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: RELEVANCE_HEX[tier] }}
                />
                {label}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
