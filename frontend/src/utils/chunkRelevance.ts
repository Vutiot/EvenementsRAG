export type RelevanceTier = "source" | "exact_answer" | "related" | "not_relevant";

/** Hex colors for Plotly chart bar fills. */
export const RELEVANCE_HEX: Record<RelevanceTier, string> = {
  source: "#ef4444",       // red
  exact_answer: "#f97316", // orange
  related: "#eab308",      // yellow
  not_relevant: "#bfdbfe", // light blue (blue-200)
};

/** Tailwind classes for ScoreBar fills. */
export const RELEVANCE_TW: Record<RelevanceTier, string> = {
  source: "bg-red-500",
  exact_answer: "bg-orange-500",
  related: "bg-yellow-500",
  not_relevant: "bg-blue-200",
};

/** Heuristic fallback: compute mark ratio from highlighted HTML. */
function markRatio(html: string): number {
  const plain = html.replace(/<[^>]*>/g, "");
  if (!plain.length) return 0;
  const markRegex = /<mark>([\s\S]*?)<\/mark>/gi;
  let marked = 0;
  let m: RegExpExecArray | null;
  while ((m = markRegex.exec(html)) !== null) marked += m[1]!.length;
  return marked / plain.length;
}

/** Get relevance tier for a chunk. Uses LLM tier if available, else heuristic. */
export function getRelevanceTier(
  chunkId: string,
  sourceChunkId: string | null,
  llmRelevance: string | undefined,
  highlightedHtml: string | undefined,
): RelevanceTier {
  if (sourceChunkId && chunkId === sourceChunkId) return "source";
  // Trust LLM classification if present
  if (llmRelevance === "exact_answer" || llmRelevance === "related" || llmRelevance === "not_relevant") {
    return llmRelevance;
  }
  // Heuristic fallback from <mark> density
  if (!highlightedHtml) return "not_relevant";
  const ratio = markRatio(highlightedHtml);
  if (ratio > 0.4) return "exact_answer";
  if (ratio > 0) return "related";
  return "not_relevant";
}
