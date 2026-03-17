import { useState } from "react";
import type { RetrievedChunk } from "../../api/types";
import { getRelevanceTier, RELEVANCE_TW } from "../../utils/chunkRelevance";

interface Props {
  chunks: RetrievedChunk[];
  highlightedContent?: Record<string, string>;
  highlighting?: boolean;
  sourceChunkId?: string | null;
  relevanceMap?: Record<string, string>;
}

/** Sanitize HTML to only allow <mark> tags. */
function sanitizeHighlight(html: string): string {
  return html
    .replace(/<(?!\/?mark\b)[^>]*>/gi, "");
}

function ScoreBar({
  score,
  colorOverride,
  hasNegative,
  absMax,
}: {
  score: number;
  colorOverride?: string;
  hasNegative?: boolean;
  absMax?: number;
}) {
  if (hasNegative && absMax) {
    // Diverging bar: 0 is at center (50%), bar extends left or right
    const halfPct = Math.min(Math.abs(score) / absMax, 1) * 50;
    const isPos = score >= 0;
    const color = colorOverride ?? (isPos ? "bg-green-500" : "bg-red-400");
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-gray-600 w-14 text-right shrink-0">
          {score.toFixed(3)}
        </span>
        <div className="relative h-2 w-24 rounded-full bg-gray-200">
          {/* Zero-line */}
          <div className="absolute left-1/2 top-0 h-full w-px bg-gray-400" />
          <div
            className={`absolute top-0 h-2 rounded-full ${color}`}
            style={
              isPos
                ? { left: "50%", width: `${halfPct}%` }
                : { left: `${50 - halfPct}%`, width: `${halfPct}%` }
            }
          />
        </div>
      </div>
    );
  }

  const pct = Math.round(score * 100);
  const color = colorOverride
    ?? (score >= 0.8
      ? "bg-green-500"
      : score >= 0.6
        ? "bg-yellow-500"
        : "bg-red-400");
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 rounded-full bg-gray-200">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-600">{score.toFixed(3)}</span>
    </div>
  );
}

export default function ChunkList({ chunks, highlightedContent, highlighting, sourceChunkId, relevanceMap }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  // Detect negative scores (e.g. cross-encoder reranker) for diverging bar display
  const minScore = chunks.length ? Math.min(...chunks.map((c) => c.score)) : 0;
  const maxScore = chunks.length ? Math.max(...chunks.map((c) => c.score)) : 1;
  const hasNegative = minScore < 0;
  const absMax = hasNegative ? Math.max(Math.abs(minScore), Math.abs(maxScore)) : undefined;

  const toggle = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold text-gray-700">
          Retrieved Chunks ({chunks.length})
        </h3>
        {highlighting && (
          <span className="flex items-center gap-1 text-xs text-amber-600">
            <span className="h-3 w-3 animate-spin rounded-full border-2 border-amber-500 border-t-transparent" />
            Highlighting...
          </span>
        )}
      </div>
      {chunks.map((chunk, i) => {
        const isOpen = expanded.has(chunk.chunk_id);
        const highlighted = highlightedContent?.[chunk.chunk_id];

        // Determine bar color override when highlighting data is available
        let barColor: string | undefined;
        if (highlightedContent && Object.keys(highlightedContent).length > 0) {
          const tier = getRelevanceTier(
            chunk.chunk_id,
            sourceChunkId ?? null,
            relevanceMap?.[chunk.chunk_id],
            highlightedContent?.[chunk.chunk_id],
          );
          barColor = RELEVANCE_TW[tier];
        }

        return (
          <div
            key={chunk.chunk_id}
            className="rounded border border-gray-200 bg-white"
          >
            <button
              onClick={() => toggle(chunk.chunk_id)}
              className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-gray-50"
            >
              <div className="flex items-center gap-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-gray-100 text-xs font-medium text-gray-600">
                  {i + 1}
                </span>
                <div>
                  <span className="text-sm font-medium text-gray-900">
                    {chunk.article_title}
                  </span>
                  <span className="ml-2 text-xs text-gray-400">
                    chunk #{chunk.chunk_index}
                  </span>
                </div>
              </div>
              <ScoreBar score={chunk.score} colorOverride={barColor} hasNegative={hasNegative} absMax={absMax} />
            </button>
            {isOpen && (
              <div className="border-t border-gray-100 px-4 py-3">
                {highlighted ? (
                  <p
                    className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed [&>mark]:bg-yellow-200 [&>mark]:px-0.5 [&>mark]:rounded"
                    dangerouslySetInnerHTML={{ __html: sanitizeHighlight(highlighted) }}
                  />
                ) : (
                  <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                    {chunk.content}
                  </p>
                )}
                <div className="mt-2 flex items-center gap-3 text-xs text-gray-400">
                  <span>ID: {chunk.chunk_id}</span>
                  <a
                    href={chunk.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-blue-500 hover:underline"
                  >
                    Source
                  </a>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
