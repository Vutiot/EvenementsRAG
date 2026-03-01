import { useState } from "react";
import type { RetrievedChunk } from "../../api/types";

interface Props {
  chunks: RetrievedChunk[];
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    score >= 0.8
      ? "bg-green-500"
      : score >= 0.6
        ? "bg-yellow-500"
        : "bg-red-400";
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 w-24 rounded-full bg-gray-200">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-600">{score.toFixed(3)}</span>
    </div>
  );
}

export default function ChunkList({ chunks }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

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
      <h3 className="text-sm font-semibold text-gray-700">
        Retrieved Chunks ({chunks.length})
      </h3>
      {chunks.map((chunk, i) => {
        const isOpen = expanded.has(chunk.chunk_id);
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
              <ScoreBar score={chunk.score} />
            </button>
            {isOpen && (
              <div className="border-t border-gray-100 px-4 py-3">
                <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                  {chunk.content}
                </p>
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
