import type { RetrievedChunk, DatasetQuestion } from "../../api/types";
import { RELEVANCE_TW } from "../../utils/chunkRelevance";

interface Props {
  /** GT chunk fetched from the eval dataset's collection. */
  gtChunk: RetrievedChunk | null;
  /** The picked question — used for fallback labels when the chunk is unavailable. */
  pickedQuestion: DatasetQuestion | null;
  /** Highlighted HTML (with <mark> tags) for this chunk, if available. */
  highlightedContent?: string;
}

function sanitizeHighlight(html: string): string {
  return html.replace(/<(?!\/?mark\b)[^>]*>/gi, "");
}

export default function GTChunkPanel({
  gtChunk,
  pickedQuestion,
  highlightedContent,
}: Props) {
  const articleTitle =
    gtChunk?.article_title ?? pickedQuestion?.source_article ?? "Unknown article";
  const sourceUrl = gtChunk?.source_url;
  const chunkIndex = gtChunk?.chunk_index ?? pickedQuestion?.source_chunk_index;
  const chunkId = gtChunk?.chunk_id ?? pickedQuestion?.source_chunk_id;

  return (
    <div className="rounded border border-red-200 bg-red-50/40">
      <div className="flex items-center gap-2 min-w-0 px-4 py-3 border-b border-red-100">
        <span className={`h-2.5 w-2.5 rounded-full ${RELEVANCE_TW.source} shrink-0`} />
        <span className="text-xs font-semibold uppercase tracking-wide text-red-700 shrink-0">
          Ground Truth
        </span>
        <span className="text-sm font-medium text-gray-900 truncate">
          {articleTitle}
        </span>
        {chunkIndex !== undefined && (
          <span className="text-xs text-gray-400 shrink-0">
            chunk #{chunkIndex}
          </span>
        )}
      </div>
      <div className="px-4 py-3">
        {gtChunk ? (
          highlightedContent ? (
            <p
              className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed [&>mark]:bg-yellow-200 [&>mark]:px-0.5 [&>mark]:rounded"
              dangerouslySetInnerHTML={{ __html: sanitizeHighlight(highlightedContent) }}
            />
          ) : (
            <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
              {gtChunk.content}
            </p>
          )
        ) : (
          <p className="text-sm italic text-gray-500">
            Loading ground-truth chunk...
          </p>
        )}
        <div className="mt-2 flex items-center gap-3 text-xs text-gray-400">
          {chunkId && <span>ID: {chunkId}</span>}
          {sourceUrl && (
            <a
              href={sourceUrl}
              target="_blank"
              rel="noreferrer"
              className="text-blue-500 hover:underline"
            >
              Source
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
