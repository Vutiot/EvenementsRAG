/** Table showing past benchmark run results with sweep grouping + active run placeholder. */

import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import type { ResultFileInfo } from "../../api/types";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface ActiveRun {
  status: "running" | "complete" | "error";
  progress: { current: number; total: number };
  error?: string;
}

interface Props {
  results: ResultFileInfo[];
  activeRun: ActiveRun | null;
}

type DisplayRow =
  | { kind: "normal"; result: ResultFileInfo }
  | { kind: "sweep_parent"; result: ResultFileInfo; childCount: number }
  | {
      kind: "sweep_child";
      result: ResultFileInfo;
      parentSweepId: string;
      childIndex: number;
      isLast: boolean;
      isBest: boolean;
    };

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function fmt(v: number | null | undefined, digits = 3): string {
  if (v == null) return "\u2014";
  return v.toFixed(digits);
}

function fmtTime(s: number | null | undefined): string {
  if (s == null) return "\u2014";
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s`;
}

function truncate(s: string | null | undefined, maxLen = 20): string {
  if (!s) return "\u2014";
  return s.length > maxLen ? s.slice(0, maxLen) + "..." : s;
}

function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return "\u2014";
  try {
    const d = new Date(ts);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

/** Short display for embedding model names */
function shortModel(s: string | null | undefined): string {
  if (!s) return "\u2014";
  const m = s.match(/MiniLM-L\d+/i);
  if (m) return m[0];
  const b = s.match(/bge-\w+/i);
  if (b) return b[0];
  return truncate(s, 16);
}

/* ------------------------------------------------------------------ */
/* Row grouping                                                        */
/* ------------------------------------------------------------------ */

function buildDisplayRows(results: ResultFileInfo[]): DisplayRow[] {
  // Build a set of filenames that are children of some sweep
  const childFilenames = new Set<string>();
  for (const r of results) {
    if (r.sweep_meta) {
      for (const fn of r.sweep_meta.child_filenames) {
        childFilenames.add(fn);
      }
    }
  }

  const rows: DisplayRow[] = [];
  const resultsByFilename = new Map(results.map((r) => [r.filename, r]));

  for (const r of results) {
    // Skip children at the top level — they'll be rendered under their parent
    if (childFilenames.has(r.filename)) continue;

    if (r.sweep_meta) {
      // Sweep parent
      rows.push({
        kind: "sweep_parent",
        result: r,
        childCount: r.sweep_meta.child_filenames.length,
      });
      // Prepare child rows (order matches child_filenames)
      const children = r.sweep_meta.child_filenames
        .map((fn) => resultsByFilename.get(fn))
        .filter((c): c is ResultFileInfo => c != null);
      // Find best MRR among children
      const mrrs = children.map((c) => c.avg_mrr ?? 0);
      const maxMrr = Math.max(...mrrs);
      const hasBest = maxMrr > 0;
      for (let i = 0; i < children.length; i++) {
        const child = children[i];
        if (!child) continue;
        rows.push({
          kind: "sweep_child",
          result: child,
          parentSweepId: r.sweep_meta.sweep_id,
          childIndex: i,
          isLast: i === children.length - 1,
          isBest: hasBest && (child.avg_mrr ?? 0) === maxMrr,
        });
      }
    } else {
      rows.push({ kind: "normal", result: r });
    }
  }
  return rows;
}

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const TOTAL_COLS = 27;

const stickyCls = "sticky left-0 z-10 bg-white";
const stickyStyle: React.CSSProperties = {
  boxShadow: "2px 0 5px -2px rgba(0,0,0,0.1)",
};
const stickyHeaderCls = "sticky left-0 z-20 bg-gray-50";

/* ------------------------------------------------------------------ */
/* Swept param stacked display                                         */
/* ------------------------------------------------------------------ */

function StackedValues({ values }: { values: unknown[] }) {
  return (
    <div className="flex flex-col gap-0.5">
      {values.map((v, i) => (
        <span key={i} className="text-xs leading-tight">
          {String(v)}
        </span>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Chevron icon                                                        */
/* ------------------------------------------------------------------ */

function ChevronIcon({ expanded }: { expanded: boolean }) {
  return (
    <svg
      className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${expanded ? "rotate-90" : ""}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Param cell renderer — handles swept vs normal                       */
/* ------------------------------------------------------------------ */

function ParamCell({
  value,
  sweptValues,
  format,
}: {
  value: unknown;
  sweptValues?: unknown[];
  format?: "model" | "float" | "truncate";
}) {
  if (sweptValues) {
    return (
      <td className="px-3 py-2 font-mono text-xs text-gray-600">
        <StackedValues values={sweptValues} />
      </td>
    );
  }
  let display: string;
  if (value == null) {
    display = "\u2014";
  } else if (format === "model") {
    display = shortModel(String(value));
  } else if (format === "float") {
    display = Number(value).toFixed(2);
  } else if (format === "truncate") {
    display = truncate(String(value), 18);
  } else {
    display = String(value);
  }
  return <td className="px-3 py-2 font-mono text-xs text-gray-600">{display}</td>;
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function RunHistoryTable({ results, activeRun }: Props) {
  const navigate = useNavigate();
  const [expandedSweeps, setExpandedSweeps] = useState<Set<string>>(new Set());

  const displayRows = useMemo(() => buildDisplayRows(results), [results]);

  const toggleSweep = (sweepId: string) => {
    setExpandedSweeps((prev) => {
      const next = new Set(prev);
      if (next.has(sweepId)) next.delete(sweepId);
      else next.add(sweepId);
      return next;
    });
  };

  /** Get swept param values for a given config key, or undefined if not swept */
  const getSweptValues = (
    r: ResultFileInfo,
    configKey: string,
  ): unknown[] | undefined => {
    return r.sweep_meta?.swept_params?.[configKey];
  };

  return (
    <div className="rounded border border-gray-200 bg-white overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-100 text-left text-xs font-medium uppercase tracking-wider text-gray-400 bg-gray-50">
            <th className="px-3 py-2 w-8"></th>
            <th className={`px-3 py-2 ${stickyHeaderCls}`} style={stickyStyle}>
              Name
            </th>
            <th className="px-3 py-2">Type</th>
            <th className="px-3 py-2">Timestamp</th>
            <th className="px-3 py-2">Dataset</th>
            <th className="px-3 py-2">Eval Dataset</th>
            <th className="px-3 py-2">Technique</th>
            <th className="px-3 py-2">Chunk Size</th>
            <th className="px-3 py-2">Chunk Overlap</th>
            <th className="px-3 py-2">Embedding</th>
            <th className="px-3 py-2">Dist. Metric</th>
            <th className="px-3 py-2">Backend</th>
            <th className="px-3 py-2">Top K</th>
            <th className="px-3 py-2">Reranker</th>
            <th className="px-3 py-2">Reranker Model</th>
            <th className="px-3 py-2">Rerank K</th>
            <th className="px-3 py-2">Sparse Wt</th>
            <th className="px-3 py-2">Sparse Type</th>
            <th className="px-3 py-2">Fusion</th>
            <th className="px-3 py-2 text-right">Qs</th>
            <th className="px-3 py-2 text-right">MRR</th>
            <th className="px-3 py-2 text-right">R@5</th>
            <th className="px-3 py-2 text-right">R@10</th>
            <th className="px-3 py-2 text-right">Doc P@5</th>
            <th className="px-3 py-2 text-right">Doc MRR</th>
            <th className="px-3 py-2 text-right">Ctx Prec</th>
            <th className="px-3 py-2 text-right">Time</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-50">
          {/* Active run placeholder */}
          {activeRun && activeRun.status === "running" && (
            <tr className="bg-blue-50/50">
              <td className="px-3 py-2">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
              </td>
              <td
                className={`px-3 py-2 text-gray-500 ${stickyCls} !bg-blue-50/50`}
                style={stickyStyle}
              >
                Running...
              </td>
              <td className="px-3 py-2" colSpan={5}>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{
                        width:
                          activeRun.progress.total > 0
                            ? `${(activeRun.progress.current / activeRun.progress.total) * 100}%`
                            : "0%",
                      }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 whitespace-nowrap">
                    {activeRun.progress.current}/{activeRun.progress.total}
                  </span>
                </div>
              </td>
              <td className="px-3 py-2" colSpan={TOTAL_COLS - 7}></td>
            </tr>
          )}

          {activeRun && activeRun.status === "error" && (
            <tr className="bg-red-50/50">
              <td className="px-3 py-2">
                <span className="text-red-500 text-base">!</span>
              </td>
              <td
                className={`px-3 py-2 text-red-600 text-xs ${stickyCls} !bg-red-50/50`}
                style={stickyStyle}
              >
                Error
              </td>
              <td className="px-3 py-2 text-red-600 text-xs" colSpan={TOTAL_COLS - 2}>
                {activeRun.error || "Unknown error"}
              </td>
            </tr>
          )}

          {/* Data rows */}
          {displayRows.map((row) => {
            if (row.kind === "normal") {
              return (
                <NormalRow
                  key={row.result.filename}
                  r={row.result}
                  navigate={navigate}
                />
              );
            }

            if (row.kind === "sweep_parent") {
              const sweepId = row.result.sweep_meta!.sweep_id;
              const expanded = expandedSweeps.has(sweepId);
              return (
                <SweepParentRow
                  key={`sweep-${sweepId}`}
                  r={row.result}
                  childCount={row.childCount}
                  expanded={expanded}
                  onToggle={() => toggleSweep(sweepId)}
                  getSweptValues={getSweptValues}
                />
              );
            }

            // sweep_child — only render if parent is expanded
            const expanded = expandedSweeps.has(row.parentSweepId);
            if (!expanded) return null;
            return (
              <SweepChildRow
                key={row.result.filename}
                r={row.result}
                childIndex={row.childIndex}
                isLast={row.isLast}
                isBest={row.isBest}
                navigate={navigate}
              />
            );
          })}

          {results.length === 0 && !activeRun && (
            <tr>
              <td colSpan={TOTAL_COLS} className="px-3 py-8 text-center text-gray-400">
                No benchmark results yet. Run a benchmark to get started.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Normal row                                                          */
/* ------------------------------------------------------------------ */

function NormalRow({
  r,
  navigate,
}: {
  r: ResultFileInfo;
  navigate: ReturnType<typeof useNavigate>;
}) {
  return (
    <tr
      onClick={() => navigate(`/benchmarks?file=${encodeURIComponent(r.filename)}`)}
      className="cursor-pointer hover:bg-gray-50 transition-colors"
    >
      <td className="px-3 py-2">
        <span className="text-green-500 text-sm">&#10003;</span>
      </td>
      <td
        className={`px-3 py-2 font-medium text-gray-700 whitespace-nowrap ${stickyCls}`}
        style={stickyStyle}
      >
        {truncate(r.run_name ?? r.phase_name, 28)}
      </td>
      <td className="px-3 py-2">
        <span className="inline-block rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
          bench
        </span>
      </td>
      <td className="px-3 py-2 text-gray-500 whitespace-nowrap">
        {formatTimestamp(r.timestamp)}
      </td>
      <td className="px-3 py-2 font-mono text-xs text-gray-600">
        {r.config_summary?.dataset_name ?? "\u2014"}
      </td>
      <td className="px-3 py-2 text-xs text-gray-500">
        {r.eval_dataset_name ?? "\u2014"}
      </td>
      <td className="px-3 py-2">
        {r.config_summary?.technique ? (
          <span className="inline-block rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
            {r.config_summary.technique}
          </span>
        ) : (
          "\u2014"
        )}
      </td>
      <ParamCell value={r.config_summary?.chunk_size} />
      <ParamCell value={r.config_summary?.chunk_overlap} />
      <ParamCell value={r.config_summary?.embedding_model} format="model" />
      <ParamCell value={r.config_summary?.distance_metric} />
      <ParamCell value={r.config_summary?.backend} />
      <ParamCell value={r.config_summary?.top_k} />
      <ParamCell value={r.config_summary?.reranker_type} />
      <ParamCell value={r.config_summary?.reranker_model} format="truncate" />
      <ParamCell value={r.config_summary?.rerank_top_k} />
      <ParamCell value={r.config_summary?.sparse_weight} format="float" />
      <ParamCell value={r.config_summary?.sparse_type} />
      <ParamCell value={r.config_summary?.fusion_method} />
      <td className="px-3 py-2 text-right text-gray-600">{r.total_questions}</td>
      <td className="px-3 py-2 text-right font-mono text-gray-700">{fmt(r.avg_mrr)}</td>
      <td className="px-3 py-2 text-right font-mono text-gray-700">
        {fmt(r.avg_recall_at_5)}
      </td>
      <td className="px-3 py-2 text-right font-mono text-gray-700">
        {fmt(r.avg_recall_at_10)}
      </td>
      <td className="px-3 py-2 text-right font-mono text-gray-700">
        {fmt(r.avg_doc_precision_at_5)}
      </td>
      <td className="px-3 py-2 text-right font-mono text-gray-700">
        {fmt(r.avg_doc_mrr)}
      </td>
      <td className="px-3 py-2 text-right font-mono text-gray-700">
        {fmt(r.avg_context_precision)}
      </td>
      <td className="px-3 py-2 text-right text-gray-500">
        {fmtTime(r.total_wall_time_s)}
      </td>
    </tr>
  );
}

/* ------------------------------------------------------------------ */
/* Sweep parent row                                                    */
/* ------------------------------------------------------------------ */

function SweepParentRow({
  r,
  childCount,
  expanded,
  onToggle,
  getSweptValues,
}: {
  r: ResultFileInfo;
  childCount: number;
  expanded: boolean;
  onToggle: () => void;
  getSweptValues: (r: ResultFileInfo, key: string) => unknown[] | undefined;
}) {
  const cs = r.config_summary;

  return (
    <tr
      onClick={onToggle}
      className="cursor-pointer hover:bg-gray-50 transition-colors bg-white"
    >
      <td className="px-3 py-2">
        <ChevronIcon expanded={expanded} />
      </td>
      <td
        className={`px-3 py-2 font-medium text-gray-700 whitespace-nowrap ${stickyCls}`}
        style={stickyStyle}
      >
        <div>
          {truncate(r.run_name ?? r.phase_name, 28)}
          <div className="text-xs text-gray-400 font-normal">{childCount} configs</div>
        </div>
      </td>
      <td className="px-3 py-2">
        <span className="inline-block rounded-full bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-700">
          sweep
        </span>
      </td>
      <td className="px-3 py-2 text-gray-500 whitespace-nowrap">
        {formatTimestamp(r.timestamp)}
      </td>
      <td className="px-3 py-2 font-mono text-xs text-gray-600">
        {cs?.dataset_name ?? "\u2014"}
      </td>
      <td className="px-3 py-2 text-xs text-gray-500">
        {r.eval_dataset_name ?? "\u2014"}
      </td>
      <td className="px-3 py-2">
        {cs?.technique ? (
          <span className="inline-block rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
            {cs.technique}
          </span>
        ) : (
          "\u2014"
        )}
      </td>
      <ParamCell value={cs?.chunk_size} sweptValues={getSweptValues(r, "chunking.chunk_size")} />
      <ParamCell value={cs?.chunk_overlap} sweptValues={getSweptValues(r, "chunking.chunk_overlap")} />
      <ParamCell value={cs?.embedding_model} format="model" sweptValues={getSweptValues(r, "embedding.model_name")} />
      <ParamCell value={cs?.distance_metric} sweptValues={getSweptValues(r, "vector_db.distance_metric")} />
      <ParamCell value={cs?.backend} sweptValues={getSweptValues(r, "vector_db.backend")} />
      <ParamCell value={cs?.top_k} sweptValues={getSweptValues(r, "retrieval.top_k")} />
      <ParamCell value={cs?.reranker_type} sweptValues={getSweptValues(r, "reranker.type")} />
      <ParamCell value={cs?.reranker_model} format="truncate" sweptValues={getSweptValues(r, "reranker.model_name")} />
      <ParamCell value={cs?.rerank_top_k} sweptValues={getSweptValues(r, "retrieval.rerank_k")} />
      <ParamCell value={cs?.sparse_weight} format="float" sweptValues={getSweptValues(r, "retrieval.sparse_weight")} />
      <ParamCell value={cs?.sparse_type} sweptValues={getSweptValues(r, "retrieval.sparse_type")} />
      <ParamCell value={cs?.fusion_method} sweptValues={getSweptValues(r, "retrieval.fusion_method")} />
      {/* Metrics show dashes for sweep parent */}
      <td className="px-3 py-2 text-right text-gray-400">{"\u2014"}</td>
      <td className="px-3 py-2 text-right font-mono text-gray-400">---</td>
      <td className="px-3 py-2 text-right font-mono text-gray-400">---</td>
      <td className="px-3 py-2 text-right font-mono text-gray-400">---</td>
      <td className="px-3 py-2 text-right font-mono text-gray-400">{"\u2014"}</td>
      <td className="px-3 py-2 text-right font-mono text-gray-400">{"\u2014"}</td>
      <td className="px-3 py-2 text-right font-mono text-gray-400">{"\u2014"}</td>
      <td className="px-3 py-2 text-right text-gray-400">---</td>
    </tr>
  );
}

/* ------------------------------------------------------------------ */
/* Sweep child row                                                     */
/* ------------------------------------------------------------------ */

function TrophyIcon() {
  return (
    <svg className="w-3.5 h-3.5 text-green-500 inline-block ml-0.5" fill="currentColor" viewBox="0 0 20 20">
      <path d="M10 1a1 1 0 0 1 1 1v1h3a1 1 0 0 1 1 1v2a4 4 0 0 1-3.071 3.886A4.992 4.992 0 0 1 10 12a4.992 4.992 0 0 1-1.929-1.114A4 4 0 0 1 5 7V4a1 1 0 0 1 1-1h3V2a1 1 0 0 1 1-1zm-4 4v2a2 2 0 0 0 1.298 1.87A5.025 5.025 0 0 1 7 7.5V5H6zm8 0h-1v2.5a5.025 5.025 0 0 1-.298 1.37A2 2 0 0 0 14 7V5zM8 14h4v1H8v-1zm-1 2a1 1 0 0 0-1 1v1h8v-1a1 1 0 0 0-1-1H7z" />
    </svg>
  );
}

function SweepChildRow({
  r,
  childIndex,
  isLast,
  isBest,
  navigate,
}: {
  r: ResultFileInfo;
  childIndex: number;
  isLast: boolean;
  isBest: boolean;
  navigate: ReturnType<typeof useNavigate>;
}) {
  const bestBg = isBest ? "bg-green-50/50" : "bg-gray-50/50";
  const bestStickyBg = isBest ? "!bg-green-50/80" : "!bg-gray-50/80";

  return (
    <tr
      onClick={() => navigate(`/benchmarks?file=${encodeURIComponent(r.filename)}`)}
      className={`cursor-pointer hover:bg-gray-100/50 transition-colors ${bestBg} ${isLast ? "border-b-2 border-gray-200" : ""}`}
    >
      <td className="px-3 py-2 text-center">
        <span className="text-xs text-gray-400 font-mono">
          #{childIndex + 1}
          {isBest && <TrophyIcon />}
        </span>
      </td>
      <td
        className={`px-3 py-2 text-gray-500 text-sm whitespace-nowrap ${stickyCls} ${bestStickyBg}`}
        style={stickyStyle}
      >
        <span className="pl-3">{truncate(r.phase_name, 24)}</span>
      </td>
      <td className="px-3 py-2"></td>
      <td className="px-3 py-2 text-gray-500 whitespace-nowrap">
        {formatTimestamp(r.timestamp)}
      </td>
      <td className="px-3 py-2 font-mono text-xs text-gray-600">
        {r.config_summary?.dataset_name ?? "\u2014"}
      </td>
      <td className="px-3 py-2 text-xs text-gray-500">
        {r.eval_dataset_name ?? "\u2014"}
      </td>
      <td className="px-3 py-2">
        {r.config_summary?.technique ? (
          <span className="inline-block rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
            {r.config_summary.technique}
          </span>
        ) : (
          "\u2014"
        )}
      </td>
      <ParamCell value={r.config_summary?.chunk_size} />
      <ParamCell value={r.config_summary?.chunk_overlap} />
      <ParamCell value={r.config_summary?.embedding_model} format="model" />
      <ParamCell value={r.config_summary?.distance_metric} />
      <ParamCell value={r.config_summary?.backend} />
      <ParamCell value={r.config_summary?.top_k} />
      <ParamCell value={r.config_summary?.reranker_type} />
      <ParamCell value={r.config_summary?.reranker_model} format="truncate" />
      <ParamCell value={r.config_summary?.rerank_top_k} />
      <ParamCell value={r.config_summary?.sparse_weight} format="float" />
      <ParamCell value={r.config_summary?.sparse_type} />
      <ParamCell value={r.config_summary?.fusion_method} />
      <td className="px-3 py-2 text-right text-gray-600">{r.total_questions}</td>
      <td className={`px-3 py-2 text-right font-mono ${isBest ? "text-green-600 font-semibold" : "text-gray-700"}`}>
        {fmt(r.avg_mrr)}
      </td>
      <td className={`px-3 py-2 text-right font-mono ${isBest ? "text-green-600 font-semibold" : "text-gray-700"}`}>
        {fmt(r.avg_recall_at_5)}
      </td>
      <td className={`px-3 py-2 text-right font-mono ${isBest ? "text-green-600 font-semibold" : "text-gray-700"}`}>
        {fmt(r.avg_recall_at_10)}
      </td>
      <td className={`px-3 py-2 text-right font-mono ${isBest ? "text-green-600 font-semibold" : "text-gray-700"}`}>
        {fmt(r.avg_doc_precision_at_5)}
      </td>
      <td className={`px-3 py-2 text-right font-mono ${isBest ? "text-green-600 font-semibold" : "text-gray-700"}`}>
        {fmt(r.avg_doc_mrr)}
      </td>
      <td className={`px-3 py-2 text-right font-mono ${isBest ? "text-green-600 font-semibold" : "text-gray-700"}`}>
        {fmt(r.avg_context_precision)}
      </td>
      <td className={`px-3 py-2 text-right ${isBest ? "text-green-600 font-semibold" : "text-gray-500"}`}>
        {fmtTime(r.total_wall_time_s)}
      </td>
    </tr>
  );
}
