/** Table showing past benchmark run results + an optional active run placeholder. */

import { useNavigate } from "react-router-dom";
import type { ResultFileInfo } from "../../api/types";

interface ActiveRun {
  status: "running" | "complete" | "error";
  progress: { current: number; total: number };
  error?: string;
}

interface Props {
  results: ResultFileInfo[];
  activeRun: ActiveRun | null;
}

function fmt(v: number | null | undefined, digits = 3): string {
  if (v == null) return "—";
  return v.toFixed(digits);
}

function fmtTime(s: number | null | undefined): string {
  if (s == null) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s`;
}

function truncate(s: string | null | undefined, maxLen = 20): string {
  if (!s) return "—";
  return s.length > maxLen ? s.slice(0, maxLen) + "..." : s;
}

function formatTimestamp(ts: string | null | undefined): string {
  if (!ts) return "—";
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

export default function RunHistoryTable({ results, activeRun }: Props) {
  const navigate = useNavigate();

  return (
    <div className="rounded border border-gray-200 bg-white overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-100 text-left text-xs font-medium uppercase tracking-wider text-gray-400">
            <th className="px-3 py-2 w-8"></th>
            <th className="px-3 py-2">Timestamp</th>
            <th className="px-3 py-2">Name</th>
            <th className="px-3 py-2">Dataset</th>
            <th className="px-3 py-2">Technique</th>
            <th className="px-3 py-2">Chunk</th>
            <th className="px-3 py-2">Top K</th>
            <th className="px-3 py-2">LLM</th>
            <th className="px-3 py-2 text-right">Qs</th>
            <th className="px-3 py-2 text-right">MRR</th>
            <th className="px-3 py-2 text-right">R@5</th>
            <th className="px-3 py-2 text-right">R@10</th>
            <th className="px-3 py-2 text-right">Time</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-50">
          {/* Active run placeholder row */}
          {activeRun && activeRun.status === "running" && (
            <tr className="bg-blue-50/50">
              <td className="px-3 py-2">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
              </td>
              <td className="px-3 py-2 text-gray-500">Running...</td>
              <td className="px-3 py-2" colSpan={5}>
                <div className="flex items-center gap-2">
                  <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 rounded-full transition-all duration-300"
                      style={{
                        width: activeRun.progress.total > 0
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
              <td className="px-3 py-2" colSpan={5}></td>
            </tr>
          )}

          {activeRun && activeRun.status === "error" && (
            <tr className="bg-red-50/50">
              <td className="px-3 py-2">
                <span className="text-red-500 text-base">!</span>
              </td>
              <td className="px-3 py-2 text-red-600 text-xs" colSpan={12}>
                Error: {activeRun.error || "Unknown error"}
              </td>
            </tr>
          )}

          {/* Result rows */}
          {results.map((r) => (
            <tr
              key={r.filename}
              onClick={() => navigate(`/benchmarks?file=${encodeURIComponent(r.filename)}`)}
              className="cursor-pointer hover:bg-gray-50 transition-colors"
            >
              <td className="px-3 py-2">
                <span className="text-green-500 text-sm">&#10003;</span>
              </td>
              <td className="px-3 py-2 text-gray-500 whitespace-nowrap">
                {formatTimestamp(r.timestamp)}
              </td>
              <td className="px-3 py-2 font-medium text-gray-700">
                {truncate(r.phase_name, 24)}
              </td>
              <td className="px-3 py-2 text-gray-600">
                {r.config_summary?.dataset_name ?? "—"}
              </td>
              <td className="px-3 py-2">
                {r.config_summary?.technique ? (
                  <span className="inline-block rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
                    {r.config_summary.technique}
                  </span>
                ) : (
                  "—"
                )}
              </td>
              <td className="px-3 py-2 text-gray-600">
                {r.config_summary?.chunk_size ?? "—"}
              </td>
              <td className="px-3 py-2 text-gray-600">
                {r.config_summary?.top_k ?? "—"}
              </td>
              <td className="px-3 py-2 text-gray-500 text-xs">
                {truncate(r.config_summary?.llm_model, 18)}
              </td>
              <td className="px-3 py-2 text-right text-gray-600">
                {r.total_questions}
              </td>
              <td className="px-3 py-2 text-right font-mono text-gray-700">
                {fmt(r.avg_mrr)}
              </td>
              <td className="px-3 py-2 text-right font-mono text-gray-700">
                {fmt(r.avg_recall_at_5)}
              </td>
              <td className="px-3 py-2 text-right font-mono text-gray-700">
                {fmt(r.avg_recall_at_10)}
              </td>
              <td className="px-3 py-2 text-right text-gray-500">
                {fmtTime(r.total_wall_time_s)}
              </td>
            </tr>
          ))}

          {results.length === 0 && !activeRun && (
            <tr>
              <td colSpan={13} className="px-3 py-8 text-center text-gray-400">
                No benchmark results yet. Run a benchmark to get started.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
