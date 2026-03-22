import { Link } from "react-router-dom";
import type { DatasetInfo, SweepConfigCompleteEvent } from "../../api/types";
import type { BenchPhase, SweepPhase, ActiveRun, SweepProgress } from "./types";

export interface ExecutionPanelProps {
  mode: "benchmark" | "sweep";
  filteredDatasets: DatasetInfo[];
  selectedDatasetId: string;
  onDatasetChange: (id: string) => void;
  runName: string;
  onRunNameChange: (name: string) => void;
  isRunning: boolean;
  onRun: () => void;
  onCancel: () => void;
  disabled: boolean;
  // Benchmark-specific
  benchPhase?: BenchPhase;
  activeRun?: ActiveRun | null;
  // Sweep-specific
  combinationCount?: number;
  sweepPhase?: SweepPhase;
  sweepProgress?: SweepProgress | null;
  sweepResults?: SweepConfigCompleteEvent[];
}

export default function ExecutionPanel({
  mode,
  filteredDatasets,
  selectedDatasetId,
  onDatasetChange,
  runName,
  onRunNameChange,
  isRunning,
  onRun,
  onCancel,
  disabled,
  benchPhase,
  activeRun,
  combinationCount = 1,
  sweepPhase,
  sweepProgress,
  sweepResults = [],
}: ExecutionPanelProps) {
  const isBenchmark = mode === "benchmark";

  const buttonText = isBenchmark
    ? "Run Benchmark"
    : `Run Sweep (${combinationCount} config${combinationCount !== 1 ? "s" : ""})`;

  return (
    <div className="rounded border border-gray-200 bg-white p-4">
      {/* ── Header row: dataset + run name + button ── */}
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <label className="block text-xs font-medium text-gray-500 mb-1">
            Eval Dataset
          </label>
          <select
            value={selectedDatasetId}
            onChange={(e) => onDatasetChange(e.target.value)}
            className="w-full rounded border-gray-300 bg-white px-3 py-2 text-sm shadow-sm
                       focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            disabled={isRunning}
          >
            <option value="">Select eval dataset...</option>
            {filteredDatasets.map((ds) => (
              <option key={ds.id} value={ds.id}>
                {ds.name} ({ds.total_questions} questions)
              </option>
            ))}
          </select>
        </div>

        <div className="flex-1">
          <label className="block text-xs font-medium text-gray-500 mb-1">
            Run Name <span className="text-gray-400">(optional)</span>
          </label>
          <input
            type="text"
            value={runName}
            onChange={(e) => onRunNameChange(e.target.value)}
            placeholder="Auto-generated if empty"
            className="w-full rounded border-gray-300 bg-white px-3 py-2 text-sm shadow-sm
                       focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            disabled={isRunning}
          />
        </div>

        <div className="shrink-0 pt-5">
          {isRunning ? (
            <button
              onClick={onCancel}
              className="rounded bg-red-600 px-5 py-2 text-sm font-medium text-white
                         hover:bg-red-700 transition-colors"
            >
              Cancel
            </button>
          ) : (
            <button
              onClick={onRun}
              disabled={disabled}
              className="rounded bg-blue-600 px-5 py-2 text-sm font-medium text-white
                         hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                         transition-colors"
            >
              {buttonText}
            </button>
          )}
        </div>
      </div>

      {/* ── Benchmark progress ── */}
      {isBenchmark && activeRun && activeRun.status === "running" && activeRun.progress.total > 0 && (
        <div className="mt-3">
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-300"
                style={{
                  width: `${(activeRun.progress.current / activeRun.progress.total) * 100}%`,
                }}
              />
            </div>
            <span className="text-xs text-gray-500 whitespace-nowrap font-mono">
              {activeRun.progress.current}/{activeRun.progress.total}
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            {benchPhase === "ensuring"
              ? "Preparing collection..."
              : `Evaluating questions... (${Math.round(
                  (activeRun.progress.current / activeRun.progress.total) * 100
                )}%)`}
          </p>
        </div>
      )}

      {isBenchmark && benchPhase === "ensuring" && (
        <div className="mt-3 flex items-center gap-2">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
          <span className="text-sm text-gray-500">Preparing collection...</span>
        </div>
      )}

      {/* ── Sweep two-level progress ── */}
      {!isBenchmark && sweepProgress && isRunning && (
        <div className="mt-3 space-y-2">
          {/* Config-level */}
          <div>
            <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
              <span>Configs</span>
              <span className="font-mono">
                {sweepResults.length}/{sweepProgress.totalConfigs}
              </span>
            </div>
            <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-600 rounded-full transition-all duration-300"
                style={{
                  width: `${(sweepResults.length / sweepProgress.totalConfigs) * 100}%`,
                }}
              />
            </div>
          </div>
          {/* Question-level */}
          <div>
            <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
              <span>
                Config {sweepProgress.configIndex} of {sweepProgress.totalConfigs}
              </span>
              <span className="font-mono">
                {sweepProgress.questionIndex}/{sweepProgress.totalQuestions}
              </span>
            </div>
            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-400 rounded-full transition-all duration-300"
                style={{
                  width: sweepProgress.totalQuestions > 0
                    ? `${(sweepProgress.questionIndex / sweepProgress.totalQuestions) * 100}%`
                    : "0%",
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* ── Sweep completed configs table ── */}
      {!isBenchmark && sweepResults.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b text-gray-500 text-left">
                <th className="pb-1.5 pr-2">#</th>
                <th className="pb-1.5 pr-2">Params</th>
                <th className="pb-1.5 pr-2 text-right">Doc MRR</th>
                <th className="pb-1.5 pr-2 text-right">Ctx Prec</th>
                <th className="pb-1.5 pr-2 text-right">Time</th>
                <th className="pb-1.5">Status</th>
              </tr>
            </thead>
            <tbody>
              {sweepResults.map((r) => (
                <tr key={r.config_index} className="border-b border-gray-50">
                  <td className="py-1 pr-2 text-gray-400">{r.config_index}</td>
                  <td className="py-1 pr-2 font-mono truncate max-w-[220px]" title={
                    Object.entries(r.params)
                      .map(([k, v]) => `${k.split(".").pop()}=${v}`)
                      .join(", ")
                  }>
                    {Object.entries(r.params)
                      .map(([k, v]) => `${k.split(".").pop()}=${v}`)
                      .join(", ")}
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {r.status === "ok" ? (r.avg_doc_mrr?.toFixed(4) ?? "\u2014") : "\u2014"}
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {r.status === "ok" ? (r.avg_context_precision?.toFixed(4) ?? "\u2014") : "\u2014"}
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {r.status === "ok" ? `${r.total_wall_time_s?.toFixed(1)}s` : "\u2014"}
                  </td>
                  <td className="py-1">
                    {r.status === "ok" ? (
                      <span className="text-green-600 font-medium">OK</span>
                    ) : (
                      <span className="text-red-500" title={r.error}>ERR</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Completion message ── */}
      {isBenchmark && benchPhase === "complete" && (
        <p className="mt-3 text-sm text-green-600">
          Benchmark completed. View results in{" "}
          <Link to="/runs" className="underline hover:text-green-700">
            Run History
          </Link>.
        </p>
      )}

      {!isBenchmark && sweepPhase === "complete" && (
        <p className="mt-3 text-sm text-green-600">
          Sweep completed ({sweepResults.filter((r) => r.status === "ok").length}/
          {sweepResults.length} configs succeeded). View results in{" "}
          <Link to="/runs" className="underline hover:text-green-700">
            Run History
          </Link>.
        </p>
      )}
    </div>
  );
}
