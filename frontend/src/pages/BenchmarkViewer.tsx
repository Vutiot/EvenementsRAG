import { useEffect, useState, useCallback } from "react";
import type { NormalizedBenchmarkResult, ResultFileInfo, BenchmarkConfig } from "../api/types";
import { getResultFiles, getResult } from "../api/client";
import PageHeader from "../components/layout/PageHeader";
import ConfigSummary from "../components/config/ConfigSummary";
import ResultFileSelector from "../components/benchmarks/ResultFileSelector";
import ResultSummaryCards from "../components/benchmarks/ResultSummaryCards";
import MetricsByTypeChart from "../components/benchmarks/MetricsByTypeChart";
import QuestionExplorer from "../components/benchmarks/QuestionExplorer";
import RagasMetricsGrid from "../components/benchmarks/RagasMetricsGrid";
import LatencyBreakdown from "../components/results/LatencyBreakdown";

export default function BenchmarkViewer() {
  const [files, setFiles] = useState<ResultFileInfo[]>([]);
  const [filesLoading, setFilesLoading] = useState(true);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [result, setResult] = useState<NormalizedBenchmarkResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getResultFiles()
      .then(setFiles)
      .catch((e) => setError(e.message))
      .finally(() => setFilesLoading(false));
  }, []);

  const handleSelect = useCallback((filename: string) => {
    setSelectedFile(filename);
    setLoading(true);
    setError(null);
    getResult(filename)
      .then(setResult)
      .catch((e) => {
        setError(e.message);
        setResult(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const latency = result?.metrics_summary?.latency as
    | Record<string, number>
    | undefined;

  const ragasAgg = result?.metrics_summary?.ragas as
    | Record<string, unknown>
    | undefined;
  const ragasAvgs: Record<string, number> | null = ragasAgg
    ? (Object.fromEntries(
        Object.entries(ragasAgg).filter(
          ([k, v]) => k.startsWith("avg_") && typeof v === "number",
        ),
      ) as Record<string, number>)
    : null;

  return (
    <div className="flex h-full">
      {/* Left sidebar: file selector */}
      <aside className="w-72 shrink-0 border-r border-slate-200 bg-slate-50/30 p-3 overflow-y-auto">
        <ResultFileSelector
          files={files}
          selectedFile={selectedFile}
          onSelect={handleSelect}
          loading={filesLoading}
        />

        {/* Config summary for new-format results */}
        {result?.config && (
          <div className="mt-4">
            <h3 className="mb-2 px-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Run Configuration
            </h3>
            <ConfigSummary config={result.config as unknown as BenchmarkConfig} />
          </div>
        )}
      </aside>

      {/* Main content area */}
      <main className="flex-1 overflow-y-auto p-6">
        <PageHeader
          title="Benchmark Result Viewer"
          description="Browse and compare saved benchmark results."
        />

        {error && (
          <div className="mb-4 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-20">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-300 border-t-amber-500" />
          </div>
        )}

        {!loading && !result && !error && (
          <div className="rounded border border-slate-200 bg-white p-12 text-center text-sm text-slate-400">
            Select a result file from the left panel to begin exploring.
          </div>
        )}

        {!loading && result && (
          <div className="space-y-5">
            {/* Header badge */}
            <div className="flex items-center gap-3">
              <h2 className="font-mono text-sm font-medium text-slate-700">
                {result.filename}
              </h2>
              <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500">
                {result.format}
              </span>
              {result.timestamp && (
                <span className="text-[10px] text-slate-400">
                  {new Date(result.timestamp).toLocaleString()}
                </span>
              )}
            </div>

            {/* Summary cards */}
            <ResultSummaryCards result={result} />

            {/* Metrics by type chart */}
            <MetricsByTypeChart metricsByType={result.metrics_by_type} />

            {/* Latency summary (new format only) */}
            {latency && (
              <div className="rounded border border-slate-200 bg-white p-4">
                <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Latency Summary (p50)
                </h3>
                <LatencyBreakdown
                  retrievalMs={latency.retrieval_p50_ms ?? result.avg_retrieval_time_ms}
                  generationMs={latency.generation_p50_ms ?? 0}
                />
              </div>
            )}

            {/* Aggregated RAGAS metrics (new format only) */}
            {ragasAvgs && Object.keys(ragasAvgs).length > 0 && (
              <div className="rounded border border-slate-200 bg-white p-4">
                <RagasMetricsGrid metrics={ragasAvgs} title="Aggregated RAGAS Metrics" />
              </div>
            )}

            {/* Question explorer */}
            <QuestionExplorer
              questions={result.per_question}
              isLegacy={result.format === "legacy"}
            />
          </div>
        )}
      </main>
    </div>
  );
}
