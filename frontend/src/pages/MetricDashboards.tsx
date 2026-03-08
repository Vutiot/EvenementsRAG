import { useEffect, useState, useCallback, useMemo } from "react";
import type { NormalizedBenchmarkResult, ResultFileInfo, BenchmarkConfig } from "../api/types";
import { getResultFiles, getResult } from "../api/client";
import PageHeader from "../components/layout/PageHeader";
import ConfigSummary from "../components/config/ConfigSummary";
import ResultFileSelector from "../components/benchmarks/ResultFileSelector";
import MetricTabs, { type TabKey } from "../components/metrics/MetricTabs";
import RetrievalTab from "../components/metrics/RetrievalTab";
import GenerationTab from "../components/metrics/GenerationTab";
import LatencyTab from "../components/metrics/LatencyTab";
import RagasTab from "../components/metrics/RagasTab";

export default function MetricDashboards() {
  const [files, setFiles] = useState<ResultFileInfo[]>([]);
  const [filesLoading, setFilesLoading] = useState(true);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [result, setResult] = useState<NormalizedBenchmarkResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabKey>("retrieval");

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
      .then((r) => {
        setResult(r);
        setActiveTab("retrieval");
      })
      .catch((e) => {
        setError(e.message);
        setResult(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const hasRetrieval = result != null && result.per_question.length > 0;
  const hasGeneration = useMemo(
    () => result?.per_question.some((q) => q.generation_metrics != null) ?? false,
    [result],
  );
  const hasLatency = useMemo(
    () => result?.per_question.some((q) => q.retrieval_time_ms != null) ?? false,
    [result],
  );
  const hasRagas = useMemo(
    () => result?.per_question.some((q) => q.ragas_metrics != null) ?? false,
    [result],
  );

  return (
    <div className="flex h-full">
      {/* Left sidebar */}
      <aside className="w-72 shrink-0 border-r border-slate-200 bg-slate-50/30 p-3 overflow-y-auto">
        <ResultFileSelector
          files={files}
          selectedFile={selectedFile}
          onSelect={handleSelect}
          loading={filesLoading}
        />
        {result?.config && (
          <div className="mt-4">
            <h3 className="mb-2 px-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Run Configuration
            </h3>
            <ConfigSummary config={result.config as unknown as BenchmarkConfig} />
          </div>
        )}
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6">
        <PageHeader
          title="Metric Dashboards"
          description="Detailed metric views by category: retrieval, generation, latency."
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
            Select a result file from the left panel to begin exploring metrics.
          </div>
        )}

        {!loading && result && (
          <div>
            {/* Header badge */}
            <div className="flex items-center gap-3 mb-4">
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

            {/* Tab bar */}
            <MetricTabs
              activeTab={activeTab}
              onTabChange={setActiveTab}
              hasRetrieval={hasRetrieval}
              hasGeneration={hasGeneration}
              hasLatency={hasLatency}
              hasRagas={hasRagas}
            />

            {/* Active tab content */}
            {activeTab === "retrieval" && <RetrievalTab result={result} />}
            {activeTab === "generation" && <GenerationTab result={result} />}
            {activeTab === "latency" && <LatencyTab result={result} />}
            {activeTab === "ragas" && <RagasTab result={result} />}
          </div>
        )}
      </main>
    </div>
  );
}
