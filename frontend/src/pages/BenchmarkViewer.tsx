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

/* ------------------------------------------------------------------ */
/* Precision Tiers section                                             */
/* ------------------------------------------------------------------ */

function PrecisionTiers({ result }: { result: NormalizedBenchmarkResult }) {
  const docP5 = result.avg_doc_precision_at_k?.["5"];
  const docMrr = result.avg_doc_mrr;
  const docRecall5 = result.avg_doc_recall_at_k?.["5"];
  const chunkP5 = result.avg_chunk_precision_at_k?.["5"];
  const chunkHit5 = result.avg_chunk_hit_at_k?.["5"];

  const ragasAgg = result.metrics_summary?.ragas as
    | Record<string, unknown>
    | undefined;
  const ctxPrec = ragasAgg?.avg_context_precision as number | undefined;

  const entityAgg = result.metrics_summary?.entity as
    | Record<string, unknown>
    | undefined;
  const entP5 = entityAgg?.avg_entity_precision_at_5 as number | undefined;
  const entR5 = entityAgg?.avg_entity_recall_at_5 as number | undefined;
  const entMrr = entityAgg?.avg_entity_mrr as number | undefined;

  // Only show if at least one tier has data
  const hasDocTier = docP5 != null || docMrr != null || docRecall5 != null;
  const hasChunkTier = chunkP5 != null || chunkHit5 != null;
  const hasCtxTier = ctxPrec != null;
  const hasEntityTier = entP5 != null || entR5 != null || entMrr != null;

  if (!hasDocTier && !hasChunkTier && !hasCtxTier && !hasEntityTier) return null;

  const fmtVal = (v: number | null | undefined): string =>
    v != null ? v.toFixed(4) : "\u2014";

  return (
    <div className="rounded border border-slate-200 bg-white p-4">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
        Precision Tiers
      </h3>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Tier 1: Document-level */}
        <div className="rounded border border-purple-100 bg-purple-50/30 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-purple-500 mb-2">
            Tier 1 &mdash; Document
          </p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Doc P@5</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(docP5)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Doc MRR</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(docMrr)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Doc R@5</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(docRecall5)}</span>
            </div>
          </div>
        </div>

        {/* Tier 2: Chunk-level */}
        <div className="rounded border border-blue-100 bg-blue-50/30 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-blue-500 mb-2">
            Tier 2 &mdash; Chunk
          </p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Chunk P@5</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(chunkP5)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Chunk Hit@5</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(chunkHit5)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">MRR</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(result.avg_mrr)}</span>
            </div>
          </div>
        </div>

        {/* Tier 3: Context (LLM-judged) */}
        <div className="rounded border border-amber-100 bg-amber-50/30 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-amber-600 mb-2">
            Tier 3 &mdash; Context (LLM)
          </p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Ctx Precision</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(ctxPrec)}</span>
            </div>
          </div>
        </div>

        {/* Tier 4: Entity-level (LLM NER) */}
        <div className="rounded border border-emerald-100 bg-emerald-50/30 p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-emerald-600 mb-2">
            Tier 4 &mdash; Entity (NER)
          </p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-500">Ent P@5</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(entP5)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Ent R@5</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(entR5)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Ent MRR</span>
              <span className="font-mono font-medium text-slate-800">{fmtVal(entMrr)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */

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

            {/* Precision Tiers */}
            <PrecisionTiers result={result} />

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
