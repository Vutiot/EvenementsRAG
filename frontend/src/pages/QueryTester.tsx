import { useState, useCallback } from "react";
import PageHeader from "../components/layout/PageHeader";
import PresetSelector from "../components/config/PresetSelector";
import ConfigSummary from "../components/config/ConfigSummary";
import ChunkList from "../components/results/ChunkList";
import GeneratedAnswer from "../components/results/GeneratedAnswer";
import LatencyBreakdown from "../components/results/LatencyBreakdown";
import ChunkScoresChart from "../components/results/ChunkScoresChart";
import { getPresetConfig, executeQuery } from "../api/client";
import type { BenchmarkConfig, QueryResult } from "../api/types";

export default function QueryTester() {
  const [preset, setPreset] = useState("");
  const [config, setConfig] = useState<BenchmarkConfig | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePresetChange = useCallback(async (filename: string) => {
    setPreset(filename);
    setResult(null);
    setError(null);
    if (!filename) {
      setConfig(null);
      return;
    }
    try {
      const cfg = await getPresetConfig(filename);
      setConfig(cfg);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setConfig(null);
    }
  }, []);

  const handleExecute = useCallback(async () => {
    if (!query.trim() || !preset) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await executeQuery(query, preset);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [query, preset]);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <PageHeader
        title="Query Tester"
        description="Test individual queries against the RAG system with different configurations."
      />

      <div className="grid grid-cols-12 gap-6">
        {/* Left: Config panel */}
        <div className="col-span-4 space-y-4">
          <PresetSelector selected={preset} onSelect={handlePresetChange} />
          <ConfigSummary config={config} />
        </div>

        {/* Right: Query + Results */}
        <div className="col-span-8 space-y-4">
          {/* Query input */}
          <div className="rounded border border-gray-200 bg-white p-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Query
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. What happened on D-Day?"
              rows={3}
              className="w-full rounded border-gray-300 px-3 py-2 text-sm shadow-sm
                         focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                         placeholder:text-gray-400"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleExecute();
              }}
            />
            <div className="mt-2 flex items-center justify-between">
              <span className="text-xs text-gray-400">Ctrl+Enter to execute</span>
              <button
                onClick={handleExecute}
                disabled={loading || !query.trim() || !preset}
                className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white
                           hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                           transition-colors"
              >
                {loading ? "Executing..." : "Execute"}
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {/* Loading spinner */}
          {loading && (
            <div className="flex items-center justify-center py-12">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <div className="space-y-4">
              <GeneratedAnswer answer={result.generated_answer} />
              <LatencyBreakdown
                retrievalMs={result.retrieval_time_ms}
                generationMs={result.generation_time_ms}
              />
              <ChunkScoresChart chunks={result.retrieved_chunks} />
              <ChunkList chunks={result.retrieved_chunks} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
