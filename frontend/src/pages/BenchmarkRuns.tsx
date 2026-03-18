import { useState, useCallback, useEffect, useMemo } from "react";
import PageHeader from "../components/layout/PageHeader";
import PresetSelector from "../components/config/PresetSelector";
import ConfigSummary from "../components/config/ConfigSummary";
import ParameterModal from "../components/config/ParameterModal";
import RunHistoryTable from "../components/benchmarks/RunHistoryTable";
import {
  getPresetConfig,
  getDatasets,
  getDatasetRegistry,
  getResultFiles,
  ensureCollection,
  runBenchmark,
} from "../api/client";
import type {
  BenchmarkConfig,
  DatasetInfo,
  DatasetRegistryEntry,
  EnsureCollectionRequest,
  ResultFileInfo,
} from "../api/types";
import { deepMerge, setOverridePath, countOverrides } from "../utils/configHelpers";

type RunPhase = "idle" | "ensuring" | "running" | "complete";

interface ActiveRun {
  status: "running" | "complete" | "error";
  progress: { current: number; total: number };
  error?: string;
}

export default function BenchmarkRuns() {
  // Config state
  const [preset, setPreset] = useState("");
  const [baseConfig, setBaseConfig] = useState<BenchmarkConfig | null>(null);
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [paramsOpen, setParamsOpen] = useState(false);

  // Dataset state
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [registryMap, setRegistryMap] = useState<Record<string, DatasetRegistryEntry>>({});

  // Run state
  const [phase, setPhase] = useState<RunPhase>("idle");
  const [activeRun, setActiveRun] = useState<ActiveRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  // History
  const [results, setResults] = useState<ResultFileInfo[]>([]);

  const overrideCount = useMemo(() => countOverrides(overrides), [overrides]);

  const effectiveConfig = useMemo(() => {
    if (!baseConfig) return null;
    if (overrideCount === 0) return baseConfig;
    return deepMerge(
      baseConfig as unknown as Record<string, unknown>,
      overrides,
    ) as unknown as BenchmarkConfig;
  }, [baseConfig, overrides, overrideCount]);

  // Load datasets, registry, and results on mount
  useEffect(() => {
    getDatasets()
      .then((r) => setDatasets(r.datasets.filter((d) => d.status === "completed")))
      .catch(() => {});
    getDatasetRegistry()
      .then((r) => {
        const map: Record<string, DatasetRegistryEntry> = {};
        for (const d of r.datasets) map[d.name] = d;
        setRegistryMap(map);
      })
      .catch(() => {});
    loadResults();
  }, []);

  const loadResults = useCallback(() => {
    getResultFiles()
      .then((files) => {
        // Sort by timestamp descending (newest first)
        const sorted = [...files].sort((a, b) => {
          if (!a.timestamp && !b.timestamp) return 0;
          if (!a.timestamp) return 1;
          if (!b.timestamp) return -1;
          return b.timestamp.localeCompare(a.timestamp);
        });
        setResults(sorted);
      })
      .catch(() => {});
  }, []);

  // Filter eval datasets by the currently-selected raw dataset
  const currentDatasetName = effectiveConfig?.dataset.dataset_name;
  const registryEntry = currentDatasetName ? registryMap[currentDatasetName] : null;
  const filteredDatasets = registryEntry
    ? datasets.filter((ds) => registryEntry.collections.includes(ds.collection_name))
    : datasets;

  // Reset eval selection when filtered list changes
  useEffect(() => {
    if (selectedDatasetId && !filteredDatasets.some((ds) => ds.id === selectedDatasetId)) {
      setSelectedDatasetId("");
    }
  }, [filteredDatasets, selectedDatasetId]);

  const handlePresetChange = useCallback(async (filename: string) => {
    setPreset(filename);
    setError(null);
    setOverrides({});
    if (!filename) {
      setBaseConfig(null);
      return;
    }
    try {
      const cfg = await getPresetConfig(filename);
      setBaseConfig(cfg);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setBaseConfig(null);
    }
  }, []);

  const handleOverrideChange = useCallback((path: string, value: unknown) => {
    setOverrides((prev) => setOverridePath(prev, path, value));
  }, []);

  const handleResetOverrides = useCallback(() => {
    setOverrides({});
  }, []);

  const handleRunBenchmark = useCallback(async () => {
    if (!preset || !effectiveConfig || !selectedDatasetId) return;

    setPhase("ensuring");
    setError(null);
    setActiveRun({ status: "running", progress: { current: 0, total: 0 } });

    let finalOverrides: Record<string, unknown> =
      overrideCount > 0 ? { ...overrides } : {};

    try {
      // Ensure collection exists
      const ec = effectiveConfig;
      const req: EnsureCollectionRequest = {
        dataset_name: ec.dataset.dataset_name,
        backend: ec.vector_db.backend,
        chunk_size: ec.chunking.chunk_size,
        chunk_overlap: ec.chunking.chunk_overlap,
        embedding_model: ec.embedding.model_name,
        embedding_dimension: ec.embedding.dimension,
        distance_metric: ec.vector_db.distance_metric,
      };
      const ensureRes = await ensureCollection(req);
      finalOverrides = {
        ...finalOverrides,
        dataset: {
          ...((finalOverrides.dataset as Record<string, unknown>) ?? {}),
          collection_name: ensureRes.collection_name,
        },
      };
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
      setActiveRun(null);
      return;
    }

    setPhase("running");

    const controller = runBenchmark(
      {
        preset,
        config_overrides: Object.keys(finalOverrides).length > 0 ? finalOverrides : null,
        eval_dataset_id: selectedDatasetId,
      },
      {
        onStarted: (e) => {
          setActiveRun({ status: "running", progress: { current: 0, total: e.total_questions } });
        },
        onProgress: (e) => {
          setActiveRun({
            status: "running",
            progress: { current: e.question_index, total: e.total_questions },
          });
        },
        onComplete: () => {
          setActiveRun(null);
          setPhase("complete");
          // Refresh results list
          loadResults();
        },
        onError: (msg) => {
          setActiveRun({ status: "error", progress: { current: 0, total: 0 }, error: msg });
          setError(msg);
          setPhase("idle");
        },
      },
    );

    setAbortController(controller);
  }, [preset, effectiveConfig, selectedDatasetId, overrides, overrideCount, loadResults]);

  const handleCancel = useCallback(() => {
    abortController?.abort();
    setPhase("idle");
    setActiveRun(null);
  }, [abortController]);

  const isRunning = phase === "ensuring" || phase === "running";

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <PageHeader
        title="Benchmark Runs"
        description="Run full benchmarks across eval datasets and compare results."
      />

      <div className="grid grid-cols-12 gap-6 mb-6">
        {/* Left: Config panel */}
        <div className="col-span-4 space-y-4">
          <PresetSelector selected={preset} onSelect={handlePresetChange} />

          {/* Parameters button + reset */}
          {baseConfig && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setParamsOpen(true)}
                className="flex items-center gap-1.5 border border-gray-300 rounded-lg px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 transition"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75" />
                </svg>
                Parameters
                {overrideCount > 0 && (
                  <span className="bg-amber-100 text-amber-700 rounded-full text-xs px-1.5 py-0.5 font-medium">
                    {overrideCount}
                  </span>
                )}
              </button>
              {overrideCount > 0 && (
                <button
                  onClick={handleResetOverrides}
                  className="text-xs text-amber-600 hover:text-amber-800 transition"
                >
                  Reset ({overrideCount})
                </button>
              )}
            </div>
          )}

          <ConfigSummary config={effectiveConfig} />
        </div>

        {/* Right: Run controls */}
        <div className="col-span-8 space-y-4">
          <div className="rounded border border-gray-200 bg-white p-4">
            <div className="flex items-center gap-3">
              {/* Eval dataset dropdown */}
              <div className="flex-1">
                <label className="block text-xs font-medium text-gray-500 mb-1">
                  Eval Dataset
                </label>
                <select
                  value={selectedDatasetId}
                  onChange={(e) => setSelectedDatasetId(e.target.value)}
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

              {/* Run / Cancel button */}
              <div className="shrink-0 pt-5">
                {isRunning ? (
                  <button
                    onClick={handleCancel}
                    className="rounded bg-red-600 px-5 py-2 text-sm font-medium text-white
                               hover:bg-red-700 transition-colors"
                  >
                    Cancel
                  </button>
                ) : (
                  <button
                    onClick={handleRunBenchmark}
                    disabled={!preset || !selectedDatasetId}
                    className="rounded bg-blue-600 px-5 py-2 text-sm font-medium text-white
                               hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                               transition-colors"
                  >
                    Run Benchmark
                  </button>
                )}
              </div>
            </div>

            {/* Progress bar */}
            {activeRun && activeRun.status === "running" && activeRun.progress.total > 0 && (
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
                  {phase === "ensuring"
                    ? "Preparing collection..."
                    : `Evaluating questions... (${Math.round(
                        (activeRun.progress.current / activeRun.progress.total) * 100
                      )}%)`}
                </p>
              </div>
            )}

            {phase === "ensuring" && (
              <div className="mt-3 flex items-center gap-2">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
                <span className="text-sm text-gray-500">Preparing collection...</span>
              </div>
            )}

            {phase === "complete" && (
              <p className="mt-3 text-sm text-green-600">
                Benchmark completed. Results saved and added to history below.
              </p>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Run history table */}
      <div>
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400 mb-3">
          Run History
        </h2>
        <RunHistoryTable
          results={results}
          activeRun={activeRun}
        />
      </div>

      {/* Parameter tuning modal */}
      {baseConfig && (
        <ParameterModal
          open={paramsOpen}
          onClose={() => setParamsOpen(false)}
          baseConfig={baseConfig}
          overrides={overrides}
          onOverrideChange={handleOverrideChange}
          onReset={handleResetOverrides}
          hideSections={new Set(["Results"])}
        />
      )}
    </div>
  );
}
