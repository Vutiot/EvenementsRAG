import { useState, useCallback, useEffect, useMemo } from "react";
import PageHeader from "../components/layout/PageHeader";
import PresetSelector from "../components/config/PresetSelector";
import ConfigSummary from "../components/config/ConfigSummary";
import ParameterModal from "../components/config/ParameterModal";
import ChunkList from "../components/results/ChunkList";
import GeneratedAnswer from "../components/results/GeneratedAnswer";
import LatencyBreakdown from "../components/results/LatencyBreakdown";
import ChunkScoresChart from "../components/results/ChunkScoresChart";
import { getPresetConfig, executeQuery, ensureCollection, getDatasets, getDataset } from "../api/client";
import type { BenchmarkConfig, DatasetInfo, DatasetQuestion, EnsureCollectionRequest, QueryResult } from "../api/types";

// ── Helpers ──────────────────────────────────────────────────────────

/** Deep merge overrides into base (immutable — returns new object). */
function deepMerge(
  base: Record<string, unknown>,
  overrides: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...base };
  for (const key of Object.keys(overrides)) {
    const bv = base[key];
    const ov = overrides[key];
    if (
      bv != null &&
      ov != null &&
      typeof bv === "object" &&
      !Array.isArray(bv) &&
      typeof ov === "object" &&
      !Array.isArray(ov)
    ) {
      result[key] = deepMerge(
        bv as Record<string, unknown>,
        ov as Record<string, unknown>,
      );
    } else {
      result[key] = ov;
    }
  }
  return result;
}

/** Set a dotted path in a nested override object. Removes path if value is undefined. */
function setOverridePath(
  overrides: Record<string, unknown>,
  path: string,
  value: unknown,
): Record<string, unknown> {
  const parts = path.split(".");
  const head = parts[0] as string;
  if (parts.length === 1) {
    const next = { ...overrides };
    if (value === undefined) {
      delete next[head];
    } else {
      next[head] = value;
    }
    return next;
  }

  const rest = parts.slice(1).join(".");
  const child = (overrides[head] ?? {}) as Record<string, unknown>;
  const updated = setOverridePath(child, rest, value);

  const next = { ...overrides };
  // Remove empty sub-objects
  if (Object.keys(updated).length === 0) {
    delete next[head];
  } else {
    next[head] = updated;
  }
  return next;
}

/** Count leaf values in a nested override object. */
function countOverrides(obj: Record<string, unknown>): number {
  let count = 0;
  for (const v of Object.values(obj)) {
    if (v != null && typeof v === "object" && !Array.isArray(v)) {
      count += countOverrides(v as Record<string, unknown>);
    } else {
      count += 1;
    }
  }
  return count;
}

// ── Collection-affecting override detection ──────────────────────────

const COLLECTION_AFFECTING_PATHS = [
  "dataset.dataset_name",
  "chunking.chunk_size",
  "chunking.chunk_overlap",
  "embedding.model_name",
  "vector_db.distance_metric",
  "vector_db.backend",
];

/** Resolve a dotted path in a nested object. */
function getPath(obj: Record<string, unknown>, path: string): unknown {
  const parts = path.split(".");
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

function hasCollectionAffectingOverrides(
  overrides: Record<string, unknown>,
): boolean {
  return COLLECTION_AFFECTING_PATHS.some(
    (p) => getPath(overrides, p) !== undefined,
  );
}

type Phase = "idle" | "ensuring" | "querying";

// ── Component ────────────────────────────────────────────────────────

export default function QueryTester() {
  const [preset, setPreset] = useState("");
  const [baseConfig, setBaseConfig] = useState<BenchmarkConfig | null>(null);
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [paramsOpen, setParamsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Dataset selector state
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [datasetQuestions, setDatasetQuestions] = useState<DatasetQuestion[]>([]);

  const overrideCount = useMemo(() => countOverrides(overrides), [overrides]);

  const effectiveConfig = useMemo(() => {
    if (!baseConfig) return null;
    if (overrideCount === 0) return baseConfig;
    return deepMerge(
      baseConfig as unknown as Record<string, unknown>,
      overrides,
    ) as unknown as BenchmarkConfig;
  }, [baseConfig, overrides, overrideCount]);

  // Load datasets on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    getDatasets()
      .then((r) => setDatasets(r.datasets.filter((d) => d.status === "completed")))
      .catch(() => {});
  }, []);

  const handleDatasetChange = useCallback(async (dsId: string) => {
    setSelectedDatasetId(dsId);
    setDatasetQuestions([]);
    if (!dsId) return;
    try {
      const detail = await getDataset(dsId);
      setDatasetQuestions(detail.questions);
    } catch {
      /* ignore */
    }
  }, []);

  const handlePickQuestion = useCallback((q: DatasetQuestion) => {
    setQuery(q.question);
  }, []);

  const handlePresetChange = useCallback(async (filename: string) => {
    setPreset(filename);
    setResult(null);
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

  const handleExecute = useCallback(async () => {
    if (!query.trim() || !preset || !effectiveConfig) return;
    setPhase("ensuring");
    setError(null);
    setResult(null);

    let finalOverrides = overrideCount > 0 ? { ...overrides } : undefined;

    try {
      // Phase 1: ensure collection exists if collection-affecting params changed
      if (overrideCount > 0 && hasCollectionAffectingOverrides(overrides)) {
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
        // Inject the derived collection_name into overrides
        finalOverrides = {
          ...(finalOverrides ?? {}),
          dataset: {
            ...((finalOverrides?.dataset as Record<string, unknown>) ?? {}),
            collection_name: ensureRes.collection_name,
          },
        };
      }

      // Phase 2: execute query
      setPhase("querying");
      const res = await executeQuery(query, preset, finalOverrides);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPhase("idle");
    }
  }, [query, preset, overrides, overrideCount, effectiveConfig]);

  const handleOverrideChange = useCallback((path: string, value: unknown) => {
    setOverrides((prev) => setOverridePath(prev, path, value));
  }, []);

  const handleResetOverrides = useCallback(() => {
    setOverrides({});
  }, []);

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

          {/* Dataset selector */}
          <div className="rounded border border-gray-200 bg-white p-3">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dataset (optional)
            </label>
            <select
              value={selectedDatasetId}
              onChange={(e) => handleDatasetChange(e.target.value)}
              className="w-full rounded border-gray-300 bg-white px-3 py-1.5 text-sm shadow-sm
                         focus:border-blue-500 focus:ring-1 focus:ring-blue-500 mb-2"
            >
              <option value="">No dataset</option>
              {datasets.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} ({ds.total_questions}q)
                </option>
              ))}
            </select>
            {datasetQuestions.length > 0 && (
              <div className="max-h-48 overflow-y-auto space-y-1">
                {datasetQuestions.map((q) => (
                  <button
                    key={q.id}
                    onClick={() => handlePickQuestion(q)}
                    className="w-full text-left px-2 py-1.5 rounded text-xs text-gray-700
                               hover:bg-blue-50 hover:text-blue-700 transition-colors truncate"
                    title={q.question}
                  >
                    <span className="inline-block rounded px-1 py-0.5 mr-1 text-xs font-medium bg-gray-100 text-gray-500">
                      {q.type}
                    </span>
                    {q.question}
                  </button>
                ))}
              </div>
            )}
          </div>
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
                disabled={phase !== "idle" || !query.trim() || !preset}
                className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white
                           hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                           transition-colors"
              >
                {phase === "ensuring"
                  ? "Preparing..."
                  : phase === "querying"
                    ? "Executing..."
                    : "Execute"}
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
          {phase !== "idle" && (
            <div className="flex flex-col items-center justify-center py-12 gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
              <span className="text-sm text-gray-500">
                {phase === "ensuring"
                  ? "Preparing collection (indexing if needed)..."
                  : "Executing query..."}
              </span>
            </div>
          )}

          {/* Results */}
          {result && phase === "idle" && (
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

      {/* Parameter tuning modal */}
      {baseConfig && (
        <ParameterModal
          open={paramsOpen}
          onClose={() => setParamsOpen(false)}
          baseConfig={baseConfig}
          overrides={overrides}
          onOverrideChange={handleOverrideChange}
          onReset={handleResetOverrides}
        />
      )}
    </div>
  );
}
