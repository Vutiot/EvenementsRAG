import { useState, useCallback, useEffect, useMemo } from "react";
import PageHeader from "../components/layout/PageHeader";
import PresetSelector from "../components/config/PresetSelector";
import ConfigSummary from "../components/config/ConfigSummary";
import ParameterModal from "../components/config/ParameterModal";
import QuestionPickerModal from "../components/config/QuestionPickerModal";
import ChunkList from "../components/results/ChunkList";
import GeneratedAnswer from "../components/results/GeneratedAnswer";
import LatencyBreakdown from "../components/results/LatencyBreakdown";
import ChunkScoresChart from "../components/results/ChunkScoresChart";
import {
  getPresetConfig,
  executeQuery,
  ensureCollection,
  getDatasets,
  getDataset,
  getDatasetRegistry,
  highlightChunks,
} from "../api/client";
import type {
  BenchmarkConfig,
  DatasetInfo,
  DatasetQuestion,
  DatasetRegistryEntry,
  EnsureCollectionRequest,
  QueryResult,
} from "../api/types";
import { deepMerge, setOverridePath, countOverrides } from "../utils/configHelpers";

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

  // Dataset registry for filtering eval datasets by selected dataset
  const [registryMap, setRegistryMap] = useState<Record<string, DatasetRegistryEntry>>({});

  // Question picker modal
  const [questionPickerOpen, setQuestionPickerOpen] = useState(false);

  // Picked question tracking for chunk score colors
  const [pickedQuestion, setPickedQuestion] = useState<DatasetQuestion | null>(null);
  const [isQueryEdited, setIsQueryEdited] = useState(false);

  // Highlight state
  const [highlightedChunks, setHighlightedChunks] = useState<Record<string, string>>({});
  const [relevanceMap, setRelevanceMap] = useState<Record<string, string>>({});
  const [highlighting, setHighlighting] = useState(false);

  const overrideCount = useMemo(() => countOverrides(overrides), [overrides]);

  const effectiveConfig = useMemo(() => {
    if (!baseConfig) return null;
    if (overrideCount === 0) return baseConfig;
    return deepMerge(
      baseConfig as unknown as Record<string, unknown>,
      overrides,
    ) as unknown as BenchmarkConfig;
  }, [baseConfig, overrides, overrideCount]);

  // Load datasets and registry on mount
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
      setDatasetQuestions([]);
      setPickedQuestion(null);
    }
  }, [filteredDatasets, selectedDatasetId]);

  // Track whether query has been edited away from the picked question
  useEffect(() => {
    if (pickedQuestion) {
      setIsQueryEdited(query !== pickedQuestion.question);
    }
  }, [query, pickedQuestion]);

  const sourceChunkId =
    pickedQuestion && !isQueryEdited ? pickedQuestion.source_chunk_id : null;

  const handleDatasetChange = useCallback(async (dsId: string) => {
    setSelectedDatasetId(dsId);
    setDatasetQuestions([]);
    setPickedQuestion(null);
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
    setPickedQuestion(q);
    setIsQueryEdited(false);
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
    setHighlightedChunks({});
    setRelevanceMap({});
    setHighlighting(false);

    let finalOverrides: Record<string, unknown> =
      overrideCount > 0 ? { ...overrides } : {};

    try {
      // Always ensure collection exists (idempotent — returns fast if already present)
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
        ...finalOverrides,
        dataset: {
          ...((finalOverrides.dataset as Record<string, unknown>) ?? {}),
          collection_name: ensureRes.collection_name,
        },
      };

      // Execute query
      setPhase("querying");
      const res = await executeQuery(query, preset, finalOverrides);
      setResult(res);

      // Auto-trigger chunk highlighting if enabled
      if (ec.generation.highlight_chunks && res.retrieved_chunks.length > 0) {
        setHighlighting(true);
        try {
          const hlRes = await highlightChunks(
            query,
            res.retrieved_chunks.map((c) => ({
              chunk_id: c.chunk_id,
              content: c.content,
            })),
            ec.generation.model,
          );
          const hlMap: Record<string, string> = {};
          const relMap: Record<string, string> = {};
          for (const hl of hlRes.highlighted_chunks) {
            hlMap[hl.chunk_id] = hl.highlighted_content;
            if (hl.relevance) relMap[hl.chunk_id] = hl.relevance;
          }
          setHighlightedChunks(hlMap);
          setRelevanceMap(relMap);
        } catch {
          // Highlighting is best-effort
        } finally {
          setHighlighting(false);
        }
      }
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

  const highlightedChunkIds = useMemo(
    () => Object.keys(highlightedChunks),
    [highlightedChunks],
  );

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
            <div className="mt-2 flex items-center justify-between gap-2">
              <span className="text-xs text-gray-400 shrink-0">Ctrl+Enter to execute</span>
              {/* Eval query controls */}
              <div className="flex items-center gap-1.5 min-w-0 flex-1 justify-end">
                <select
                  value={selectedDatasetId}
                  onChange={(e) => handleDatasetChange(e.target.value)}
                  className="rounded border-gray-300 bg-white px-2 py-1.5 text-xs shadow-sm
                             focus:border-blue-500 focus:ring-1 focus:ring-blue-500 max-w-[160px]"
                >
                  <option value="">Eval dataset...</option>
                  {filteredDatasets.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name} ({ds.total_questions}q)
                    </option>
                  ))}
                </select>
                {datasetQuestions.length > 0 && (
                  <button
                    onClick={() => setQuestionPickerOpen(true)}
                    className="rounded border border-gray-300 bg-white px-2 py-1.5 text-xs shadow-sm
                               hover:bg-gray-50 transition text-gray-700"
                  >
                    Pick question ({datasetQuestions.length})
                  </button>
                )}
              </div>
              <button
                onClick={handleExecute}
                disabled={phase !== "idle" || !query.trim() || !preset}
                className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white
                           hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                           transition-colors shrink-0"
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
              <ChunkScoresChart
                chunks={result.retrieved_chunks}
                sourceChunkId={sourceChunkId}
                highlightedChunkIds={highlightedChunkIds}
                highlightedContent={highlightedChunks}
                relevanceMap={relevanceMap}
              />
              <ChunkList
                chunks={result.retrieved_chunks}
                highlightedContent={highlightedChunks}
                highlighting={highlighting}
                sourceChunkId={sourceChunkId}
                relevanceMap={relevanceMap}
              />
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

      {/* Question picker modal */}
      <QuestionPickerModal
        open={questionPickerOpen}
        onClose={() => setQuestionPickerOpen(false)}
        questions={datasetQuestions}
        onSelect={handlePickQuestion}
      />
    </div>
  );
}
