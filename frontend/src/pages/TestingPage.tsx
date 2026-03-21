import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { Link } from "react-router-dom";
import ModeSwitcher from "../components/testing/ModeSwitcher";
import type { TestingMode } from "../components/testing/ModeSwitcher";
import ParameterModal from "../components/config/ParameterModal";
import ConfigBadges from "../components/config/ConfigBadges";
import QuestionPickerModal from "../components/config/QuestionPickerModal";
import ChunkList from "../components/results/ChunkList";
import StreamingAnswer from "../components/results/StreamingAnswer";
import LatencyBreakdown from "../components/results/LatencyBreakdown";
import ChunkScoresChart from "../components/results/ChunkScoresChart";
import {
  getPresetConfig,
  ensureCollection,
  executeQueryStreaming,
  getDatasets,
  getDataset,
  getDatasetRegistry,
  highlightChunks,
  runBenchmark,
} from "../api/client";
import type {
  BenchmarkConfig,
  DatasetInfo,
  DatasetQuestion,
  DatasetRegistryEntry,
  EnsureCollectionRequest,
  RetrievedChunk,
} from "../api/types";
import { deepMerge, setOverridePath, countOverrides } from "../utils/configHelpers";

// ── Types ─────────────────────────────────────────────────────────────

type QueryPhase = "idle" | "ensuring" | "querying";
type BenchPhase = "idle" | "ensuring" | "running" | "complete";

interface ActiveRun {
  status: "running" | "complete" | "error";
  progress: { current: number; total: number };
  error?: string;
}

// ── Component ─────────────────────────────────────────────────────────

export default function TestingPage() {
  // ── Mode ──────────────────────────────────────────────────────────
  const [mode, setMode] = useState<TestingMode>("query");

  // ── Shared config state ───────────────────────────────────────────
  const [preset, setPreset] = useState("");
  const [baseConfig, setBaseConfig] = useState<BenchmarkConfig | null>(null);
  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [paramsOpen, setParamsOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Shared dataset state ──────────────────────────────────────────
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [registryMap, setRegistryMap] = useState<Record<string, DatasetRegistryEntry>>({});
  const [datasetQuestions, setDatasetQuestions] = useState<DatasetQuestion[]>([]);

  // ── Query-mode state ──────────────────────────────────────────────
  const [query, setQuery] = useState("");
  const [queryPhase, setQueryPhase] = useState<QueryPhase>("idle");
  const [questionPickerOpen, setQuestionPickerOpen] = useState(false);
  const [pickedQuestion, setPickedQuestion] = useState<DatasetQuestion | null>(null);
  const [isQueryEdited, setIsQueryEdited] = useState(false);
  const [highlightedChunks, setHighlightedChunks] = useState<Record<string, string>>({});
  const [relevanceMap, setRelevanceMap] = useState<Record<string, string>>({});
  const [highlighting, setHighlighting] = useState(false);

  // ── Streaming state ────────────────────────────────────────────────
  const [streamingTokens, setStreamingTokens] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingChunks, setStreamingChunks] = useState<RetrievedChunk[] | null>(null);
  const [streamRetrievalMs, setStreamRetrievalMs] = useState(0);
  const [streamGenerationMs, setStreamGenerationMs] = useState<number | null>(null);
  const [streamError, setStreamError] = useState<string | undefined>(undefined);
  const streamControllerRef = useRef<AbortController | null>(null);

  // ── Benchmark-mode state ──────────────────────────────────────────
  const [benchPhase, setBenchPhase] = useState<BenchPhase>("idle");
  const [activeRun, setActiveRun] = useState<ActiveRun | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const [runName, setRunName] = useState("");

  // ── Computed values ───────────────────────────────────────────────
  const overrideCount = useMemo(() => countOverrides(overrides), [overrides]);

  const effectiveConfig = useMemo(() => {
    if (!baseConfig) return null;
    if (overrideCount === 0) return baseConfig;
    return deepMerge(
      baseConfig as unknown as Record<string, unknown>,
      overrides,
    ) as unknown as BenchmarkConfig;
  }, [baseConfig, overrides, overrideCount]);

  const currentDatasetName = effectiveConfig?.dataset.dataset_name;
  const registryEntry = currentDatasetName ? registryMap[currentDatasetName] : null;
  const filteredDatasets = registryEntry
    ? datasets.filter((ds) => registryEntry.collections.includes(ds.collection_name))
    : datasets;

  const sourceChunkId =
    pickedQuestion && !isQueryEdited ? pickedQuestion.source_chunk_id : null;

  const highlightedChunkIds = useMemo(
    () => Object.keys(highlightedChunks),
    [highlightedChunks],
  );

  const hideSections = useMemo(() => {
    if (mode === "benchmark" || mode === "sweep") return new Set(["Results"]);
    return undefined;
  }, [mode]);

  const isBenchRunning = benchPhase === "ensuring" || benchPhase === "running";

  // ── Effects ───────────────────────────────────────────────────────

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

  // Auto-load default preset on mount
  useEffect(() => {
    handlePresetChange("default.yaml");
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  // ── Shared handlers ───────────────────────────────────────────────

  const handleModeChange = useCallback((newMode: TestingMode) => {
    setError(null);
    setMode(newMode);
  }, []);

  const handlePresetChange = useCallback(async (filename: string) => {
    setPreset(filename);
    setStreamingChunks(null);
    setStreamingTokens([]);
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

  // ── Query-mode handler ────────────────────────────────────────────

  const handleQueryExecute = useCallback(async () => {
    if (!query.trim() || !preset || !effectiveConfig) return;
    window.scrollTo({ top: 0, behavior: "smooth" });

    // Abort any previous streaming request
    streamControllerRef.current?.abort();

    setQueryPhase("ensuring");
    setError(null);
    setStreamingChunks(null);
    setStreamingTokens([]);
    setIsStreaming(false);
    setStreamRetrievalMs(0);
    setStreamGenerationMs(null);
    setStreamError(undefined);
    setHighlightedChunks({});
    setRelevanceMap({});
    setHighlighting(false);

    let finalOverrides: Record<string, unknown> =
      overrideCount > 0 ? { ...overrides } : {};

    try {
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
      setQueryPhase("idle");
      return;
    }

    setQueryPhase("querying");

    const ec = effectiveConfig;
    // Store chunks in a local variable for highlighting in the completion callback
    let retrievedChunks: RetrievedChunk[] = [];

    const controller = executeQueryStreaming(
      query,
      preset,
      Object.keys(finalOverrides).length > 0 ? finalOverrides : undefined,
      {
        onRetrievalComplete: (e) => {
          retrievedChunks = e.chunks;
          setStreamingChunks(e.chunks);
          setStreamRetrievalMs(e.retrieval_time_ms);
          if (ec.generation.enabled) {
            setIsStreaming(true);
          } else {
            setQueryPhase("idle");
          }
        },
        onGenerationToken: (e) => {
          setStreamingTokens((prev) => [...prev, e.token]);
        },
        onGenerationComplete: async (e) => {
          setIsStreaming(false);
          setStreamGenerationMs(e.generation_time_ms);
          setQueryPhase("idle");

          // Trigger highlighting (best-effort)
          if (ec.generation.highlight_chunks && retrievedChunks.length > 0) {
            setHighlighting(true);
            try {
              const hlRes = await highlightChunks(
                query,
                retrievedChunks.map((c) => ({
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
              // best-effort
            } finally {
              setHighlighting(false);
            }
          }
        },
        onError: (msg) => {
          setIsStreaming(false);
          setStreamError(msg);
          setError(msg);
          setQueryPhase("idle");
        },
      },
    );

    streamControllerRef.current = controller;
  }, [query, preset, overrides, overrideCount, effectiveConfig]);

  // ── Benchmark-mode handlers ───────────────────────────────────────

  const handleBenchmarkRun = useCallback(async () => {
    if (!preset || !effectiveConfig || !selectedDatasetId) return;
    window.scrollTo({ top: 0, behavior: "smooth" });

    setBenchPhase("ensuring");
    setError(null);
    setActiveRun({ status: "running", progress: { current: 0, total: 0 } });

    let finalOverrides: Record<string, unknown> =
      overrideCount > 0 ? { ...overrides } : {};

    try {
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
      setBenchPhase("idle");
      setActiveRun(null);
      return;
    }

    setBenchPhase("running");

    const controller = runBenchmark(
      {
        preset,
        config_overrides: Object.keys(finalOverrides).length > 0 ? finalOverrides : null,
        eval_dataset_id: selectedDatasetId,
        ...(runName ? { name: runName } : {}),
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
          setBenchPhase("complete");
          setRunName("");
        },
        onError: (msg) => {
          setActiveRun({ status: "error", progress: { current: 0, total: 0 }, error: msg });
          setError(msg);
          setBenchPhase("idle");
        },
      },
    );

    setAbortController(controller);
  }, [preset, effectiveConfig, selectedDatasetId, overrides, overrideCount, runName]);

  const handleBenchmarkCancel = useCallback(() => {
    abortController?.abort();
    setBenchPhase("idle");
    setActiveRun(null);
  }, [abortController]);

  // ── Render ────────────────────────────────────────────────────────

  return (
    <div className="p-6 max-w-3xl mx-auto">
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Testing</h1>
        <p className="mt-1 text-sm text-gray-500">
          Test queries, run benchmarks, and execute parameter sweeps.
        </p>
        <div className="mt-4 flex justify-center">
          <ModeSwitcher mode={mode} onModeChange={handleModeChange} />
        </div>

        {/* Config bar */}
        <div className="mt-4 flex flex-col items-center gap-2">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setParamsOpen(true)}
              className="flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              </svg>
              Configure Pipeline
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
          <ConfigBadges config={effectiveConfig} />
        </div>
      </div>

      {/* ── Query Mode ─────────────────────────────────────────────── */}
      {mode === "query" && (
        <div key="query" className="animate-fade-in-up space-y-4">
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
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleQueryExecute();
                  }}
                />
                <div className="mt-2 flex items-center justify-between gap-2">
                  <span className="text-xs text-gray-400 shrink-0">Ctrl+Enter to execute</span>
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
                    onClick={handleQueryExecute}
                    disabled={queryPhase !== "idle" || !query.trim() || !effectiveConfig}
                    className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white
                               hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                               transition-colors shrink-0"
                  >
                    {queryPhase === "ensuring"
                      ? "Preparing..."
                      : queryPhase === "querying"
                        ? "Executing..."
                        : "Execute"}
                  </button>
                </div>
              </div>

              {/* Error */}
              {error && mode === "query" && (
                <div className="rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              )}

              {/* Loading spinner — only during collection preparation */}
              {queryPhase === "ensuring" && (
                <div className="flex flex-col items-center justify-center py-12 gap-3">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
                  <span className="text-sm text-gray-500">
                    Preparing collection (indexing if needed)...
                  </span>
                </div>
              )}

              {/* Retrieving indicator — after ensure, before chunks arrive */}
              {queryPhase === "querying" && !streamingChunks && (
                <div className="flex flex-col items-center justify-center py-12 gap-3">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
                  <span className="text-sm text-gray-500">Retrieving chunks...</span>
                </div>
              )}

              {/* Results — appear as soon as chunks arrive */}
              {streamingChunks && (
                <div className="space-y-4">
                  <LatencyBreakdown
                    retrievalMs={streamRetrievalMs}
                    generationMs={streamGenerationMs ?? 0}
                  />
                  <ChunkScoresChart
                    chunks={streamingChunks}
                    sourceChunkId={sourceChunkId}
                    highlightedChunkIds={highlightedChunkIds}
                    highlightedContent={highlightedChunks}
                    relevanceMap={relevanceMap}
                  />
                  <ChunkList
                    chunks={streamingChunks}
                    highlightedContent={highlightedChunks}
                    highlighting={highlighting}
                    sourceChunkId={sourceChunkId}
                    relevanceMap={relevanceMap}
                  />
                  {(isStreaming || streamingTokens.length > 0) && (
                    <StreamingAnswer
                      tokens={streamingTokens}
                      isStreaming={isStreaming}
                      error={streamError}
                    />
                  )}
                </div>
              )}
        </div>
      )}

      {/* ── Benchmark Mode ─────────────────────────────────────────── */}
      {mode === "benchmark" && (
        <div key="benchmark" className="animate-fade-in-up space-y-4">
              <div className="rounded border border-gray-200 bg-white p-4">
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Eval Dataset
                    </label>
                    <select
                      value={selectedDatasetId}
                      onChange={(e) => setSelectedDatasetId(e.target.value)}
                      className="w-full rounded border-gray-300 bg-white px-3 py-2 text-sm shadow-sm
                                 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                      disabled={isBenchRunning}
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
                      onChange={(e) => setRunName(e.target.value)}
                      placeholder="Auto-generated if empty"
                      className="w-full rounded border-gray-300 bg-white px-3 py-2 text-sm shadow-sm
                                 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                      disabled={isBenchRunning}
                    />
                  </div>

                  <div className="shrink-0 pt-5">
                    {isBenchRunning ? (
                      <button
                        onClick={handleBenchmarkCancel}
                        className="rounded bg-red-600 px-5 py-2 text-sm font-medium text-white
                                   hover:bg-red-700 transition-colors"
                      >
                        Cancel
                      </button>
                    ) : (
                      <button
                        onClick={handleBenchmarkRun}
                        disabled={!effectiveConfig || !selectedDatasetId}
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
                      {benchPhase === "ensuring"
                        ? "Preparing collection..."
                        : `Evaluating questions... (${Math.round(
                            (activeRun.progress.current / activeRun.progress.total) * 100
                          )}%)`}
                    </p>
                  </div>
                )}

                {benchPhase === "ensuring" && (
                  <div className="mt-3 flex items-center gap-2">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
                    <span className="text-sm text-gray-500">Preparing collection...</span>
                  </div>
                )}

                {benchPhase === "complete" && (
                  <p className="mt-3 text-sm text-green-600">
                    Benchmark completed. View results in{" "}
                    <Link to="/runs" className="underline hover:text-green-700">
                      Run History
                    </Link>.
                  </p>
                )}
              </div>

              {/* Error */}
              {error && mode === "benchmark" && (
                <div className="rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              )}
        </div>
      )}

      {/* ── Sweep Mode (placeholder) ───────────────────────────────── */}
      {mode === "sweep" && (
        <div key="sweep" className="animate-fade-in-up">
          <div className="rounded border border-gray-200 bg-white p-8 text-center text-sm text-gray-400">
            Sweep mode — multi-select parameters and run cartesian product benchmarks.
            <br />
            Coming in E6-F3.
          </div>
        </div>
      )}

      {/* ── Shared modals ──────────────────────────────────────────── */}
      <ParameterModal
        open={paramsOpen}
        onClose={() => setParamsOpen(false)}
        baseConfig={baseConfig}
        overrides={overrides}
        onOverrideChange={handleOverrideChange}
        onReset={handleResetOverrides}
        hideSections={hideSections}
        selectedPreset={preset}
        onPresetChange={handlePresetChange}
      />

      <QuestionPickerModal
        open={questionPickerOpen}
        onClose={() => setQuestionPickerOpen(false)}
        questions={datasetQuestions}
        onSelect={handlePickQuestion}
      />
    </div>
  );
}
