import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import ModeSwitcher from "../components/testing/ModeSwitcher";
import type { TestingMode } from "../components/testing/ModeSwitcher";
import ParameterModal from "../components/config/ParameterModal";
import ConfigBadges from "../components/config/ConfigBadges";
import QuestionPickerModal from "../components/config/QuestionPickerModal";
import ExecutionPanel from "../components/testing/ExecutionPanel";
import CollectionPreview from "../components/testing/CollectionPreview";
import ChunkList from "../components/results/ChunkList";
import StreamingAnswer from "../components/results/StreamingAnswer";
import LatencyBreakdown from "../components/results/LatencyBreakdown";
import { EChartWrapper, buildBarChart } from "../components/charts";
import {
  getPresetConfig,
  ensureCollection,
  executeQueryStreaming,
  getDatasets,
  getDataset,
  getDatasetRegistry,
  getCollections,
  highlightChunks,
  runBenchmark,
  runSweep,
} from "../api/client";
import type {
  BenchmarkConfig,
  CollectionInfo,
  DatasetInfo,
  DatasetQuestion,
  DatasetRegistryEntry,
  EnsureCollectionRequest,
  RetrievedChunk,
  SweepConfigCompleteEvent,
} from "../api/types";
import {
  deepMerge,
  setOverridePath,
  countOverrides,
  computeSweepCombinations,
  extractSweepParams,
} from "../utils/configHelpers";
import { deriveCollectionName } from "../constants/paramOptions";
import type { BenchPhase, SweepPhase, ActiveRun, SweepProgress } from "../components/testing/types";

// ── Types ─────────────────────────────────────────────────────────────

type QueryPhase = "idle" | "ensuring" | "querying";

// ── Component ─────────────────────────────────────────────────────────

export default function TestingPage() {
  // ── Mode ──────────────────────────────────────────────────────────
  const [mode, setMode] = useState<TestingMode>("query");

  // ── Shared config state ───────────────────────────────────────────
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

  // ── Sweep-mode state ──────────────────────────────────────────────
  const [sweepPhase, setSweepPhase] = useState<SweepPhase>("idle");
  const [sweepProgress, setSweepProgress] = useState<SweepProgress | null>(null);
  const [sweepResults, setSweepResults] = useState<SweepConfigCompleteEvent[]>([]);
  const [sweepAbort, setSweepAbort] = useState<AbortController | null>(null);

  // ── Warning & info state (shared by benchmark & sweep) ──────────
  const [warnings, setWarnings] = useState<string[]>([]);
  const [infoMessages, setInfoMessages] = useState<string[]>([]);

  // ── Collection preview state (sweep) ────────────────────────────
  const [existingCollections, setExistingCollections] = useState<CollectionInfo[]>([]);

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


  const hideSections = useMemo(() => {
    if (mode === "benchmark" || mode === "sweep") return new Set(["Results"]);
    return undefined;
  }, [mode]);

  const isBenchRunning = benchPhase === "ensuring" || benchPhase === "running";
  const isSweepRunning = sweepPhase === "running";

  const combinationCount = useMemo(
    () => (mode === "sweep" ? computeSweepCombinations(overrides) : 1),
    [overrides, mode],
  );

  // ── Effects ───────────────────────────────────────────────────────

  // Load registry on mount
  useEffect(() => {
    getDatasetRegistry()
      .then((r) => {
        const map: Record<string, DatasetRegistryEntry> = {};
        for (const d of r.datasets) map[d.name] = d;
        setRegistryMap(map);
      })
      .catch(() => {});
  }, []);

  // Load datasets — filtered by collection in benchmark mode, unfiltered in sweep mode
  useEffect(() => {
    let collectionFilter: string | undefined;
    if (mode === "benchmark" && effectiveConfig) {
      collectionFilter = deriveCollectionName(
        effectiveConfig.dataset.dataset_name,
        effectiveConfig.vector_db.backend,
        effectiveConfig.chunking.chunk_size,
        effectiveConfig.chunking.chunk_overlap,
        effectiveConfig.embedding.model_name,
        effectiveConfig.vector_db.distance_metric,
      );
    }
    getDatasets(collectionFilter)
      .then((r) => setDatasets(r.datasets.filter((d) => d.status === "completed")))
      .catch(() => {});
  }, [mode, effectiveConfig]);

  // Auto-load default config on mount
  useEffect(() => {
    loadDefaultConfig();
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

  // Fetch existing collections for sweep collection preview
  useEffect(() => {
    if (mode !== "sweep") return;
    getCollections()
      .then((r) => setExistingCollections(r.collections))
      .catch(() => {});
  }, [mode, effectiveConfig]);

  // ── Shared handlers ───────────────────────────────────────────────

  const handleModeChange = useCallback((newMode: TestingMode) => {
    setError(null);
    setWarnings([]);
    setInfoMessages([]);
    setMode(newMode);
  }, []);

  const loadDefaultConfig = useCallback(async () => {
    setStreamingChunks(null);
    setStreamingTokens([]);
    setError(null);
    setOverrides({});
    try {
      const cfg = await getPresetConfig("default.yaml");
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
    if (!query.trim() || !effectiveConfig) return;
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
      "default.yaml",
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
                ec.generation.model ?? undefined,
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
  }, [query, overrides, overrideCount, effectiveConfig]);

  // ── Benchmark-mode handlers ───────────────────────────────────────

  const handleBenchmarkRun = useCallback(async () => {
    if (!effectiveConfig || !selectedDatasetId) return;
    window.scrollTo({ top: 0, behavior: "smooth" });

    setBenchPhase("ensuring");
    setError(null);
    setWarnings([]);
    setInfoMessages([]);
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
        preset: "default.yaml",
        config_overrides: Object.keys(finalOverrides).length > 0 ? finalOverrides : null,
        eval_dataset_id: selectedDatasetId,
        ...(runName ? { name: runName } : {}),
      },
      {
        onStarted: (e) => {
          setActiveRun({ status: "running", progress: { current: 0, total: e.total_questions } });
        },
        onWarning: (e) => {
          setWarnings((prev) => [...prev, e.message]);
        },
        onInfo: (e) => {
          setInfoMessages((prev) => [...prev, e.message]);
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
  }, [effectiveConfig, selectedDatasetId, overrides, overrideCount, runName]);

  const handleBenchmarkCancel = useCallback(() => {
    abortController?.abort();
    setBenchPhase("idle");
    setActiveRun(null);
  }, [abortController]);

  // ── Sweep-mode handlers ───────────────────────────────────────────

  const handleSweepRun = useCallback(() => {
    if (!effectiveConfig || !selectedDatasetId) return;

    const { sweepParams, configOverrides } = extractSweepParams(overrides);

    setSweepPhase("running");
    setSweepProgress(null);
    setSweepResults([]);
    setError(null);
    setWarnings([]);
    setInfoMessages([]);

    const controller = runSweep(
      {
        preset: "default.yaml",
        sweep_params: sweepParams,
        eval_dataset_id: selectedDatasetId,
        config_overrides: Object.keys(configOverrides).length > 0 ? configOverrides : null,
        ...(runName ? { name: runName } : {}),
      },
      {
        onWarning: (e) => {
          setWarnings((prev) => [...prev, e.message]);
        },
        onInfo: (e) => {
          setInfoMessages((prev) => [...prev, e.message]);
        },
        onSweepStarted: (e) => {
          setSweepProgress({
            configIndex: 0,
            totalConfigs: e.total_configs,
            questionIndex: 0,
            totalQuestions: e.total_questions_per_config,
          });
        },
        onConfigStarted: (e) => {
          setSweepProgress((prev) => prev && ({
            ...prev,
            configIndex: e.config_index,
            questionIndex: 0,
          }));
        },
        onConfigProgress: (e) => {
          setSweepProgress({
            configIndex: e.config_index,
            totalConfigs: e.total_configs,
            questionIndex: e.question_index,
            totalQuestions: e.total_questions,
          });
        },
        onConfigComplete: (e) => {
          setSweepResults((prev) => [...prev, e]);
        },
        onSweepComplete: () => {
          setSweepPhase("complete");
          setRunName("");
        },
        onError: (msg) => {
          setError(msg);
          setSweepPhase("idle");
        },
      },
    );

    setSweepAbort(controller);
  }, [effectiveConfig, selectedDatasetId, overrides, runName]);

  const handleSweepCancel = useCallback(() => {
    sweepAbort?.abort();
    setSweepPhase("idle");
    setSweepProgress(null);
  }, [sweepAbort]);

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
          {mode === "sweep" && combinationCount > 1 && (
            <span className="inline-flex items-center gap-1 bg-blue-100 text-blue-700 rounded-full px-2.5 py-0.5 text-xs font-medium">
              {combinationCount} configs
            </span>
          )}
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
                  {streamingChunks.length > 0 && (
                    <EChartWrapper
                      option={buildBarChart({
                        categories: streamingChunks.map((_, i) => `#${i + 1}`),
                        series: [{
                          name: "Score",
                          data: streamingChunks.map((c) => c.score),
                        }],
                        title: "Chunk Similarity Scores",
                        horizontal: true,
                      })}
                      height={Math.max(200, streamingChunks.length * 28 + 80)}
                    />
                  )}
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
          <ExecutionPanel
            mode="benchmark"
            filteredDatasets={filteredDatasets}
            selectedDatasetId={selectedDatasetId}
            onDatasetChange={(id) => setSelectedDatasetId(id)}
            runName={runName}
            onRunNameChange={setRunName}
            isRunning={isBenchRunning}
            onRun={handleBenchmarkRun}
            onCancel={handleBenchmarkCancel}
            disabled={!effectiveConfig || !selectedDatasetId}
            benchPhase={benchPhase}
            activeRun={activeRun}
            warnings={warnings}
            infoMessages={infoMessages}
          />
          {error && mode === "benchmark" && (
            <div className="rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>
      )}

      {/* ── Sweep Mode ─────────────────────────────────────────────── */}
      {mode === "sweep" && (
        <div key="sweep" className="animate-fade-in-up space-y-4">
          {effectiveConfig && (
            <CollectionPreview
              overrides={overrides}
              baseConfig={effectiveConfig}
              existingCollections={existingCollections}
            />
          )}
          <ExecutionPanel
            mode="sweep"
            filteredDatasets={filteredDatasets}
            selectedDatasetId={selectedDatasetId}
            onDatasetChange={(id) => setSelectedDatasetId(id)}
            runName={runName}
            onRunNameChange={setRunName}
            isRunning={isSweepRunning}
            onRun={handleSweepRun}
            onCancel={handleSweepCancel}
            disabled={!effectiveConfig || !selectedDatasetId}
            combinationCount={combinationCount}
            sweepPhase={sweepPhase}
            sweepProgress={sweepProgress}
            sweepResults={sweepResults}
            warnings={warnings}
            infoMessages={infoMessages}
          />
          {error && mode === "sweep" && (
            <div className="rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}
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
        multiSelect={mode === "sweep"}
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
