/** Centered modal for tuning RAG pipeline parameters — 5-section structure. */

import { useState, useEffect, useCallback, useMemo } from "react";
import ParamChips from "./ParamChips";
import MultiSelectChips from "./MultiSelectChips";
import ParamSlider from "./ParamSlider";
import CollectionSection from "./CollectionSection";
import type { BenchmarkConfig } from "../../api/types";
import type { ParsedCollectionParams } from "../../constants/paramOptions";
import {
  DATASET_OPTIONS,
  TECHNIQUE_OPTIONS,
  SPARSE_TYPE_OPTIONS,
  SPARSE_WEIGHT_OPTIONS,
  FUSION_OPTIONS,
  RETRIEVER_K_NO_RERANKER,
  RETRIEVER_K_WITH_RERANKER,
  RERANK_TOP_K_OPTIONS,
  LLM_MODELS,
  MAX_TOKENS_OPTIONS,
  EMBEDDING_DIMENSION_MAP,
  RERANKER_TYPE_OPTIONS,
  RERANKER_MODEL_OPTIONS,
  CHUNK_SIZE_OPTIONS,
  CHUNK_OVERLAP_VALUES,
  EMBEDDING_MODELS,
  DISTANCE_OPTIONS,
  BACKEND_OPTIONS,
} from "../../constants/paramOptions";
import { computeSweepCombinations, isGenerationDisabled } from "../../utils/configHelpers";

// ── Helpers ──────────────────────────────────────────────────────────

/** Resolve a dotted config path to a value. */
function getPath(obj: Record<string, unknown>, path: string): unknown {
  const parts = path.split(".");
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

// Field mapping from CollectionSection field names → config paths
const COLLECTION_FIELD_TO_PATH: Record<string, string> = {
  backend: "vector_db.backend",
  chunkSize: "chunking.chunk_size",
  chunkOverlap: "chunking.chunk_overlap",
  embeddingModel: "embedding.model_name",
  distanceMetric: "vector_db.distance_metric",
};

// ── System Prompt Presets ────────────────────────────────────────────

const SYSTEM_PROMPT_PRESETS: { label: string; value: string }[] = [
  { label: "Default (Historian)", value: "You are a knowledgeable historian assistant." },
  { label: "Concise Analyst", value: "You are a concise military analyst. Answer with bullet points and hard facts. Avoid filler." },
  { label: "Detailed Researcher", value: "You are a thorough historical researcher. Provide detailed, well-sourced answers with full context, dates, and cross-references between events." },
  { label: "Teacher", value: "You are a history teacher explaining concepts to a student. Use simple language, provide examples, and build understanding step by step." },
  { label: "Custom", value: "__custom__" },
];

// ── Prompt Pieces ────────────────────────────────────────────────────

const PROMPT_PIECES = [
  { key: "citation", label: "Citation", default: "Cite chunk(s) used to answer, you can cite several chunks for one answer." },
  { key: "relevance", label: "Relevance", default: "If no chunk relevant, do not answer and say you don't know instead." },
  { key: "misc", label: "Misc", default: "When answering, pay special attention to chronological order, cause-and-effect relationships, and the geopolitical context of events." },
] as const;

// ── Types ────────────────────────────────────────────────────────────

interface ParameterModalProps {
  open: boolean;
  onClose: () => void;
  baseConfig: BenchmarkConfig | null;
  overrides: Record<string, unknown>;
  onOverrideChange: (path: string, value: unknown) => void;
  onReset: () => void;
  hideSections?: Set<string>;
  multiSelect?: boolean;
}

// ── Component ────────────────────────────────────────────────────────

export default function ParameterModal({
  open,
  onClose,
  baseConfig,
  overrides,
  onOverrideChange,
  onReset,
  hideSections,
  multiSelect = false,
}: ParameterModalProps) {
  // When a collection is imported, stores the imported values as the new preset baseline for chips
  const [collectionPresetOverride, setCollectionPresetOverride] = useState<Record<string, unknown> | null>(null);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Clear collection preset override when all overrides are cleared
  useEffect(() => {
    if (Object.keys(overrides).length === 0) {
      setCollectionPresetOverride(null);
    }
  }, [overrides]);

  const combinationCount = useMemo(
    () => (multiSelect ? computeSweepCombinations(overrides) : 1),
    [overrides, multiSelect],
  );

  if (!open) return null;

  const hasConfig = baseConfig != null;
  const base = (baseConfig ?? {}) as unknown as Record<string, unknown>;

  /** Get effective value (override wins, else base). */
  function effective(path: string): unknown {
    const overrideVal = getPath(overrides, path);
    if (overrideVal !== undefined) return overrideVal;
    return getPath(base, path);
  }

  /** True when at least one eval metric toggle requires LLM generation. */
  const needsGeneration =
    !!(effective("evaluation.compute_context_precision") as boolean) ||
    !!(effective("evaluation.compute_entity_metrics") as boolean);

  /** Dynamic Retriever Top K options based on reranker selection. */
  const hasReranker = (effective("reranker.type") as string) !== "none";
  const topKOptions = hasReranker ? RETRIEVER_K_WITH_RERANKER : RETRIEVER_K_NO_RERANKER;

  /** Get base/preset value. */
  function preset(path: string): unknown {
    return getPath(base, path);
  }

  /** Handle a change: if same as preset, remove override; else set it. */
  function handleChange(path: string, value: unknown) {
    const presetVal = preset(path);
    if (value === presetVal) {
      onOverrideChange(path, undefined);
    } else {
      onOverrideChange(path, value);
    }
  }

  // ── Multi-select helpers ──

  /** Get effective value as array for multi-select mode. */
  function effectiveArray<T>(path: string): T[] {
    const overrideVal = getPath(overrides, path);
    if (Array.isArray(overrideVal)) return overrideVal as T[];
    const pv = getPath(base, path);
    return [pv as T];
  }

  /** Handle array change: if [presetVal], remove override; else set array. */
  function handleArrayChange(path: string, values: unknown[]) {
    const presetVal = preset(path);
    if (values.length === 1 && values[0] === presetVal) {
      onOverrideChange(path, undefined);
    } else {
      onOverrideChange(path, values);
    }
  }

  const technique = effective("retrieval.technique") as string;

  // ── Collection section callbacks ──

  const handleCollectionParamChange = (field: string, value: string | number) => {
    const path = COLLECTION_FIELD_TO_PATH[field];
    if (path) handleChange(path, value);
    // Auto-update embedding dimension
    if (field === "embeddingModel") {
      const dim = EMBEDDING_DIMENSION_MAP[value as string];
      if (dim !== undefined) handleChange("embedding.dimension", dim);
    }
  };

  const handleCollectionSelect = (
    collectionName: string,
    params: ParsedCollectionParams | null,
  ) => {
    if (!collectionName) {
      handleChange("dataset.collection_name", undefined);
      return;
    }
    handleChange("dataset.collection_name", collectionName);
    if (params) {
      handleChange("vector_db.backend", params.backend);
      handleChange("chunking.chunk_size", params.chunkSize);
      handleChange("chunking.chunk_overlap", params.chunkOverlap);
      handleChange("embedding.model_name", params.embeddingModel);
      handleChange("vector_db.distance_metric", params.distanceMetric);
      const dim = EMBEDDING_DIMENSION_MAP[params.embeddingModel];
      if (dim) handleChange("embedding.dimension", dim);
      // Store imported values as the new baseline so chips show blue
      setCollectionPresetOverride({
        backend: params.backend,
        chunkSize: params.chunkSize,
        chunkOverlap: params.chunkOverlap,
        embeddingModel: params.embeddingModel,
        distanceMetric: params.distanceMetric,
      });
    }
  };

  // Overlap options filtered by chunk size (single-select only)
  const overlapOptions = CHUNK_OVERLAP_VALUES.map((v) => ({
    value: v,
    label: String(v),
    disabled: !multiSelect && v >= (effective("chunking.chunk_size") as number),
  }));

  return (
    // Backdrop
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {/* Modal panel */}
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3">
          <h2 className="text-lg font-semibold text-gray-900">
            {multiSelect ? "Sweep Parameters" : "Configure Pipeline"}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition p-1"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-6 pb-4 space-y-5">
          {!hasConfig && (
            <p className="text-sm text-gray-400 text-center py-8">
              Loading configuration...
            </p>
          )}

          {hasConfig && (<>
          {/* ── Part 1: Dataset ── */}
          <Section title="Dataset">
            <ParamChips
              label="Dataset"
              options={DATASET_OPTIONS}
              value={effective("dataset.dataset_name") as string}
              presetValue={preset("dataset.dataset_name") as string}
              onChange={(v) => {
                handleChange("dataset.dataset_name", v);
                // Clear collection_name when dataset changes
                handleChange("dataset.collection_name", undefined);
              }}
            />
          </Section>

          {/* ── Part 2: Collection ── */}
          <Section title="Collection">
            {multiSelect ? (
              /* Sweep mode: direct multi-select chips, no import/derived name */
              <div className="space-y-3">
                <ParamChips
                  label="Backend"
                  options={BACKEND_OPTIONS}
                  value={effective("vector_db.backend") as string}
                  presetValue={preset("vector_db.backend") as string}
                  onChange={(v) => handleChange("vector_db.backend", v)}
                />
                <MultiSelectChips
                  label="Chunk Size"
                  options={CHUNK_SIZE_OPTIONS}
                  values={effectiveArray<number>("chunking.chunk_size")}
                  presetValue={preset("chunking.chunk_size") as number}
                  onChange={(vs) => handleArrayChange("chunking.chunk_size", vs)}
                />
                <MultiSelectChips
                  label="Overlap"
                  options={overlapOptions}
                  values={effectiveArray<number>("chunking.chunk_overlap")}
                  presetValue={preset("chunking.chunk_overlap") as number}
                  onChange={(vs) => handleArrayChange("chunking.chunk_overlap", vs)}
                />
                <MultiSelectChips
                  label="Embedding"
                  options={EMBEDDING_MODELS}
                  values={effectiveArray<string>("embedding.model_name")}
                  presetValue={preset("embedding.model_name") as string}
                  onChange={(vs) => handleArrayChange("embedding.model_name", vs)}
                />
                <MultiSelectChips
                  label="Distance"
                  options={DISTANCE_OPTIONS}
                  values={effectiveArray<string>("vector_db.distance_metric")}
                  presetValue={preset("vector_db.distance_metric") as string}
                  onChange={(vs) => handleArrayChange("vector_db.distance_metric", vs)}
                />
              </div>
            ) : (
              <CollectionSection
                datasetName={effective("dataset.dataset_name") as string}
                backend={effective("vector_db.backend") as string}
                chunkSize={effective("chunking.chunk_size") as number}
                chunkOverlap={effective("chunking.chunk_overlap") as number}
                embeddingModel={effective("embedding.model_name") as string}
                distanceMetric={effective("vector_db.distance_metric") as string}
                presetValues={
                  collectionPresetOverride
                    ? {
                        backend: collectionPresetOverride.backend as string,
                        chunkSize: collectionPresetOverride.chunkSize as number,
                        chunkOverlap: collectionPresetOverride.chunkOverlap as number,
                        embeddingModel: collectionPresetOverride.embeddingModel as string,
                        distanceMetric: collectionPresetOverride.distanceMetric as string,
                      }
                    : {
                        backend: preset("vector_db.backend") as string,
                        chunkSize: preset("chunking.chunk_size") as number,
                        chunkOverlap: preset("chunking.chunk_overlap") as number,
                        embeddingModel: preset("embedding.model_name") as string,
                        distanceMetric: preset("vector_db.distance_metric") as string,
                      }
                }
                onParamChange={handleCollectionParamChange}
                onCollectionSelect={handleCollectionSelect}
              />
            )}
          </Section>

          {/* ── Part 3: Pipeline ── */}
          <Section title="Pipeline">
            {/* Retrieval */}
            <div className="space-y-3">
              <h4 className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
                Retrieval
              </h4>
              {multiSelect ? (
                <>
                  <MultiSelectChips
                    label="Technique"
                    options={TECHNIQUE_OPTIONS}
                    values={effectiveArray<string>("retrieval.technique")}
                    presetValue={preset("retrieval.technique") as string}
                    onChange={(vs) => handleArrayChange("retrieval.technique", vs)}
                  />
                  <MultiSelectChips
                    label="Top K"
                    options={topKOptions}
                    values={effectiveArray<number>("retrieval.top_k")}
                    presetValue={preset("retrieval.top_k") as number}
                    onChange={(vs) => handleArrayChange("retrieval.top_k", vs)}
                  />
                  {/* Show hybrid params if hybrid is among selected techniques */}
                  {(Array.isArray(effective("retrieval.technique"))
                    ? (effective("retrieval.technique") as string[]).includes("hybrid")
                    : technique === "hybrid") && (
                    <>
                      <MultiSelectChips
                        label="Sparse Weight"
                        options={SPARSE_WEIGHT_OPTIONS}
                        values={effectiveArray<number>("retrieval.sparse_weight")}
                        presetValue={preset("retrieval.sparse_weight") as number}
                        onChange={(vs) => handleArrayChange("retrieval.sparse_weight", vs)}
                      />
                      <MultiSelectChips
                        label="Sparse Type"
                        options={SPARSE_TYPE_OPTIONS}
                        values={effectiveArray<string>("retrieval.sparse_type")}
                        presetValue={(preset("retrieval.sparse_type") ?? "bm25") as string}
                        onChange={(vs) => handleArrayChange("retrieval.sparse_type", vs)}
                      />
                      <MultiSelectChips
                        label="Fusion"
                        options={FUSION_OPTIONS}
                        values={effectiveArray<string>("retrieval.fusion_method")}
                        presetValue={preset("retrieval.fusion_method") as string}
                        onChange={(vs) => handleArrayChange("retrieval.fusion_method", vs)}
                      />
                    </>
                  )}
                </>
              ) : (
                <>
                  <ParamChips
                    label="Technique"
                    options={TECHNIQUE_OPTIONS}
                    value={effective("retrieval.technique") as string}
                    presetValue={preset("retrieval.technique") as string}
                    onChange={(v) => handleChange("retrieval.technique", v)}
                  />
                  <ParamChips
                    label="Top K"
                    options={topKOptions}
                    value={effective("retrieval.top_k") as number}
                    presetValue={preset("retrieval.top_k") as number}
                    onChange={(v) => handleChange("retrieval.top_k", v)}
                  />
                  {technique === "hybrid" && (
                    <>
                      <ParamSlider
                        label="Sparse Weight"
                        min={0}
                        max={1}
                        step={0.05}
                        value={effective("retrieval.sparse_weight") as number}
                        presetValue={preset("retrieval.sparse_weight") as number}
                        onChange={(v) => {
                          handleChange("retrieval.sparse_weight", v);
                          handleChange("retrieval.dense_weight", +(1 - v).toFixed(2));
                        }}
                      />
                      <ParamChips
                        label="Sparse Type"
                        options={SPARSE_TYPE_OPTIONS}
                        value={(effective("retrieval.sparse_type") ?? "bm25") as string}
                        presetValue={(preset("retrieval.sparse_type") ?? "bm25") as string}
                        onChange={(v) => handleChange("retrieval.sparse_type", v)}
                      />
                      <ParamChips
                        label="Fusion"
                        options={FUSION_OPTIONS}
                        value={effective("retrieval.fusion_method") as string}
                        presetValue={preset("retrieval.fusion_method") as string}
                        onChange={(v) => handleChange("retrieval.fusion_method", v)}
                      />
                    </>
                  )}
                </>
              )}
            </div>

            {/* Reranking */}
            <div className="space-y-3 mt-4">
              <h4 className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
                Reranking
              </h4>
              {multiSelect ? (
                <>
                  <MultiSelectChips
                    label="Reranker"
                    options={RERANKER_TYPE_OPTIONS}
                    values={effectiveArray<string>("reranker.type")}
                    presetValue={preset("reranker.type") as string}
                    onChange={(vs) => handleArrayChange("reranker.type", vs)}
                  />
                  <MultiSelectChips
                    label="Top K Rerank"
                    options={RERANK_TOP_K_OPTIONS}
                    values={effectiveArray<number>("generation.top_k_chunks")}
                    presetValue={preset("generation.top_k_chunks") as number}
                    onChange={(vs) => handleArrayChange("generation.top_k_chunks", vs)}
                  />
                </>
              ) : (
                <>
                  <ParamChips
                    label="Reranker"
                    options={RERANKER_TYPE_OPTIONS}
                    value={effective("reranker.type") as string}
                    presetValue={preset("reranker.type") as string}
                    onChange={(v) => {
                      handleChange("reranker.type", v);
                      if (v === "none") {
                        handleChange("reranker.model_name", null);
                        handleChange("retrieval.top_k", 20);
                      } else {
                        const models = RERANKER_MODEL_OPTIONS[v as string];
                        if (models?.[0]) {
                          handleChange("reranker.model_name", models[0].value);
                        }
                        handleChange("retrieval.top_k", 100);
                      }
                    }}
                  />
                  {(effective("reranker.type") as string) !== "none" && (
                    <>
                      <ParamChips
                        label="Model"
                        options={RERANKER_MODEL_OPTIONS[effective("reranker.type") as string] ?? []}
                        value={effective("reranker.model_name") as string}
                        presetValue={preset("reranker.model_name") as string}
                        onChange={(v) => handleChange("reranker.model_name", v)}
                      />
                      <ParamChips
                        label="Top K Rerank"
                        options={RERANK_TOP_K_OPTIONS}
                        value={effective("generation.top_k_chunks") as number}
                        presetValue={preset("generation.top_k_chunks") as number}
                        onChange={(v) => handleChange("generation.top_k_chunks", v)}
                      />
                    </>
                  )}
                </>
              )}
            </div>

            {/* Generation (visible only when an eval metric needs LLM) */}
            {needsGeneration && (
            <div className="space-y-3 mt-4">
              <h4 className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
                Generation
              </h4>
              <ParamChips
                label="LLM Model"
                options={LLM_MODELS}
                value={effective("generation.model") as string}
                presetValue={preset("generation.model") as string}
                onChange={(v) => {
                  if (v === "__none__") {
                    handleChange("generation.model", "__none__");
                    handleChange("generation.enabled", false);
                  } else {
                    handleChange("generation.model", v);
                    handleChange("generation.enabled", true);
                  }
                }}
              />
              {!isGenerationDisabled(effective("generation.model") as string) && (
              <>
              <ParamSlider
                label="Temperature"
                min={0}
                max={2}
                step={0.1}
                value={effective("generation.temperature") as number}
                presetValue={preset("generation.temperature") as number}
                onChange={(v) => handleChange("generation.temperature", v)}
              />
              <ParamChips
                label="Max Tokens"
                options={MAX_TOKENS_OPTIONS}
                value={effective("generation.max_tokens") as number}
                presetValue={preset("generation.max_tokens") as number}
                onChange={(v) => handleChange("generation.max_tokens", v)}
              />
              </>
              )}
            </div>
            )}
          </Section>

          {/* ── Part 4: System Prompt (hidden when generation disabled) ── */}
          {needsGeneration && !isGenerationDisabled(effective("generation.model") as string) && (
          <Section title="System Prompt">
            <SystemPromptSection
              currentPrompt={(effective("generation.system_prompt") as string) ?? ""}
              onPromptChange={(v) => handleChange("generation.system_prompt", v)}
            />
          </Section>
          )}

          {/* ── Part 5: Evaluation Metrics (always visible) ── */}
          <Section title="Evaluation">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm font-medium text-gray-700">LLM Context Precision</span>
                <p className="text-xs text-gray-400">RAGAS context_precision (requires LLM calls)</p>
              </div>
              <button
                onClick={() =>
                  handleChange(
                    "evaluation.compute_context_precision",
                    !(effective("evaluation.compute_context_precision") as boolean),
                  )
                }
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  (effective("evaluation.compute_context_precision") as boolean)
                    ? "bg-blue-600"
                    : "bg-gray-200"
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    (effective("evaluation.compute_context_precision") as boolean)
                      ? "translate-x-6"
                      : "translate-x-1"
                  }`}
                />
              </button>
            </div>
            <div className="flex items-center justify-between mt-3">
              <div>
                <span className="text-sm font-medium text-gray-700">Entity Metrics (LLM NER)</span>
                <p className="text-xs text-gray-400">Entity precision/recall via LLM entity extraction</p>
              </div>
              <button
                onClick={() =>
                  handleChange(
                    "evaluation.compute_entity_metrics",
                    !(effective("evaluation.compute_entity_metrics") as boolean),
                  )
                }
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  (effective("evaluation.compute_entity_metrics") as boolean)
                    ? "bg-blue-600"
                    : "bg-gray-200"
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    (effective("evaluation.compute_entity_metrics") as boolean)
                      ? "translate-x-6"
                      : "translate-x-1"
                  }`}
                />
              </button>
            </div>
            {/* RAGAS Repeat Count — visible when RAGAS is enabled */}
            {(effective("evaluation.compute_ragas") as boolean) !== false && (
            <div className="flex items-center justify-between mt-3">
              <div>
                <span className="text-sm font-medium text-gray-700">RAGAS Repeat Count</span>
                <p className="text-xs text-gray-400">Run LLM evaluation N times and average (1 = no repeat)</p>
              </div>
              <input
                type="number"
                min={1}
                max={10}
                value={(effective("evaluation.ragas_repeat_count") as number) ?? 1}
                onChange={(e) =>
                  handleChange(
                    "evaluation.ragas_repeat_count",
                    Math.max(1, Math.min(10, parseInt(e.target.value) || 1)),
                  )
                }
                className="w-16 rounded border border-gray-300 px-2 py-1 text-sm text-center font-mono text-gray-700 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
            )}
          </Section>

          {/* ── Part 6: Results (hidden when generation disabled) ── */}
          {!hideSections?.has("Results") && needsGeneration && !isGenerationDisabled(effective("generation.model") as string) && (
          <Section title="Results">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm font-medium text-gray-700">Highlight Chunks</span>
                <p className="text-xs text-gray-400">Use LLM to highlight relevant passages</p>
              </div>
              <button
                onClick={() =>
                  handleChange(
                    "generation.highlight_chunks",
                    !(effective("generation.highlight_chunks") as boolean),
                  )
                }
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  (effective("generation.highlight_chunks") as boolean)
                    ? "bg-blue-600"
                    : "bg-gray-200"
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    (effective("generation.highlight_chunks") as boolean)
                      ? "translate-x-6"
                      : "translate-x-1"
                  }`}
                />
              </button>
            </div>
          </Section>
          )}
          {/* ── Sweep combination counter ── */}
          {multiSelect && (
            <div className="rounded-lg bg-blue-50 border border-blue-200 px-4 py-2.5 text-sm text-blue-800 flex items-center justify-between">
              <span className="font-medium">Sweep Combinations</span>
              <span className="font-mono text-lg font-bold">{combinationCount}</span>
            </div>
          )}
          </>)}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-100">
          <button
            onClick={onReset}
            className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 transition"
          >
            Reset to Default
          </button>
          <button
            onClick={onClose}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 transition"
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  );
}

// ── System Prompt sub-component ──────────────────────────────────────

function SystemPromptSection({
  currentPrompt,
  onPromptChange,
}: {
  currentPrompt: string;
  onPromptChange: (value: string) => void;
}) {
  // Detect which preset matches current prompt
  const matchedPreset = SYSTEM_PROMPT_PRESETS.find(
    (p) => p.value !== "__custom__" && p.value === currentPrompt,
  );
  const defaultPreset = SYSTEM_PROMPT_PRESETS[0]!;
  const initialKey = !currentPrompt
    ? defaultPreset.label
    : matchedPreset
      ? matchedPreset.label
      : "custom";

  const [selectedPreset, setSelectedPreset] = useState(initialKey);

  // Populate textarea with default text on first render when prompt is empty
  useEffect(() => {
    if (!currentPrompt) {
      onPromptChange(defaultPreset.value);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const [selectedPiece, setSelectedPiece] = useState<string>(PROMPT_PIECES[0].key);
  const [pieceText, setPieceText] = useState<string>(PROMPT_PIECES[0].default);

  const handleAddPiece = useCallback(() => {
    if (!pieceText.trim()) return;
    const updated = currentPrompt
      ? currentPrompt.trimEnd() + "\n" + pieceText.trim()
      : pieceText.trim();
    onPromptChange(updated);
    if (selectedPreset !== "custom") setSelectedPreset("custom");
  }, [pieceText, currentPrompt, onPromptChange, selectedPreset]);

  const handlePresetChange = useCallback(
    (presetKey: string) => {
      setSelectedPreset(presetKey);
      if (presetKey === "custom") {
        // Keep current text
      } else {
        const found = SYSTEM_PROMPT_PRESETS.find((p) => p.label === presetKey);
        if (found && found.value !== "__custom__") {
          onPromptChange(found.value);
        }
      }
    },
    [onPromptChange],
  );

  return (
    <div className="space-y-3">
      {/* Preset dropdown */}
      <select
        value={selectedPreset}
        onChange={(e) => handlePresetChange(e.target.value)}
        className="w-full rounded border-gray-300 bg-white px-3 py-1.5 text-sm shadow-sm
                   focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
      >
        {SYSTEM_PROMPT_PRESETS.map((p) => (
          <option key={p.label} value={p.value === "__custom__" ? "custom" : p.label}>
            {p.label}
          </option>
        ))}
      </select>

      {/* Editable textarea */}
      <textarea
        value={currentPrompt}
        onChange={(e) => {
          onPromptChange(e.target.value);
          if (selectedPreset !== "custom") setSelectedPreset("custom");
        }}
        placeholder="Enter a system prompt..."
        rows={3}
        className="w-full rounded border-gray-300 px-3 py-2 text-sm shadow-sm
                   focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                   placeholder:text-gray-400"
      />

      {/* Prompt pieces */}
      <div className="space-y-2">
        <h4 className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
          Prompt Pieces
        </h4>
        <div className="flex items-center gap-2">
          <select
            value={selectedPiece}
            onChange={(e) => {
              setSelectedPiece(e.target.value);
              const piece = PROMPT_PIECES.find((p) => p.key === e.target.value);
              if (piece) setPieceText(piece.default);
            }}
            className="rounded border-gray-300 bg-white px-2 py-1.5 text-sm shadow-sm
                       focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          >
            {PROMPT_PIECES.map((piece) => (
              <option key={piece.key} value={piece.key}>
                {piece.label}
              </option>
            ))}
          </select>
          <button
            onClick={handleAddPiece}
            className="shrink-0 rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white
                       hover:bg-blue-700 transition"
            title="Append piece to system prompt"
          >
            +
          </button>
        </div>
        <textarea
          value={pieceText}
          onChange={(e) => setPieceText(e.target.value)}
          rows={2}
          className="w-full rounded border-gray-200 px-2 py-1.5 text-xs shadow-sm
                     focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                     bg-white text-gray-600"
        />
      </div>
    </div>
  );
}

// ── Section divider ──────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400">
        {title}
      </h3>
      {children}
    </div>
  );
}
