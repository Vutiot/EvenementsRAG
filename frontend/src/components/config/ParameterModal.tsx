/** Centered modal for tuning RAG pipeline parameters — 3-part structure. */

import { useState, useEffect } from "react";
import ParamChips from "./ParamChips";
import ParamSlider from "./ParamSlider";
import CollectionSection, { type CollectionMode } from "./CollectionSection";
import type { BenchmarkConfig } from "../../api/types";
import type { ParsedCollectionParams } from "../../constants/paramOptions";
import {
  DATASET_OPTIONS,
  TECHNIQUE_OPTIONS,
  SPARSE_TYPE_OPTIONS,
  FUSION_OPTIONS,
  TOP_K_OPTIONS,
  LLM_MODELS,
  MAX_TOKENS_OPTIONS,
  EMBEDDING_DIMENSION_MAP,
} from "../../constants/paramOptions";

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

// ── Types ────────────────────────────────────────────────────────────

interface ParameterModalProps {
  open: boolean;
  onClose: () => void;
  baseConfig: BenchmarkConfig;
  overrides: Record<string, unknown>;
  onOverrideChange: (path: string, value: unknown) => void;
  onReset: () => void;
}

// ── Component ────────────────────────────────────────────────────────

export default function ParameterModal({
  open,
  onClose,
  baseConfig,
  overrides,
  onOverrideChange,
  onReset,
}: ParameterModalProps) {
  const [collectionMode, setCollectionMode] = useState<CollectionMode>("create");

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Reset collection mode when all overrides are cleared
  useEffect(() => {
    if (Object.keys(overrides).length === 0) {
      setCollectionMode("create");
    }
  }, [overrides]);

  if (!open) return null;

  const base = baseConfig as unknown as Record<string, unknown>;

  /** Get effective value (override wins, else base). */
  function effective(path: string): unknown {
    const overrideVal = getPath(overrides, path);
    if (overrideVal !== undefined) return overrideVal;
    return getPath(base, path);
  }

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
    }
  };

  const handleModeChange = (mode: CollectionMode) => {
    setCollectionMode(mode);
    if (mode === "create") {
      // Clear explicit collection_name when switching to create (ensureCollection will derive it)
      handleChange("dataset.collection_name", undefined);
    }
  };

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
          <h2 className="text-lg font-semibold text-gray-900">Tune Parameters</h2>
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
            <CollectionSection
              datasetName={effective("dataset.dataset_name") as string}
              backend={effective("vector_db.backend") as string}
              chunkSize={effective("chunking.chunk_size") as number}
              chunkOverlap={effective("chunking.chunk_overlap") as number}
              embeddingModel={effective("embedding.model_name") as string}
              distanceMetric={effective("vector_db.distance_metric") as string}
              presetValues={{
                backend: preset("vector_db.backend") as string,
                chunkSize: preset("chunking.chunk_size") as number,
                chunkOverlap: preset("chunking.chunk_overlap") as number,
                embeddingModel: preset("embedding.model_name") as string,
                distanceMetric: preset("vector_db.distance_metric") as string,
              }}
              onParamChange={handleCollectionParamChange}
              onCollectionSelect={handleCollectionSelect}
              mode={collectionMode}
              onModeChange={handleModeChange}
            />
          </Section>

          {/* ── Part 3: Pipeline ── */}
          <Section title="Pipeline">
            {/* Retrieval */}
            <div className="space-y-3">
              <h4 className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
                Retrieval
              </h4>
              <ParamChips
                label="Technique"
                options={TECHNIQUE_OPTIONS}
                value={effective("retrieval.technique") as string}
                presetValue={preset("retrieval.technique") as string}
                onChange={(v) => handleChange("retrieval.technique", v)}
              />
              <ParamChips
                label="Top K Chunks"
                options={TOP_K_OPTIONS}
                value={effective("generation.top_k_chunks") as number}
                presetValue={preset("generation.top_k_chunks") as number}
                onChange={(v) => handleChange("generation.top_k_chunks", v)}
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
            </div>

            {/* Generation */}
            <div className="space-y-3 mt-4">
              <h4 className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
                Generation
              </h4>
              <ParamChips
                label="LLM Model"
                options={LLM_MODELS}
                value={effective("generation.model") as string}
                presetValue={preset("generation.model") as string}
                onChange={(v) => handleChange("generation.model", v)}
              />
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
            </div>
          </Section>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-100">
          <button
            onClick={onReset}
            className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 transition"
          >
            Reset to Preset
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
