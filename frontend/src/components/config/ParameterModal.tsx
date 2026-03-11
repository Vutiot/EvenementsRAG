/** Centered modal for tuning RAG pipeline parameters. */

import { useEffect } from "react";
import ParamChips from "./ParamChips";
import ParamSlider from "./ParamSlider";
import type { BenchmarkConfig } from "../../api/types";

// ── Value registries (mirror backend constants) ──────────────────────

const DATASET_OPTIONS = [
  { value: "wiki_10k", label: "wiki_10k" },
  { value: "octank", label: "octank" },
];

const CHUNK_SIZE_OPTIONS = [
  { value: 256, label: "256" },
  { value: 512, label: "512" },
  { value: 1024, label: "1024" },
];

const CHUNK_OVERLAP_VALUES = [0, 50, 128, 256] as const;

const TECHNIQUE_OPTIONS = [
  { value: "vanilla", label: "vanilla" },
  { value: "hybrid", label: "hybrid" },
];

const TOP_K_OPTIONS = [
  { value: 3, label: "3" },
  { value: 5, label: "5" },
  { value: 10, label: "10" },
];

const FUSION_OPTIONS = [
  { value: "rrf", label: "rrf" },
  { value: "weighted_sum", label: "weighted_sum" },
];

const LLM_MODELS: { value: string; label: string }[] = [
  { value: "mistralai/mistral-small-3.1-24b-instruct:free", label: "Mistral" },
  { value: "meta-llama/llama-3.1-8b-instruct:free", label: "Llama" },
  { value: "google/gemma-2-9b-it:free", label: "Gemma" },
];

const MAX_TOKENS_OPTIONS = [
  { value: 512, label: "512" },
  { value: 1000, label: "1000" },
  { value: 2000, label: "2000" },
  { value: 4000, label: "4000" },
];

const EMBEDDING_MODELS: { value: string; label: string }[] = [
  { value: "sentence-transformers/all-MiniLM-L6-v2", label: "MiniLM-L6" },
  { value: "sentence-transformers/all-MiniLM-L12-v2", label: "MiniLM-L12" },
  { value: "BAAI/bge-small-en-v1.5", label: "BGE-Small" },
  { value: "BAAI/bge-base-en-v1.5", label: "BGE-Base" },
];

const EMBEDDING_DIMENSION_MAP: Record<string, number> = {
  "sentence-transformers/all-MiniLM-L6-v2": 384,
  "sentence-transformers/all-MiniLM-L12-v2": 384,
  "BAAI/bge-small-en-v1.5": 384,
  "BAAI/bge-base-en-v1.5": 768,
};

const DISTANCE_OPTIONS = [
  { value: "cosine", label: "cosine" },
  { value: "euclidean", label: "euclidean" },
  { value: "dot_product", label: "dot_product" },
];

const BACKEND_OPTIONS = [
  { value: "qdrant", label: "qdrant" },
  { value: "faiss", label: "faiss" },
  { value: "pgvector", label: "pgvector", disabled: true },
];

// ── Helpers ──────────────────────────────────────────────────────────

/** Resolve a dotted config path to a value, e.g. "chunking.chunk_size" → 512 */
function getPath(obj: Record<string, unknown>, path: string): unknown {
  const parts = path.split(".");
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

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
  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  // Build flat effective config for reading current values
  const base = baseConfig as unknown as Record<string, unknown>;

  /** Get effective value (override wins, else base) */
  function effective(path: string): unknown {
    const overrideVal = getPath(overrides, path);
    if (overrideVal !== undefined) return overrideVal;
    return getPath(base, path);
  }

  /** Get base/preset value */
  function preset(path: string): unknown {
    return getPath(base, path);
  }

  /** Handle a change: if same as preset, remove override; else set it */
  function handleChange(path: string, value: unknown) {
    const presetVal = preset(path);
    if (value === presetVal) {
      // Remove this override
      onOverrideChange(path, undefined);
    } else {
      onOverrideChange(path, value);
    }
  }

  const technique = effective("retrieval.technique") as string;
  const chunkSize = effective("chunking.chunk_size") as number;

  const overlapOptions = CHUNK_OVERLAP_VALUES.map((v) => ({
    value: v,
    label: String(v),
    disabled: v >= chunkSize,
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
          {/* ── Dataset ── */}
          <Section title="Dataset">
            <ParamChips
              label="Dataset"
              options={DATASET_OPTIONS}
              value={effective("dataset.dataset_name") as string}
              presetValue={preset("dataset.dataset_name") as string}
              onChange={(v) => handleChange("dataset.dataset_name", v)}
            />
          </Section>

          {/* ── Chunking ── */}
          <Section title="Chunking">
            <ParamChips
              label="Chunk Size"
              options={CHUNK_SIZE_OPTIONS}
              value={effective("chunking.chunk_size") as number}
              presetValue={preset("chunking.chunk_size") as number}
              onChange={(v) => handleChange("chunking.chunk_size", v)}
            />
            <ParamChips
              label="Overlap"
              options={overlapOptions}
              value={effective("chunking.chunk_overlap") as number}
              presetValue={preset("chunking.chunk_overlap") as number}
              onChange={(v) => handleChange("chunking.chunk_overlap", v)}
            />
          </Section>

          {/* ── Retrieval ── */}
          <Section title="Retrieval">
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
                  onChange={(v) => handleChange("retrieval.sparse_weight", v)}
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
          </Section>

          {/* ── Generation ── */}
          <Section title="Generation">
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
          </Section>

          {/* ── Embedding & Vector DB ── */}
          <Section title="Embedding & Vector DB">
            <ParamChips
              label="Embedding"
              options={EMBEDDING_MODELS}
              value={effective("embedding.model_name") as string}
              presetValue={preset("embedding.model_name") as string}
              onChange={(v) => {
                handleChange("embedding.model_name", v);
                const dim = EMBEDDING_DIMENSION_MAP[v as string];
                if (dim !== undefined) handleChange("embedding.dimension", dim);
              }}
            />
            <ParamChips
              label="Distance"
              options={DISTANCE_OPTIONS}
              value={effective("vector_db.distance_metric") as string}
              presetValue={preset("vector_db.distance_metric") as string}
              onChange={(v) => handleChange("vector_db.distance_metric", v)}
            />
            <ParamChips
              label="Backend"
              options={BACKEND_OPTIONS}
              value={effective("vector_db.backend") as string}
              presetValue={preset("vector_db.backend") as string}
              onChange={(v) => handleChange("vector_db.backend", v)}
            />
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
