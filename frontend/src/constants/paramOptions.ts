/** Shared parameter option constants — mirrors backend registries. */

// ── Dataset ──────────────────────────────────────────────────────────

export const DATASET_OPTIONS = [
  { value: "wiki_10k", label: "wiki_10k" },
  { value: "octank", label: "octank" },
];

// ── Chunking ─────────────────────────────────────────────────────────

export const CHUNK_SIZE_OPTIONS = [
  { value: 256, label: "256" },
  { value: 512, label: "512" },
  { value: 1024, label: "1024" },
];

export const CHUNK_OVERLAP_VALUES = [0, 50, 128, 256] as const;

// ── Embedding ────────────────────────────────────────────────────────

export const EMBEDDING_MODELS: { value: string; label: string }[] = [
  { value: "sentence-transformers/all-MiniLM-L6-v2", label: "MiniLM-L6" },
  { value: "sentence-transformers/all-MiniLM-L12-v2", label: "MiniLM-L12" },
  { value: "BAAI/bge-small-en-v1.5", label: "BGE-Small" },
  { value: "BAAI/bge-base-en-v1.5", label: "BGE-Base" },
];

export const EMBEDDING_DIMENSION_MAP: Record<string, number> = {
  "sentence-transformers/all-MiniLM-L6-v2": 384,
  "sentence-transformers/all-MiniLM-L12-v2": 384,
  "BAAI/bge-small-en-v1.5": 384,
  "BAAI/bge-base-en-v1.5": 768,
};

export const EMBEDDING_SHORT_NAMES: Record<string, string> = {
  "sentence-transformers/all-MiniLM-L6-v2": "minilm_l6",
  "sentence-transformers/all-MiniLM-L12-v2": "minilm_l12",
  "BAAI/bge-small-en-v1.5": "bge_small",
  "BAAI/bge-base-en-v1.5": "bge_base",
};

/** Reverse lookup: short name → full model name */
const SHORT_NAME_TO_MODEL: Record<string, string> = Object.fromEntries(
  Object.entries(EMBEDDING_SHORT_NAMES).map(([k, v]) => [v, k]),
);

// ── Vector DB ────────────────────────────────────────────────────────

export const DISTANCE_OPTIONS = [
  { value: "cosine", label: "cosine" },
  { value: "euclidean", label: "euclidean" },
  { value: "dot_product", label: "dot_product" },
];

export const BACKEND_OPTIONS = [
  { value: "qdrant", label: "qdrant" },
  { value: "faiss", label: "faiss" },
  { value: "pgvector", label: "pgvector", disabled: true },
];

// ── Retrieval ────────────────────────────────────────────────────────

export const TECHNIQUE_OPTIONS = [
  { value: "vanilla", label: "vanilla" },
  { value: "hybrid", label: "hybrid" },
];

export const SPARSE_TYPE_OPTIONS = [
  { value: "bm25", label: "BM25" },
  { value: "tfidf", label: "TF-IDF" },
];

export const FUSION_OPTIONS = [
  { value: "rrf", label: "rrf" },
  { value: "weighted_sum", label: "weighted_sum" },
];

export const TOP_K_OPTIONS = [
  { value: 3, label: "3" },
  { value: 5, label: "5" },
  { value: 10, label: "10" },
];

// ── Generation ───────────────────────────────────────────────────────

export const LLM_MODELS: { value: string; label: string }[] = [
  { value: "mistralai/mistral-small-3.1-24b-instruct:free", label: "Mistral" },
  { value: "meta-llama/llama-3.1-8b-instruct:free", label: "Llama" },
  { value: "google/gemma-2-9b-it:free", label: "Gemma" },
];

export const MAX_TOKENS_OPTIONS = [
  { value: 512, label: "512" },
  { value: 1000, label: "1000" },
  { value: 2000, label: "2000" },
  { value: 4000, label: "4000" },
];

// ── Collection name derivation ───────────────────────────────────────
// Mirrors backend CollectionService.derive_collection_name

const _LEGACY_KEY = JSON.stringify([
  "wiki_10k", "qdrant", 512, 50,
  "sentence-transformers/all-MiniLM-L6-v2", "cosine",
]);
const _LEGACY_NAMES: Record<string, string> = { [_LEGACY_KEY]: "ww2_events_10000" };

export function deriveCollectionName(
  datasetName: string,
  backend = "qdrant",
  chunkSize = 512,
  chunkOverlap = 50,
  embeddingModel = "sentence-transformers/all-MiniLM-L6-v2",
  distanceMetric = "cosine",
): string {
  const key = JSON.stringify([
    datasetName, backend, chunkSize, chunkOverlap, embeddingModel, distanceMetric,
  ]);
  const legacy = _LEGACY_NAMES[key];
  if (legacy) return legacy;

  const embShort =
    EMBEDDING_SHORT_NAMES[embeddingModel] ??
    embeddingModel.split("/").pop()?.toLowerCase() ??
    embeddingModel;
  return `${datasetName}_${backend}_cs${chunkSize}_co${chunkOverlap}_${embShort}_${distanceMetric}`;
}

// ── Parse collection name → params ───────────────────────────────────

export interface ParsedCollectionParams {
  dataset: string;
  backend: string;
  chunkSize: number;
  chunkOverlap: number;
  embeddingModel: string;
  distanceMetric: string;
}

const _LEGACY_REVERSE: Record<string, ParsedCollectionParams> = {
  ww2_events_10000: {
    dataset: "wiki_10k",
    backend: "qdrant",
    chunkSize: 512,
    chunkOverlap: 50,
    embeddingModel: "sentence-transformers/all-MiniLM-L6-v2",
    distanceMetric: "cosine",
  },
};

export function parseCollectionName(name: string): ParsedCollectionParams | null {
  if (name in _LEGACY_REVERSE) return _LEGACY_REVERSE[name] ?? null;

  const match = name.match(
    /^(.+?)_(qdrant|faiss|pgvector)_cs(\d+)_co(\d+)_(.+?)_(cosine|euclidean|dot_product)$/,
  );
  if (!match) return null;

  const embShort = match[5] as string;
  const embeddingModel = SHORT_NAME_TO_MODEL[embShort] ?? embShort;

  return {
    dataset: match[1] as string,
    backend: match[2] as string,
    chunkSize: parseInt(match[3] as string),
    chunkOverlap: parseInt(match[4] as string),
    embeddingModel,
    distanceMetric: match[6] as string,
  };
}
