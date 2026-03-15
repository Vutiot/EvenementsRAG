/** Shared collection selector / creator — used in ParameterModal and CollectionManager. */

import { useState, useEffect } from "react";
import ParamChips from "./ParamChips";
import { getCollections } from "../../api/client";
import type { CollectionInfo } from "../../api/types";
import {
  CHUNK_SIZE_OPTIONS,
  CHUNK_OVERLAP_VALUES,
  EMBEDDING_MODELS,
  DISTANCE_OPTIONS,
  BACKEND_OPTIONS,
  deriveCollectionName,
  parseCollectionName,
  type ParsedCollectionParams,
} from "../../constants/paramOptions";

export type CollectionMode = "create" | "select";

interface CollectionSectionProps {
  datasetName: string;
  backend: string;
  chunkSize: number;
  chunkOverlap: number;
  embeddingModel: string;
  distanceMetric: string;
  presetValues?: {
    backend?: string;
    chunkSize?: number;
    chunkOverlap?: number;
    embeddingModel?: string;
    distanceMetric?: string;
  };
  onParamChange: (field: string, value: string | number) => void;
  onCollectionSelect?: (collectionName: string, params: ParsedCollectionParams | null) => void;
  mode?: CollectionMode;
  onModeChange?: (mode: CollectionMode) => void;
  backendsAvailable?: string[];
  showCreateButton?: boolean;
  onCreateCollection?: () => void;
  creating?: boolean;
}

const BACKEND_BADGE: Record<string, string> = {
  qdrant: "bg-blue-100 text-blue-700",
  faiss: "bg-emerald-100 text-emerald-700",
  pgvector: "bg-purple-100 text-purple-700",
};

export default function CollectionSection({
  datasetName,
  backend,
  chunkSize,
  chunkOverlap,
  embeddingModel,
  distanceMetric,
  presetValues,
  onParamChange,
  onCollectionSelect,
  mode = "create",
  onModeChange,
  backendsAvailable,
  showCreateButton = false,
  onCreateCollection,
  creating = false,
}: CollectionSectionProps) {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCollection, setSelectedCollection] = useState("");

  // Fetch collections when entering select mode
  useEffect(() => {
    if (mode !== "select") return;
    setLoading(true);
    getCollections()
      .then((res) => setCollections(res.collections))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [mode]);

  // Reset selection when dataset changes
  useEffect(() => {
    setSelectedCollection("");
  }, [datasetName]);

  // Clear selection when leaving select mode
  useEffect(() => {
    if (mode !== "select") setSelectedCollection("");
  }, [mode]);

  // Filter collections by dataset
  const defaultCol = deriveCollectionName(datasetName);
  const filteredCollections = collections.filter(
    (c) => c.name.startsWith(datasetName) || c.name === defaultCol,
  );

  // Derived name preview (create mode)
  const derivedName = deriveCollectionName(
    datasetName, backend, chunkSize, chunkOverlap, embeddingModel, distanceMetric,
  );

  // Overlap options filtered by chunk size
  const overlapOptions = CHUNK_OVERLAP_VALUES.map((v) => ({
    value: v,
    label: String(v),
    disabled: v >= chunkSize,
  }));

  // Backend options with availability
  const backendOptions = backendsAvailable
    ? BACKEND_OPTIONS.map((opt) => ({
        ...opt,
        disabled: opt.disabled || !backendsAvailable.includes(opt.value),
      }))
    : BACKEND_OPTIONS;

  const handleSelect = (name: string) => {
    setSelectedCollection(name);
    const parsed = name ? parseCollectionName(name) : null;
    onCollectionSelect?.(name, parsed);
  };

  const selectedInfo = collections.find((c) => c.name === selectedCollection);

  return (
    <div className="space-y-3">
      {/* Mode tabs */}
      {onModeChange && (
        <div className="flex gap-1 rounded-lg bg-gray-100 p-0.5">
          <button
            onClick={() => onModeChange("select")}
            className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition ${
              mode === "select"
                ? "bg-white text-gray-900 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Select Existing
          </button>
          <button
            onClick={() => onModeChange("create")}
            className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition ${
              mode === "create"
                ? "bg-white text-gray-900 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Create New
          </button>
        </div>
      )}

      {mode === "select" ? (
        <div className="space-y-2">
          <select
            value={selectedCollection}
            onChange={(e) => handleSelect(e.target.value)}
            className="w-full rounded border-gray-300 bg-white px-3 py-1.5 text-sm shadow-sm
                       focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            disabled={loading}
          >
            <option value="">
              {loading ? "Loading..." : "Choose a collection..."}
            </option>
            {filteredCollections.map((c) => (
              <option key={`${c.backend}::${c.name}`} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>

          {/* Info badges for selected collection */}
          {selectedInfo && (
            <div className="flex flex-wrap gap-2">
              <span
                className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${
                  BACKEND_BADGE[selectedInfo.backend] ?? "bg-gray-100 text-gray-600"
                }`}
              >
                {selectedInfo.backend}
              </span>
              {selectedInfo.points_count != null && (
                <span className="inline-block rounded-full px-2.5 py-0.5 text-xs font-medium bg-gray-100 text-gray-600">
                  {selectedInfo.points_count.toLocaleString()} vectors
                </span>
              )}
              {selectedInfo.distance && (
                <span className="inline-block rounded-full px-2.5 py-0.5 text-xs font-medium bg-gray-100 text-gray-600">
                  {selectedInfo.distance}
                </span>
              )}
              {selectedInfo.vector_size != null && (
                <span className="inline-block rounded-full px-2.5 py-0.5 text-xs font-medium bg-gray-100 text-gray-600">
                  {selectedInfo.vector_size}d
                </span>
              )}
            </div>
          )}

          {!loading && filteredCollections.length === 0 && (
            <p className="text-xs text-gray-400">
              No collections found for {datasetName}.
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <ParamChips
            label="Backend"
            options={backendOptions}
            value={backend}
            presetValue={presetValues?.backend ?? backend}
            onChange={(v) => onParamChange("backend", v)}
          />
          <ParamChips
            label="Chunk Size"
            options={CHUNK_SIZE_OPTIONS}
            value={chunkSize}
            presetValue={presetValues?.chunkSize ?? chunkSize}
            onChange={(v) => onParamChange("chunkSize", v)}
          />
          <ParamChips
            label="Overlap"
            options={overlapOptions}
            value={chunkOverlap}
            presetValue={presetValues?.chunkOverlap ?? chunkOverlap}
            onChange={(v) => onParamChange("chunkOverlap", v)}
          />
          <ParamChips
            label="Embedding"
            options={EMBEDDING_MODELS}
            value={embeddingModel}
            presetValue={presetValues?.embeddingModel ?? embeddingModel}
            onChange={(v) => onParamChange("embeddingModel", v)}
          />
          <ParamChips
            label="Distance"
            options={DISTANCE_OPTIONS}
            value={distanceMetric}
            presetValue={presetValues?.distanceMetric ?? distanceMetric}
            onChange={(v) => onParamChange("distanceMetric", v)}
          />

          {/* Collection name preview */}
          <div className="flex items-center gap-3">
            <span className="w-28 shrink-0 text-sm text-gray-600">Collection</span>
            <span className="font-mono text-xs text-gray-700 truncate">{derivedName}</span>
          </div>

          {showCreateButton && onCreateCollection && (
            <div className="flex items-center gap-3 pt-1">
              <button
                onClick={onCreateCollection}
                disabled={creating}
                className="rounded bg-blue-600 px-5 py-2 text-sm font-medium text-white
                           hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                           transition-colors"
              >
                {creating ? "Creating..." : "Create & Index"}
              </button>
              {creating && (
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
                  Indexing...
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
