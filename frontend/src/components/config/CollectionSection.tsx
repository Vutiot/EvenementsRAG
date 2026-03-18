/** Unified collection section — Import button + always-visible param chips + derived name. */

import { useState, useEffect } from "react";
import ParamChips from "./ParamChips";
import CollectionPickerModal from "./CollectionPickerModal";
import { getCollections } from "../../api/client";
import type { CollectionInfo } from "../../api/types";
import {
  CHUNK_SIZE_OPTIONS,
  CHUNK_OVERLAP_VALUES,
  EMBEDDING_MODELS,
  DISTANCE_OPTIONS,
  BACKEND_OPTIONS,
  deriveCollectionName,
  type ParsedCollectionParams,
} from "../../constants/paramOptions";

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
  backendsAvailable?: string[];
  showCreateButton?: boolean;
  onCreateCollection?: () => void;
  creating?: boolean;
}

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
  backendsAvailable,
  showCreateButton = false,
  onCreateCollection,
  creating = false,
}: CollectionSectionProps) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const [collections, setCollections] = useState<CollectionInfo[]>([]);

  // Fetch collections once on mount to detect existence
  useEffect(() => {
    getCollections()
      .then((res) => setCollections(res.collections))
      .catch(() => {});
  }, []);

  // Derived name from current params
  const derivedName = deriveCollectionName(
    datasetName, backend, chunkSize, chunkOverlap, embeddingModel, distanceMetric,
  );

  // Check if derived name matches an existing collection
  const exists = collections.some((c) => c.name === derivedName);

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

  const handlePickerSelect = (name: string, params: ParsedCollectionParams | null) => {
    onCollectionSelect?.(name, params);
    // Refresh collection list to keep existence detection up to date
    getCollections()
      .then((res) => setCollections(res.collections))
      .catch(() => {});
  };

  return (
    <div className="space-y-3">
      {/* Import button + collection name row */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setPickerOpen(true)}
          className="shrink-0 rounded border border-gray-300 bg-white px-2.5 py-1 text-xs font-medium text-gray-600
                     hover:bg-gray-50 hover:border-gray-400 transition"
        >
          Import Collection
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-gray-700 truncate">{derivedName}</span>
            {exists ? (
              <span className="shrink-0 inline-block rounded-full bg-green-100 px-2 py-0.5 text-[10px] font-medium text-green-700">
                exists
              </span>
            ) : (
              <span className="shrink-0 inline-block rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-600">
                new
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Param chips — always visible */}
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

      {/* Collection picker modal */}
      <CollectionPickerModal
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        datasetName={datasetName}
        onSelect={handlePickerSelect}
      />
    </div>
  );
}
