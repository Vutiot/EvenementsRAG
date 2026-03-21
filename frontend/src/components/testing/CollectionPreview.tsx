import { useMemo } from "react";
import type { BenchmarkConfig, CollectionInfo } from "../../api/types";
import { computeSweepCollections } from "../../utils/configHelpers";

interface CollectionPreviewProps {
  overrides: Record<string, unknown>;
  baseConfig: BenchmarkConfig;
  existingCollections: CollectionInfo[];
}

export default function CollectionPreview({
  overrides,
  baseConfig,
  existingCollections,
}: CollectionPreviewProps) {
  const collections = useMemo(
    () => computeSweepCollections(overrides, baseConfig as unknown as Record<string, unknown>),
    [overrides, baseConfig],
  );

  const existingNames = useMemo(
    () => new Set(existingCollections.map((c) => c.name)),
    [existingCollections],
  );

  const existingCount = collections.filter((c) => existingNames.has(c.collectionName)).length;
  const newCount = collections.length - existingCount;

  if (collections.length <= 1) return null;

  return (
    <div className="rounded border border-gray-200 bg-white p-3">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-medium text-gray-700">Collections</h4>
        <span className="text-xs text-gray-500">
          {collections.length} total
          {existingCount > 0 && (
            <span className="text-green-600 ml-1">({existingCount} existing)</span>
          )}
          {newCount > 0 && (
            <span className="text-amber-600 ml-1">({newCount} new)</span>
          )}
        </span>
      </div>
      <div className="max-h-48 overflow-y-auto space-y-1">
        {collections.map((c) => {
          const exists = existingNames.has(c.collectionName);
          return (
            <div
              key={c.collectionName}
              className="flex items-center justify-between text-xs py-1"
            >
              <span className="font-mono text-gray-600 truncate mr-2">
                {c.collectionName}
              </span>
              <span
                className={`shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-medium ${
                  exists
                    ? "bg-green-100 text-green-700"
                    : "bg-amber-100 text-amber-700"
                }`}
              >
                {exists ? "exists" : "new"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
