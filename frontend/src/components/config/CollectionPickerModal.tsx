/** Modal for picking an existing collection — follows QuestionPickerModal pattern. */

import { useState, useEffect } from "react";
import { getCollections } from "../../api/client";
import type { CollectionInfo } from "../../api/types";
import type { ParsedCollectionParams } from "../../constants/paramOptions";
import { parseCollectionName } from "../../constants/paramOptions";

interface Props {
  open: boolean;
  onClose: () => void;
  datasetName: string;
  onSelect: (name: string, params: ParsedCollectionParams | null) => void;
}

const BACKEND_BADGE: Record<string, string> = {
  qdrant: "bg-blue-100 text-blue-700",
  faiss: "bg-emerald-100 text-emerald-700",
  pgvector: "bg-purple-100 text-purple-700",
};

export default function CollectionPickerModal({ open, onClose, datasetName, onSelect }: Props) {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(false);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Fetch collections when modal opens
  useEffect(() => {
    if (!open) return;
    setLoading(true);
    getCollections()
      .then((res) => setCollections(res.collections))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [open]);

  if (!open) return null;

  // Filter by dataset name
  const filtered = collections.filter(
    (c) => c.name.startsWith(datasetName) || c.name.includes(datasetName),
  );

  const handleSelect = (c: CollectionInfo) => {
    const parsed = parseCollectionName(c.name);
    onSelect(c.name, parsed);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-white rounded-2xl shadow-2xl max-w-3xl w-full max-h-[85vh] overflow-hidden mx-4 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3 shrink-0">
          <h2 className="text-lg font-semibold text-gray-900">
            Import Collection
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

        {/* Table */}
        <div className="overflow-y-auto flex-1 px-6 pb-4">
          {loading ? (
            <div className="flex items-center justify-center py-8 gap-2">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
              <span className="text-sm text-gray-500">Loading collections...</span>
            </div>
          ) : filtered.length === 0 ? (
            <p className="py-8 text-center text-sm text-gray-400">
              No collections found for "{datasetName}".
            </p>
          ) : (
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="border-b border-gray-200 text-left text-xs font-medium uppercase tracking-wider text-gray-400">
                  <th className="py-2 pr-2">Name</th>
                  <th className="py-2 pr-2">Backend</th>
                  <th className="py-2 pr-2 text-right">Vectors</th>
                  <th className="py-2 pr-2">Distance</th>
                  <th className="py-2 text-right">Dimension</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((c) => (
                  <tr
                    key={`${c.backend}::${c.name}`}
                    onClick={() => handleSelect(c)}
                    className="border-b border-gray-50 hover:bg-blue-50 cursor-pointer transition-colors"
                  >
                    <td className="py-2 pr-2 font-mono text-xs text-gray-700">{c.name}</td>
                    <td className="py-2 pr-2">
                      <span
                        className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                          BACKEND_BADGE[c.backend] ?? "bg-gray-100 text-gray-600"
                        }`}
                      >
                        {c.backend}
                      </span>
                    </td>
                    <td className="py-2 pr-2 text-right text-gray-600">
                      {c.points_count != null ? c.points_count.toLocaleString() : "—"}
                    </td>
                    <td className="py-2 pr-2 text-gray-600">{c.distance ?? "—"}</td>
                    <td className="py-2 text-right text-gray-600">
                      {c.vector_size != null ? `${c.vector_size}d` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
