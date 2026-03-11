import { useEffect, useState, useCallback, useMemo } from "react";
import PageHeader from "../components/layout/PageHeader";
import ParamChips from "../components/config/ParamChips";
import { getCollections, createCollection, deleteCollection } from "../api/client";
import type { CollectionInfo } from "../api/types";

// ── Embedding model dimension lookup ─────────────────────────────────

const EMBEDDING_MODELS = [
  { value: "all-MiniLM-L6-v2", label: "MiniLM-L6 (384d)", dimension: 384 },
  { value: "all-MiniLM-L12-v2", label: "MiniLM-L12 (384d)", dimension: 384 },
  { value: "BAAI/bge-small-en-v1.5", label: "BGE-small (384d)", dimension: 384 },
  { value: "BAAI/bge-base-en-v1.5", label: "BGE-base (768d)", dimension: 768 },
];

const DIMENSION_MAP: Record<string, number> = Object.fromEntries(
  EMBEDDING_MODELS.map((m) => [m.value, m.dimension]),
);

// ── Backend badge colors ─────────────────────────────────────────────

const BACKEND_BADGE: Record<string, string> = {
  qdrant: "bg-blue-100 text-blue-700",
  faiss: "bg-emerald-100 text-emerald-700",
  pgvector: "bg-purple-100 text-purple-700",
};

// ── Component ────────────────────────────────────────────────────────

export default function CollectionManager() {
  // Collection list state
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [backendsAvailable, setBackendsAvailable] = useState<string[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  // Create form state
  const [dataset, setDataset] = useState<string>("wiki_10k");
  const [backend, setBackend] = useState<string>("qdrant");
  const [chunkSize, setChunkSize] = useState<number>(512);
  const [chunkOverlap, setChunkOverlap] = useState<number>(50);
  const [embeddingModel, setEmbeddingModel] = useState<string>("all-MiniLM-L6-v2");
  const [distanceMetric, setDistanceMetric] = useState<string>("cosine");
  const [collectionName, setCollectionName] = useState<string>("");
  const [nameManuallyEdited, setNameManuallyEdited] = useState(false);

  // Create action state
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createSuccess, setCreateSuccess] = useState<string | null>(null);

  // Delete confirmation state
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  // Auto-generate collection name
  const autoName = useMemo(
    () => `${dataset}_${backend}_cs${chunkSize}_co${chunkOverlap}`,
    [dataset, backend, chunkSize, chunkOverlap],
  );

  useEffect(() => {
    if (!nameManuallyEdited) setCollectionName(autoName);
  }, [autoName, nameManuallyEdited]);

  // Fetch collection list
  const refreshCollections = useCallback(async () => {
    setListLoading(true);
    setListError(null);
    try {
      const res = await getCollections();
      setCollections(res.collections);
      setBackendsAvailable(res.backends_available);
    } catch (e: unknown) {
      setListError(e instanceof Error ? e.message : String(e));
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshCollections();
  }, [refreshCollections]);

  // Create handler
  const handleCreate = useCallback(async () => {
    if (!collectionName.trim()) return;
    setCreating(true);
    setCreateError(null);
    setCreateSuccess(null);
    try {
      const res = await createCollection({
        dataset_name: dataset,
        collection_name: collectionName,
        backend,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        embedding_model: embeddingModel,
        embedding_dimension: DIMENSION_MAP[embeddingModel] ?? 384,
        distance_metric: distanceMetric,
      });
      setCreateSuccess(res.message);
      setNameManuallyEdited(false);
      await refreshCollections();
    } catch (e: unknown) {
      setCreateError(e instanceof Error ? e.message : String(e));
    } finally {
      setCreating(false);
    }
  }, [
    collectionName, dataset, backend, chunkSize, chunkOverlap,
    embeddingModel, distanceMetric, refreshCollections,
  ]);

  // Delete handler
  const handleDelete = useCallback(
    async (b: string, name: string) => {
      setDeleteLoading(true);
      try {
        await deleteCollection(b, name);
        await refreshCollections();
      } catch (e: unknown) {
        setListError(e instanceof Error ? e.message : String(e));
      } finally {
        setDeleteLoading(false);
        setDeletingKey(null);
      }
    },
    [refreshCollections],
  );

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <PageHeader
        title="Collections"
        description="Create, inspect, and delete vector collections across backends."
      />

      {/* ── Create Collection ──────────────────────────────────────── */}
      <section className="rounded border border-gray-200 bg-white p-5 mb-6">
        <h2 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">
          Create Collection
        </h2>

        <div className="space-y-3">
          <ParamChips
            label="Dataset"
            options={[
              { value: "wiki_10k", label: "wiki_10k" },
              { value: "octank", label: "octank" },
            ]}
            value={dataset}
            presetValue={dataset}
            onChange={setDataset}
          />

          <ParamChips
            label="Backend"
            options={[
              { value: "qdrant", label: "Qdrant", disabled: !backendsAvailable.includes("qdrant") },
              { value: "faiss", label: "FAISS", disabled: !backendsAvailable.includes("faiss") },
              { value: "pgvector", label: "pgvector", disabled: !backendsAvailable.includes("pgvector") },
            ]}
            value={backend}
            presetValue={backend}
            onChange={setBackend}
          />

          <ParamChips
            label="Chunk Size"
            options={[
              { value: 256, label: "256" },
              { value: 512, label: "512" },
              { value: 1024, label: "1024" },
            ]}
            value={chunkSize}
            presetValue={chunkSize}
            onChange={setChunkSize}
          />

          <ParamChips
            label="Chunk Overlap"
            options={[
              { value: 0, label: "0" },
              { value: 50, label: "50" },
              { value: 128, label: "128" },
              { value: 256, label: "256" },
            ]}
            value={chunkOverlap}
            presetValue={chunkOverlap}
            onChange={setChunkOverlap}
          />

          <ParamChips
            label="Embedding"
            options={EMBEDDING_MODELS}
            value={embeddingModel}
            presetValue={embeddingModel}
            onChange={setEmbeddingModel}
          />

          <ParamChips
            label="Distance"
            options={[
              { value: "cosine", label: "Cosine" },
              { value: "euclidean", label: "Euclidean" },
              { value: "dot_product", label: "Dot Product" },
            ]}
            value={distanceMetric}
            presetValue={distanceMetric}
            onChange={setDistanceMetric}
          />

          {/* Collection name */}
          <div className="flex items-start gap-3">
            <span className="w-28 shrink-0 pt-1.5 text-sm text-gray-600">
              Name
            </span>
            <div className="flex-1 flex gap-3 items-center">
              <input
                type="text"
                value={collectionName}
                onChange={(e) => {
                  setCollectionName(e.target.value);
                  setNameManuallyEdited(true);
                }}
                className="flex-1 rounded border border-gray-300 px-3 py-1.5 text-sm
                           focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              />
              {nameManuallyEdited && (
                <button
                  type="button"
                  onClick={() => {
                    setNameManuallyEdited(false);
                    setCollectionName(autoName);
                  }}
                  className="text-xs text-gray-400 hover:text-gray-600 transition"
                >
                  Reset
                </button>
              )}
            </div>
          </div>

          {/* Create button */}
          <div className="flex items-center gap-4 pt-2">
            <button
              onClick={handleCreate}
              disabled={creating || !collectionName.trim()}
              className="rounded bg-blue-600 px-5 py-2 text-sm font-medium text-white
                         hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                         transition-colors"
            >
              {creating ? "Creating..." : "Create & Index"}
            </button>

            {creating && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
                Indexing dataset — this may take a few minutes...
              </div>
            )}
          </div>
        </div>

        {/* Alerts */}
        {createError && (
          <div className="mt-4 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {createError}
          </div>
        )}
        {createSuccess && (
          <div className="mt-4 rounded border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
            {createSuccess}
          </div>
        )}
      </section>

      {/* ── Collections Table ──────────────────────────────────────── */}
      <section className="rounded border border-gray-200 bg-white p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-900 uppercase tracking-wider">
            Existing Collections
          </h2>
          <button
            onClick={refreshCollections}
            disabled={listLoading}
            className="text-xs text-gray-500 hover:text-gray-700 transition"
          >
            {listLoading ? "Loading..." : "Refresh"}
          </button>
        </div>

        {listError && (
          <div className="mb-4 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {listError}
          </div>
        )}

        {listLoading && collections.length === 0 ? (
          <div className="flex items-center justify-center py-12">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
          </div>
        ) : collections.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-8">
            No collections found. Create one above or start a vector backend.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  <th className="pb-2 pr-4">Name</th>
                  <th className="pb-2 pr-4">Backend</th>
                  <th className="pb-2 pr-4">Distance</th>
                  <th className="pb-2 pr-4 text-right">Dimension</th>
                  <th className="pb-2 pr-4 text-right">Vectors</th>
                  <th className="pb-2 w-20"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {collections.map((c) => {
                  const key = `${c.backend}::${c.name}`;
                  const isConfirming = deletingKey === key;

                  return (
                    <tr key={key} className="hover:bg-gray-50 transition-colors">
                      <td className="py-2.5 pr-4 font-mono text-xs text-gray-800">
                        {c.name}
                      </td>
                      <td className="py-2.5 pr-4">
                        <span
                          className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${
                            BACKEND_BADGE[c.backend] ?? "bg-gray-100 text-gray-600"
                          }`}
                        >
                          {c.backend}
                        </span>
                      </td>
                      <td className="py-2.5 pr-4 text-gray-600">
                        {c.distance ?? "—"}
                      </td>
                      <td className="py-2.5 pr-4 text-right text-gray-600">
                        {c.vector_size ?? "—"}
                      </td>
                      <td className="py-2.5 pr-4 text-right text-gray-600">
                        {c.points_count != null
                          ? c.points_count.toLocaleString()
                          : "—"}
                      </td>
                      <td className="py-2.5 text-right">
                        {isConfirming ? (
                          <span className="flex items-center justify-end gap-1.5">
                            <button
                              onClick={() => handleDelete(c.backend, c.name)}
                              disabled={deleteLoading}
                              className="text-xs text-red-600 font-medium hover:text-red-800 transition"
                            >
                              {deleteLoading ? "..." : "Confirm"}
                            </button>
                            <button
                              onClick={() => setDeletingKey(null)}
                              className="text-xs text-gray-400 hover:text-gray-600 transition"
                            >
                              Cancel
                            </button>
                          </span>
                        ) : (
                          <button
                            onClick={() => setDeletingKey(key)}
                            className="text-xs text-gray-400 hover:text-red-600 transition"
                          >
                            Delete
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
