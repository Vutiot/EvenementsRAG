/** Fetch wrapper for /api/* endpoints. */

import type {
  BenchmarkConfig,
  CollectionCreateRequest,
  CollectionCreateResponse,
  CollectionListResponse,
  DatasetCreateRequest,
  DatasetDetail,
  DatasetInfo,
  DatasetProgressEvent,
  DatasetRegistryEntry,
  EnsureCollectionRequest,
  EnsureCollectionResponse,
  HighlightChunksResponse,
  NormalizedBenchmarkResult,
  PresetInfo,
  QueryResult,
  ResultFileInfo,
} from "./types";

const BASE = "/api";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export function getHealth(): Promise<{ status: string }> {
  return fetchJSON(`${BASE}/health`);
}

export function getPresets(): Promise<PresetInfo[]> {
  return fetchJSON(`${BASE}/presets`);
}

export function getPresetConfig(filename: string): Promise<BenchmarkConfig> {
  return fetchJSON(`${BASE}/presets/${encodeURIComponent(filename)}`);
}

export function getResultFiles(): Promise<ResultFileInfo[]> {
  return fetchJSON(`${BASE}/results`);
}

export function getResult(filename: string): Promise<NormalizedBenchmarkResult> {
  // Encode each segment separately to preserve '/' for subdirectory paths
  const encoded = filename.split("/").map(encodeURIComponent).join("/");
  return fetchJSON(`${BASE}/results/${encoded}`);
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

export function getCollections(): Promise<CollectionListResponse> {
  return fetchJSON(`${BASE}/collections`);
}

export function createCollection(
  request: CollectionCreateRequest,
): Promise<CollectionCreateResponse> {
  return fetchJSON(`${BASE}/collections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export function ensureCollection(
  request: EnsureCollectionRequest,
): Promise<EnsureCollectionResponse> {
  return fetchJSON(`${BASE}/ensure-collection`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export function deleteCollection(
  backend: string,
  name: string,
): Promise<{ status: string; collection_name: string; backend: string }> {
  return fetchJSON(
    `${BASE}/collections/${encodeURIComponent(backend)}/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

export function getDatasetRegistry(): Promise<{ datasets: DatasetRegistryEntry[] }> {
  return fetchJSON(`${BASE}/datasets/registry`);
}

export function getDatasets(): Promise<{ datasets: DatasetInfo[] }> {
  return fetchJSON(`${BASE}/datasets`);
}

export function getDataset(id: string): Promise<DatasetDetail> {
  return fetchJSON(`${BASE}/datasets/${encodeURIComponent(id)}`);
}

export function deleteDataset(
  id: string,
): Promise<{ status: string; dataset_id: string }> {
  return fetchJSON(`${BASE}/datasets/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

/**
 * Start dataset generation via SSE streaming.
 * Returns an AbortController to cancel the request.
 */
export function generateDataset(
  request: DatasetCreateRequest,
  callbacks: {
    onProgress: (e: DatasetProgressEvent) => void;
    onCategoryComplete: (e: { category: string; generated: number; total: number }) => void;
    onComplete: (e: { dataset_id: string; total_generated: number }) => void;
    onError: (msg: string) => void;
  },
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/datasets/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        callbacks.onError(`${res.status}: ${text}`);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError("No response body");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let currentEvent = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6));
            if (currentEvent === "progress") callbacks.onProgress(data);
            else if (currentEvent === "category_complete") callbacks.onCategoryComplete(data);
            else if (currentEvent === "complete") callbacks.onComplete(data);
            else if (currentEvent === "error") callbacks.onError(data.message);
          }
        }
      }
    } catch (err: unknown) {
      if ((err as Error).name !== "AbortError") {
        callbacks.onError(err instanceof Error ? err.message : String(err));
      }
    }
  })();

  return controller;
}

export function executeQuery(
  query: string,
  preset: string,
  configOverrides?: Record<string, unknown>,
): Promise<QueryResult> {
  return fetchJSON(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      preset,
      config_overrides: configOverrides ?? null,
    }),
  });
}

export function highlightChunks(
  query: string,
  chunks: { chunk_id: string; content: string }[],
  model?: string,
): Promise<HighlightChunksResponse> {
  return fetchJSON(`${BASE}/query/highlight-chunks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      chunks,
      ...(model ? { model } : {}),
    }),
  });
}
