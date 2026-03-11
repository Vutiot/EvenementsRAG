/** Fetch wrapper for /api/* endpoints. */

import type {
  BenchmarkConfig,
  CollectionCreateRequest,
  CollectionCreateResponse,
  CollectionListResponse,
  EnsureCollectionRequest,
  EnsureCollectionResponse,
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
