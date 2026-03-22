/** Fetch wrapper for /api/* endpoints. */

import type {
  BenchmarkCompleteEvent,
  BenchmarkConfig,
  BenchmarkProgressEvent,
  BenchmarkRunRequest,
  BenchmarkStartedEvent,
  CollectionCreateRequest,
  CollectionCreateResponse,
  CollectionErrorEvent,
  CollectionListResponse,
  CollectionProgressEvent,
  CollectionSSEEvent,
  DatasetCreateRequest,
  DatasetDetail,
  DatasetInfo,
  DatasetProgressEvent,
  DatasetRegistryEntry,
  EnsureCollectionRequest,
  EnsureCollectionResponse,
  EnsureCollectionsRequest,
  HighlightChunksResponse,
  NormalizedBenchmarkResult,
  PresetInfo,

  ResultFileInfo,
  SweepCompleteEvent,
  SweepConfigCompleteEvent,
  SweepConfigProgressEvent,
  SweepConfigStartedEvent,
  SweepRunRequest,
  SweepStartedEvent,
  WarningEvent,
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

export function getDatasets(collectionName?: string): Promise<{ datasets: DatasetInfo[] }> {
  const url = collectionName
    ? `${BASE}/datasets?collection_name=${encodeURIComponent(collectionName)}`
    : `${BASE}/datasets`;
  return fetchJSON(url);
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
 * Ensure collections exist, creating missing ones via SSE streaming.
 * Returns an AbortController to cancel the request.
 */
export function ensureCollections(
  request: EnsureCollectionsRequest,
  callbacks: {
    onExists: (e: CollectionSSEEvent) => void;
    onCreating: (e: CollectionSSEEvent) => void;
    onCreated: (e: CollectionSSEEvent) => void;
    onProgress?: (e: CollectionProgressEvent) => void;
    onError: (e: CollectionErrorEvent) => void;
    onDone: () => void;
  },
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/datasets/ensure-collections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!res.ok) {
        const text = await res.text();
        callbacks.onError({ name: "", index: 0, total: 0, error: `${res.status}: ${text}` });
        callbacks.onDone();
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        callbacks.onError({ name: "", index: 0, total: 0, error: "No response body" });
        callbacks.onDone();
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
            switch (currentEvent) {
              case "collection_exists":
                callbacks.onExists(data);
                break;
              case "collection_creating":
                callbacks.onCreating(data);
                break;
              case "collection_created":
                callbacks.onCreated(data);
                break;
              case "collection_progress":
                callbacks.onProgress?.(data);
                break;
              case "collection_error":
                callbacks.onError(data);
                break;
              case "error":
                callbacks.onError({ name: "", index: 0, total: 0, error: data.message });
                break;
            }
          }
        }
      }

      callbacks.onDone();
    } catch (err: unknown) {
      if ((err as Error).name !== "AbortError") {
        callbacks.onError({
          name: "",
          index: 0,
          total: 0,
          error: err instanceof Error ? err.message : String(err),
        });
      }
      callbacks.onDone();
    }
  })();

  return controller;
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

// ---------------------------------------------------------------------------
// Benchmark runs
// ---------------------------------------------------------------------------

/**
 * Start a benchmark run via SSE streaming.
 * Returns an AbortController to cancel the request.
 */
export function runBenchmark(
  request: BenchmarkRunRequest,
  callbacks: {
    onStarted: (e: BenchmarkStartedEvent) => void;
    onProgress: (e: BenchmarkProgressEvent) => void;
    onComplete: (e: BenchmarkCompleteEvent) => void;
    onWarning?: (e: WarningEvent) => void;
    onError: (msg: string) => void;
  },
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/benchmark/run`, {
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
            if (currentEvent === "started") callbacks.onStarted(data);
            else if (currentEvent === "warning") callbacks.onWarning?.(data);
            else if (currentEvent === "progress") callbacks.onProgress(data);
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

// ---------------------------------------------------------------------------
// Sweep runs
// ---------------------------------------------------------------------------

/**
 * Start a sweep run (cartesian product of params) via SSE streaming.
 * Returns an AbortController to cancel the request.
 */
export function runSweep(
  request: SweepRunRequest,
  callbacks: {
    onSweepStarted: (e: SweepStartedEvent) => void;
    onConfigStarted: (e: SweepConfigStartedEvent) => void;
    onConfigProgress: (e: SweepConfigProgressEvent) => void;
    onConfigComplete: (e: SweepConfigCompleteEvent) => void;
    onSweepComplete: (e: SweepCompleteEvent) => void;
    onWarning?: (e: WarningEvent) => void;
    onError: (msg: string) => void;
  },
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/benchmark/sweep`, {
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
            switch (currentEvent) {
              case "warning":
                callbacks.onWarning?.(data);
                break;
              case "sweep_started":
                callbacks.onSweepStarted(data);
                break;
              case "config_started":
                callbacks.onConfigStarted(data);
                break;
              case "config_progress":
                callbacks.onConfigProgress(data);
                break;
              case "config_complete":
                callbacks.onConfigComplete(data);
                break;
              case "sweep_complete":
                callbacks.onSweepComplete(data);
                break;
              case "error":
                callbacks.onError(data.message);
                break;
            }
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

// ---------------------------------------------------------------------------
// Query execution
// ---------------------------------------------------------------------------

/**
 * Execute a query with SSE-streamed generation tokens.
 * Returns an AbortController to cancel the request.
 */
export function executeQueryStreaming(
  query: string,
  preset: string,
  configOverrides: Record<string, unknown> | undefined,
  callbacks: {
    onRetrievalComplete: (e: import("./types").QueryStreamRetrievalEvent) => void;
    onGenerationToken: (e: import("./types").QueryStreamTokenEvent) => void;
    onGenerationComplete: (e: import("./types").QueryStreamCompleteEvent) => void;
    onError: (msg: string) => void;
  },
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          preset,
          config_overrides: configOverrides ?? null,
        }),
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
            if (currentEvent === "retrieval_complete")
              callbacks.onRetrievalComplete(data);
            else if (currentEvent === "generation_token")
              callbacks.onGenerationToken(data);
            else if (currentEvent === "generation_complete")
              callbacks.onGenerationComplete(data);
            else if (currentEvent === "error")
              callbacks.onError(data.message);
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
