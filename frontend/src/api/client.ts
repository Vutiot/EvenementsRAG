/** Fetch wrapper for /api/* endpoints. */

import type { BenchmarkConfig, PresetInfo, QueryResult } from "./types";

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
