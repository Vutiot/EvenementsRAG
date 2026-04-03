import type { NormalizedBenchmarkResult } from "../../api/types";
import { ALL_METRIC_KEYS, getMetricValue } from "../../constants/metricGroups";

// ─── Types ───────────────────────────────────────────────────────────

export interface SweepConfig {
  filename: string;
  label: string;
  params: Record<string, unknown>;
  metrics: Record<string, number>;
  result: NormalizedBenchmarkResult;
}

// ─── Param abbreviations (mirrors backend sweep_service.py) ──────────

const PARAM_SHORT: Record<string, string> = {
  "chunking.chunk_size": "cs",
  "chunking.chunk_overlap": "co",
  "embedding.model_name": "embed",
  "retrieval.top_k": "k",
  "retrieval.sparse_weight": "sw",
  "retrieval.sparse_type": "st",
  "retrieval.fusion_method": "fm",
  "reranker.type": "reranker",
  "vector_db.distance_metric": "dm",
  "vector_db.backend": "backend",
};

export function paramShortName(path: string): string {
  return PARAM_SHORT[path] ?? path.split(".").pop() ?? path;
}

// ─── Dot-path config extraction ──────────────────────────────────────

export function configParamValue(
  result: NormalizedBenchmarkResult,
  paramPath: string,
): unknown {
  if (!result.config) return undefined;
  const parts = paramPath.split(".");
  let cur: unknown = result.config;
  for (const p of parts) {
    if (cur == null || typeof cur !== "object") return undefined;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

// ─── Smart labeling ──────────────────────────────────────────────────

export function smartLabel(
  configs: { params: Record<string, unknown> }[],
  sweptParams: Record<string, unknown[]>,
): string[] {
  // Use swept param keys as varying keys
  const varyingKeys = Object.keys(sweptParams).filter(
    (k) => (sweptParams[k]?.length ?? 0) > 1,
  );

  if (varyingKeys.length === 0) {
    return configs.map((_, i) => `Config ${i + 1}`);
  }

  const pickedKeys = varyingKeys.slice(0, 3);
  return configs.map((c) =>
    pickedKeys
      .map((k) => {
        const short = paramShortName(k);
        const val = c.params[k];
        // Shorten model names
        const display =
          typeof val === "string" && val.includes("/")
            ? val.split("/").pop()
            : String(val ?? "?");
        return `${short}=${display}`;
      })
      .join(" · "),
  );
}

// ─── Build SweepConfig array ─────────────────────────────────────────

export function buildSweepConfigs(
  children: NormalizedBenchmarkResult[],
  sweptParams: Record<string, unknown[]>,
): SweepConfig[] {
  const paramKeys = Object.keys(sweptParams);

  // Extract params and metrics for each child
  const configs: SweepConfig[] = children.map((result) => {
    const params: Record<string, unknown> = {};
    for (const key of paramKeys) {
      params[key] = configParamValue(result, key);
    }

    const metrics: Record<string, number> = {};
    for (const mKey of ALL_METRIC_KEYS) {
      const v = getMetricValue(result, mKey);
      if (v !== null) metrics[mKey] = v;
    }

    return { filename: result.filename, label: "", params, metrics, result };
  });

  // Assign smart labels
  const labels = smartLabel(configs, sweptParams);
  for (let i = 0; i < configs.length; i++) {
    configs[i]!.label = labels[i] ?? `Config ${i + 1}`;
  }

  return configs;
}

// ─── Filter configs ──────────────────────────────────────────────────

export function filterConfigs(
  configs: SweepConfig[],
  filters: Record<string, unknown[]>,
): SweepConfig[] {
  const activeFilters = Object.entries(filters).filter(
    ([, vals]) => vals.length > 0,
  );
  if (activeFilters.length === 0) return configs;

  return configs.filter((c) =>
    activeFilters.every(([key, vals]) => vals.includes(c.params[key])),
  );
}

// ─── Unique param values ─────────────────────────────────────────────

export function uniqueParamValues(
  configs: SweepConfig[],
  paramKey: string,
): unknown[] {
  const set = new Set<string>();
  const values: unknown[] = [];
  for (const c of configs) {
    const v = c.params[paramKey];
    const s = String(v);
    if (!set.has(s)) {
      set.add(s);
      values.push(v);
    }
  }
  return values.sort((a, b) =>
    typeof a === "number" && typeof b === "number"
      ? a - b
      : String(a).localeCompare(String(b)),
  );
}
