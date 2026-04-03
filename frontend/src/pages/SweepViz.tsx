import { useEffect, useState, useCallback, useMemo } from "react";
import { useParams } from "react-router-dom";
import type { SweepMeta } from "../api/types";
import { getResultFiles, getResult } from "../api/client";
import PageHeader from "../components/layout/PageHeader";
import { MetricFilterBar } from "../components/charts";
import {
  RETRIEVAL_METRICS,
  GENERATION_METRICS,
  ALL_METRIC_KEYS,
} from "../constants/metricGroups";
import {
  buildSweepConfigs,
  filterConfigs,
  uniqueParamValues,
  paramShortName,
} from "../components/viz/sweepUtils";
import type { SweepConfig } from "../components/viz/sweepUtils";
import SweepParameterCharts from "../components/viz/SweepParameterCharts";
import SweepRunComparison from "../components/viz/SweepRunComparison";
import SweepHeatmapScatter from "../components/viz/SweepHeatmapScatter";
import SweepDataTable from "../components/viz/SweepDataTable";

type ViewMode = "parameters" | "comparison";

export default function SweepViz() {
  const { sweepId } = useParams<{ sweepId: string }>();

  const [sweepMeta, setSweepMeta] = useState<SweepMeta | null>(null);
  const [sweepName, setSweepName] = useState<string>("");
  const [configs, setConfigs] = useState<SweepConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [activeView, setActiveView] = useState<ViewMode>("parameters");
  const [activeMetrics, setActiveMetrics] = useState<Set<string>>(
    () => new Set(ALL_METRIC_KEYS),
  );
  const [filters, setFilters] = useState<Record<string, unknown[]>>({});

  // ── Data loading ────────────────────────────────────────────────
  useEffect(() => {
    if (!sweepId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    (async () => {
      const files = await getResultFiles();
      const parent = files.find(
        (f) => f.sweep_meta?.sweep_id === decodeURIComponent(sweepId),
      );
      if (!parent?.sweep_meta) {
        if (!cancelled) setError("Sweep not found");
        if (!cancelled) setLoading(false);
        return;
      }

      if (!cancelled) {
        setSweepMeta(parent.sweep_meta);
        setSweepName(parent.run_name ?? parent.phase_name);
      }

      const children = await Promise.all(
        parent.sweep_meta.child_filenames.map((fn) =>
          getResult(fn).catch(() => null),
        ),
      );

      if (cancelled) return;

      const loaded = children.filter(
        (c): c is NonNullable<typeof c> => c !== null,
      );
      setConfigs(buildSweepConfigs(loaded, parent.sweep_meta.swept_params));
      setLoading(false);
    })().catch((e: Error) => {
      if (!cancelled) {
        setError(e.message);
        setLoading(false);
      }
    });

    return () => { cancelled = true; };
  }, [sweepId]);

  // ── Metric toggle ───────────────────────────────────────────────
  const toggleMetric = useCallback((key: string) => {
    setActiveMetrics((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const toggleAllMetrics = useCallback((keys: string[], active: boolean) => {
    setActiveMetrics((prev) => {
      const next = new Set(prev);
      for (const k of keys) {
        if (active) next.add(k);
        else next.delete(k);
      }
      return next;
    });
  }, []);

  // ── Filter toggle ───────────────────────────────────────────────
  const toggleFilter = useCallback((paramKey: string, val: unknown) => {
    setFilters((prev) => {
      const cur = prev[paramKey] ?? [];
      const next = cur.includes(val)
        ? cur.filter((v) => v !== val)
        : [...cur, val];
      return { ...prev, [paramKey]: next };
    });
  }, []);

  // ── Filtered configs ────────────────────────────────────────────
  const filtered = useMemo(
    () => filterConfigs(configs, filters),
    [configs, filters],
  );

  // ── Render ──────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="rounded border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      </div>
    );
  }

  if (!sweepMeta || configs.length === 0) {
    return (
      <div className="p-6">
        <p className="text-sm text-slate-500">No sweep data to display.</p>
      </div>
    );
  }

  const sweptParams = sweepMeta.swept_params;

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <PageHeader
        title={sweepName}
        description={`${configs.length} configurations · ${Object.keys(sweptParams).length} swept parameters`}
      />

      {/* Filter Panel */}
      <details className="rounded border border-slate-200 bg-white">
        <summary className="cursor-pointer px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800">
          Filters
          {Object.values(filters).some((v) => v.length > 0) && (
            <span className="ml-2 rounded-full bg-blue-500 px-2 py-0.5 text-xs text-white">
              {Object.values(filters).filter((v) => v.length > 0).length}
            </span>
          )}
        </summary>
        <div className="space-y-3 border-t border-slate-100 px-4 py-3">
          {Object.entries(sweptParams).map(([paramKey]) => {
            const vals = uniqueParamValues(configs, paramKey);
            if (vals.length < 2) return null;
            const active = filters[paramKey] ?? [];
            return (
              <div key={paramKey}>
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                  {paramShortName(paramKey)}
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {vals.map((v) => {
                    const s = String(v);
                    const isActive = active.includes(v);
                    return (
                      <button
                        key={s}
                        type="button"
                        onClick={() => toggleFilter(paramKey, v)}
                        className={`rounded-full border px-2.5 py-0.5 text-xs font-medium font-mono transition-colors ${
                          isActive
                            ? "border-blue-200 bg-blue-100 text-blue-700"
                            : "border-gray-200 bg-gray-50 text-gray-500"
                        }`}
                      >
                        {s}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </details>

      {/* View toggle */}
      <div className="flex items-center gap-4">
        <div className="inline-flex rounded border border-slate-200 bg-white">
          {(["parameters", "comparison"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setActiveView(mode)}
              className={`px-4 py-1.5 text-sm font-medium transition-colors ${
                activeView === mode
                  ? "bg-blue-500 text-white"
                  : "text-slate-600 hover:bg-slate-50"
              } ${mode === "parameters" ? "rounded-l" : "rounded-r"}`}
            >
              {mode === "parameters" ? "By Parameter" : "By Run"}
            </button>
          ))}
        </div>
        <span className="text-xs text-slate-400">
          {filtered.length}/{configs.length} configs shown
        </span>
      </div>

      {/* Metric Filter */}
      <MetricFilterBar
        metricGroups={[RETRIEVAL_METRICS, GENERATION_METRICS]}
        activeMetrics={activeMetrics}
        onToggle={toggleMetric}
        onToggleAll={toggleAllMetrics}
      />

      {/* Views */}
      {activeView === "parameters" ? (
        <div className="space-y-8">
          <SweepParameterCharts
            configs={filtered}
            sweptParams={sweptParams}
            activeMetrics={activeMetrics}
          />
          <SweepDataTable
            configs={filtered}
            sweptParams={sweptParams}
            activeMetrics={activeMetrics}
          />
        </div>
      ) : (
        <div className="space-y-8">
          <SweepRunComparison
            configs={filtered}
            activeMetrics={activeMetrics}
          />
          <SweepHeatmapScatter
            configs={filtered}
            sweptParams={sweptParams}
            activeMetrics={activeMetrics}
          />
        </div>
      )}
    </div>
  );
}
