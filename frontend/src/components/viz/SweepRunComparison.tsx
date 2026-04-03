import { useMemo, useState } from "react";
import {
  EChartWrapper,
  buildRadarChart,
  buildParallelChart,
  buildHeatmapChart,
  SERIES_COLORS,
} from "../charts";
import type { HeatmapDataPoint } from "../charts";
import { ALL_METRIC_KEYS, HIGHER_IS_BETTER } from "../../constants/metricGroups";
import { metricDisplayName } from "../charts/chartBuilders";
import type { SweepConfig } from "./sweepUtils";

interface Props {
  configs: SweepConfig[];
  activeMetrics: Set<string>;
}

type ViewType = "radar" | "parallel" | "heatmap";

function autoView(count: number): ViewType {
  if (count <= 5) return "radar";
  if (count <= 25) return "parallel";
  return "heatmap";
}

function activeMetricKeys(active: Set<string>): string[] {
  return ALL_METRIC_KEYS.filter((k) => active.has(k));
}

export default function SweepRunComparison({ configs, activeMetrics }: Props) {
  const metricKeys = useMemo(
    () => activeMetricKeys(activeMetrics),
    [activeMetrics],
  );

  const defaultView = useMemo(() => autoView(configs.length), [configs.length]);
  const [viewOverride, setViewOverride] = useState<ViewType | null>(null);
  const view = viewOverride ?? defaultView;

  const [sortMetric, setSortMetric] = useState<string | null>(null);

  // Metrics that actually have data in at least one config
  const availableMetrics = useMemo(
    () => metricKeys.filter((k) => configs.some((c) => c.metrics[k] !== undefined)),
    [metricKeys, configs],
  );

  // ── Radar ───────────────────────────────────────────────────────
  const radarOption = useMemo(() => {
    if (view !== "radar" || availableMetrics.length < 3) return null;
    return buildRadarChart({
      indicators: availableMetrics.map((k) => ({
        name: metricDisplayName(k),
        max: 1,
      })),
      series: configs.map((c, i) => ({
        name: c.label,
        values: availableMetrics.map((k) => c.metrics[k] ?? 0),
        color: SERIES_COLORS[i % SERIES_COLORS.length],
      })),
    });
  }, [view, configs, availableMetrics]);

  // ── Parallel ────────────────────────────────────────────────────
  const parallelOption = useMemo(() => {
    if (view !== "parallel" || availableMetrics.length < 2) return null;
    return buildParallelChart({
      dimensions: availableMetrics.map((k) => ({
        name: metricDisplayName(k),
        min: 0,
        max: 1,
      })),
      data: configs.map((c) =>
        availableMetrics.map((k) => c.metrics[k] ?? 0),
      ),
      colors: configs.map(
        (_, i) => SERIES_COLORS[i % SERIES_COLORS.length]!,
      ),
    });
  }, [view, configs, availableMetrics]);

  // ── Heatmap matrix ──────────────────────────────────────────────
  const heatmapOption = useMemo(() => {
    if (view !== "heatmap" || availableMetrics.length === 0) return null;

    // Sort configs by selected metric
    let sorted = configs;
    if (sortMetric && availableMetrics.includes(sortMetric)) {
      const dir = HIGHER_IS_BETTER[sortMetric] !== false ? -1 : 1;
      sorted = [...configs].sort(
        (a, b) => dir * ((a.metrics[sortMetric] ?? 0) - (b.metrics[sortMetric] ?? 0)),
      );
    }

    const xData = availableMetrics.map(metricDisplayName);
    const yData = sorted.map((c) => c.label);
    const data: HeatmapDataPoint[] = [];

    for (let yi = 0; yi < sorted.length; yi++) {
      const c = sorted[yi]!;
      for (let xi = 0; xi < availableMetrics.length; xi++) {
        const v = c.metrics[availableMetrics[xi]!];
        if (v !== undefined) data.push([xi, yi, v]);
      }
    }

    return buildHeatmapChart({ xData, yData, data, min: 0, max: 1 });
  }, [view, configs, availableMetrics, sortMetric]);

  if (configs.length === 0 || availableMetrics.length === 0) {
    return (
      <p className="text-sm text-slate-400 italic">No metric data for comparison.</p>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold text-slate-800">Run Comparison</h2>
        <div className="inline-flex gap-1">
          {(["radar", "parallel", "heatmap"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setViewOverride(t)}
              className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${
                view === t
                  ? "border-blue-200 bg-blue-100 text-blue-700"
                  : "border-gray-200 bg-gray-50 text-gray-500"
              }`}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {view === "heatmap" && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">Sort by:</span>
          <select
            className="rounded border border-slate-200 px-2 py-1 text-xs text-slate-600"
            value={sortMetric ?? ""}
            onChange={(e) => setSortMetric(e.target.value || null)}
          >
            <option value="">Default</option>
            {availableMetrics.map((k) => (
              <option key={k} value={k}>
                {metricDisplayName(k)}
              </option>
            ))}
          </select>
        </div>
      )}

      {view === "radar" && radarOption && (
        <EChartWrapper option={radarOption} height={420} />
      )}
      {view === "parallel" && parallelOption && (
        <EChartWrapper option={parallelOption} height={400} />
      )}
      {view === "heatmap" && heatmapOption && (
        <EChartWrapper
          option={heatmapOption}
          height={Math.max(300, configs.length * 24 + 100)}
        />
      )}
    </div>
  );
}
