import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { EChartWrapper, buildHeatmapChart, SERIES_COLORS } from "../charts";
import type { HeatmapDataPoint } from "../charts";
import { ALL_METRIC_KEYS } from "../../constants/metricGroups";
import { metricDisplayName } from "../charts/chartBuilders";
import { paramShortName, uniqueParamValues } from "./sweepUtils";
import type { SweepConfig } from "./sweepUtils";

interface Props {
  configs: SweepConfig[];
  sweptParams: Record<string, unknown[]>;
  activeMetrics: Set<string>;
}

function activeKeys(active: Set<string>): string[] {
  return ALL_METRIC_KEYS.filter((k) => active.has(k));
}

/* ── Parameter Heatmap ─────────────────────────────────────────────── */

function ParamHeatmap({
  configs,
  sweptParams,
  metricKeys,
}: {
  configs: SweepConfig[];
  sweptParams: Record<string, unknown[]>;
  metricKeys: string[];
}) {
  const paramKeys = Object.keys(sweptParams).filter(
    (k) => (sweptParams[k]?.length ?? 0) > 1,
  );

  const [xParam, setXParam] = useState(paramKeys[0] ?? "");
  const [yParam, setYParam] = useState(paramKeys[1] ?? paramKeys[0] ?? "");
  const [metric, setMetric] = useState(metricKeys[0] ?? "");

  const option = useMemo(() => {
    if (!xParam || !yParam || !metric) return null;

    const xVals = uniqueParamValues(configs, xParam).map(String);
    const yVals = uniqueParamValues(configs, yParam).map(String);
    const data: HeatmapDataPoint[] = [];

    for (const c of configs) {
      const xi = xVals.indexOf(String(c.params[xParam]));
      const yi = yVals.indexOf(String(c.params[yParam]));
      const v = c.metrics[metric];
      if (xi >= 0 && yi >= 0 && v !== undefined) {
        data.push([xi, yi, v]);
      }
    }

    return buildHeatmapChart({
      xData: xVals,
      yData: yVals,
      data,
      min: 0,
      max: 1,
      title: `${metricDisplayName(metric)} by ${paramShortName(xParam)} × ${paramShortName(yParam)}`,
    });
  }, [configs, xParam, yParam, metric]);

  if (paramKeys.length < 2) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-slate-700">Parameter Heatmap</h3>
      <div className="flex flex-wrap gap-3">
        <label className="text-xs text-slate-500">
          X-Axis
          <select
            className="ml-1 rounded border border-slate-200 px-2 py-1 text-xs"
            value={xParam}
            onChange={(e) => setXParam(e.target.value)}
          >
            {paramKeys.map((k) => (
              <option key={k} value={k}>{paramShortName(k)}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500">
          Y-Axis
          <select
            className="ml-1 rounded border border-slate-200 px-2 py-1 text-xs"
            value={yParam}
            onChange={(e) => setYParam(e.target.value)}
          >
            {paramKeys.map((k) => (
              <option key={k} value={k}>{paramShortName(k)}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500">
          Metric
          <select
            className="ml-1 rounded border border-slate-200 px-2 py-1 text-xs"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            {metricKeys.map((k) => (
              <option key={k} value={k}>{metricDisplayName(k)}</option>
            ))}
          </select>
        </label>
      </div>
      {option && <EChartWrapper option={option} height={350} />}
    </div>
  );
}

/* ── Scatter Plot ──────────────────────────────────────────────────── */

function MetricScatter({
  configs,
  sweptParams,
  metricKeys,
}: {
  configs: SweepConfig[];
  sweptParams: Record<string, unknown[]>;
  metricKeys: string[];
}) {
  const [xMetric, setXMetric] = useState(metricKeys[0] ?? "");
  const [yMetric, setYMetric] = useState(metricKeys[1] ?? metricKeys[0] ?? "");
  const groupParam = Object.keys(sweptParams).find(
    (k) => (sweptParams[k]?.length ?? 0) > 1,
  );

  const option = useMemo((): EChartsOption | null => {
    if (!xMetric || !yMetric) return null;

    // Group configs by the first swept param for coloring
    const groups = new Map<string, SweepConfig[]>();
    for (const c of configs) {
      const gKey = groupParam ? String(c.params[groupParam]) : "all";
      if (!groups.has(gKey)) groups.set(gKey, []);
      groups.get(gKey)!.push(c);
    }

    const series = Array.from(groups.entries()).map(([name, cfgs], i) => ({
      type: "scatter" as const,
      name,
      data: cfgs.map((c) => [
        c.metrics[xMetric] ?? 0,
        c.metrics[yMetric] ?? 0,
      ]),
      symbolSize: 12,
      itemStyle: { color: SERIES_COLORS[i % SERIES_COLORS.length] },
    }));

    return {
      tooltip: {
        trigger: "item" as const,
        formatter(p: unknown): string {
          const params = p as { seriesName: string; value: [number, number] };
          return [
            `<strong>${params.seriesName}</strong>`,
            `${metricDisplayName(xMetric)}: ${params.value[0].toFixed(3)}`,
            `${metricDisplayName(yMetric)}: ${params.value[1].toFixed(3)}`,
          ].join("<br/>");
        },
      },
      legend: { top: 0 },
      grid: { top: 40, left: 60, right: 20, bottom: 50, containLabel: true },
      xAxis: {
        type: "value" as const,
        name: metricDisplayName(xMetric),
        nameLocation: "center" as const,
        nameGap: 30,
      },
      yAxis: {
        type: "value" as const,
        name: metricDisplayName(yMetric),
        nameLocation: "center" as const,
        nameGap: 45,
      },
      series,
    };
  }, [configs, xMetric, yMetric, groupParam]);

  if (metricKeys.length < 2) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-slate-700">Metric Scatter</h3>
      <div className="flex flex-wrap gap-3">
        <label className="text-xs text-slate-500">
          X
          <select
            className="ml-1 rounded border border-slate-200 px-2 py-1 text-xs"
            value={xMetric}
            onChange={(e) => setXMetric(e.target.value)}
          >
            {metricKeys.map((k) => (
              <option key={k} value={k}>{metricDisplayName(k)}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-500">
          Y
          <select
            className="ml-1 rounded border border-slate-200 px-2 py-1 text-xs"
            value={yMetric}
            onChange={(e) => setYMetric(e.target.value)}
          >
            {metricKeys.map((k) => (
              <option key={k} value={k}>{metricDisplayName(k)}</option>
            ))}
          </select>
        </label>
        {groupParam && (
          <span className="text-xs text-slate-400 self-end">
            colored by {paramShortName(groupParam)}
          </span>
        )}
      </div>
      {option && <EChartWrapper option={option} height={380} />}
    </div>
  );
}

/* ── Main ──────────────────────────────────────────────────────────── */

export default function SweepHeatmapScatter({
  configs,
  sweptParams,
  activeMetrics,
}: Props) {
  const metricKeys = useMemo(
    () => activeKeys(activeMetrics).filter((k) => configs.some((c) => c.metrics[k] !== undefined)),
    [activeMetrics, configs],
  );

  if (configs.length === 0 || metricKeys.length === 0) return null;

  return (
    <div className="space-y-8">
      <ParamHeatmap configs={configs} sweptParams={sweptParams} metricKeys={metricKeys} />
      <MetricScatter configs={configs} sweptParams={sweptParams} metricKeys={metricKeys} />
    </div>
  );
}
