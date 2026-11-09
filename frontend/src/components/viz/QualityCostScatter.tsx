import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { EChartWrapper, formatMetricValue } from "../charts";
import {
  EFFICIENCY_METRIC_KEYS,
  QUALITY_METRIC_KEYS,
  metricLabel,
} from "../../constants/metricGroups";
import type { SweepConfig } from "./sweepUtils";

interface Props {
  configs: SweepConfig[];
}

/** Sweep-wide trade-off: one point per run, quality metric against cost or latency per question. */
export default function QualityCostScatter({ configs }: Props) {
  const availableQuality = QUALITY_METRIC_KEYS.filter((k) => configs.some((c) => c.metrics[k] !== undefined));
  const availableCost = EFFICIENCY_METRIC_KEYS.filter((k) => configs.some((c) => c.metrics[k] !== undefined));

  const [yKey, setYKey] = useState<string>(
    availableQuality.includes("answer_correctness") ? "answer_correctness" : availableQuality[0] ?? "",
  );
  const [xKey, setXKey] = useState<string>(availableCost[0] ?? "");

  const option = useMemo((): EChartsOption | null => {
    if (!xKey || !yKey) return null;
    const points = configs
      .filter((c) => c.metrics[xKey] !== undefined && c.metrics[yKey] !== undefined)
      .map((c) => ({ name: c.label, value: [c.metrics[xKey]!, c.metrics[yKey]!] }));
    if (points.length === 0) return null;
    return {
      tooltip: {
        trigger: "item",
        formatter: (p: unknown) => {
          const item = p as { name: string; value: [number, number] };
          return `${item.name}<br/>${metricLabel(yKey)}: ${formatMetricValue(yKey, item.value[1])}<br/>${metricLabel(xKey)}: ${formatMetricValue(xKey, item.value[0])}`;
        },
      },
      grid: { left: 56, right: 24, top: 24, bottom: 48 },
      xAxis: { type: "value", name: metricLabel(xKey), nameLocation: "middle", nameGap: 30, scale: true },
      yAxis: { type: "value", name: metricLabel(yKey), min: 0, max: 1 },
      series: [
        {
          type: "scatter",
          symbolSize: 12,
          data: points,
          label: { show: points.length <= 12, position: "right", formatter: "{b}", fontSize: 10, color: "#64748b" },
        },
      ],
    };
  }, [configs, xKey, yKey]);

  if (availableQuality.length === 0 || availableCost.length === 0) return null;

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold text-slate-800">Quality vs Cost</h2>
        <select
          value={yKey}
          onChange={(e) => setYKey(e.target.value)}
          className="rounded border border-slate-200 bg-white px-2 py-1 text-xs"
        >
          {availableQuality.map((k) => (
            <option key={k} value={k}>{metricLabel(k)}</option>
          ))}
        </select>
        <span className="text-xs text-slate-400">against</span>
        <select
          value={xKey}
          onChange={(e) => setXKey(e.target.value)}
          className="rounded border border-slate-200 bg-white px-2 py-1 text-xs"
        >
          {availableCost.map((k) => (
            <option key={k} value={k}>{metricLabel(k)}</option>
          ))}
        </select>
        <span className="text-xs text-slate-400">best runs sit top-left</span>
      </div>
      {option ? (
        <EChartWrapper option={option} height={380} />
      ) : (
        <p className="text-sm italic text-slate-400">No run has both values.</p>
      )}
    </section>
  );
}
