import type { EChartsOption } from "echarts";
import { SERIES_COLORS } from "./EChartWrapper";

// ─── Types ───────────────────────────────────────────────────────────

export interface BarSeriesInput {
  name: string;
  data: number[];
  stack?: string;
  color?: string;
}

export interface BuildBarChartParams {
  categories: string[];
  series: BarSeriesInput[];
  yAxisFormat?: "percent" | "ms" | "number";
  title?: string;
  horizontal?: boolean;
}

export interface RadarIndicator {
  name: string;
  max: number;
}

export interface RadarSeriesInput {
  name: string;
  values: number[];
  color?: string;
}

export interface BuildRadarChartParams {
  indicators: RadarIndicator[];
  series: RadarSeriesInput[];
}

export type HeatmapDataPoint = [number, number, number];

export interface BuildHeatmapChartParams {
  xData: string[];
  yData: string[];
  data: HeatmapDataPoint[];
  min?: number;
  max?: number;
  title?: string;
}

export interface ParallelDimension {
  name: string;
  min?: number;
  max?: number;
  type?: "value" | "category";
  data?: string[];
}

export interface BuildParallelChartParams {
  dimensions: ParallelDimension[];
  data: number[][];
  colors?: string[];
}

export interface BuildBoxplotChartParams {
  categories: string[];
  data: number[][];
  title?: string;
}

// ─── Internal helpers ────────────────────────────────────────────────

function titleCase(s: string): string {
  return s
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function computeBoxplotStats(
  values: number[],
): [number, number, number, number, number] {
  if (values.length === 0) return [0, 0, 0, 0, 0];
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) {
    const v = sorted[0]!;
    return [v, v, v, v, v];
  }
  const q = (p: number): number => {
    const idx = (p / 100) * (sorted.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo]!;
    return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (idx - lo);
  };
  return [sorted[0]!, q(25), q(50), q(75), sorted[sorted.length - 1]!];
}

function valueAxisFormatter(
  format: "percent" | "ms" | "number" | undefined,
): ((v: number) => string) | undefined {
  if (format === "percent") return (v: number) => `${(v * 100).toFixed(0)}%`;
  if (format === "ms") return (v: number) => `${v.toFixed(0)} ms`;
  return undefined;
}

// ─── Utility exports ─────────────────────────────────────────────────

export function colorScale(
  value: number,
  min: number,
  max: number,
  higherIsBetter: boolean,
): string {
  const range = max - min;
  let t = range === 0 ? 0.5 : (value - min) / range;
  t = Math.max(0, Math.min(1, t));
  if (!higherIsBetter) t = 1 - t;

  // red (#ef4444) → yellow (#fbbf24) → green (#10b981)
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const s = t * 2;
    r = Math.round(239 + (251 - 239) * s);
    g = Math.round(68 + (191 - 68) * s);
    b = Math.round(68 + (36 - 68) * s);
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(251 + (16 - 251) * s);
    g = Math.round(191 + (185 - 191) * s);
    b = Math.round(36 + (129 - 36) * s);
  }

  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

export function formatMetricValue(key: string, value: number): string {
  if (key.endsWith("_ms") || /_p\d+_ms$/.test(key)) {
    return `${value.toFixed(1)} ms`;
  }
  if (key.includes("wall_time") || key.endsWith("_s")) {
    return `${value.toFixed(1)} s`;
  }
  if (
    key.startsWith("recall_at_") ||
    key.startsWith("ndcg_at_") ||
    key === "mrr" ||
    key === "doc_mrr" ||
    key.includes("precision") ||
    key.includes("hit_at_")
  ) {
    return value.toFixed(4);
  }
  return value.toFixed(3);
}

const SPECIAL_NAMES: Record<string, string> = {
  mrr: "MRR",
  doc_mrr: "Doc MRR",
  ndcg: "NDCG",
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
  answer_correctness: "Answer Correctness",
  answer_similarity: "Answer Similarity",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
  context_relevance: "Context Relevance",
  coherence: "Coherence",
  correctness: "Correctness",
  conciseness: "Conciseness",
  harmfulness: "Harmfulness",
  maliciousness: "Maliciousness",
};

export function metricDisplayName(key: string): string {
  const exact = SPECIAL_NAMES[key];
  if (exact) return exact;

  // _at_N → @N (e.g. recall_at_5 → Recall@5)
  const atMatch = key.match(/^(.+?)_at_(\d+)$/);
  if (atMatch) {
    const base = atMatch[1]!;
    const n = atMatch[2]!;
    const baseName = SPECIAL_NAMES[base];
    return baseName ? `${baseName}@${n}` : `${titleCase(base)}@${n}`;
  }

  // percentile: retrieval_p50_ms → Retrieval P50
  const pMatch = key.match(/^(.+?)_(p\d+)_ms$/);
  if (pMatch) {
    const base = pMatch[1]!;
    const percentile = pMatch[2]!;
    return `${titleCase(base)} ${percentile.toUpperCase()}`;
  }

  // strip _ms suffix: retrieval_time_ms → Retrieval Time
  if (key.endsWith("_ms")) {
    return titleCase(key.slice(0, -3));
  }

  return titleCase(key);
}

// ─── Factory: Bar Chart ──────────────────────────────────────────────

export function buildBarChart(params: BuildBarChartParams): EChartsOption {
  const { categories, series, yAxisFormat, title, horizontal = false } = params;

  const categoryAxis = { type: "category" as const, data: categories };
  const valueAxis = {
    type: "value" as const,
    axisLabel: {
      formatter: valueAxisFormatter(yAxisFormat),
    },
  };

  const echartsSeriesList = series.map((s) => ({
    type: "bar" as const,
    name: s.name,
    data: s.data,
    ...(s.stack ? { stack: s.stack } : {}),
    ...(s.color ? { itemStyle: { color: s.color } } : {}),
  }));

  return {
    ...(title ? { title: { text: title } } : {}),
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
    },
    ...(series.length > 1
      ? { legend: { top: title ? 30 : 0 } }
      : {}),
    grid: {
      top: (title ? 40 : 10) + (series.length > 1 ? 30 : 0),
      left: 60,
      right: 20,
      bottom: horizontal ? 40 : 60,
      containLabel: true,
    },
    xAxis: horizontal ? valueAxis : categoryAxis,
    yAxis: horizontal ? categoryAxis : valueAxis,
    series: echartsSeriesList,
  };
}

// ─── Factory: Radar Chart ────────────────────────────────────────────

export function buildRadarChart(params: BuildRadarChartParams): EChartsOption {
  const { indicators, series } = params;

  return {
    tooltip: { trigger: "item" },
    ...(series.length > 1 ? { legend: { top: 0 } } : {}),
    radar: {
      shape: "polygon" as const,
      indicator: indicators.map((ind) => ({
        name: ind.name,
        max: ind.max,
      })),
    },
    series: [
      {
        type: "radar" as const,
        data: series.map((s, i) => ({
          name: s.name,
          value: s.values,
          lineStyle: {
            color: s.color ?? SERIES_COLORS[i % SERIES_COLORS.length],
          },
          areaStyle: {
            color: s.color ?? SERIES_COLORS[i % SERIES_COLORS.length],
            opacity: 0.15,
          },
          itemStyle: {
            color: s.color ?? SERIES_COLORS[i % SERIES_COLORS.length],
          },
        })),
      },
    ],
  };
}

// ─── Factory: Heatmap Chart ──────────────────────────────────────────

export function buildHeatmapChart(
  params: BuildHeatmapChartParams,
): EChartsOption {
  const { xData, yData, data, title } = params;

  const values = data.map((d) => d[2]);
  const dataMin =
    params.min ?? (values.length > 0 ? Math.min(...values) : 0);
  const dataMax =
    params.max ?? (values.length > 0 ? Math.max(...values) : 1);
  const safeMin = dataMin === dataMax ? dataMin - 0.1 : dataMin;
  const safeMax = dataMin === dataMax ? dataMax + 0.1 : dataMax;

  return {
    ...(title ? { title: { text: title } } : {}),
    tooltip: {
      formatter(p: unknown): string {
        const params = p as {
          value: [number, number, number];
          name: string;
        };
        const [xi, yi, val] = params.value;
        const xLabel = xData[xi] ?? String(xi);
        const yLabel = yData[yi] ?? String(yi);
        return `${yLabel} / ${xLabel}<br/><strong>${val.toFixed(3)}</strong>`;
      },
    },
    grid: {
      top: title ? 50 : 20,
      left: 20,
      right: 60,
      bottom: 60,
      containLabel: true,
    },
    xAxis: {
      type: "category" as const,
      data: xData,
      splitArea: { show: true },
    },
    yAxis: {
      type: "category" as const,
      data: yData,
      splitArea: { show: true },
    },
    visualMap: {
      min: safeMin,
      max: safeMax,
      calculable: true,
      orient: "vertical" as const,
      right: 0,
      top: "center",
      inRange: {
        color: ["#ef4444", "#fbbf24", "#10b981"],
      },
    },
    series: [
      {
        type: "heatmap" as const,
        data,
        label: {
          show: true,
          formatter(p: unknown): string {
            const params = p as { value: [number, number, number] };
            return params.value[2].toFixed(2);
          },
        },
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.25)" },
        },
      },
    ],
  };
}

// ─── Factory: Parallel Coordinates ───────────────────────────────────

export function buildParallelChart(
  params: BuildParallelChartParams,
): EChartsOption {
  const { dimensions, data, colors } = params;

  const parallelAxis = dimensions.map((dim, i) => ({
    dim: i,
    name: dim.name,
    ...(dim.min !== undefined ? { min: dim.min } : {}),
    ...(dim.max !== undefined ? { max: dim.max } : {}),
    ...(dim.type ? { type: dim.type } : {}),
    ...(dim.data ? { data: dim.data } : {}),
  }));

  const seriesList = data.map((row, i) => ({
    type: "parallel" as const,
    data: [row],
    lineStyle: {
      color:
        colors?.[i] ??
        SERIES_COLORS[i % SERIES_COLORS.length],
      width: 1.5,
      opacity: 0.6,
    },
    emphasis: {
      lineStyle: { width: 3, opacity: 1 },
    },
  }));

  return {
    tooltip: { trigger: "item" as const },
    parallel: {
      left: 60,
      right: 60,
      top: 40,
      bottom: 40,
    },
    parallelAxis,
    series: seriesList,
  };
}

// ─── Factory: Box Plot ───────────────────────────────────────────────

export function buildBoxplotChart(
  params: BuildBoxplotChartParams,
): EChartsOption {
  const { categories, data, title } = params;

  const boxData: [number, number, number, number, number][] = [];
  const outlierData: [number, number][] = [];

  for (let i = 0; i < data.length; i++) {
    const raw = data[i]!;
    const stats = computeBoxplotStats(raw);
    boxData.push(stats);

    const [, q1, , q3] = stats;
    const iqr = q3 - q1;
    const lo = q1 - 1.5 * iqr;
    const hi = q3 + 1.5 * iqr;
    for (const v of raw) {
      if (v < lo || v > hi) {
        outlierData.push([i, v]);
      }
    }
  }

  return {
    ...(title ? { title: { text: title } } : {}),
    tooltip: {
      trigger: "item",
      formatter(p: unknown): string {
        const params = p as {
          seriesType: string;
          name: string;
          value: number[];
        };
        if (params.seriesType === "boxplot") {
          const [min, q1, med, q3, max] = params.value.slice(1);
          return [
            `<strong>${params.name}</strong>`,
            `Max: ${max?.toFixed(1)} ms`,
            `Q3: ${q3?.toFixed(1)} ms`,
            `Median: ${med?.toFixed(1)} ms`,
            `Q1: ${q1?.toFixed(1)} ms`,
            `Min: ${min?.toFixed(1)} ms`,
          ].join("<br/>");
        }
        return `Outlier: ${(params.value[1] ?? 0).toFixed(1)} ms`;
      },
    },
    grid: {
      top: title ? 50 : 20,
      left: 20,
      right: 20,
      bottom: 60,
      containLabel: true,
    },
    xAxis: { type: "category" as const, data: categories },
    yAxis: {
      type: "value" as const,
      axisLabel: { formatter: (v: number) => `${v.toFixed(0)} ms` },
    },
    series: [
      {
        type: "boxplot" as const,
        data: boxData,
      },
      ...(outlierData.length > 0
        ? [
            {
              type: "scatter" as const,
              data: outlierData,
              itemStyle: {
                color: SERIES_COLORS[4],
                opacity: 0.6,
              },
              symbolSize: 5,
            },
          ]
        : []),
    ],
  };
}
