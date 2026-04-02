import ReactECharts from "echarts-for-react";
import * as echarts from "echarts/core";
import type { EChartsOption } from "echarts";

const THEME_NAME = "evenementsrag";

const SERIES_COLORS = [
  "#3b82f6",
  "#6366f1",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#14b8a6",
  "#ec4899",
  "#f97316",
  "#0ea5e9",
];

let themeRegistered = false;

function ensureTheme() {
  if (themeRegistered) return;
  echarts.registerTheme(THEME_NAME, {
    color: SERIES_COLORS,
    textStyle: { fontFamily: "DM Sans, sans-serif", color: "#64748b" },
    title: {
      textStyle: { fontFamily: "DM Sans, sans-serif", color: "#334155", fontWeight: 500 },
    },
    categoryAxis: {
      axisLine: { lineStyle: { color: "#e2e8f0" } },
      axisTick: { lineStyle: { color: "#e2e8f0" } },
      axisLabel: { color: "#64748b" },
      splitLine: { lineStyle: { color: "#f1f5f9" } },
    },
    valueAxis: {
      axisLine: { lineStyle: { color: "#e2e8f0" } },
      axisTick: { lineStyle: { color: "#e2e8f0" } },
      axisLabel: { color: "#64748b" },
      splitLine: { lineStyle: { color: "#f1f5f9" } },
    },
    legend: {
      textStyle: { fontFamily: "DM Sans, sans-serif", color: "#64748b" },
    },
    tooltip: {
      backgroundColor: "#ffffff",
      borderColor: "#e2e8f0",
      textStyle: { fontFamily: "DM Sans, sans-serif", color: "#334155" },
    },
  });
  themeRegistered = true;
}

interface EChartWrapperProps {
  option: EChartsOption;
  height?: number | string;
  loading?: boolean;
  onEvents?: Record<string, (params: unknown) => void>;
  className?: string;
}

export default function EChartWrapper({
  option,
  height = 350,
  loading = false,
  onEvents,
  className,
}: EChartWrapperProps) {
  ensureTheme();

  return (
    <div className={`relative ${className ?? ""}`}>
      {loading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/80 rounded">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-500 border-t-transparent" />
        </div>
      )}
      <ReactECharts
        option={option}
        theme={THEME_NAME}
        style={{ height }}
        opts={{ renderer: "svg" }}
        onEvents={onEvents}
        notMerge
      />
    </div>
  );
}

export { SERIES_COLORS, THEME_NAME };
export type { EChartWrapperProps };
