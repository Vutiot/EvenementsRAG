import { useMemo, useState } from "react";
import type { NormalizedBenchmarkResult } from "../../api/types";
import {
  EChartWrapper,
  buildHeatmapChart,
  buildBoxplotChart,
} from "../charts";
import type { HeatmapDataPoint } from "../charts";
import {
  RETRIEVAL_METRICS,
  GENERATION_METRICS,
} from "../../constants/metricGroups";
import type { MetricDef } from "../../constants/metricGroups";
import { exportAsLatex, exportAsMarkdown } from "../../utils/tableExport";

interface Props {
  result: NormalizedBenchmarkResult;
  activeRetrievalMetrics: Set<string>;
  activeGenerationMetrics: Set<string>;
}

/* ── helpers ───────────────────────────────────────────────────────── */

function activeDefsFrom(group: typeof RETRIEVAL_METRICS, active: Set<string>): MetricDef[] {
  return group.subGroups.flatMap((sg) => sg.metrics).filter((m) => active.has(m.key));
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + "\u2026" : s;
}

/* ── Toast helper ──────────────────────────────────────────────────── */

function CopyToast({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return (
    <span className="ml-2 inline-block rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 animate-pulse">
      Copied!
    </span>
  );
}

/* ── Main component ────────────────────────────────────────────────── */

export default function BenchHeatmap({
  result,
  activeRetrievalMetrics,
  activeGenerationMetrics,
}: Props) {
  const [toast, setToast] = useState<"latex" | "md" | null>(null);

  const retMetrics = useMemo(
    () => activeDefsFrom(RETRIEVAL_METRICS, activeRetrievalMetrics),
    [activeRetrievalMetrics],
  );
  const genMetrics = useMemo(
    () => activeDefsFrom(GENERATION_METRICS, activeGenerationMetrics),
    [activeGenerationMetrics],
  );
  const allMetrics = useMemo(() => [...retMetrics, ...genMetrics], [retMetrics, genMetrics]);

  /* Build heatmap data */
  const { xData, yData, data, tableHeaders, tableRows } = useMemo(() => {
    const questions = result.per_question;
    const xData = allMetrics.map((m) => m.label);
    const yData = questions.map((q) => truncate(q.question, 50));
    const hmData: HeatmapDataPoint[] = [];

    const headers = ["Question", ...allMetrics.map((m) => m.label)];
    const rows: (string | number)[][] = [];

    for (let qi = 0; qi < questions.length; qi++) {
      const q = questions[qi]!;
      const row: (string | number)[] = [truncate(q.question, 80)];

      for (let mi = 0; mi < allMetrics.length; mi++) {
        const m = allMetrics[mi]!;
        // Try retrieval metrics, then RAGAS metrics
        const v = q.metrics[m.key] ?? q.ragas_metrics?.[m.key] ?? undefined;
        if (v !== undefined) {
          hmData.push([mi, qi, v]);
          row.push(v);
        } else {
          row.push("");
        }
      }
      rows.push(row);
    }

    return { xData, yData, data: hmData, tableHeaders: headers, tableRows: rows };
  }, [result, allMetrics]);

  /* Heatmap chart option */
  const heatmapOption = useMemo(() => {
    if (allMetrics.length === 0 || result.per_question.length === 0) return null;
    const opt = buildHeatmapChart({ xData, yData, data, min: 0, max: 1 });
    // Add toolbox for save-as-image
    return {
      ...opt,
      toolbox: {
        feature: { saveAsImage: { title: "Save" } },
        right: 70,
        top: 0,
      },
    };
  }, [xData, yData, data, allMetrics.length, result.per_question.length]);

  /* Latency box plot */
  const boxplotOption = useMemo(() => {
    const retTimes = result.per_question
      .map((q) => q.retrieval_time_ms)
      .filter((v): v is number => v !== null);
    const genTimes = result.per_question
      .map((q) => q.generation_time_ms)
      .filter((v): v is number => v !== null);

    if (retTimes.length === 0 && genTimes.length === 0) return null;

    const categories: string[] = [];
    const boxData: number[][] = [];
    if (retTimes.length > 0) {
      categories.push("Retrieval");
      boxData.push(retTimes);
    }
    if (genTimes.length > 0) {
      categories.push("Generation");
      boxData.push(genTimes);
    }
    return buildBoxplotChart({ categories, data: boxData, title: "Latency Distribution" });
  }, [result]);

  /* Export handlers */
  const handleExport = (format: "latex" | "md") => {
    const text =
      format === "latex"
        ? exportAsLatex(tableHeaders, tableRows)
        : exportAsMarkdown(tableHeaders, tableRows);
    navigator.clipboard.writeText(text).then(() => {
      setToast(format);
      setTimeout(() => setToast(null), 2000);
    });
  };

  if (allMetrics.length === 0) return null;

  return (
    <div className="space-y-6">
      {/* Heatmap section */}
      {heatmapOption && (
        <section className="space-y-3">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-slate-800">
              Per-Question Heatmap
            </h2>
            <button
              type="button"
              onClick={() => handleExport("latex")}
              className="rounded border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              Export LaTeX
            </button>
            <CopyToast visible={toast === "latex"} />
            <button
              type="button"
              onClick={() => handleExport("md")}
              className="rounded border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
            >
              Export Markdown
            </button>
            <CopyToast visible={toast === "md"} />
          </div>
          <EChartWrapper
            option={heatmapOption}
            height={Math.max(300, result.per_question.length * 22 + 100)}
          />
        </section>
      )}

      {/* Latency box plot */}
      {boxplotOption && (
        <section className="space-y-2">
          <EChartWrapper option={boxplotOption} height={350} />
        </section>
      )}
    </div>
  );
}
