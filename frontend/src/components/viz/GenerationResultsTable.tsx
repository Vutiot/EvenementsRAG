import { useMemo } from "react";
import type { NormalizedBenchmarkResult, NormalizedQuestion } from "../../api/types";
import { GENERATION_METRICS, getMetricValue } from "../../constants/metricGroups";
import type { MetricDef } from "../../constants/metricGroups";
import ResultsTable from "./ResultsTable";

interface Props {
  result: NormalizedBenchmarkResult;
  activeMetrics: Set<string>;
}

function flatActive(active: Set<string>): MetricDef[] {
  return GENERATION_METRICS.subGroups
    .flatMap((sg) => sg.metrics)
    .filter((m) => active.has(m.key));
}

/** Average q.ragas_metrics[key] across questions of a given type. */
function avgPerType(
  questions: NormalizedQuestion[],
  type: string,
  key: string,
): number | null {
  const vals = questions
    .filter((q) => q.type === type)
    .map((q) => q.ragas_metrics?.[key])
    .filter((v): v is number => v !== undefined && v !== null);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export default function GenerationResultsTable({ result, activeMetrics }: Props) {
  const metrics = useMemo(() => flatActive(activeMetrics), [activeMetrics]);

  const types = useMemo(() => {
    const set = new Set<string>();
    for (const q of result.per_question) {
      if (q.type) set.add(q.type);
    }
    return Array.from(set).sort();
  }, [result.per_question]);

  const { headers, rows } = useMemo(() => {
    const headers = ["Metric", "Overall", ...types];
    const rows: (string | number | null)[][] = [];

    for (const m of metrics) {
      const overall = getMetricValue(result, m.key);
      const perType = types.map((t) => avgPerType(result.per_question, t, m.key));
      const hasAny =
        overall !== null || perType.some((v) => v !== null);
      if (!hasAny) continue;

      rows.push([m.label, overall, ...perType]);
    }
    return { headers, rows };
  }, [result, metrics, types]);

  if (metrics.length === 0) return null;

  return (
    <ResultsTable
      title="Generation Results — Overall and by Question Type"
      headers={headers}
      rows={rows}
      filenameStem={`${result.phase_name}_generation_results`}
    />
  );
}
