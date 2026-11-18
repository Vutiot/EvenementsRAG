import { useMemo } from "react";
import type { NormalizedBenchmarkResult } from "../../api/types";
import { RETRIEVAL_METRICS, getMetricValue } from "../../constants/metricGroups";
import type { MetricDef } from "../../constants/metricGroups";
import ResultsTable from "./ResultsTable";

interface Props {
  result: NormalizedBenchmarkResult;
  activeMetrics: Set<string>;
}

function flatActive(active: Set<string>): MetricDef[] {
  return RETRIEVAL_METRICS.subGroups
    .flatMap((sg) => sg.metrics)
    .filter((m) => active.has(m.key));
}

export default function RetrievalResultsTable({ result, activeMetrics }: Props) {
  const metrics = useMemo(() => flatActive(activeMetrics), [activeMetrics]);
  const types = useMemo(
    () => Object.keys(result.metrics_by_type).sort(),
    [result.metrics_by_type],
  );

  const { headers, rows } = useMemo(() => {
    const headers = ["Metric", "Overall", ...types];
    const rows: (string | number | null)[][] = [];

    for (const m of metrics) {
      const overall = getMetricValue(result, m.key);
      const perType = types.map(
        (t) => result.metrics_by_type[t]?.[m.key] ?? null,
      );
      // Skip rows with no data anywhere
      const hasAny =
        overall !== null || perType.some((v) => v !== null && v !== undefined);
      if (!hasAny) continue;

      rows.push([m.label, overall, ...perType]);
    }
    return { headers, rows };
  }, [result, metrics, types]);

  if (metrics.length === 0) return null;

  return (
    <ResultsTable
      title="Retrieval Results — Overall and by Question Type"
      headers={headers}
      rows={rows}
      filenameStem={`${result.phase_name}_retrieval_results`}
    />
  );
}
