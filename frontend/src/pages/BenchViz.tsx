import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import type { NormalizedBenchmarkResult } from "../api/types";
import type { BenchmarkConfig } from "../api/types";
import { getResult } from "../api/client";
import PageHeader from "../components/layout/PageHeader";
import ConfigBadges from "../components/config/ConfigBadges";
import { MetricFilterBar } from "../components/charts";
import RetrievalBarCharts from "../components/viz/RetrievalBarCharts";
import GenerationBarCharts from "../components/viz/GenerationBarCharts";
import BenchHeatmap from "../components/viz/BenchHeatmap";
import {
  RETRIEVAL_METRICS,
  GENERATION_METRICS,
} from "../constants/metricGroups";

function allKeys(group: typeof RETRIEVAL_METRICS): Set<string> {
  return new Set(group.subGroups.flatMap((sg) => sg.metrics.map((m) => m.key)));
}

export default function BenchViz() {
  const { filename } = useParams<{ filename: string }>();

  const [result, setResult] = useState<NormalizedBenchmarkResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [activeRetrieval, setActiveRetrieval] = useState<Set<string>>(
    () => allKeys(RETRIEVAL_METRICS),
  );
  const [activeGeneration, setActiveGeneration] = useState<Set<string>>(
    () => allKeys(GENERATION_METRICS),
  );

  useEffect(() => {
    if (!filename) return;
    setLoading(true);
    setError(null);
    getResult(decodeURIComponent(filename))
      .then(setResult)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [filename]);

  const makeToggle = useCallback(
    (setter: React.Dispatch<React.SetStateAction<Set<string>>>) =>
      (key: string) => {
        setter((prev) => {
          const next = new Set(prev);
          if (next.has(key)) next.delete(key);
          else next.add(key);
          return next;
        });
      },
    [],
  );

  const makeToggleAll = useCallback(
    (setter: React.Dispatch<React.SetStateAction<Set<string>>>) =>
      (keys: string[], active: boolean) => {
        setter((prev) => {
          const next = new Set(prev);
          for (const k of keys) {
            if (active) next.add(k);
            else next.delete(k);
          }
          return next;
        });
      },
    [],
  );

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

  if (!result) {
    return (
      <div className="p-6">
        <p className="text-sm text-slate-500">No result to display.</p>
      </div>
    );
  }

  const ts = result.timestamp
    ? new Date(result.timestamp).toLocaleString()
    : null;

  return (
    <div className="space-y-8 p-6">
      {/* Header */}
      <div>
        <PageHeader
          title={result.phase_name}
          description={
            [ts, `${result.total_questions} questions`]
              .filter(Boolean)
              .join(" \u00b7 ")
          }
        />
        <ConfigBadges config={result.config as BenchmarkConfig | null} />
      </div>

      {/* Retrieval Evaluation */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">
          Retrieval Evaluation
        </h2>
        <MetricFilterBar
          metricGroups={[RETRIEVAL_METRICS]}
          activeMetrics={activeRetrieval}
          onToggle={makeToggle(setActiveRetrieval)}
          onToggleAll={makeToggleAll(setActiveRetrieval)}
        />
        <RetrievalBarCharts result={result} activeMetrics={activeRetrieval} />
      </section>

      {/* Generation Evaluation */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">
          Generation Evaluation
        </h2>
        <MetricFilterBar
          metricGroups={[GENERATION_METRICS]}
          activeMetrics={activeGeneration}
          onToggle={makeToggle(setActiveGeneration)}
          onToggleAll={makeToggleAll(setActiveGeneration)}
        />
        <GenerationBarCharts result={result} activeMetrics={activeGeneration} />
      </section>

      {/* Cross-section: Heatmap + Latency */}
      <BenchHeatmap
        result={result}
        activeRetrievalMetrics={activeRetrieval}
        activeGenerationMetrics={activeGeneration}
      />
    </div>
  );
}
