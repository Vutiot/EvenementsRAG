import { useMemo } from "react";
import { EChartWrapper, buildBarChart } from "../charts";
import {
  RETRIEVAL_METRICS,
  GENERATION_METRICS,
} from "../../constants/metricGroups";
import type { MetricDef } from "../../constants/metricGroups";
import { paramShortName, uniqueParamValues } from "./sweepUtils";
import type { SweepConfig } from "./sweepUtils";

interface Props {
  configs: SweepConfig[];
  sweptParams: Record<string, unknown[]>;
  activeMetrics: Set<string>;
}

function activeDefsFrom(
  groups: typeof RETRIEVAL_METRICS[],
  active: Set<string>,
): MetricDef[] {
  return groups.flatMap((g) =>
    g.subGroups.flatMap((sg) => sg.metrics.filter((m) => active.has(m.key))),
  );
}

/** For a given primary param, group configs by secondary param values */
function groupBySecondary(
  configs: SweepConfig[],
  primaryParam: string,
  sweptParams: Record<string, unknown[]>,
): { seriesLabel: string; configsByPrimary: Map<string, SweepConfig> }[] {
  const secondaryKeys = Object.keys(sweptParams).filter(
    (k) => k !== primaryParam && (sweptParams[k]?.length ?? 0) > 1,
  );

  if (secondaryKeys.length === 0) {
    // No secondary params — one series for all
    const map = new Map<string, SweepConfig>();
    for (const c of configs) map.set(String(c.params[primaryParam]), c);
    return [{ seriesLabel: "Value", configsByPrimary: map }];
  }

  // Group by secondary param combo
  const groups = new Map<string, { label: string; map: Map<string, SweepConfig> }>();
  for (const c of configs) {
    const secLabel = secondaryKeys
      .slice(0, 2)
      .map((k) => `${paramShortName(k)}=${String(c.params[k])}`)
      .join(" · ");
    if (!groups.has(secLabel)) {
      groups.set(secLabel, { label: secLabel, map: new Map() });
    }
    groups.get(secLabel)!.map.set(String(c.params[primaryParam]), c);
  }

  return Array.from(groups.values()).map((g) => ({
    seriesLabel: g.label,
    configsByPrimary: g.map,
  }));
}

function ParameterSection({
  paramKey,
  configs,
  sweptParams,
  metrics,
}: {
  paramKey: string;
  configs: SweepConfig[];
  sweptParams: Record<string, unknown[]>;
  metrics: MetricDef[];
}) {
  const charts = useMemo(() => {
    const primaryValues = uniqueParamValues(configs, paramKey).map(String);
    const groups = groupBySecondary(configs, paramKey, sweptParams);

    return metrics
      .map((m) => {
        const series = groups
          .map((g) => ({
            name: g.seriesLabel,
            data: primaryValues.map(
              (pv) => g.configsByPrimary.get(pv)?.metrics[m.key] ?? 0,
            ),
          }))
          .filter((s) => s.data.some((v) => v > 0));

        if (series.length === 0) return null;

        return {
          key: m.key,
          option: buildBarChart({
            categories: primaryValues,
            series,
            title: m.label,
          }),
        };
      })
      .filter((c): c is NonNullable<typeof c> => c !== null);
  }, [paramKey, configs, sweptParams, metrics]);

  if (charts.length === 0) return null;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-slate-700">
        By {paramShortName(paramKey)}
      </h3>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {charts.map((c) => (
          <EChartWrapper key={c.key} option={c.option} height={300} />
        ))}
      </div>
    </div>
  );
}

export default function SweepParameterCharts({
  configs,
  sweptParams,
  activeMetrics,
}: Props) {
  const retMetrics = useMemo(
    () => activeDefsFrom([RETRIEVAL_METRICS], activeMetrics),
    [activeMetrics],
  );
  const genMetrics = useMemo(
    () => activeDefsFrom([GENERATION_METRICS], activeMetrics),
    [activeMetrics],
  );

  const paramKeys = Object.keys(sweptParams).filter(
    (k) => (sweptParams[k]?.length ?? 0) > 1,
  );

  if (configs.length === 0 || paramKeys.length === 0) {
    return (
      <p className="text-sm text-slate-400 italic">No swept parameters to chart.</p>
    );
  }

  return (
    <div className="space-y-8">
      {retMetrics.length > 0 && (
        <section className="space-y-6">
          <h2 className="text-lg font-semibold text-slate-800">Retrieval Evaluation</h2>
          {paramKeys.map((pk) => (
            <ParameterSection
              key={pk}
              paramKey={pk}
              configs={configs}
              sweptParams={sweptParams}
              metrics={retMetrics}
            />
          ))}
        </section>
      )}

      {genMetrics.length > 0 && (
        <section className="space-y-6">
          <h2 className="text-lg font-semibold text-slate-800">Generation Evaluation</h2>
          {paramKeys.map((pk) => (
            <ParameterSection
              key={pk}
              paramKey={pk}
              configs={configs}
              sweptParams={sweptParams}
              metrics={genMetrics}
            />
          ))}
        </section>
      )}
    </div>
  );
}
