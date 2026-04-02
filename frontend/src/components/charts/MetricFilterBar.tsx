import type { MetricGroup } from "../../constants/metricGroups";

interface Props {
  metricGroups: MetricGroup[];
  activeMetrics: Set<string>;
  onToggle: (key: string) => void;
  onToggleAll: (keys: string[], active: boolean) => void;
}

export default function MetricFilterBar({
  metricGroups,
  activeMetrics,
  onToggle,
  onToggleAll,
}: Props) {
  const totalCount = metricGroups.reduce(
    (n, g) => n + g.subGroups.reduce((m, sg) => m + sg.metrics.length, 0),
    0,
  );
  const activeCount = activeMetrics.size;

  return (
    <div className="space-y-3">
      <p className="text-xs font-medium text-slate-500">
        {activeCount}/{totalCount} metrics selected
      </p>

      {metricGroups.map((group) => (
        <div key={group.name} className="space-y-2">
          {group.subGroups.map((sg) => {
            const groupKeys = sg.metrics.map((m) => m.key);
            const allActive = groupKeys.every((k) => activeMetrics.has(k));

            return (
              <div key={sg.name} className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => onToggleAll(groupKeys, !allActive)}
                  className="mr-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400 hover:text-slate-600"
                >
                  {sg.name}
                </button>

                {sg.metrics.map((m) => {
                  const active = activeMetrics.has(m.key);
                  return (
                    <button
                      key={m.key}
                      type="button"
                      onClick={() => onToggle(m.key)}
                      className={`rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors ${
                        active
                          ? "border-blue-200 bg-blue-100 text-blue-700"
                          : "border-gray-200 bg-gray-50 text-gray-400 line-through"
                      }`}
                    >
                      {m.label}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}
