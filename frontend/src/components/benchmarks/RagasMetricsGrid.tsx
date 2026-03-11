interface Props {
  metrics: Record<string, number>;
  title?: string;
}

function colorClass(v: number): string {
  if (v >= 0.7) return "border-l-green-500 text-green-700 bg-green-50/40";
  if (v >= 0.4) return "border-l-yellow-500 text-yellow-700 bg-yellow-50/40";
  return "border-l-red-500 text-red-700 bg-red-50/40";
}

function formatLabel(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function RagasMetricsGrid({ metrics, title = "RAGAS Metrics" }: Props) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) return null;

  return (
    <div>
      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
        {title}
      </h4>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        {entries.map(([key, value]) => (
          <div
            key={key}
            className={`rounded border border-slate-200 border-l-3 px-3 py-2 ${colorClass(value)}`}
          >
            <p className="text-[10px] font-medium uppercase tracking-wide opacity-70">
              {formatLabel(key)}
            </p>
            <p className="mt-0.5 font-mono text-sm font-medium">
              {value.toFixed(3)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
