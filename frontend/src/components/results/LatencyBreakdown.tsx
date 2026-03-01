interface Props {
  retrievalMs: number;
  generationMs: number;
}

function MetricCard({
  label,
  value,
  unit,
}: {
  label: string;
  value: number;
  unit: string;
}) {
  return (
    <div className="rounded border border-gray-200 bg-white px-4 py-3 text-center">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className="mt-1 text-xl font-semibold text-gray-900">
        {value.toFixed(1)}
        <span className="text-sm font-normal text-gray-400 ml-0.5">{unit}</span>
      </p>
    </div>
  );
}

export default function LatencyBreakdown({ retrievalMs, generationMs }: Props) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-2">Latency</h3>
      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Retrieval" value={retrievalMs} unit="ms" />
        <MetricCard label="Generation" value={generationMs} unit="ms" />
        <MetricCard label="Total" value={retrievalMs + generationMs} unit="ms" />
      </div>
    </div>
  );
}
