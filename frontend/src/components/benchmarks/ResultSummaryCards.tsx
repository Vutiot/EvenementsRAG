import type { NormalizedBenchmarkResult } from "../../api/types";

interface CardProps {
  label: string;
  value: string;
  accent: string; // tailwind border-l color class
  delay: number;
}

function MetricCard({ label, value, accent, delay }: CardProps) {
  return (
    <div
      className={`rounded border border-slate-200 bg-white px-4 py-3 border-l-3 ${accent}
                  opacity-0 animate-fade-in-up`}
      style={{ animationDelay: `${delay}ms` }}
    >
      <p className="text-[10px] font-medium uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-1 font-mono text-lg font-medium text-slate-900">{value}</p>
    </div>
  );
}

interface Props {
  result: NormalizedBenchmarkResult;
}

export default function ResultSummaryCards({ result }: Props) {
  const recall5 = result.avg_recall_at_k["5"];
  const ndcg5 = result.avg_ndcg["5"];

  const cards: Omit<CardProps, "delay">[] = [
    {
      label: "MRR",
      value: result.avg_mrr.toFixed(4),
      accent: "border-l-blue-500",
    },
    {
      label: "Recall@5",
      value: recall5 != null ? recall5.toFixed(4) : "—",
      accent: "border-l-blue-500",
    },
    {
      label: "NDCG@5",
      value: ndcg5 != null ? ndcg5.toFixed(4) : "—",
      accent: "border-l-green-500",
    },
    {
      label: "Avg Retrieval",
      value: `${result.avg_retrieval_time_ms.toFixed(1)} ms`,
      accent: "border-l-amber-500",
    },
    {
      label: "Questions",
      value: String(result.total_questions),
      accent: "border-l-slate-400",
    },
  ];

  if (result.total_wall_time_s != null) {
    cards.push({
      label: "Wall Time",
      value: `${result.total_wall_time_s.toFixed(1)}s`,
      accent: "border-l-amber-500",
    });
  }

  return (
    <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
      {cards.map((c, i) => (
        <MetricCard key={c.label} {...c} delay={i * 60} />
      ))}
    </div>
  );
}
