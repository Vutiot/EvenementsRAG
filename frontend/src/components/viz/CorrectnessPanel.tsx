import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import type { NormalizedBenchmarkResult, NormalizedQuestion } from "../../api/types";
import { EChartWrapper, formatMetricValue } from "../charts";
import { getMetricValue } from "../../constants/metricGroups";

interface Props {
  result: NormalizedBenchmarkResult;
}

interface CorrectnessSummary {
  answer_correctness?: number | null;
  answer_correctness_jev?: number | null;
  judge_agreement?: number | null;
  judge_kappa?: number | null;
  jev_threshold?: number;
  jev_snapshot?: string;
  judge_model?: string;
  num_questions_scored?: number;
  llm_judge_errors?: number;
  jev_errors?: number;
  agreement_detail?: {
    n: number;
    both_correct?: number;
    both_wrong?: number;
    llm_only?: number;
    jev_only?: number;
  };
}

function pct(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`;
}

function StatCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded border border-slate-200 bg-white px-4 py-3">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-1 font-mono text-lg font-medium text-slate-900">{value}</p>
      {hint && <p className="text-[11px] text-slate-400">{hint}</p>}
    </div>
  );
}

function Verdict({ v }: { v: number | null | undefined }) {
  if (v === null || v === undefined) return <span className="text-slate-300">{"—"}</span>;
  return v === 1 ? (
    <span className="font-medium text-green-600">yes</span>
  ) : (
    <span className="font-medium text-red-500">no</span>
  );
}

function failedCriteria(q: NormalizedQuestion): string {
  const crit = q.correctness?.llm?.criteria;
  if (!crit) return "";
  return Object.entries(crit)
    .filter(([, ok]) => !ok)
    .map(([name]) => name)
    .join(", ");
}

/** Per-run view of answer correctness: both judges, their agreement, cost and latency. */
export default function CorrectnessPanel({ result }: Props) {
  const summary = (result.metrics_summary?.correctness ?? null) as CorrectnessSummary | null;
  const [onlyDisagreements, setOnlyDisagreements] = useState(false);

  const scored = useMemo(
    () => result.per_question.filter((q) => q.correctness != null),
    [result],
  );

  /* Jev P(yes) distribution split by the LLM judge verdict: shows where the threshold should sit */
  const histogramOption = useMemo((): EChartsOption | null => {
    const bins = 10;
    const yes = new Array<number>(bins).fill(0);
    const no = new Array<number>(bins).fill(0);
    let any = false;
    for (const q of scored) {
      const p = q.correctness?.jev?.prob_yes;
      const llm = q.correctness?.llm?.correct;
      if (typeof p !== "number" || (llm !== 0 && llm !== 1)) continue;
      any = true;
      const b = Math.min(bins - 1, Math.floor(p * bins));
      if (llm === 1) yes[b]! += 1;
      else no[b]! += 1;
    }
    if (!any) return null;
    const categories = Array.from({ length: bins }, (_, i) => `${(i / bins).toFixed(1)}–${((i + 1) / bins).toFixed(1)}`);
    const threshold = summary?.jev_threshold ?? 0.5;
    return {
      title: { text: "Jev P(yes) by LLM-judge verdict", left: 0, textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis" },
      legend: { top: 0, right: 0 },
      grid: { left: 40, right: 16, top: 56, bottom: 32 },
      xAxis: { type: "category", data: categories },
      yAxis: { type: "value", minInterval: 1 },
      series: [
        {
          name: "LLM judge: correct",
          type: "bar",
          stack: "v",
          data: yes,
          itemStyle: { color: "#10b981" },
          markLine: {
            symbol: "none",
            label: { formatter: `threshold ${threshold}` },
            lineStyle: { color: "#64748b", type: "dashed" },
            data: [{ xAxis: Math.min(bins - 1, Math.floor(threshold * bins)) }],
          },
        },
        { name: "LLM judge: wrong", type: "bar", stack: "v", data: no, itemStyle: { color: "#ef4444" } },
      ],
    };
  }, [scored, summary]);

  const rows = useMemo(
    () => (onlyDisagreements ? scored.filter((q) => q.correctness?.agree === false) : scored),
    [scored, onlyDisagreements],
  );

  const cost = getMetricValue(result, "cost_per_question_usd");
  const latency = getMetricValue(result, "latency_per_question_ms");

  if (!summary && scored.length === 0) {
    return (
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Cost / question" value={cost !== null ? formatMetricValue("cost_per_question_usd", cost) : "—"} />
          <StatCard label="Latency / question" value={latency !== null ? formatMetricValue("latency_per_question_ms", latency) : "—"} />
        </div>
        <div className="rounded border border-dashed border-slate-300 bg-slate-50 p-8 text-center text-sm text-slate-400">
          No answer-correctness data for this run (generation off, or a run recorded before the two-judge evaluation).
        </div>
      </div>
    );
  }

  const detail = summary?.agreement_detail;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Correct (LLM judge)" value={pct(summary?.answer_correctness)} hint={summary?.judge_model} />
        <StatCard
          label="Correct (Jev)"
          value={pct(summary?.answer_correctness_jev)}
          hint={summary?.jev_threshold !== undefined ? `P(yes) ≥ ${summary.jev_threshold}` : undefined}
        />
        <StatCard
          label="Judge agreement"
          value={pct(summary?.judge_agreement)}
          hint={summary?.judge_kappa != null ? `κ = ${summary.judge_kappa.toFixed(2)}` : undefined}
        />
        <StatCard label="Scored questions" value={String(summary?.num_questions_scored ?? scored.length)}
          hint={(summary?.llm_judge_errors || summary?.jev_errors) ? `errors: LLM ${summary?.llm_judge_errors ?? 0}, Jev ${summary?.jev_errors ?? 0}` : undefined} />
        <StatCard label="Cost / question" value={cost !== null ? formatMetricValue("cost_per_question_usd", cost) : "—"} hint="generation + judges" />
        <StatCard label="Latency / question" value={latency !== null ? formatMetricValue("latency_per_question_ms", latency) : "—"} hint="retrieval + generation" />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {detail && detail.n > 0 && (
          <div className="rounded border border-slate-200 bg-white p-4">
            <p className="mb-3 text-sm font-medium text-slate-600">LLM judge vs Jev ({detail.n} questions)</p>
            <table className="w-full text-center text-sm">
              <thead>
                <tr className="text-xs text-slate-400">
                  <th></th>
                  <th className="pb-2 font-medium">Jev: yes</th>
                  <th className="pb-2 font-medium">Jev: no</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                <tr>
                  <th className="pr-2 text-left text-xs font-medium text-slate-400">LLM: yes</th>
                  <td className="rounded bg-green-50 py-3 text-green-700">{detail.both_correct ?? 0}</td>
                  <td className="rounded bg-amber-50 py-3 text-amber-700">{detail.llm_only ?? 0}</td>
                </tr>
                <tr>
                  <th className="pr-2 text-left text-xs font-medium text-slate-400">LLM: no</th>
                  <td className="rounded bg-amber-50 py-3 text-amber-700">{detail.jev_only ?? 0}</td>
                  <td className="rounded bg-green-50 py-3 text-green-700">{detail.both_wrong ?? 0}</td>
                </tr>
              </tbody>
            </table>
            {summary?.jev_snapshot && (
              <p className="mt-3 text-[11px] text-slate-400">Jev snapshot: {summary.jev_snapshot}</p>
            )}
          </div>
        )}
        {histogramOption && (
          <div className="lg:col-span-2">
            <EChartWrapper option={histogramOption} height={260} />
          </div>
        )}
      </div>

      {scored.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-medium text-slate-600">Per-question verdicts</h3>
            <label className="flex items-center gap-1.5 text-xs text-slate-500">
              <input
                type="checkbox"
                checked={onlyDisagreements}
                onChange={(e) => setOnlyDisagreements(e.target.checked)}
              />
              Only disagreements
            </label>
            <span className="text-xs text-slate-400">{rows.length} shown</span>
          </div>
          <div className="max-h-[480px] overflow-auto rounded border border-slate-200">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-slate-50 text-slate-500">
                <tr>
                  <th className="px-2 py-1.5">Question</th>
                  <th className="px-2 py-1.5">Type</th>
                  <th className="px-2 py-1.5">Answer</th>
                  <th className="px-2 py-1.5 text-center">Hit@10</th>
                  <th className="px-2 py-1.5 text-center">LLM</th>
                  <th className="px-2 py-1.5">Failed criteria</th>
                  <th className="px-2 py-1.5 text-center">Jev</th>
                  <th className="px-2 py-1.5 text-right">P(yes)</th>
                  <th className="px-2 py-1.5 text-right">Cost</th>
                  <th className="px-2 py-1.5 text-right">Latency</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((q) => {
                  const c = q.correctness!;
                  const disagree = c.agree === false;
                  return (
                    <tr key={q.question_id} className={`border-t border-slate-100 align-top ${disagree ? "bg-amber-50/60" : ""}`}>
                      <td className="max-w-[260px] px-2 py-1.5 text-slate-700" title={q.question}>{q.question}</td>
                      <td className="px-2 py-1.5 text-slate-500">{q.type}</td>
                      <td className="max-w-[320px] truncate px-2 py-1.5 text-slate-600" title={q.generated_answer ?? ""}>
                        {q.generated_answer ?? "—"}
                      </td>
                      <td className="px-2 py-1.5 text-center"><Verdict v={q.metrics["hit_at_10"]} /></td>
                      <td className="px-2 py-1.5 text-center" title={c.llm?.error ?? ""}><Verdict v={c.llm?.correct} /></td>
                      <td className="px-2 py-1.5 font-mono text-[11px] text-red-500">{failedCriteria(q)}</td>
                      <td className="px-2 py-1.5 text-center" title={c.jev?.error ?? ""}><Verdict v={c.jev?.correct} /></td>
                      <td className="px-2 py-1.5 text-right font-mono">
                        {typeof c.jev?.prob_yes === "number" ? c.jev.prob_yes.toFixed(2) : "—"}
                      </td>
                      <td className="px-2 py-1.5 text-right font-mono">
                        {typeof q.cost_usd === "number" ? formatMetricValue("cost_usd", q.cost_usd) : "—"}
                      </td>
                      <td className="whitespace-nowrap px-2 py-1.5 text-right font-mono">
                        {typeof q.latency_ms === "number" ? `${q.latency_ms.toFixed(0)} ms` : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
