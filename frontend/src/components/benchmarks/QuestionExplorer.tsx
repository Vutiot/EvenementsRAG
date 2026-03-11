import { useState, useMemo } from "react";
import type { NormalizedQuestion } from "../../api/types";
import GeneratedAnswer from "../results/GeneratedAnswer";
import RagasMetricsGrid from "./RagasMetricsGrid";

interface Props {
  questions: NormalizedQuestion[];
  isLegacy: boolean;
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-0.5">
      <span className="text-slate-500">{label}</span>
      <span className="font-mono text-slate-800">{value}</span>
    </div>
  );
}

function ContextPanel({ index, text }: { index: number; text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded border border-slate-200 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs text-slate-600 hover:bg-slate-50 transition-colors"
      >
        <span className={`transform transition-transform text-[10px] ${open ? "rotate-90" : ""}`}>
          &#9654;
        </span>
        <span className="font-mono text-slate-400">#{index + 1}</span>
        <span className="truncate">{text.slice(0, 120)}...</span>
      </button>
      {open && (
        <div className="border-t border-slate-100 px-3 py-2 text-xs text-slate-700 leading-relaxed whitespace-pre-wrap bg-slate-50/50">
          {text}
        </div>
      )}
    </div>
  );
}

export default function QuestionExplorer({ questions, isLegacy }: Props) {
  const [filter, setFilter] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const filtered = useMemo(() => {
    if (!filter) return questions;
    const lower = filter.toLowerCase();
    return questions.filter(
      (q) =>
        q.question.toLowerCase().includes(lower) ||
        q.question_id.toLowerCase().includes(lower) ||
        q.type.toLowerCase().includes(lower),
    );
  }, [questions, filter]);

  const selected = useMemo(
    () => questions.find((q) => q.question_id === selectedId) ?? null,
    [questions, selectedId],
  );

  return (
    <div className="rounded border border-slate-200 bg-white overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-100 bg-slate-50/50">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
          Question Explorer
        </h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter questions..."
            className="flex-1 rounded border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700
                       placeholder:text-slate-300 focus:border-amber-400 focus:outline-none focus:ring-1 focus:ring-amber-200"
          />
          <select
            value={selectedId ?? ""}
            onChange={(e) => setSelectedId(e.target.value || null)}
            className="flex-[2] min-w-0 rounded border border-slate-200 px-2.5 py-1.5 text-xs text-slate-700
                       focus:border-amber-400 focus:outline-none focus:ring-1 focus:ring-amber-200"
          >
            <option value="">Select a question ({filtered.length})</option>
            {filtered.map((q) => (
              <option key={q.question_id} value={q.question_id}>
                [{q.question_id}] {q.type} — {q.question.slice(0, 80)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {selected ? (
        <div className="p-4 space-y-4">
          {/* Question header */}
          <div>
            <p className="text-sm font-medium text-slate-900 leading-relaxed">
              {selected.question}
            </p>
            <div className="mt-1.5 flex flex-wrap gap-2 text-[10px]">
              <span className="rounded bg-blue-100 px-1.5 py-0.5 font-medium text-blue-700">
                {selected.type}
              </span>
              {selected.difficulty && (
                <span className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-500">
                  {selected.difficulty}
                </span>
              )}
              {selected.source_article && (
                <span className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-500 truncate max-w-[200px]">
                  {selected.source_article}
                </span>
              )}
            </div>
          </div>

          {/* Per-question metrics */}
          <div>
            <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Retrieval Metrics
            </h4>
            <div className="rounded border border-slate-200 bg-slate-50/50 px-3 py-2 text-xs divide-y divide-slate-100">
              {Object.entries(selected.metrics).map(([k, v]) => (
                <MetricRow key={k} label={k} value={v.toFixed(4)} />
              ))}
              {selected.retrieval_time_ms != null && (
                <MetricRow label="retrieval_time_ms" value={selected.retrieval_time_ms.toFixed(1)} />
              )}
              {selected.ground_truth_count != null && (
                <MetricRow label="ground_truth_count" value={String(selected.ground_truth_count)} />
              )}
              {selected.retrieved_count != null && (
                <MetricRow label="retrieved_count" value={String(selected.retrieved_count)} />
              )}
            </div>
          </div>

          {/* Retrieved contexts */}
          <div>
            <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Retrieved Contexts
            </h4>
            {selected.retrieved_contexts && selected.retrieved_contexts.length > 0 ? (
              <div className="space-y-1">
                {selected.retrieved_contexts.map((ctx, i) => (
                  <ContextPanel key={i} index={i} text={ctx} />
                ))}
              </div>
            ) : (
              <p className="text-xs text-slate-400 italic">
                {isLegacy
                  ? "Contexts not available for legacy results."
                  : "No retrieved contexts recorded."}
              </p>
            )}
          </div>

          {/* Generated answer */}
          {selected.generated_answer ? (
            <GeneratedAnswer answer={selected.generated_answer} />
          ) : (
            <div>
              <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                Generated Answer
              </h4>
              <p className="text-xs text-slate-400 italic">
                {isLegacy
                  ? "Generation not available for legacy results."
                  : "No generated answer recorded."}
              </p>
            </div>
          )}

          {/* Generation metrics */}
          {selected.generation_metrics && Object.keys(selected.generation_metrics).length > 0 && (
            <div>
              <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                Generation Metrics
              </h4>
              <div className="rounded border border-slate-200 bg-slate-50/50 px-3 py-2 text-xs divide-y divide-slate-100">
                {Object.entries(selected.generation_metrics).map(([k, v]) => (
                  <MetricRow key={k} label={k} value={v.toFixed(4)} />
                ))}
              </div>
            </div>
          )}

          {/* RAGAS per-question */}
          {selected.ragas_metrics && Object.keys(selected.ragas_metrics).length > 0 && (
            <RagasMetricsGrid metrics={selected.ragas_metrics} title="Per-Question RAGAS" />
          )}
        </div>
      ) : (
        <div className="p-8 text-center text-sm text-slate-400">
          Select a question from the dropdown above to inspect its metrics, retrieved contexts, and generated answer.
        </div>
      )}
    </div>
  );
}
