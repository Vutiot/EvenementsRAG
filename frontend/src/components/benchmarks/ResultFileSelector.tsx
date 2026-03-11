import type { ResultFileInfo } from "../../api/types";

interface Props {
  files: ResultFileInfo[];
  selectedFile: string | null;
  onSelect: (filename: string) => void;
  loading: boolean;
}

function groupByPhase(files: ResultFileInfo[]): Record<string, ResultFileInfo[]> {
  const groups: Record<string, ResultFileInfo[]> = {};
  for (const f of files) {
    const key = f.phase_name.replace(/_/g, " ");
    if (!groups[key]) groups[key] = [];
    groups[key].push(f);
  }
  return groups;
}

function SkeletonRow() {
  return (
    <div className="px-3 py-2.5 animate-pulse">
      <div className="h-3 w-3/4 rounded bg-slate-200" />
      <div className="mt-1.5 h-2 w-1/2 rounded bg-slate-100" />
    </div>
  );
}

export default function ResultFileSelector({
  files,
  selectedFile,
  onSelect,
  loading,
}: Props) {
  if (loading) {
    return (
      <div className="rounded border border-slate-200 bg-white divide-y divide-slate-100">
        <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
          Result Files
        </div>
        {Array.from({ length: 5 }).map((_, i) => (
          <SkeletonRow key={i} />
        ))}
      </div>
    );
  }

  if (files.length === 0) {
    return (
      <div className="rounded border border-slate-200 bg-white p-4 text-center text-sm text-slate-400">
        No result files found in <code className="font-mono text-xs">results/</code>
      </div>
    );
  }

  const grouped = groupByPhase(files);

  return (
    <div className="rounded border border-slate-200 bg-white overflow-hidden">
      <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-slate-400 bg-slate-50 border-b border-slate-100">
        Result Files
      </div>
      <div className="max-h-[calc(100vh-12rem)] overflow-y-auto divide-y divide-slate-50">
        {Object.entries(grouped).map(([phase, items]) => (
          <div key={phase}>
            <div className="px-3 py-1.5 text-[10px] font-medium uppercase tracking-widest text-slate-400 bg-slate-50/50">
              {phase}
            </div>
            {items.map((f) => {
              const active = f.filename === selectedFile;
              return (
                <button
                  key={f.filename}
                  onClick={() => onSelect(f.filename)}
                  className={`w-full text-left px-3 py-2.5 transition-all duration-200 border-l-3 ${
                    active
                      ? "border-l-amber-500 bg-amber-50/50"
                      : "border-l-transparent hover:bg-slate-50 hover:border-l-slate-300"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-mono truncate ${active ? "text-amber-800" : "text-slate-700"}`}>
                      {f.filename}
                    </span>
                    <span className="ml-2 shrink-0 rounded-full bg-slate-100 px-1.5 py-0.5 text-[10px] font-mono text-slate-500">
                      {f.avg_mrr.toFixed(3)}
                    </span>
                  </div>
                  <div className="mt-0.5 flex items-center gap-2 text-[10px] text-slate-400">
                    <span>{f.total_questions}q</span>
                    <span className="rounded bg-slate-100 px-1 py-px">{f.format === "benchmark_result" ? "new" : "legacy"}</span>
                  </div>
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
