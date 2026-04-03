import { useMemo, useState } from "react";
import { ALL_METRIC_KEYS, HIGHER_IS_BETTER } from "../../constants/metricGroups";
import { metricDisplayName } from "../charts/chartBuilders";
import { paramShortName } from "./sweepUtils";
import type { SweepConfig } from "./sweepUtils";
import { downloadCSV } from "../../utils/csvExport";
import { exportAsLatex, exportAsMarkdown } from "../../utils/tableExport";

interface Props {
  configs: SweepConfig[];
  sweptParams: Record<string, unknown[]>;
  activeMetrics: Set<string>;
}

function activeKeys(active: Set<string>): string[] {
  return ALL_METRIC_KEYS.filter((k) => active.has(k));
}

export default function SweepDataTable({
  configs,
  sweptParams,
  activeMetrics,
}: Props) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const metricKeys = useMemo(
    () => activeKeys(activeMetrics).filter((k) => configs.some((c) => c.metrics[k] !== undefined)),
    [activeMetrics, configs],
  );
  const paramKeys = Object.keys(sweptParams);

  // Sort
  const sorted = useMemo(() => {
    if (!sortKey) return configs;
    return [...configs].sort((a, b) => {
      const av = a.metrics[sortKey] ?? -Infinity;
      const bv = b.metrics[sortKey] ?? -Infinity;
      return sortAsc ? av - bv : bv - av;
    });
  }, [configs, sortKey, sortAsc]);

  // Best/worst per metric
  const bestWorst = useMemo(() => {
    const bw: Record<string, { best: number; worst: number }> = {};
    for (const k of metricKeys) {
      const vals = configs.map((c) => c.metrics[k]).filter((v): v is number => v !== undefined);
      if (vals.length === 0) continue;
      const hib = HIGHER_IS_BETTER[k] !== false;
      bw[k] = {
        best: hib ? Math.max(...vals) : Math.min(...vals),
        worst: hib ? Math.min(...vals) : Math.max(...vals),
      };
    }
    return bw;
  }, [configs, metricKeys]);

  const handleSort = (key: string) => {
    if (sortKey === key) setSortAsc((p) => !p);
    else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  // Export helpers
  const buildRows = () => {
    const headers = [
      "Rank",
      "Config",
      ...paramKeys.map(paramShortName),
      ...metricKeys.map(metricDisplayName),
    ];
    const rows = sorted.map((c, i) => [
      i + 1,
      c.label,
      ...paramKeys.map((k) => String(c.params[k] ?? "")),
      ...metricKeys.map((k) => c.metrics[k] ?? ""),
    ] as (string | number)[]);
    return { headers, rows };
  };

  const handleCSV = () => {
    const csvRows = sorted.map((c, i) => {
      const row: Record<string, string | number | null> = {
        rank: i + 1,
        config: c.label,
      };
      for (const k of paramKeys) row[paramShortName(k)] = String(c.params[k] ?? "");
      for (const k of metricKeys) row[metricDisplayName(k)] = c.metrics[k] ?? null;
      return row;
    });
    downloadCSV(csvRows, "sweep_results.csv");
  };

  const handleCopy = (format: "latex" | "md") => {
    const { headers, rows } = buildRows();
    const text = format === "latex" ? exportAsLatex(headers, rows) : exportAsMarkdown(headers, rows);
    navigator.clipboard.writeText(text).then(() => {
      setToast(format);
      setTimeout(() => setToast(null), 2000);
    });
  };

  if (configs.length === 0 || metricKeys.length === 0) return null;

  return (
    <details open={configs.length <= 20}>
      <summary className="cursor-pointer text-lg font-semibold text-slate-800 hover:text-slate-600">
        Data Table ({configs.length} configs)
      </summary>

      <div className="mt-3 space-y-2">
        {/* Export buttons */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleCSV}
            className="rounded border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            Export CSV
          </button>
          <button
            type="button"
            onClick={() => handleCopy("latex")}
            className="rounded border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            Export LaTeX
          </button>
          <button
            type="button"
            onClick={() => handleCopy("md")}
            className="rounded border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
          >
            Export Markdown
          </button>
          {toast && (
            <span className="rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 animate-pulse">
              Copied!
            </span>
          )}
        </div>

        {/* Table */}
        <div className="overflow-x-auto rounded border border-slate-200">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-50 text-left">
                <th className="px-2 py-2 font-medium text-slate-500">#</th>
                <th className="px-2 py-2 font-medium text-slate-500">Config</th>
                {paramKeys.map((k) => (
                  <th key={k} className="px-2 py-2 font-medium text-slate-500 whitespace-nowrap">
                    {paramShortName(k)}
                  </th>
                ))}
                {metricKeys.map((k) => (
                  <th
                    key={k}
                    className="cursor-pointer px-2 py-2 font-medium text-slate-500 hover:text-slate-800 whitespace-nowrap"
                    onClick={() => handleSort(k)}
                  >
                    {metricDisplayName(k)}
                    {sortKey === k && (sortAsc ? " \u25B2" : " \u25BC")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((c, i) => (
                <tr key={c.filename} className="border-t border-slate-100">
                  <td className="px-2 py-1.5 text-slate-400">{i + 1}</td>
                  <td className="max-w-[200px] truncate px-2 py-1.5 font-mono text-slate-700">
                    {c.label}
                  </td>
                  {paramKeys.map((k) => (
                    <td key={k} className="px-2 py-1.5 font-mono text-slate-600">
                      {String(c.params[k] ?? "\u2014")}
                    </td>
                  ))}
                  {metricKeys.map((k) => {
                    const v = c.metrics[k];
                    const bw = bestWorst[k];
                    let cls = "text-slate-700";
                    if (v !== undefined && bw) {
                      if (v === bw.best) cls = "bg-green-50 text-green-700 font-semibold";
                      else if (v === bw.worst) cls = "bg-red-50 text-red-700";
                    }
                    return (
                      <td key={k} className={`px-2 py-1.5 font-mono text-center ${cls}`}>
                        {v !== undefined ? v.toFixed(4) : "\u2014"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  );
}
