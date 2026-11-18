import { useState } from "react";
import { exportAsLatex, exportAsMarkdown } from "../../utils/tableExport";

interface Props {
  title: string;
  headers: string[];
  rows: (string | number | null)[][];
  filenameStem: string;
}

function CopyToast({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return (
    <span className="ml-2 inline-block rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 animate-pulse">
      Copied!
    </span>
  );
}

function fmt(v: number | null): string {
  return v === null ? "—" : v.toFixed(3);
}

export default function ResultsTable({ title, headers, rows, filenameStem }: Props) {
  const [toast, setToast] = useState<"latex" | "md" | null>(null);

  if (rows.length === 0) return null;

  const handleExport = (format: "latex" | "md") => {
    const exportRows = rows.map((row) =>
      row.map((cell) => (cell === null ? "" : cell)),
    );
    const text =
      format === "latex"
        ? exportAsLatex(headers, exportRows)
        : exportAsMarkdown(headers, exportRows);
    navigator.clipboard.writeText(text).then(() => {
      setToast(format);
      setTimeout(() => setToast(null), 2000);
    });
  };

  return (
    <section className="space-y-3">
      <div className="flex items-center gap-3">
        <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
        <button
          type="button"
          onClick={() => handleExport("latex")}
          className="rounded border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
          aria-label={`Copy ${filenameStem} as LaTeX`}
        >
          Export LaTeX
        </button>
        <CopyToast visible={toast === "latex"} />
        <button
          type="button"
          onClick={() => handleExport("md")}
          className="rounded border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
          aria-label={`Copy ${filenameStem} as Markdown`}
        >
          Export Markdown
        </button>
        <CopyToast visible={toast === "md"} />
      </div>

      <div className="mx-auto overflow-x-auto rounded border border-slate-200">
        <table className="w-full border-collapse text-center text-xs">
          <thead>
            <tr className="bg-slate-50">
              {headers.map((h, i) => (
                <th
                  key={`${h}-${i}`}
                  className={`px-3 py-2 font-semibold text-slate-600 ${
                    i === 0
                      ? "sticky left-0 z-10 bg-slate-50 text-left"
                      : ""
                  }`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr
                key={`row-${ri}`}
                className="border-t border-slate-100 hover:bg-slate-50/40"
              >
                {row.map((cell, ci) => (
                  <td
                    key={`cell-${ri}-${ci}`}
                    className={`px-3 py-1.5 ${
                      ci === 0
                        ? "sticky left-0 bg-white text-left font-medium text-slate-700"
                        : "font-mono text-slate-700"
                    }`}
                  >
                    {ci === 0
                      ? String(cell ?? "")
                      : typeof cell === "number"
                        ? fmt(cell)
                        : cell === null || cell === ""
                          ? "—"
                          : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
