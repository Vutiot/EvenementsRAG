/**
 * Export a data table as LaTeX tabular or GitHub-flavored Markdown.
 */

function latexCellColor(v: number): string {
  // Map 0-1 value to red/yellow/green via RGB
  let r: number, g: number, b: number;
  const t = Math.max(0, Math.min(1, v));
  if (t < 0.5) {
    const s = t * 2;
    r = Math.round(239 + (251 - 239) * s);
    g = Math.round(68 + (191 - 68) * s);
    b = Math.round(68 + (36 - 68) * s);
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(251 + (16 - 251) * s);
    g = Math.round(191 + (185 - 191) * s);
    b = Math.round(36 + (129 - 36) * s);
  }
  return `\\cellcolor[RGB]{${r},${g},${b}}`;
}

function escapeLatex(s: string): string {
  return s
    .replace(/\\/g, "\\textbackslash{}")
    .replace(/[&%$#_{}~^]/g, (c) => `\\${c}`);
}

export function exportAsLatex(
  headers: string[],
  rows: (string | number)[][],
): string {
  const cols = headers.map(() => "l").join("|");
  const lines: string[] = [];

  lines.push(`\\begin{tabular}{|${cols}|}`);
  lines.push("\\hline");
  lines.push(headers.map((h) => `\\textbf{${escapeLatex(h)}}`).join(" & ") + " \\\\");
  lines.push("\\hline");

  for (const row of rows) {
    const cells = row.map((cell) => {
      if (typeof cell === "number") {
        return `${latexCellColor(cell)} ${cell.toFixed(3)}`;
      }
      return escapeLatex(String(cell));
    });
    lines.push(cells.join(" & ") + " \\\\");
  }

  lines.push("\\hline");
  lines.push("\\end{tabular}");
  return lines.join("\n");
}

export function exportAsMarkdown(
  headers: string[],
  rows: (string | number)[][],
): string {
  const escape = (s: string) => s.replace(/\|/g, "\\|");

  const headerLine = "| " + headers.map(escape).join(" | ") + " |";
  const sepLine = "| " + headers.map(() => "---").join(" | ") + " |";

  const dataLines = rows.map((row) => {
    const cells = row.map((cell) =>
      typeof cell === "number" ? cell.toFixed(3) : escape(String(cell)),
    );
    return "| " + cells.join(" | ") + " |";
  });

  return [headerLine, sepLine, ...dataLines].join("\n");
}
