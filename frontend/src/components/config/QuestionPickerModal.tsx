/** Table-based modal for picking an evaluation question. */

import { useEffect } from "react";
import type { DatasetQuestion } from "../../api/types";

interface Props {
  open: boolean;
  onClose: () => void;
  questions: DatasetQuestion[];
  onSelect: (q: DatasetQuestion) => void;
}

const DIFFICULTY_COLORS: Record<string, string> = {
  easy: "bg-green-100 text-green-700",
  medium: "bg-yellow-100 text-yellow-700",
  hard: "bg-red-100 text-red-700",
};

const CATEGORY_COLORS: Record<string, string> = {
  factual: "bg-blue-100 text-blue-700",
  temporal: "bg-purple-100 text-purple-700",
  comparative: "bg-orange-100 text-orange-700",
  entity_centric: "bg-teal-100 text-teal-700",
  relationship: "bg-pink-100 text-pink-700",
  analytical: "bg-indigo-100 text-indigo-700",
};

function truncate(s: string | undefined | null, max: number) {
  if (!s) return "";
  return s.length > max ? s.slice(0, max) + "..." : s;
}

export default function QuestionPickerModal({ open, onClose, questions, onSelect }: Props) {
  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-white rounded-2xl shadow-2xl max-w-5xl w-full max-h-[85vh] overflow-hidden mx-4 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-5 pb-3 shrink-0">
          <h2 className="text-lg font-semibold text-gray-900">
            Pick a Question ({questions.length})
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition p-1"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Table */}
        <div className="overflow-y-auto flex-1 px-6 pb-4">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-white">
              <tr className="border-b border-gray-200 text-left text-xs font-medium uppercase tracking-wider text-gray-400">
                <th className="py-2 pr-2 w-8">#</th>
                <th className="py-2 pr-2">Category</th>
                <th className="py-2 pr-2">Difficulty</th>
                <th className="py-2 pr-2">Source Article</th>
                <th className="py-2 pr-2">Chunk</th>
                <th className="py-2 pr-2">Hint</th>
                <th className="py-2">Question</th>
              </tr>
            </thead>
            <tbody>
              {questions.map((q, i) => (
                <tr
                  key={q.id}
                  onClick={() => {
                    onSelect(q);
                    onClose();
                  }}
                  className="border-b border-gray-50 hover:bg-blue-50 cursor-pointer transition-colors"
                >
                  <td className="py-2 pr-2 text-gray-400 font-mono text-xs">{i + 1}</td>
                  <td className="py-2 pr-2">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                        CATEGORY_COLORS[q.type] ?? "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {q.type}
                    </span>
                  </td>
                  <td className="py-2 pr-2">
                    <span
                      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                        DIFFICULTY_COLORS[q.difficulty] ?? "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {q.difficulty}
                    </span>
                  </td>
                  <td className="py-2 pr-2 text-gray-600 text-xs" title={q.source_article}>
                    {truncate(q.source_article, 30)}
                  </td>
                  <td className="py-2 pr-2 text-gray-500 text-xs font-mono">
                    #{q.source_chunk_index ?? 0}
                  </td>
                  <td className="py-2 pr-2 text-gray-500 text-xs" title={q.expected_answer_hint}>
                    {truncate(q.expected_answer_hint, 50)}
                  </td>
                  <td className="py-2 text-gray-800 text-xs" title={q.question}>
                    {truncate(q.question, 60)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
