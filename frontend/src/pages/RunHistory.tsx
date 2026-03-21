import { useState, useCallback, useEffect } from "react";
import PageHeader from "../components/layout/PageHeader";
import RunHistoryTable from "../components/benchmarks/RunHistoryTable";
import { getResultFiles } from "../api/client";
import type { ResultFileInfo } from "../api/types";

export default function RunHistory() {
  const [results, setResults] = useState<ResultFileInfo[]>([]);

  const loadResults = useCallback(() => {
    getResultFiles()
      .then((files) => {
        const sorted = [...files].sort((a, b) => {
          if (!a.timestamp && !b.timestamp) return 0;
          if (!a.timestamp) return 1;
          if (!b.timestamp) return -1;
          return b.timestamp.localeCompare(a.timestamp);
        });
        setResults(sorted);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    loadResults();
  }, [loadResults]);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <PageHeader
        title="Run History"
        description="View and compare past benchmark and sweep results."
      />
      <RunHistoryTable results={results} activeRun={null} />
    </div>
  );
}
