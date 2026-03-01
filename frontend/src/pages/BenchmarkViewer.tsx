import PageHeader from "../components/layout/PageHeader";

export default function BenchmarkViewer() {
  return (
    <div className="p-6 max-w-7xl mx-auto">
      <PageHeader
        title="Benchmark Result Viewer"
        description="Browse and compare saved benchmark results."
      />
      <div className="rounded border border-gray-200 bg-white p-8 text-center text-sm text-gray-400">
        Coming in E3-F2-T1
      </div>
    </div>
  );
}
