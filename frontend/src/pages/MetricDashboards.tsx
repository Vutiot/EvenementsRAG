import PageHeader from "../components/layout/PageHeader";

export default function MetricDashboards() {
  return (
    <div className="p-6 max-w-7xl mx-auto">
      <PageHeader
        title="Metric Dashboards"
        description="Detailed metric views by category: retrieval, generation, latency."
      />
      <div className="rounded border border-gray-200 bg-white p-8 text-center text-sm text-gray-400">
        Coming in E3-F2-T2
      </div>
    </div>
  );
}
