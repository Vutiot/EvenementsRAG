import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  {
    group: "Query",
    items: [{ to: "/query", label: "Query Tester" }],
  },
  {
    group: "Data",
    items: [
      { to: "/collections", label: "Collections" },
      { to: "/evaluations", label: "Evaluations" },
    ],
  },
  {
    group: "Benchmark",
    items: [
      { to: "/runs", label: "Runs" },
      { to: "/benchmarks", label: "Result Viewer" },
      { to: "/metrics", label: "Metric Dashboards" },
      { to: "/sweeps", label: "Sweep Visualizer" },
    ],
  },
];

export default function Sidebar() {
  return (
    <nav className="w-56 flex-shrink-0 bg-gray-900 text-gray-300 flex flex-col">
      <div className="px-4 py-5 border-b border-gray-700">
        <h1 className="text-lg font-bold text-white tracking-tight">
          EvenementsRAG
        </h1>
        <p className="text-xs text-gray-500 mt-0.5">WW2 RAG Benchmarking</p>
      </div>

      <div className="flex-1 overflow-y-auto py-4">
        {NAV_ITEMS.map((section) => (
          <div key={section.group} className="mb-4">
            <h2 className="px-4 text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
              {section.group}
            </h2>
            {section.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `block px-4 py-2 text-sm transition-colors ${
                    isActive
                      ? "bg-gray-800 text-white border-l-2 border-blue-400"
                      : "hover:bg-gray-800 hover:text-white"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        ))}
      </div>
    </nav>
  );
}
