import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import QueryTester from "./pages/QueryTester";
import BenchmarkViewer from "./pages/BenchmarkViewer";
import MetricDashboards from "./pages/MetricDashboards";
import SweepVisualizer from "./pages/SweepVisualizer";

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/query" replace />} />
          <Route path="/query" element={<QueryTester />} />
          <Route path="/benchmarks" element={<BenchmarkViewer />} />
          <Route path="/metrics" element={<MetricDashboards />} />
          <Route path="/sweeps" element={<SweepVisualizer />} />
        </Routes>
      </main>
    </div>
  );
}
