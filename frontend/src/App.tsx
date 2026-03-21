import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import TestingPage from "./pages/TestingPage";
import BenchmarkViewer from "./pages/BenchmarkViewer";
import RunHistory from "./pages/RunHistory";
import MetricDashboards from "./pages/MetricDashboards";
import CollectionManager from "./pages/CollectionManager";
import DatasetManager from "./pages/DatasetManager";

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/testing" replace />} />
          <Route path="/testing" element={<TestingPage />} />
          <Route path="/collections" element={<CollectionManager />} />
          <Route path="/evaluations" element={<DatasetManager />} />
          <Route path="/runs" element={<RunHistory />} />
          <Route path="/benchmarks" element={<BenchmarkViewer />} />
          <Route path="/metrics" element={<MetricDashboards />} />
          {/* Backward-compat redirects */}
          <Route path="/query" element={<Navigate to="/testing" replace />} />
          <Route path="/sweeps" element={<Navigate to="/testing" replace />} />
        </Routes>
      </main>
    </div>
  );
}
