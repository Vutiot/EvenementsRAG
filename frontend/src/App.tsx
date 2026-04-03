import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import TestingPage from "./pages/TestingPage";
import RunHistory from "./pages/RunHistory";
import CollectionManager from "./pages/CollectionManager";
import DatasetManager from "./pages/DatasetManager";
import BenchViz from "./pages/BenchViz";
import SweepViz from "./pages/SweepViz";

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
          <Route path="/benchmarks" element={<Navigate to="/runs" replace />} />
          <Route path="/bench-viz/:filename" element={<BenchViz />} />
          <Route path="/sweep-viz/:sweepId" element={<SweepViz />} />
          <Route path="/metrics" element={<Navigate to="/runs" replace />} />
          {/* Backward-compat redirects */}
          <Route path="/query" element={<Navigate to="/testing" replace />} />
          <Route path="/sweeps" element={<Navigate to="/testing" replace />} />
        </Routes>
      </main>
    </div>
  );
}
