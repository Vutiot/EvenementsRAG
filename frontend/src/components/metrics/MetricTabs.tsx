export type TabKey = "retrieval" | "generation" | "latency" | "ragas";

interface TabDef {
  key: TabKey;
  label: string;
  enabledProp: "hasRetrieval" | "hasGeneration" | "hasLatency" | "hasRagas";
}

const TABS: TabDef[] = [
  { key: "retrieval", label: "Retrieval", enabledProp: "hasRetrieval" },
  { key: "generation", label: "Generation", enabledProp: "hasGeneration" },
  { key: "latency", label: "Latency", enabledProp: "hasLatency" },
  { key: "ragas", label: "RAGAS", enabledProp: "hasRagas" },
];

interface Props {
  activeTab: TabKey;
  onTabChange: (tab: TabKey) => void;
  hasRetrieval: boolean;
  hasGeneration: boolean;
  hasLatency: boolean;
  hasRagas: boolean;
}

export default function MetricTabs({
  activeTab,
  onTabChange,
  ...flags
}: Props) {
  return (
    <div className="flex gap-1 border-b border-slate-200 mb-5">
      {TABS.map(({ key, label, enabledProp }) => {
        const enabled = flags[enabledProp];
        const active = activeTab === key;
        if (!enabled && key === "ragas") return null; // hide RAGAS tab entirely when unavailable
        return (
          <button
            key={key}
            disabled={!enabled}
            onClick={() => enabled && onTabChange(key)}
            className={`px-4 py-2 text-sm font-medium transition-colors -mb-px ${
              active
                ? "border-b-2 border-amber-500 text-amber-700"
                : enabled
                  ? "text-slate-500 hover:text-slate-700"
                  : "text-slate-300 cursor-not-allowed"
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
