export type TestingMode = "query" | "benchmark" | "sweep";

const MODES: { value: TestingMode; label: string }[] = [
  { value: "query", label: "Query" },
  { value: "benchmark", label: "Benchmark" },
  { value: "sweep", label: "Sweep" },
];

interface Props {
  mode: TestingMode;
  onModeChange: (mode: TestingMode) => void;
}

export default function ModeSwitcher({ mode, onModeChange }: Props) {
  const activeIndex = MODES.findIndex((m) => m.value === mode);

  return (
    <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1 relative">
      {/* Sliding background indicator */}
      <div
        className="absolute top-1 bottom-1 rounded-md bg-blue-600 transition-all duration-200"
        style={{
          width: `calc(${100 / MODES.length}% - 2px)`,
          left: `calc(${activeIndex * (100 / MODES.length)}% + 1px)`,
        }}
      />
      {MODES.map((m) => (
        <button
          key={m.value}
          onClick={() => onModeChange(m.value)}
          className={`relative z-10 px-6 py-2 text-sm font-medium rounded-md transition-colors duration-200 ${
            mode === m.value
              ? "text-white"
              : "text-gray-600 hover:bg-gray-50"
          }`}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
