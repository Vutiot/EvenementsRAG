/** Reusable labeled slider for numeric parameter values. */

interface ParamSliderProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  presetValue: number;
  onChange: (value: number) => void;
}

export default function ParamSlider({
  label,
  min,
  max,
  step,
  value,
  presetValue,
  onChange,
}: ParamSliderProps) {
  const isOverridden = value !== presetValue;
  const accent = isOverridden ? "#d97706" : "#2563eb"; // amber-600 / blue-600

  return (
    <div className="flex items-center gap-3">
      <span className="w-28 shrink-0 text-sm text-gray-600">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="flex-1 h-2 rounded-lg appearance-none bg-gray-200 cursor-pointer"
        style={{ accentColor: accent }}
      />
      <span
        className={`w-12 text-right font-mono text-sm ${
          isOverridden ? "text-amber-700" : "text-gray-700"
        }`}
      >
        {Number.isInteger(step) ? value : value.toFixed(1)}
      </span>
    </div>
  );
}
