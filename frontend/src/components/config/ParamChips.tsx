/** Reusable chip-group picker for parameter values. */

interface ParamChipsProps<T extends string | number> {
  label: string;
  options: { value: T; label: string; disabled?: boolean }[];
  value: T;
  presetValue: T;
  onChange: (value: T) => void;
}

export default function ParamChips<T extends string | number>({
  label,
  options,
  value,
  presetValue,
  onChange,
}: ParamChipsProps<T>) {
  return (
    <div className="flex items-start gap-3">
      <span className="w-28 shrink-0 pt-1.5 text-sm text-gray-600">{label}</span>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => {
          const isActive = opt.value === value;
          const isOverridden = isActive && value !== presetValue;
          const disabled = opt.disabled ?? false;

          let cls =
            "rounded-lg px-3 py-1.5 text-sm border transition select-none ";
          if (disabled) {
            cls += "bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed";
          } else if (isActive && isOverridden) {
            cls += "bg-amber-50 border-amber-400 text-amber-700 font-medium cursor-pointer";
          } else if (isActive) {
            cls += "bg-blue-50 border-blue-400 text-blue-700 font-medium cursor-pointer";
          } else {
            cls += "bg-gray-50 border-gray-200 text-gray-600 hover:border-gray-400 cursor-pointer";
          }

          return (
            <button
              key={String(opt.value)}
              type="button"
              disabled={disabled}
              className={cls}
              onClick={() => !disabled && onChange(opt.value)}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
