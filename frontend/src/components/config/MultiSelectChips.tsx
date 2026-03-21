/** Multi-select chip-group picker for sweep parameter values. */

interface MultiSelectChipsProps<T extends string | number> {
  label: string;
  options: { value: T; label: string; disabled?: boolean }[];
  values: T[];
  presetValue: T;
  onChange: (values: T[]) => void;
}

export default function MultiSelectChips<T extends string | number>({
  label,
  options,
  values,
  presetValue,
  onChange,
}: MultiSelectChipsProps<T>) {
  const isOverridden =
    values.length !== 1 || values[0] !== presetValue;

  return (
    <div className="flex items-start gap-3">
      <span className="w-28 shrink-0 pt-1.5 text-sm text-gray-600">{label}</span>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => {
          const isSelected = values.includes(opt.value);
          const disabled = opt.disabled ?? false;

          let cls =
            "rounded-lg px-3 py-1.5 text-sm border transition select-none ";
          if (disabled) {
            cls += "bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed";
          } else if (isSelected && isOverridden) {
            cls += "bg-amber-50 border-amber-400 text-amber-700 font-medium cursor-pointer";
          } else if (isSelected) {
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
              onClick={() => {
                if (disabled) return;
                if (isSelected) {
                  if (values.length <= 1) return;
                  onChange(values.filter((v) => v !== opt.value));
                } else {
                  onChange([...values, opt.value]);
                }
              }}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
