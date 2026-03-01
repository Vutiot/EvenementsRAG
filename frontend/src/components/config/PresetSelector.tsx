import { useEffect, useState } from "react";
import { getPresets } from "../../api/client";
import type { PresetInfo } from "../../api/types";

/** Group presets by inferred category from filename. */
function categorize(presets: PresetInfo[]): Record<string, PresetInfo[]> {
  const groups: Record<string, PresetInfo[]> = {};
  for (const p of presets) {
    let category = "Base";
    if (p.filename.startsWith("sweep_cs")) category = "Chunk Sweep";
    else if (p.filename.startsWith("wiki_dm_")) category = "Distance Metric";
    else if (p.filename.startsWith("wiki_em_")) category = "Embedding Model";
    else if (
      p.filename.startsWith("wiki_temp") ||
      p.filename.startsWith("wiki_topk") ||
      p.filename.startsWith("wiki_model_")
    )
      category = "Generation";

    if (!groups[category]) groups[category] = [];
    groups[category]!.push(p);
  }
  return groups;
}

interface Props {
  selected: string;
  onSelect: (filename: string) => void;
}

export default function PresetSelector({ selected, onSelect }: Props) {
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPresets()
      .then(setPresets)
      .catch((e: unknown) =>
        setError(e instanceof Error ? e.message : String(e)),
      );
  }, []);

  if (error) {
    return <p className="text-sm text-red-600">Failed to load presets: {error}</p>;
  }

  const grouped = categorize(presets);

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Configuration Preset
      </label>
      <select
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full rounded border-gray-300 bg-white px-3 py-2 text-sm shadow-sm
                   focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
      >
        <option value="">Select a preset...</option>
        {Object.entries(grouped).map(([group, items]) => (
          <optgroup key={group} label={group}>
            {items.map((p) => (
              <option key={p.filename} value={p.filename}>
                {p.name}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  );
}
