import { useEffect, useState, useCallback, useMemo } from "react";
import PageHeader from "../components/layout/PageHeader";
import {
  getDatasetRegistry,
  getDatasets,
  getDataset,
  deleteDataset,
  generateDataset,
} from "../api/client";
import type {
  DatasetRegistryEntry,
  DatasetInfo,
  DatasetDetail,
  DatasetProgressEvent,
} from "../api/types";
import { hashColor } from "../utils/hashColor";

// ── Preset types ──────────────────────────────────────────────────────

interface CategoryPresetEntry {
  type: string;
  prompt: string;
  count: number;
}

interface CategoryPreset {
  name: string;
  categories: CategoryPresetEntry[];
  builtIn: boolean;
}

// ── Built-in presets ──────────────────────────────────────────────────

const BUILTIN_PRESETS: CategoryPreset[] = [
  {
    name: "Question_Types",
    builtIn: true,
    categories: [
      {
        type: "factual",
        count: 5,
        prompt: "Generate factual recall questions about specific events, dates, names, and places mentioned in the passage.",
      },
      {
        type: "temporal",
        count: 5,
        prompt: "Generate questions about chronological order, time periods, before/after relationships, and sequences of events.",
      },
      {
        type: "comparative",
        count: 5,
        prompt: "Generate questions that compare or contrast events, strategies, figures, or outcomes described in the passage.",
      },
      {
        type: "entity_centric",
        count: 5,
        prompt: "Generate questions focused on the roles, actions, and significance of specific people, organizations, or places.",
      },
      {
        type: "relationship",
        count: 5,
        prompt: "Generate questions about causal links, alliances, conflicts, and connections between entities or events.",
      },
      {
        type: "analytical",
        count: 5,
        prompt: "Generate questions requiring analysis, synthesis, or evaluation of impacts, consequences, and broader significance.",
      },
    ],
  },
  {
    name: "Query_Styles",
    builtIn: true,
    categories: [
      {
        type: "well_redacted",
        count: 30,
        prompt: "Generate a well-written, grammatically correct question that a knowledgeable user might ask about the passage. Use proper sentence structure and precise vocabulary.",
      },
      {
        type: "grammar_typos",
        count: 15,
        prompt: "Based on the passage content, write a question that a user would ask to find this information — but intentionally introduce 1-2 grammar mistakes or typos into the question itself (e.g. missing articles, wrong verb tense, swapped letters). The passage content is correct; only the question text should contain errors.",
      },
      {
        type: "name_misspelling",
        count: 15,
        prompt: "Based on the passage content, write a question that a user would ask to find this information — but intentionally misspell one person name, place name, or organization in the question itself (e.g. 'Churchil' instead of 'Churchill', 'Stalinrad' instead of 'Stalingrad'). The passage content is correct; only the question text should have the misspelling.",
      },
      {
        type: "date_place_errors",
        count: 10,
        prompt: "Based on the passage content, write a question that a user would ask to find this information — but intentionally use a slightly wrong date or place in the question itself (e.g. off by one year, wrong city). The passage content is correct; only the question text should contain the factual error.",
      },
      {
        type: "keywords",
        count: 10,
        prompt: "Generate a keyword-only query (no sentence structure) about the passage. Use 3-6 relevant keywords separated by spaces, like a search engine query. Example: 'Battle Stalingrad casualties 1942 German'.",
      },
      {
        type: "telegraphic",
        count: 10,
        prompt: "Generate a very short, telegraphic question about the passage using minimal words (5-10 words max). Drop articles, pronouns, and auxiliary verbs. Example: 'Who led D-Day invasion?' or 'Casualties Battle of Bulge?'.",
      },
      {
        type: "fragmented_french",
        count: 10,
        prompt: "Generate a question about the passage written in broken or approximate French, as if by a non-native speaker mixing French and English. Example: 'Qui a gagné la battle de Midway et pourquoi c'était important?'. The question should still be answerable from the English passage.",
      },
    ],
  },
];

// ── Constants ────────────────────────────────────────────────────────

const LLM_MODELS = [
  { value: "nvidia/nemotron-3-nano-30b-a3b:free", label: "Nemotron Nano 30B (free)" },
  { value: "mistralai/mistral-small-3.1-24b-instruct:free", label: "Mistral Small 3.1 (free)" },
  { value: "google/gemma-3-4b-it:free", label: "Gemma 3 4B (free)" },
  { value: "meta-llama/llama-4-scout:free", label: "Llama 4 Scout (free)" },
  { value: "qwen/qwen3-8b:free", label: "Qwen 3 8B (free)" },
];

const LOCALSTORAGE_KEY = "evalCategoryPresets";

// ── localStorage helpers ─────────────────────────────────────────────

function loadCustomPresets(): CategoryPreset[] {
  try {
    const raw = localStorage.getItem(LOCALSTORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as CategoryPreset[];
    return parsed.map((p) => ({ ...p, builtIn: false }));
  } catch {
    return [];
  }
}

function saveCustomPresets(presets: CategoryPreset[]) {
  localStorage.setItem(LOCALSTORAGE_KEY, JSON.stringify(presets));
}

// ── Card state type ──────────────────────────────────────────────────

interface CardState {
  id: string;
  type: string;
  prompt: string;
  count: number;
  generated: number;
  enabled: boolean;
}

let _cardIdCounter = 0;

function cardsFromPreset(preset: CategoryPreset): CardState[] {
  return preset.categories.map((c) => ({
    id: `card_${_cardIdCounter++}`,
    type: c.type,
    prompt: c.prompt,
    count: c.count,
    generated: 0,
    enabled: true,
  }));
}

// ── Component ────────────────────────────────────────────────────────

export default function DatasetManager() {
  // Dataset registry for dropdown
  const [registryDatasets, setRegistryDatasets] = useState<DatasetRegistryEntry[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetRegistryEntry | null>(null);
  const [selectedCollection, setSelectedCollection] = useState("");

  // Create form
  const [datasetName, setDatasetName] = useState("");
  const [globalModel, setGlobalModel] = useState(LLM_MODELS[0]!.value);
  const [cards, setCards] = useState<CardState[]>(() => cardsFromPreset(BUILTIN_PRESETS[0]!));

  // Preset state
  const [customPresets, setCustomPresets] = useState<CategoryPreset[]>(() => loadCustomPresets());
  const [savingPreset, setSavingPreset] = useState(false);
  const [newPresetName, setNewPresetName] = useState("");

  // Generation state
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);
  const [genSuccess, setGenSuccess] = useState<string | null>(null);

  // Datasets list
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [listLoading, setListLoading] = useState(true);

  // Detail view
  const [selectedDetail, setSelectedDetail] = useState<DatasetDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // Delete state
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);

  // ── Preset logic ────────────────────────────────────────────────────

  const allPresets = useMemo(
    () => [...BUILTIN_PRESETS, ...customPresets],
    [customPresets],
  );

  const matchedPresetName = useMemo(() => {
    for (const preset of allPresets) {
      if (preset.categories.length !== cards.length) continue;
      const match = preset.categories.every(
        (pc, i) => cards[i]!.type === pc.type && cards[i]!.prompt === pc.prompt && cards[i]!.count === pc.count,
      );
      if (match) return preset.name;
    }
    return null;
  }, [allPresets, cards]);

  const matchedPreset = useMemo(
    () => allPresets.find((p) => p.name === matchedPresetName) ?? null,
    [allPresets, matchedPresetName],
  );

  const applyPreset = useCallback((preset: CategoryPreset) => {
    setCards(cardsFromPreset(preset));
  }, []);

  const handleSavePreset = useCallback(() => {
    const name = newPresetName.trim();
    if (!name) return;
    if (allPresets.some((p) => p.name === name)) return;
    const preset: CategoryPreset = {
      name,
      builtIn: false,
      categories: cards.map((c) => ({ type: c.type, prompt: c.prompt, count: c.count })),
    };
    const updated = [...customPresets, preset];
    setCustomPresets(updated);
    saveCustomPresets(updated);
    setSavingPreset(false);
    setNewPresetName("");
  }, [newPresetName, allPresets, cards, customPresets]);

  const handleDeletePreset = useCallback(() => {
    if (!matchedPresetName || matchedPreset?.builtIn) return;
    const updated = customPresets.filter((p) => p.name !== matchedPresetName);
    setCustomPresets(updated);
    saveCustomPresets(updated);
  }, [matchedPresetName, matchedPreset, customPresets]);

  // ── Data fetching ──────────────────────────────────────────────────

  const refreshDatasets = useCallback(async () => {
    setListLoading(true);
    try {
      const res = await getDatasets();
      setDatasets(res.datasets);
    } catch {
      /* ignore */
    } finally {
      setListLoading(false);
    }
  }, []);

  useEffect(() => {
    getDatasetRegistry().then((r) => {
      setRegistryDatasets(r.datasets);
      if (r.datasets.length > 0 && !selectedDataset) {
        const first = r.datasets[0]!;
        setSelectedDataset(first);
        setSelectedCollection(first.collections[0] ?? first.default_collection);
      }
    });
    refreshDatasets();
  }, [refreshDatasets]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Card helpers ───────────────────────────────────────────────────

  const updateCard = useCallback((idx: number, patch: Partial<CardState>) => {
    setCards((prev) => prev.map((c, i) => (i === idx ? { ...c, ...patch } : c)));
  }, []);

  const addCard = useCallback(() => {
    setCards((prev) => [
      ...prev,
      {
        id: `card_${_cardIdCounter++}`,
        type: "custom",
        prompt: "",
        count: 5,
        generated: 0,
        enabled: true,
      },
    ]);
  }, []);

  const removeCard = useCallback((idx: number) => {
    setCards((prev) => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== idx)));
  }, []);

  const totalQuestions = cards.filter((c) => c.enabled).reduce((s, c) => s + c.count, 0);
  const totalGenerated = cards.filter((c) => c.enabled).reduce((s, c) => s + c.generated, 0);

  // Auto-generate default name from preset + collection + total questions
  useEffect(() => {
    if (matchedPresetName && selectedCollection) {
      setDatasetName(`${matchedPresetName.replace(/\s+/g, "_")}_${selectedCollection}_${totalQuestions}q`);
    }
  }, [matchedPresetName, selectedCollection, totalQuestions]);

  // ── Generate handler ──────────────────────────────────────────────

  const handleGenerate = useCallback(() => {
    if (!datasetName.trim() || !selectedCollection) return;
    const enabledCards = cards.filter((c) => c.enabled && c.count > 0);
    if (enabledCards.length === 0) return;

    setGenerating(true);
    setGenError(null);
    setGenSuccess(null);

    // Reset generated counters
    setCards((prev) =>
      prev.map((c) => ({ ...c, generated: 0 })),
    );

    generateDataset(
      {
        name: datasetName,
        collection_name: selectedCollection,
        categories: enabledCards.map((c) => ({
          type: c.type,
          prompt: c.prompt,
          model: globalModel,
          count: c.count,
        })),
      },
      {
        onProgress: (e: DatasetProgressEvent) => {
          setCards((prev) =>
            prev.map((c) =>
              c.type === e.category ? { ...c, generated: e.generated } : c,
            ),
          );
        },
        onCategoryComplete: () => {
          // nothing extra needed — progress already updates
        },
        onComplete: (e) => {
          setGenerating(false);
          setGenSuccess(
            `Evaluation set created with ${e.total_generated} questions.`,
          );
          refreshDatasets();
        },
        onError: (msg) => {
          setGenerating(false);
          setGenError(msg);
        },
      },
    );
  }, [datasetName, selectedCollection, cards, globalModel, refreshDatasets]);

  // ── Detail handler ─────────────────────────────────────────────────

  const handleViewDetail = useCallback(
    async (id: string) => {
      if (selectedDetail?.id === id) {
        setSelectedDetail(null);
        return;
      }
      setDetailLoading(true);
      try {
        const data = await getDataset(id);
        setSelectedDetail(data);
      } catch {
        /* ignore */
      } finally {
        setDetailLoading(false);
      }
    },
    [selectedDetail],
  );

  // ── Delete handler ─────────────────────────────────────────────────

  const handleDelete = useCallback(
    async (id: string) => {
      setDeleteLoading(true);
      try {
        await deleteDataset(id);
        if (selectedDetail?.id === id) setSelectedDetail(null);
        await refreshDatasets();
      } catch {
        /* ignore */
      } finally {
        setDeleteLoading(false);
        setDeletingId(null);
      }
    },
    [refreshDatasets, selectedDetail],
  );

  // ── Render ─────────────────────────────────────────────────────────

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <PageHeader
        title="Evaluations"
        description="Create evaluation question sets associated to a collection, with customizable categories and generation prompts."
      />

      {/* ── Create Dataset ──────────────────────────────────────────── */}
      <section className="rounded border border-gray-200 bg-white p-5 mb-6">
        <h2 className="text-sm font-semibold text-gray-900 uppercase tracking-wider mb-4">
          Create Evaluation Set
        </h2>

        <div className="space-y-4">
          {/* Name + Collection row */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Evaluation Name
              </label>
              <input
                type="text"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                placeholder="e.g. WW2 Factual 50Q"
                className="w-full rounded border border-gray-300 px-3 py-1.5 text-sm
                           focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Collection
              </label>
              <select
                value={selectedDataset?.name ?? ""}
                onChange={(e) => {
                  const entry = registryDatasets.find((d) => d.name === e.target.value);
                  setSelectedDataset(entry ?? null);
                  if (entry) {
                    setSelectedCollection(entry.collections[0] ?? entry.default_collection);
                  } else {
                    setSelectedCollection("");
                  }
                }}
                className="w-full rounded border-gray-300 bg-white px-3 py-1.5 text-sm shadow-sm
                           focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              >
                {registryDatasets.length === 0 && (
                  <option value="">No datasets available</option>
                )}
                {registryDatasets.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name} ({d.description})
                  </option>
                ))}
              </select>
              {selectedDataset && selectedDataset.collections.length > 1 && (
                <select
                  value={selectedCollection}
                  onChange={(e) => setSelectedCollection(e.target.value)}
                  className="w-full mt-1 rounded border-gray-300 bg-white px-3 py-1.5 text-sm shadow-sm
                             focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                >
                  {selectedDataset.collections.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              )}
              {selectedDataset && selectedDataset.collections.length === 0 && (
                <p className="mt-1 text-xs text-amber-600">
                  No indexed collection found for this dataset. Create one in Collections first.
                </p>
              )}
            </div>
          </div>

          {/* Preset selector row */}
          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-gray-700 shrink-0">Preset</label>
            <select
              value={matchedPresetName ?? ""}
              onChange={(e) => {
                const preset = allPresets.find((p) => p.name === e.target.value);
                if (preset) applyPreset(preset);
              }}
              className="rounded border-gray-300 bg-white px-3 py-1.5 text-sm shadow-sm
                         focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            >
              <option value="">-- Custom --</option>
              <optgroup label="Built-in">
                {BUILTIN_PRESETS.map((p) => (
                  <option key={p.name} value={p.name}>{p.name}</option>
                ))}
              </optgroup>
              {customPresets.length > 0 && (
                <optgroup label="Custom">
                  {customPresets.map((p) => (
                    <option key={p.name} value={p.name}>{p.name}</option>
                  ))}
                </optgroup>
              )}
            </select>

            {/* Save as Preset */}
            {savingPreset ? (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={newPresetName}
                  onChange={(e) => setNewPresetName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleSavePreset(); }}
                  placeholder="Preset name..."
                  autoFocus
                  className="rounded border border-gray-300 px-2 py-1 text-sm
                             focus:border-blue-500 focus:ring-1 focus:ring-blue-500 w-40"
                />
                <button
                  onClick={handleSavePreset}
                  disabled={!newPresetName.trim() || allPresets.some((p) => p.name === newPresetName.trim())}
                  className="text-xs font-medium text-blue-600 hover:text-blue-800
                             disabled:text-gray-400 disabled:cursor-not-allowed"
                >
                  Save
                </button>
                <button
                  onClick={() => { setSavingPreset(false); setNewPresetName(""); }}
                  className="text-xs text-gray-400 hover:text-gray-600"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                onClick={() => setSavingPreset(true)}
                className="text-xs font-medium text-blue-600 hover:text-blue-800"
              >
                Save as Preset
              </button>
            )}

            {/* Delete Preset (only for non-built-in matched presets) */}
            {matchedPreset && !matchedPreset.builtIn && (
              <button
                onClick={handleDeletePreset}
                className="text-xs font-medium text-red-500 hover:text-red-700"
              >
                Delete Preset
              </button>
            )}
          </div>

          {/* Global model selector */}
          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-gray-700 shrink-0">Model</label>
            <select
              value={globalModel}
              onChange={(e) => setGlobalModel(e.target.value)}
              className="rounded border-gray-300 bg-white px-3 py-1.5 text-sm shadow-sm
                         focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            >
              {LLM_MODELS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          {/* Category cards grid */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">
                Question Categories
              </span>
              <span className="text-xs text-gray-500">
                Total: {totalGenerated} / {totalQuestions} questions
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {cards.map((card, idx) => (
                <div
                  key={card.id}
                  className={`rounded-lg border p-4 transition-all ${
                    card.enabled
                      ? "border-gray-200 bg-white"
                      : "border-gray-100 bg-gray-50 opacity-50"
                  }`}
                >
                  {/* Card header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={card.enabled}
                        onChange={(e) =>
                          updateCard(idx, { enabled: e.target.checked })
                        }
                        className="rounded border-gray-300 text-blue-600
                                   focus:ring-blue-500"
                      />
                      <input
                        type="text"
                        value={card.type}
                        onChange={(e) => updateCard(idx, { type: e.target.value })}
                        disabled={!card.enabled}
                        className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold border w-28
                                    focus:bg-transparent focus:outline-none focus:ring-1 focus:ring-blue-500
                                    disabled:opacity-50 ${hashColor(card.type)}`}
                      />
                      <button
                        onClick={() => removeCard(idx)}
                        disabled={cards.length <= 1}
                        className="text-gray-300 hover:text-red-500 disabled:hover:text-gray-300
                                   disabled:cursor-not-allowed transition-colors"
                        title="Remove category"
                      >
                        <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                    {/* Live counter */}
                    <span className="text-xs font-mono text-gray-500">
                      <span className={card.generated > 0 ? "text-green-600 font-semibold" : ""}>
                        {card.generated}
                      </span>
                      {" / "}
                      {card.count}
                    </span>
                  </div>

                  {/* Prompt */}
                  <textarea
                    value={card.prompt}
                    onChange={(e) => updateCard(idx, { prompt: e.target.value })}
                    disabled={!card.enabled}
                    rows={3}
                    className="w-full rounded border border-gray-200 px-2.5 py-1.5 text-xs text-gray-700
                               focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                               disabled:bg-gray-50 disabled:text-gray-400 resize-none mb-2"
                    placeholder="Describe the generation task..."
                  />

                  {/* Count */}
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={card.count}
                      onChange={(e) =>
                        updateCard(idx, {
                          count: Math.max(1, parseInt(e.target.value) || 1),
                        })
                      }
                      disabled={!card.enabled}
                      className="w-16 rounded border border-gray-200 px-2 py-1 text-xs text-center
                                 focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                                 disabled:bg-gray-50 disabled:text-gray-400"
                    />
                    <span className="text-xs text-gray-400">questions</span>
                  </div>

                  {/* Progress bar */}
                  {generating && card.enabled && card.count > 0 && (
                    <div className="mt-2">
                      <div className="h-1.5 w-full rounded-full bg-gray-100 overflow-hidden">
                        <div
                          className="h-full rounded-full bg-blue-500 transition-all duration-300"
                          style={{
                            width: `${Math.min(100, (card.generated / card.count) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            <button
              onClick={addCard}
              className="mt-3 w-full rounded-lg border border-dashed border-gray-300 py-2 text-sm
                         text-gray-500 hover:border-blue-400 hover:text-blue-600 transition-colors"
            >
              + Add Category
            </button>
          </div>

          {/* Create button */}
          <div className="flex items-center gap-4 pt-2">
            <button
              onClick={handleGenerate}
              disabled={
                generating || !datasetName.trim() || !selectedCollection || (selectedDataset?.collections.length === 0) || totalQuestions === 0
              }
              className="rounded bg-blue-600 px-5 py-2 text-sm font-medium text-white
                         hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                         transition-colors"
            >
              {generating ? "Generating..." : "Create Evaluation Set"}
            </button>

            {generating && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
                Generating {totalGenerated} / {totalQuestions} questions...
              </div>
            )}
          </div>
        </div>

        {/* Alerts */}
        {genError && (
          <div className="mt-4 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {genError}
          </div>
        )}
        {genSuccess && (
          <div className="mt-4 rounded border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
            {genSuccess}
          </div>
        )}
      </section>

      {/* ── Existing Datasets ───────────────────────────────────────── */}
      <section className="rounded border border-gray-200 bg-white p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-gray-900 uppercase tracking-wider">
            Existing Evaluation Sets
          </h2>
          <button
            onClick={refreshDatasets}
            disabled={listLoading}
            className="text-xs text-gray-500 hover:text-gray-700 transition"
          >
            {listLoading ? "Loading..." : "Refresh"}
          </button>
        </div>

        {listLoading && datasets.length === 0 ? (
          <div className="flex items-center justify-center py-12">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
          </div>
        ) : datasets.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-8">
            No evaluation sets yet. Create one above.
          </p>
        ) : (
          <div className="space-y-2">
            {datasets.map((ds) => {
              const isExpanded = selectedDetail?.id === ds.id;
              const isConfirmingDelete = deletingId === ds.id;

              return (
                <div key={ds.id}>
                  {/* Dataset row */}
                  <div
                    className={`flex items-center justify-between rounded-lg border px-4 py-3 cursor-pointer transition-colors ${
                      isExpanded
                        ? "border-blue-300 bg-blue-50"
                        : "border-gray-200 hover:bg-gray-50"
                    }`}
                    onClick={() => handleViewDetail(ds.id)}
                  >
                    <div className="flex items-center gap-4 min-w-0">
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-gray-900 truncate">
                          {ds.name}
                        </div>
                        <div className="text-xs text-gray-500">
                          {ds.collection_name} &middot;{" "}
                          {new Date(ds.created_at).toLocaleDateString()}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 shrink-0">
                      {/* Category pills */}
                      <div className="hidden sm:flex gap-1">
                        {ds.categories.map((cat) => (
                          <span
                            key={cat.type}
                            className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium border ${hashColor(cat.type)}`}
                          >
                            {cat.type.replace("_", " ")} ({cat.generated ?? cat.count})
                          </span>
                        ))}
                      </div>

                      {/* Status badge */}
                      <span
                        className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${
                          ds.status === "completed"
                            ? "bg-green-100 text-green-700"
                            : ds.status === "generating"
                              ? "bg-yellow-100 text-yellow-700"
                              : "bg-red-100 text-red-700"
                        }`}
                      >
                        {ds.total_questions}q &middot; {ds.status}
                      </span>

                      {/* Delete */}
                      <span
                        onClick={(e) => e.stopPropagation()}
                        className="flex items-center gap-1"
                      >
                        {isConfirmingDelete ? (
                          <>
                            <button
                              onClick={() => handleDelete(ds.id)}
                              disabled={deleteLoading}
                              className="text-xs text-red-600 font-medium hover:text-red-800"
                            >
                              {deleteLoading ? "..." : "Confirm"}
                            </button>
                            <button
                              onClick={() => setDeletingId(null)}
                              className="text-xs text-gray-400 hover:text-gray-600"
                            >
                              Cancel
                            </button>
                          </>
                        ) : (
                          <button
                            onClick={() => setDeletingId(ds.id)}
                            className="text-xs text-gray-400 hover:text-red-600 transition"
                          >
                            Delete
                          </button>
                        )}
                      </span>
                    </div>
                  </div>

                  {/* Expanded detail */}
                  {isExpanded && selectedDetail && (
                    <DatasetDetailView detail={selectedDetail} />
                  )}
                  {isExpanded && detailLoading && (
                    <div className="flex items-center justify-center py-6">
                      <div className="h-6 w-6 animate-spin rounded-full border-2 border-blue-600 border-t-transparent" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}

// ── Detail sub-component ─────────────────────────────────────────────

function DatasetDetailView({ detail }: { detail: DatasetDetail }) {
  const [expandedType, setExpandedType] = useState<string | null>(null);

  const questionsByType: Record<string, typeof detail.questions> = {};
  for (const q of detail.questions) {
    if (!questionsByType[q.type]) questionsByType[q.type] = [];
    questionsByType[q.type]!.push(q);
  }

  return (
    <div className="mt-2 rounded-lg border border-gray-200 bg-white p-4 space-y-4">
      {/* Summary */}
      {detail.metadata && (
        <div className="flex items-center gap-6 text-xs text-gray-500">
          <span>
            <strong className="text-gray-700">{detail.metadata.total_generated}</strong> questions
          </span>
          <span>
            <strong className="text-gray-700">{detail.metadata.unique_articles}</strong> unique articles
          </span>
          <span>
            Generated in <strong className="text-gray-700">{detail.metadata.generation_time_s}s</strong>
          </span>
        </div>
      )}

      {/* Category cards (read-only) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {detail.categories.map((cat) => {
          const catQuestions = questionsByType[cat.type] ?? [];
          const isOpen = expandedType === cat.type;

          return (
            <div
              key={cat.type}
              className={`rounded-lg border p-3 cursor-pointer transition-colors ${
                isOpen ? "border-blue-300 bg-blue-50" : "border-gray-200 hover:border-gray-300"
              }`}
              onClick={() => setExpandedType(isOpen ? null : cat.type)}
            >
              <div className="flex items-center justify-between mb-1">
                <span
                  className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold border ${hashColor(cat.type)}`}
                >
                  {cat.type.replace("_", " ")}
                </span>
                <span className="text-xs font-mono text-gray-500">
                  {cat.generated ?? catQuestions.length} / {cat.count}
                </span>
              </div>
              <p className="text-xs text-gray-500 line-clamp-2">{cat.prompt}</p>
              <div className="mt-1 text-xs text-gray-400">
                Model: {(cat.model as string).split("/").pop()}
              </div>
            </div>
          );
        })}
      </div>

      {/* Questions table for expanded type */}
      {expandedType && questionsByType[expandedType] && (
        <div className="rounded border border-gray-200 overflow-hidden">
          <div className="bg-gray-50 px-4 py-2 text-xs font-semibold text-gray-600 uppercase tracking-wider border-b border-gray-200">
            {expandedType.replace("_", " ")} questions ({questionsByType[expandedType].length})
          </div>
          <div className="divide-y divide-gray-100 max-h-96 overflow-y-auto">
            {questionsByType[expandedType].map((q) => (
              <div key={q.id} className="px-4 py-3">
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm text-gray-900">{q.question}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      <span
                        className={`inline-block rounded px-1.5 py-0.5 mr-2 text-xs font-medium ${
                          q.difficulty === "easy"
                            ? "bg-green-100 text-green-700"
                            : q.difficulty === "medium"
                              ? "bg-yellow-100 text-yellow-700"
                              : "bg-red-100 text-red-700"
                        }`}
                      >
                        {q.difficulty}
                      </span>
                      {q.source_article}
                    </p>
                    {q.expected_answer_hint && (
                      <p className="text-xs text-gray-400 mt-1 italic">
                        Hint: {q.expected_answer_hint}
                      </p>
                    )}
                  </div>
                  <span className="text-xs text-gray-400 shrink-0 font-mono">
                    {q.id}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* All questions view when no type is expanded */}
      {!expandedType && detail.questions.length > 0 && (
        <p className="text-xs text-gray-400 text-center py-2">
          Click a category card above to view its questions.
        </p>
      )}
    </div>
  );
}
