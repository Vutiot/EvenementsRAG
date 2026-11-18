/**
 * Demo fixtures — entirely fictional data used by the offline demo build
 * (`npm run build:demo`). No backend, no LLM call, no API key.
 */
import type {
  BenchmarkConfig,
  CollectionInfo,
  DatasetDetail,
  DatasetInfo,
  DatasetQuestion,
  DatasetRegistryEntry,
  NormalizedBenchmarkResult,
  NormalizedQuestion,
  ResultFileInfo,
  RetrievedChunk,
} from "../api/types";
import { deriveCollectionName } from "../constants/paramOptions";

// ── Deterministic PRNG so the demo looks the same on every load ─────────
let seed = 42;
function rand(): number {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
}
function around(base: number, spread: number): number {
  return Math.max(0, Math.min(1, base + (rand() - 0.5) * 2 * spread));
}
const r3 = (x: number) => Math.round(x * 1000) / 1000;

// ── Base config (mirrors config/benchmarks/default.yaml) ────────────────
export const DEFAULT_CONFIG: BenchmarkConfig = {
  name: "default",
  description: "Default preset - base configuration for interactive queries",
  dataset: {
    dataset_name: "wiki_10k",
    collection_name: "wiki_10k_qdrant_cs512_co128_minilm_l6_cosine",
    questions_file: "data/evaluation/eval_10k_200q.json",
    articles_dir: null,
  },
  embedding: { model_name: "sentence-transformers/all-MiniLM-L6-v2", dimension: 384 },
  chunking: { chunk_size: 512, chunk_overlap: 128 },
  retrieval: {
    technique: "vanilla",
    top_k: 100,
    rerank_k: 20,
    sparse_weight: 0.3,
    dense_weight: 0.7,
    fusion_method: "rrf",
    sparse_type: "bm25",
  },
  reranker: { type: "cross_encoder", model_name: "cross-encoder/ms-marco-MiniLM-L-12-v2" },
  generation: {
    llm_provider: "nvidia",
    model: "nvidia/nemotron-3-nano-30b-a3b",
    temperature: 0,
    max_tokens: 512,
    top_k_chunks: 20,
    top_k_articles: null,
    prompt_template: null,
    system_prompt: null,
    highlight_chunks: false,
    enabled: true,
  },
  evaluation: {
    k_values: [1, 3, 5, 10],
    compute_ragas: false,
    compute_context_precision: false,
    compute_entity_metrics: false,
    ragas_metrics: ["faithfulness", "answer_relevancy", "context_precision"],
    ragas_evaluator_model: "mistralai/mistral-small-4-119b-2603",
    ragas_max_workers: 4,
    ragas_timeout: 120,
    ragas_repeat_count: 1,
  },
  vector_db: { backend: "qdrant", distance_metric: "cosine", connection_params: null },
};

function cloneConfig(patch: (c: BenchmarkConfig) => void): BenchmarkConfig {
  const c = JSON.parse(JSON.stringify(DEFAULT_CONFIG)) as BenchmarkConfig;
  patch(c);
  return c;
}

// ── Collections ────────────────────────────────────────────────────────
const COLLECTION_SPECS: [string, string, number, number, string, string, number][] = [
  ["wiki_10k", "qdrant", 512, 128, "sentence-transformers/all-MiniLM-L6-v2", "cosine", 48213],
  ["wiki_10k", "qdrant", 512, 50, "sentence-transformers/all-MiniLM-L6-v2", "cosine", 41877],
  ["wiki_10k", "qdrant", 256, 50, "sentence-transformers/all-MiniLM-L6-v2", "cosine", 87340],
  ["wiki_10k", "qdrant", 1024, 128, "sentence-transformers/all-MiniLM-L6-v2", "cosine", 23655],
  ["wiki_10k", "qdrant", 512, 128, "BAAI/bge-small-en-v1.5", "cosine", 48213],
  ["wiki_10k", "qdrant", 512, 128, "BAAI/bge-base-en-v1.5", "cosine", 48213],
  ["wiki_10k", "faiss", 512, 128, "sentence-transformers/all-MiniLM-L6-v2", "euclidean", 48213],
  ["octank", "qdrant", 512, 50, "sentence-transformers/all-MiniLM-L6-v2", "cosine", 1932],
  ["octank", "faiss", 256, 0, "BAAI/bge-small-en-v1.5", "dot_product", 3811],
];

const DIM: Record<string, number> = {
  "sentence-transformers/all-MiniLM-L6-v2": 384,
  "BAAI/bge-small-en-v1.5": 384,
  "BAAI/bge-base-en-v1.5": 768,
};

export const COLLECTIONS: CollectionInfo[] = COLLECTION_SPECS.map(
  ([ds, be, cs, co, emb, dist, pts], i) => ({
    name: deriveCollectionName(ds, be, cs, co, emb, dist),
    backend: be,
    vector_size: DIM[emb] ?? 384,
    distance: dist,
    points_count: pts,
    created_at: new Date(Date.UTC(2026, 2, 12 + i * 2, 9 + i, 14)).toISOString(),
  }),
);

export const REGISTRY: DatasetRegistryEntry[] = [
  {
    name: "wiki_10k",
    description: "10 000 articles Wikipedia sur la Seconde Guerre mondiale",
    default_collection: COLLECTIONS[0]!.name,
    collections: COLLECTIONS.filter((c) => c.name.startsWith("wiki_10k")).map((c) => c.name),
  },
  {
    name: "octank",
    description: "Corpus documentaire fictif Octank Financial (PDF convertis)",
    default_collection: COLLECTIONS[7]!.name,
    collections: COLLECTIONS.filter((c) => c.name.startsWith("octank")).map((c) => c.name),
  },
];

// ── Eval datasets ─────────────────────────────────────────────────────
const QUESTION_BANK: [string, string, string, string][] = [
  ["Quand a eu lieu le débarquement de Normandie ?", "factual", "easy", "Operation Overlord"],
  ["Quel général commandait la 3e armée américaine en 1944 ?", "entity", "easy", "George S. Patton"],
  ["Combien de temps a duré la bataille de Stalingrad ?", "temporal", "medium", "Battle of Stalingrad"],
  ["Pourquoi l'opération Market Garden a-t-elle échoué ?", "causal", "hard", "Operation Market Garden"],
  ["Comparez les stratégies navales japonaise et américaine à Midway.", "comparative", "hard", "Battle of Midway"],
  ["Quelles conséquences la conférence de Yalta a-t-elle eu sur l'Europe de l'Est ?", "causal", "hard", "Yalta Conference"],
  ["Qui dirigeait la France libre depuis Londres ?", "entity", "easy", "Free France"],
  ["Quel événement a déclenché l'entrée en guerre des États-Unis ?", "factual", "easy", "Attack on Pearl Harbor"],
  ["Dans quel ordre se sont déroulées El Alamein et l'opération Torch ?", "temporal", "medium", "North African campaign"],
  ["Quel rôle a joué Enigma dans la bataille de l'Atlantique ?", "analytical", "hard", "Battle of the Atlantic"],
  ["Où s'est tenue la conférence de Potsdam ?", "factual", "easy", "Potsdam Conference"],
  ["Quel lien entre le pacte germano-soviétique et l'invasion de la Pologne ?", "causal", "medium", "Molotov–Ribbentrop Pact"],
];

function makeQuestions(prefix: string, collection: string, n: number): DatasetQuestion[] {
  return Array.from({ length: n }, (_, i) => {
    const [q, type, diff, art] = QUESTION_BANK[i % QUESTION_BANK.length]!;
    return {
      id: `${prefix}_q${String(i + 1).padStart(3, "0")}`,
      question: q,
      type,
      difficulty: diff,
      expected_answer_hint: `Voir l'article « ${art} ».`,
      source_article: art,
      source_article_id: `art_${1000 + i}`,
      source_chunk_id: `${collection}::art_${1000 + i}::chunk_${i % 7}`,
      source_chunk_index: i % 7,
      char_start: 512 * (i % 7),
      char_end: 512 * (i % 7) + 480,
      model: "mistralai/mistral-small-4-119b-2603",
      generated_at: "2026-03-20T10:12:00Z",
    };
  });
}

const DATASET_DEFS: Omit<DatasetInfo, "categories" | "total_questions">[] = [
  {
    id: "ds_wiki10k_cs512_co128_200q",
    name: "wiki_10k · cs512/co128 · 200q",
    created_at: "2026-03-20T10:12:00Z",
    status: "completed",
    collection_name: COLLECTIONS[0]!.name,
    model: "mistralai/mistral-small-4-119b-2603",
  },
  {
    id: "ds_wiki10k_cs256_co50_60q",
    name: "wiki_10k · cs256/co50 · 60q",
    created_at: "2026-03-24T15:41:00Z",
    status: "completed",
    collection_name: COLLECTIONS[2]!.name,
    model: "google/gemma-3-27b-it",
  },
  {
    id: "ds_octank_cs512_co50_40q",
    name: "octank · cs512/co50 · 40q",
    created_at: "2026-03-28T08:05:00Z",
    status: "completed",
    collection_name: COLLECTIONS[7]!.name,
    model: "nvidia/nemotron-3-nano-30b-a3b",
  },
];

const CATEGORY_SPLITS: Record<string, [string, number][]> = {
  ds_wiki10k_cs512_co128_200q: [["factual", 60], ["temporal", 40], ["causal", 40], ["comparative", 30], ["entity", 30]],
  ds_wiki10k_cs256_co50_60q: [["factual", 20], ["temporal", 20], ["causal", 20]],
  ds_octank_cs512_co50_40q: [["factual", 25], ["analytical", 15]],
};

export const DATASETS: DatasetDetail[] = DATASET_DEFS.map((d) => {
  const cats = CATEGORY_SPLITS[d.id]!;
  const total = cats.reduce((a, [, n]) => a + n, 0);
  return {
    ...d,
    total_questions: total,
    system_prompt:
      "Tu es un historien. Génère des questions précises dont la réponse se trouve dans l'extrait fourni.",
    categories: cats.map(([type, count]) => ({
      type,
      prompt: `Génère une question de type « ${type} » à partir de l'extrait.`,
      model: d.model ?? "mistralai/mistral-small-4-119b-2603",
      count,
      generated: count,
    })),
    questions: makeQuestions(d.id, d.collection_name, Math.min(total, 24)),
    metadata: { total_generated: total, unique_articles: Math.round(total * 0.8), generation_time_s: total * 2.7 },
  };
});

// ── Benchmark results ─────────────────────────────────────────────────
const METRIC_TYPES = ["factual", "temporal", "causal", "comparative", "entity"];

function makeResult(
  filename: string,
  phase: string,
  ts: string,
  config: BenchmarkConfig,
  quality: number,
  withGeneration: boolean,
): NormalizedBenchmarkResult {
  const n = 24;
  const per_question: NormalizedQuestion[] = Array.from({ length: n }, (_, i) => {
    const [q, type, diff, art] = QUESTION_BANK[i % QUESTION_BANK.length]!;
    const mrr = around(quality, 0.25);
    const metrics: Record<string, number> = {
      mrr: r3(mrr),
      recall_at_1: r3(around(quality - 0.15, 0.2)),
      recall_at_3: r3(around(quality - 0.02, 0.2)),
      recall_at_5: r3(around(quality + 0.08, 0.18)),
      recall_at_10: r3(around(quality + 0.16, 0.15)),
      ndcg_at_5: r3(around(quality + 0.02, 0.2)),
      doc_precision_at_5: r3(around(quality + 0.05, 0.2)),
      doc_recall_at_5: r3(around(quality + 0.12, 0.18)),
      doc_mrr: r3(around(quality + 0.1, 0.2)),
      chunk_precision_at_5: r3(around(quality - 0.2, 0.15)),
      entity_precision_at_5: r3(around(quality - 0.05, 0.2)),
      entity_recall_at_5: r3(around(quality + 0.02, 0.2)),
      entity_mrr: r3(around(quality, 0.2)),
    };
    const ragas = withGeneration
      ? {
          faithfulness: r3(around(0.82, 0.15)),
          answer_relevancy: r3(around(0.78, 0.15)),
          answer_correctness: r3(around(quality + 0.1, 0.2)),
          context_precision: r3(around(quality + 0.05, 0.2)),
        }
      : null;
    return {
      question_id: `q${String(i + 1).padStart(3, "0")}`,
      question: q,
      type,
      difficulty: diff,
      source_article: art,
      ground_truth_count: 1 + (i % 3),
      retrieved_count: config.retrieval.top_k,
      retrieval_time_ms: Math.round(40 + rand() * 90),
      metrics,
      generated_answer: withGeneration
        ? `Réponse simulée : d'après l'article « ${art} », … (texte fictif généré pour la démo).`
        : null,
      generation_time_ms: withGeneration ? Math.round(900 + rand() * 1500) : null,
      retrieved_contexts: withGeneration ? [`Extrait fictif de « ${art} »…`] : null,
      ragas_metrics: ragas,
      ragas_metrics_std: ragas
        ? Object.fromEntries(Object.keys(ragas).map((k) => [k, r3(0.02 + rand() * 0.05)]))
        : null,
    };
  });

  const avg = (k: string) =>
    r3(per_question.reduce((a, q) => a + (q.metrics[k] ?? 0), 0) / per_question.length);
  const avgR = (k: string) =>
    r3(per_question.reduce((a, q) => a + (q.ragas_metrics?.[k] ?? 0), 0) / per_question.length);

  const metrics_by_type: Record<string, Record<string, number>> = {};
  for (const t of METRIC_TYPES) {
    metrics_by_type[t] = {
      mrr: r3(around(quality, 0.15)),
      recall_at_5: r3(around(quality + 0.08, 0.15)),
      recall_at_10: r3(around(quality + 0.15, 0.12)),
      count: per_question.filter((q) => q.type === t).length,
    };
  }

  return {
    filename,
    format: "benchmark_result",
    phase_name: phase,
    timestamp: ts,
    config: config as unknown as Record<string, unknown>,
    avg_recall_at_k: { "1": avg("recall_at_1"), "3": avg("recall_at_3"), "5": avg("recall_at_5"), "10": avg("recall_at_10") },
    avg_mrr: avg("mrr"),
    avg_ndcg: { "5": avg("ndcg_at_5") },
    avg_article_hit_at_k: null,
    avg_chunk_hit_at_k: null,
    avg_doc_precision_at_k: { "5": avg("doc_precision_at_5") },
    avg_doc_recall_at_k: { "5": avg("doc_recall_at_5") },
    avg_doc_mrr: avg("doc_mrr"),
    avg_chunk_precision_at_k: { "5": avg("chunk_precision_at_5") },
    metrics_by_type,
    per_question,
    total_questions: 200,
    avg_retrieval_time_ms: Math.round(per_question.reduce((a, q) => a + (q.retrieval_time_ms ?? 0), 0) / n),
    total_wall_time_s: Math.round(60 + rand() * 400),
    metrics_summary: {
      entity: {
        avg_entity_precision_at_5: avg("entity_precision_at_5"),
        avg_entity_recall_at_5: avg("entity_recall_at_5"),
        avg_entity_mrr: avg("entity_mrr"),
      },
      ...(withGeneration
        ? {
            ragas: {
              avg_faithfulness: avgR("faithfulness"),
              avg_answer_relevancy: avgR("answer_relevancy"),
              avg_answer_correctness: avgR("answer_correctness"),
              avg_context_precision: avgR("context_precision"),
            },
          }
        : {}),
    },
  };
}

export const RESULTS: Record<string, NormalizedBenchmarkResult> = {};
export const RESULT_FILES: ResultFileInfo[] = [];

function summarize(
  r: NormalizedBenchmarkResult,
  runName: string | null,
  evalName: string,
  mode: "direct" | "mapped",
): ResultFileInfo {
  const c = r.config as unknown as BenchmarkConfig;
  const ragas = (r.metrics_summary?.ragas ?? {}) as Record<string, number>;
  const ent = (r.metrics_summary?.entity ?? {}) as Record<string, number>;
  return {
    filename: r.filename,
    phase_name: r.phase_name,
    timestamp: r.timestamp,
    total_questions: r.total_questions,
    format: "benchmark_result",
    avg_mrr: r.avg_mrr,
    avg_recall_at_5: r.avg_recall_at_k["5"] ?? null,
    avg_recall_at_10: r.avg_recall_at_k["10"] ?? null,
    avg_doc_precision_at_5: r.avg_doc_precision_at_k?.["5"] ?? null,
    avg_doc_mrr: r.avg_doc_mrr,
    avg_chunk_precision_at_5: r.avg_chunk_precision_at_k?.["5"] ?? null,
    avg_context_precision: ragas.avg_context_precision ?? null,
    avg_entity_precision_at_5: ent.avg_entity_precision_at_5 ?? null,
    avg_entity_recall_at_5: ent.avg_entity_recall_at_5 ?? null,
    avg_entity_mrr: ent.avg_entity_mrr ?? null,
    total_wall_time_s: r.total_wall_time_s,
    config_summary: {
      technique: c.retrieval.technique,
      chunk_size: c.chunking.chunk_size,
      chunk_overlap: c.chunking.chunk_overlap,
      embedding_model: c.embedding.model_name,
      dataset_name: c.dataset.dataset_name,
      top_k: c.retrieval.top_k,
      llm_model: c.generation.enabled ? c.generation.model : null,
      distance_metric: c.vector_db.distance_metric,
      backend: c.vector_db.backend,
      reranker_type: c.reranker.type,
      reranker_model: c.reranker.model_name,
      rerank_top_k: c.retrieval.rerank_k,
      sparse_weight: c.retrieval.technique === "hybrid" ? c.retrieval.sparse_weight : null,
      sparse_type: c.retrieval.technique === "hybrid" ? c.retrieval.sparse_type : null,
      fusion_method: c.retrieval.technique === "hybrid" ? c.retrieval.fusion_method : null,
    },
    sweep_meta: null,
    run_name: runName,
    eval_dataset_name: evalName,
    evaluation_mode: mode,
  };
}

// Single benchmarks
const SINGLE_RUNS: [string, string, BenchmarkConfig, number, boolean, string][] = [
  ["Baseline vanilla MiniLM", "2026-03-22T14:03:00Z", cloneConfig(() => {}), 0.52, true, "direct"],
  [
    "Hybrid BM25 + rerank",
    "2026-03-26T09:47:00Z",
    cloneConfig((c) => { c.retrieval.technique = "hybrid"; c.retrieval.sparse_weight = 0.3; }),
    0.64,
    true,
    "direct",
  ],
  [
    "BGE-Base sans reranker",
    "2026-03-29T17:20:00Z",
    cloneConfig((c) => {
      c.embedding.model_name = "BAAI/bge-base-en-v1.5"; c.embedding.dimension = 768;
      c.reranker.type = "none"; c.reranker.model_name = null; c.retrieval.top_k = 10;
      c.generation.enabled = false;
    }),
    0.57,
    false,
    "mapped",
  ],
];

SINGLE_RUNS.forEach(([name, ts, cfg, q, gen, mode], i) => {
  const fn = `benchmarks/${cfg.retrieval.technique}_${(0xa1b2c3 + i * 7919).toString(16)}_${ts.slice(0, 10).replace(/-/g, "")}.json`;
  const r = makeResult(fn, `${cfg.retrieval.technique}_rag`, ts, cfg, q, gen);
  RESULTS[fn] = r;
  RESULT_FILES.push(summarize(r, name, DATASETS[0]!.name, mode as "direct" | "mapped"));
});

// Sweep: chunk_size × sparse_weight on hybrid
export const SWEEP_ID = "sweep_20260401_chunk_x_sparse";
const sweptParams: Record<string, unknown[]> = {
  "chunking.chunk_size": [256, 512, 1024],
  "retrieval.sparse_weight": [0.1, 0.3, 0.5],
};
const childFiles: string[] = [];
let idx = 0;
for (const cs of [256, 512, 1024]) {
  for (const sw of [0.1, 0.3, 0.5]) {
    const cfg = cloneConfig((c) => {
      c.retrieval.technique = "hybrid";
      c.chunking.chunk_size = cs;
      c.chunking.chunk_overlap = 50;
      c.retrieval.sparse_weight = sw;
      c.retrieval.dense_weight = r3(1 - sw);
      c.generation.enabled = false;
    });
    const quality = 0.48 + (cs === 512 ? 0.12 : cs === 256 ? 0.05 : 0) + (sw === 0.3 ? 0.06 : sw === 0.1 ? 0.02 : -0.01);
    const fn = `sweeps/${SWEEP_ID}/cfg${idx}_cs${cs}_sw${sw}.json`;
    const r = makeResult(fn, "hybrid_rag", `2026-04-01T1${Math.floor(idx / 3)}:${10 + idx * 4}:00Z`, cfg, quality, false);
    RESULTS[fn] = r;
    childFiles.push(fn);
    idx++;
  }
}
const best = RESULTS[childFiles[4]!]!;
RESULT_FILES.push({
  ...summarize(best, "Sweep chunk_size × sparse_weight", DATASETS[0]!.name, "mapped"),
  filename: `sweeps/${SWEEP_ID}/_sweep_meta.json`,
  timestamp: "2026-04-01T10:02:00Z",
  sweep_meta: { sweep_id: SWEEP_ID, child_filenames: childFiles, swept_params: sweptParams },
});
for (const fn of childFiles) {
  RESULT_FILES.push(summarize(RESULTS[fn]!, null, DATASETS[0]!.name, "mapped"));
}

// ── Query mode chunks ─────────────────────────────────────────────────
export function makeChunks(query: string, k: number): RetrievedChunk[] {
  const arts = ["Operation Overlord", "Normandy landings", "Omaha Beach", "Dwight D. Eisenhower", "Atlantic Wall", "Operation Fortitude", "Utah Beach", "Juno Beach"];
  return Array.from({ length: Math.min(k, 8) }, (_, i) => ({
    chunk_id: `wiki_10k::art_${2000 + i}::chunk_${i}`,
    content:
      `[Extrait fictif ${i + 1}] ${arts[i]} — Le 6 juin 1944, les forces alliées débarquent sur les côtes normandes. ` +
      `Ce passage simulé illustre le type de contenu renvoyé pour la requête « ${query.slice(0, 60)} ». ` +
      `Aucune donnée réelle n'est interrogée dans ce mode démo.`,
    score: r3(0.91 - i * 0.06),
    article_title: arts[i]!,
    source_url: `https://en.wikipedia.org/wiki/${encodeURIComponent(arts[i]!.replace(/ /g, "_"))}`,
    chunk_index: i,
  }));
}

export const DEMO_ANSWER =
  "Le débarquement de Normandie (opération Overlord) a eu lieu le 6 juin 1944. " +
  "Les forces alliées, sous le commandement suprême du général Dwight D. Eisenhower, " +
  "ont débarqué sur cinq plages (Utah, Omaha, Gold, Juno et Sword) [1][2]. " +
  "Cette réponse est un texte fictif rejoué localement : aucun modèle de langage n'a été appelé.";
