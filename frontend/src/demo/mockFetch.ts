/**
 * Offline mock of the FastAPI backend for the demo build.
 * Intercepts every `/api/*` fetch and answers with fixtures; SSE endpoints are
 * replayed as timed `ReadableStream`s. No network, no LLM, no API key.
 */
import type { BenchmarkConfig } from "../api/types";
import {
  COLLECTIONS,
  DATASETS,
  DEFAULT_CONFIG,
  DEMO_ANSWER,
  REGISTRY,
  RESULTS,
  RESULT_FILES,
  makeChunks,
} from "./fixtures";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

type SSEStep = { event: string; data: unknown; delay?: number };

function sse(steps: SSEStep[], signal?: AbortSignal | null): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      for (const s of steps) {
        if (signal?.aborted) break;
        await sleep(s.delay ?? 120);
        controller.enqueue(enc.encode(`event: ${s.event}\ndata: ${JSON.stringify(s.data)}\n\n`));
      }
      controller.close();
    },
  });
  return new Response(stream, { headers: { "Content-Type": "text/event-stream" } });
}

function body(init?: RequestInit): Record<string, unknown> {
  try {
    return JSON.parse(String(init?.body ?? "{}"));
  } catch {
    return {};
  }
}

function deepMerge(base: Record<string, unknown>, over: Record<string, unknown> | null | undefined) {
  if (!over) return base;
  for (const [k, v] of Object.entries(over)) {
    if (v && typeof v === "object" && !Array.isArray(v) && typeof base[k] === "object" && base[k]) {
      deepMerge(base[k] as Record<string, unknown>, v as Record<string, unknown>);
    } else {
      base[k] = v;
    }
  }
  return base;
}

function cartesian(params: Record<string, unknown[]>): Record<string, unknown>[] {
  return Object.entries(params).reduce<Record<string, unknown>[]>(
    (acc, [k, vals]) => acc.flatMap((a) => vals.map((v) => ({ ...a, [k]: v }))),
    [{}],
  );
}

let runCounter = 0;

async function handle(path: string, method: string, init?: RequestInit): Promise<Response> {
  await sleep(150);
  const url = new URL(path, "http://demo.local");
  const p = decodeURIComponent(url.pathname);

  if (p === "/api/health") return json({ status: "ok (demo)" });
  if (p.startsWith("/api/presets/")) return json(DEFAULT_CONFIG);

  if (p === "/api/results") return json(RESULT_FILES);
  if (p.startsWith("/api/results/")) {
    const fn = p.slice("/api/results/".length);
    const r = RESULTS[fn];
    return r ? json(r) : json({ detail: "Not found" }, 404);
  }

  if (p === "/api/collections" && method === "GET")
    return json({ collections: COLLECTIONS, backends_available: ["qdrant", "faiss"] });
  if (p === "/api/collections" && method === "POST") {
    const b = body(init);
    return json({ status: "created", collection_name: b.collection_name ?? "demo_collection", message: "Collection simulée (mode démo)" });
  }
  if (p.startsWith("/api/collections/") && method === "DELETE") {
    const [, , , backend, name] = p.split("/");
    return json({ status: "deleted (demo — rien n'a été supprimé)", collection_name: name, backend });
  }
  if (p === "/api/ensure-collection") {
    return json({ status: "exists", collection_name: DEFAULT_CONFIG.dataset.collection_name, message: "ok (demo)" });
  }

  if (p === "/api/datasets/registry") return json({ datasets: REGISTRY });
  if (p === "/api/datasets" && method === "GET") {
    const col = url.searchParams.get("collection_name");
    const list = DATASETS.filter((d) => !col || d.collection_name === col).map(
      ({ questions: _q, metadata: _m, ...info }) => info,
    );
    return json({ datasets: list });
  }
  if (p === "/api/datasets/ensure-collections") {
    const cols = (body(init).collections as unknown[]) ?? [];
    const steps: SSEStep[] = [];
    cols.forEach((_, i) => {
      const name = `collection_${i + 1}`;
      steps.push({ event: "collection_exists", data: { name, index: i, total: cols.length } });
    });
    return sse(steps, init?.signal);
  }
  if (p === "/api/datasets/generate") {
    const b = body(init);
    const cats = (b.categories as { type: string; count: number }[]) ?? [];
    const steps: SSEStep[] = [];
    let total = 0;
    for (const c of cats) {
      for (let i = 1; i <= c.count; i++) {
        steps.push({ event: "progress", data: { category: c.type, generated: i, total: c.count }, delay: 60 });
      }
      steps.push({ event: "category_complete", data: { category: c.type, generated: c.count, total: c.count } });
      total += c.count;
    }
    steps.push({ event: "complete", data: { dataset_id: `ds_demo_${Date.now()}`, total_generated: total } });
    return sse(steps, init?.signal);
  }
  if (p.startsWith("/api/datasets/")) {
    const id = p.slice("/api/datasets/".length);
    if (method === "DELETE") return json({ status: "deleted (demo)", dataset_id: id });
    const d = DATASETS.find((x) => x.id === id);
    return d ? json(d) : json({ detail: "Not found" }, 404);
  }

  if (p === "/api/query/stream") {
    const b = body(init);
    const cfg = deepMerge(
      JSON.parse(JSON.stringify(DEFAULT_CONFIG)),
      b.config_overrides as Record<string, unknown>,
    ) as unknown as BenchmarkConfig;
    const chunks = makeChunks(String(b.query ?? ""), cfg.reranker.type !== "none" ? cfg.retrieval.rerank_k : cfg.retrieval.top_k);
    const steps: SSEStep[] = [
      { event: "retrieval_complete", data: { chunks, retrieval_time_ms: 87 }, delay: 500 },
    ];
    if (cfg.generation.enabled && cfg.generation.model) {
      for (const tok of DEMO_ANSWER.match(/\S+\s*/g) ?? []) {
        steps.push({ event: "generation_token", data: { token: tok }, delay: 35 });
      }
      steps.push({ event: "generation_complete", data: { full_answer: DEMO_ANSWER, generation_time_ms: 1840 } });
    }
    return sse(steps, init?.signal);
  }
  if (p === "/api/query/highlight-chunks") {
    const b = body(init);
    const chunks = (b.chunks as { chunk_id: string; content: string }[]) ?? [];
    return json({
      highlighted_chunks: chunks.map((c, i) => ({
        chunk_id: c.chunk_id,
        highlighted_content: c.content.replace("6 juin 1944", "<mark>6 juin 1944</mark>"),
        relevance: i < 2 ? "exact_answer" : i < 4 ? "related" : "not_relevant",
      })),
    });
  }

  if (p === "/api/benchmark/run") {
    const total = 20;
    const steps: SSEStep[] = [
      { event: "info", data: { message: "Mode démo : exécution simulée, aucun appel LLM." } },
      { event: "started", data: { total_questions: total, config_hash: "demo" + (++runCounter) } },
    ];
    const types = ["factual", "temporal", "causal", "comparative", "entity"];
    for (let i = 0; i < total; i++) {
      steps.push({
        event: "progress",
        data: { question_index: i + 1, total_questions: total, question_id: `q${i + 1}`, question_type: types[i % 5], retrieval_time_ms: 50 + (i * 13) % 70 },
        delay: 110,
      });
    }
    const first = RESULT_FILES[0]!;
    steps.push({
      event: "complete",
      data: { filename: first.filename, phase_name: first.phase_name, total_questions: total, avg_mrr: first.avg_mrr, avg_recall_at_5: first.avg_recall_at_5, avg_recall_at_10: first.avg_recall_at_10, total_wall_time_s: 4.2 },
    });
    return sse(steps, init?.signal);
  }

  if (p === "/api/benchmark/sweep") {
    const b = body(init);
    const combos = cartesian((b.sweep_params as Record<string, unknown[]>) ?? {});
    const nQ = 6;
    const sweepId = `sweep_demo_${++runCounter}`;
    const steps: SSEStep[] = [
      { event: "info", data: { message: "Mode démo : sweep simulé, aucun appel LLM." } },
      { event: "sweep_started", data: { sweep_id: sweepId, total_configs: combos.length, total_questions_per_config: nQ, sweep_name: String(b.name ?? sweepId) } },
    ];
    const files: string[] = [];
    combos.forEach((params, ci) => {
      steps.push({ event: "config_started", data: { config_index: ci, total_configs: combos.length, config_hash: `h${ci}`, params, collection_name: DEFAULT_CONFIG.dataset.collection_name } });
      for (let q = 0; q < nQ; q++) {
        steps.push({ event: "config_progress", data: { config_index: ci, total_configs: combos.length, question_index: q + 1, total_questions: nQ, question_id: `q${q + 1}`, question_type: "factual", retrieval_time_ms: 60 }, delay: 70 });
      }
      const ref = RESULT_FILES[(ci % 9) + 4] ?? RESULT_FILES[0]!;
      files.push(ref.filename);
      steps.push({ event: "config_complete", data: { config_index: ci, total_configs: combos.length, status: "ok", filename: ref.filename, params, avg_mrr: ref.avg_mrr, avg_recall_at_5: ref.avg_recall_at_5, avg_recall_at_10: ref.avg_recall_at_10, avg_doc_mrr: ref.avg_doc_mrr, avg_doc_precision_at_5: ref.avg_doc_precision_at_5, total_wall_time_s: 3.1 } });
    });
    steps.push({ event: "sweep_complete", data: { sweep_id: sweepId, total_configs: combos.length, completed_configs: combos.length, result_files: files, meta_filename: `${sweepId}/_sweep_meta.json` } });
    return sse(steps, init?.signal);
  }

  return json({ detail: `Endpoint non simulé en démo : ${method} ${p}` }, 404);
}

export function installMockFetch(): void {
  const realFetch = window.fetch.bind(window);
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const path = url.replace(/^https?:\/\/[^/]+/, "");
    if (path.startsWith("/api/")) {
      return handle(path, (init?.method ?? "GET").toUpperCase(), init);
    }
    return realFetch(input, init);
  };
}
