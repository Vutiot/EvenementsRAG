/** TypeScript types matching the FastAPI response schemas. */

export interface PresetInfo {
  filename: string;
  name: string;
  description: string;
}

export interface RetrievedChunk {
  chunk_id: string;
  content: string;
  score: number;
  article_title: string;
  source_url: string;
  chunk_index: number;
}

export interface QueryResult {
  query: string;
  generated_answer: string;
  retrieved_chunks: RetrievedChunk[];
  retrieval_time_ms: number;
  generation_time_ms: number;
  config_hash: string;
}

/** Benchmark result file info from GET /api/results */
export interface ResultFileInfo {
  filename: string;
  phase_name: string;
  timestamp: string | null;
  total_questions: number;
  format: "legacy" | "benchmark_result";
  avg_mrr: number;
  avg_recall_at_5: number | null;
}

/** Single normalized question from a benchmark result */
export interface NormalizedQuestion {
  question_id: string;
  question: string;
  type: string;
  difficulty: string | null;
  source_article: string | null;
  ground_truth_count: number | null;
  retrieved_count: number | null;
  retrieval_time_ms: number | null;
  metrics: Record<string, number>;
  generated_answer: string | null;
  generation_time_ms: number | null;
  retrieved_contexts: string[] | null;
  generation_metrics: Record<string, number> | null;
  ragas_metrics: Record<string, number> | null;
}

/** Full normalized benchmark result from GET /api/results/{filename} */
export interface NormalizedBenchmarkResult {
  filename: string;
  format: "legacy" | "benchmark_result";
  phase_name: string;
  timestamp: string | null;
  config: Record<string, unknown> | null;
  avg_recall_at_k: Record<string, number>;
  avg_mrr: number;
  avg_ndcg: Record<string, number>;
  avg_article_hit_at_k: Record<string, number> | null;
  avg_chunk_hit_at_k: Record<string, number> | null;
  metrics_by_type: Record<string, Record<string, number>>;
  per_question: NormalizedQuestion[];
  total_questions: number;
  avg_retrieval_time_ms: number;
  total_wall_time_s: number | null;
  metrics_summary: Record<string, unknown> | null;
}

/** Collection management types */
export interface CollectionInfo {
  name: string;
  backend: string;
  vector_size: number | null;
  distance: string | null;
  points_count: number | null;
}

export interface CollectionListResponse {
  collections: CollectionInfo[];
  backends_available: string[];
}

export interface EnsureCollectionRequest {
  dataset_name: string;
  backend: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  embedding_dimension: number;
  distance_metric: string;
}

export interface EnsureCollectionResponse {
  status: "exists" | "created";
  collection_name: string;
  message: string;
}

export interface CollectionCreateRequest {
  dataset_name: string;
  collection_name?: string;
  backend: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  embedding_dimension: number;
  distance_metric: string;
}

export interface CollectionCreateResponse {
  status: string;
  collection_name: string;
  message: string;
}

/** Dataset registry (raw source datasets) */
export interface DatasetRegistryEntry {
  name: string;
  description: string;
  default_collection: string;
  collections: string[];
}

/** Dataset management types */
export interface DatasetCategoryConfig {
  type: string;
  prompt: string;
  model: string;
  count: number;
  generated: number;
}

export interface DatasetCreateRequest {
  name: string;
  collection_name: string;
  categories: { type: string; prompt: string; model: string; count: number }[];
}

export interface DatasetInfo {
  id: string;
  name: string;
  created_at: string;
  status: string;
  collection_name: string;
  total_questions: number;
  categories: DatasetCategoryConfig[];
}

export interface DatasetDetail extends DatasetInfo {
  questions: DatasetQuestion[];
  metadata?: {
    total_generated: number;
    unique_articles: number;
    generation_time_s: number;
  };
}

export interface DatasetQuestion {
  id: string;
  question: string;
  type: string;
  difficulty: string;
  expected_answer_hint: string;
  source_article: string;
  source_article_id: string;
  source_chunk_id: string;
  model: string;
  generated_at: string;
}

export interface DatasetProgressEvent {
  category: string;
  generated: number;
  total: number;
  question_id?: string;
}

/** Full config shape returned by GET /api/presets/{filename} */
export interface BenchmarkConfig {
  name: string;
  description: string;
  dataset: {
    dataset_name: string;
    collection_name: string;
    questions_file: string;
    articles_dir: string | null;
  };
  embedding: {
    model_name: string;
    dimension: number;
  };
  chunking: {
    chunk_size: number;
    chunk_overlap: number;
  };
  retrieval: {
    technique: string;
    top_k: number;
    rerank_k: number;
    sparse_weight: number;
    dense_weight: number;
    fusion_method: string;
    sparse_type: string;
  };
  reranker: {
    type: string;
    model_name: string | null;
  };
  generation: {
    llm_provider: string;
    model: string;
    temperature: number;
    max_tokens: number;
    top_k_chunks: number;
    top_k_articles: number | null;
    prompt_template: string | null;
    enabled: boolean;
  };
  evaluation: {
    k_values: number[];
    compute_ragas: boolean;
    compute_bert_score: boolean;
    compute_rouge: boolean;
    ragas_metrics: string[];
    ragas_evaluator_model: string;
    ragas_max_workers: number;
    ragas_timeout: number;
  };
  vector_db: {
    backend: string;
    distance_metric: string;
    connection_params: Record<string, unknown> | null;
  };
}
