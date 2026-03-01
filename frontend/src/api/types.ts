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
