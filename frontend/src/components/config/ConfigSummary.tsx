import { useState } from "react";
import type { BenchmarkConfig } from "../../api/types";

interface SectionProps {
  title: string;
  badge?: string;
  children: React.ReactNode;
}

function Section({ title, badge, children }: SectionProps) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border-b border-gray-100 last:border-0">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-3 py-2 text-sm font-medium
                   text-gray-700 hover:bg-gray-50 transition-colors"
      >
        <span className="flex items-center gap-2">
          <span className={`transform transition-transform ${open ? "rotate-90" : ""}`}>
            &#9654;
          </span>
          {title}
        </span>
        {badge && (
          <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
            {badge}
          </span>
        )}
      </button>
      {open && <div className="px-3 pb-3 text-xs text-gray-600 space-y-1">{children}</div>}
    </div>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="font-mono text-gray-800">{String(value ?? "—")}</span>
    </div>
  );
}

interface Props {
  config: BenchmarkConfig | null;
}

export default function ConfigSummary({ config }: Props) {
  if (!config) {
    return (
      <div className="rounded border border-gray-200 bg-white p-4 text-sm text-gray-400">
        Select a preset to view its configuration.
      </div>
    );
  }

  return (
    <div className="rounded border border-gray-200 bg-white divide-y divide-gray-100 text-sm">
      <Section title="Dataset" badge={config.dataset.dataset_name}>
        <Field label="Collection" value={config.dataset.collection_name} />
        <Field label="Questions file" value={config.dataset.questions_file} />
        {config.dataset.articles_dir && (
          <Field label="Articles dir" value={config.dataset.articles_dir} />
        )}
      </Section>

      <Section title="Embedding" badge={config.embedding.model_name.split("/").pop()}>
        <Field label="Model" value={config.embedding.model_name} />
        <Field label="Dimension" value={config.embedding.dimension} />
      </Section>

      <Section title="Chunking">
        <Field label="Chunk size" value={config.chunking.chunk_size} />
        <Field label="Chunk overlap" value={config.chunking.chunk_overlap} />
      </Section>

      <Section title="Retrieval" badge={config.retrieval.technique}>
        <Field label="Technique" value={config.retrieval.technique} />
        <Field label="Top K" value={config.retrieval.top_k} />
        {config.retrieval.technique === "hybrid" && (
          <>
            <Field label="Sparse weight" value={config.retrieval.sparse_weight} />
            <Field label="Dense weight" value={config.retrieval.dense_weight} />
            <Field label="Sparse type" value={config.retrieval.sparse_type ?? "bm25"} />
            <Field label="Fusion" value={config.retrieval.fusion_method} />
          </>
        )}
      </Section>

      <Section
        title="Reranker"
        badge={config.reranker.type !== "none" ? config.reranker.type : undefined}
      >
        <Field label="Type" value={config.reranker.type} />
        {config.reranker.type !== "none" && (
          <>
            <Field label="Model" value={config.reranker.model_name} />
            <Field label="Top K Rerank" value={config.generation.top_k_chunks} />
          </>
        )}
      </Section>

      <Section title="Generation" badge={config.generation.model === "__none__" ? "disabled" : config.generation.model.split("/").pop()}>
        <Field label="Provider" value={config.generation.llm_provider} />
        <Field label="Model" value={config.generation.model === "__none__" ? "None" : config.generation.model} />
        <Field label="Temperature" value={config.generation.temperature} />
        <Field label="Max tokens" value={config.generation.max_tokens} />
        <Field label="Top K chunks" value={config.generation.top_k_chunks} />
        <Field label="Enabled" value={config.generation.enabled ? "Yes" : "No"} />
      </Section>

      <Section title="Evaluation">
        <Field label="K values" value={config.evaluation.k_values.join(", ")} />
        <Field label="ROUGE" value={config.evaluation.compute_rouge ? "Yes" : "No"} />
        <Field label="BERTScore" value={config.evaluation.compute_bert_score ? "Yes" : "No"} />
        <Field label="RAGAS" value={config.evaluation.compute_ragas ? "Yes" : "No"} />
      </Section>

      <Section title="Vector DB" badge={config.vector_db.backend}>
        <Field label="Backend" value={config.vector_db.backend} />
        <Field label="Distance" value={config.vector_db.distance_metric} />
      </Section>
    </div>
  );
}
