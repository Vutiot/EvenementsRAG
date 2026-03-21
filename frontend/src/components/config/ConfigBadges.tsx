import type { BenchmarkConfig } from "../../api/types";

interface Props {
  config: BenchmarkConfig | null;
}

export default function ConfigBadges({ config }: Props) {
  if (!config) return null;

  const chips: string[] = [];

  chips.push(config.dataset.dataset_name);
  chips.push(`cs${config.chunking.chunk_size}`);
  chips.push(`co${config.chunking.chunk_overlap}`);

  // Short embedding model name (last segment after /)
  const embParts = config.embedding.model_name.split("/");
  chips.push(embParts[embParts.length - 1] ?? config.embedding.model_name);

  chips.push(config.retrieval.technique);
  chips.push(`Top K ${config.retrieval.top_k}`);

  if (config.reranker.type !== "none") {
    const rerankerShort = config.reranker.model_name
      ? config.reranker.model_name.split("/").pop()!
      : config.reranker.type;
    chips.push(`${rerankerShort} rerank ${config.retrieval.rerank_k}`);
  }

  if (config.generation.model !== "__none__") {
    const modelParts = config.generation.model.split("/");
    chips.push(modelParts[modelParts.length - 1] ?? config.generation.model);
  } else {
    chips.push("No LLM");
  }

  return (
    <div className="flex flex-wrap gap-1.5 justify-center">
      {chips.map((chip, i) => (
        <span
          key={i}
          className="bg-gray-100 text-gray-600 rounded-full px-2.5 py-0.5 text-xs font-mono"
        >
          {chip}
        </span>
      ))}
    </div>
  );
}
