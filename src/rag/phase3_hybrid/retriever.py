"""
Phase 3: Hybrid RAG Retriever

Combines dense vector search with sparse keyword search (BM25 or TF-IDF)
via Reciprocal Rank Fusion (RRF), then optionally reranks results.

Wired into ParameterizedBenchmarkRunner via _RAG_REGISTRY["hybrid"].

Usage:
    from src.rag.phase3_hybrid.retriever import HybridRetriever
    from src.benchmarks.config import BenchmarkConfig

    cfg = BenchmarkConfig.phase2_hybrid()
    retriever = HybridRetriever(
        collection_name="ww2_events_10000",
        qdrant_manager=vector_store,
        embedding_generator=emb_gen,
        config=cfg,
    )
    chunks = retriever.retrieve("What happened at D-Day?", top_k=5)
"""

from typing import Dict, List, Optional

import openai

from config.settings import settings
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.rag.base_rag import BaseRAG, RetrievedChunk
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker_factory import RerankerFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPT_TEMPLATE = """You are a knowledgeable historian assistant. Answer the question based ONLY on the provided context from historical documents. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer: Provide a clear, concise answer based on the context above. Include specific dates, names, and events when relevant."""


class HybridRetriever(BaseRAG):
    """Phase 3: Hybrid search with optional reranking."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        qdrant_manager=None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        llm_client: Optional[openai.OpenAI] = None,
        prompt_template: Optional[str] = None,
        config=None,
    ):
        """Initialize hybrid retriever.

        Args:
            collection_name: Qdrant collection name.
            qdrant_manager: A QdrantAdapter (BaseVectorStore) or QdrantManager.
            embedding_generator: EmbeddingGenerator instance.
            llm_client: OpenAI-compatible LLM client.
            prompt_template: Custom prompt template.
            config: BenchmarkConfig with retrieval/reranker sub-models.
        """
        super().__init__(name="Phase3_Hybrid_RAG")

        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self._config = config

        # Unwrap QdrantAdapter → raw QdrantManager for HybridSearcher
        # (HybridSearcher.index_collection uses qdrant.client.scroll)
        raw_manager = qdrant_manager
        if hasattr(qdrant_manager, "manager"):
            # QdrantAdapter exposes .manager property
            raw_manager = qdrant_manager.manager
        self._vector_store = qdrant_manager  # kept for generate (qdrant.search)

        # Retrieval params from config (with sensible defaults)
        retrieval_cfg = config.retrieval if config is not None else None
        sparse_weight = retrieval_cfg.sparse_weight if retrieval_cfg else 0.3
        sparse_type = retrieval_cfg.sparse_type if retrieval_cfg else "bm25"
        self._rerank_k = retrieval_cfg.rerank_k if retrieval_cfg else 20

        self._hybrid_searcher = HybridSearcher(
            qdrant_manager=raw_manager,
            embedding_generator=self.embedding_gen,
            bm25_weight=sparse_weight,
            sparse_type=sparse_type,
        )

        # Build reranker
        reranker_cfg = config.reranker if config is not None else None
        if reranker_cfg is not None:
            self._reranker = RerankerFactory.from_config(reranker_cfg)
        else:
            from src.retrieval.reranker import NoOpReranker
            self._reranker = NoOpReranker()

        self._indexed = False  # lazy BM25/TF-IDF corpus indexing

        # Verify collection exists (same guard as VanillaRetriever)
        if not self._vector_store.collection_exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist. "
                "Run indexing first."
            )

        # LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            if not settings.OPENROUTER_API_KEY:
                logger.warning(
                    "OpenRouter API key not set. Generation will fail. "
                    "Set OPENROUTER_API_KEY in .env file."
                )
            self.llm_client = openai.OpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "EvenementsRAG",
                },
            )

        logger.info(
            f"{self.name} initialized with collection '{self.collection_name}'"
        )

    # ------------------------------------------------------------------
    # BaseRAG implementation
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve *rerank_k* candidates via hybrid search, rerank, return *top_k*."""
        # Lazy index on first call
        if not self._indexed:
            self._hybrid_searcher.index_collection(self.collection_name)
            self._indexed = True

        candidates = self._hybrid_searcher.search(
            query=query,
            collection_name=self.collection_name,
            top_k=self._rerank_k,
            filter_conditions=filters,
        )

        # Convert SearchResult → RetrievedChunk
        chunk_objects = [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                content=r.payload.get("content", ""),
                score=r.score,
                metadata=r.payload,
            )
            for r in candidates
        ]

        # Rerank and truncate to top_k
        reranked = self._reranker.rerank(query, chunk_objects, top_k=top_k)

        logger.debug(
            f"HybridRetriever: {len(candidates)} candidates → "
            f"{len(reranked)} after reranking"
        )
        return reranked

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """Generate answer using LLM with retrieved context.

        Identical to VanillaRetriever.generate to maintain a consistent
        generation interface across RAG phases.
        """
        context = self.format_context(context_chunks)
        prompt = self.prompt_template.format(context=context, query=query)

        llm_model = kwargs.pop("model", None) or settings.CURRENT_LLM_MODEL

        logger.debug(f"Generating answer with {len(context_chunks)} context chunks")

        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable historian assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        # DEBUG: print exact payload sent to OpenRouter
        print("=" * 60)
        print(f"[DEBUG LLM CALL] model={llm_model}")
        print(f"[DEBUG LLM CALL] temperature={temperature}, max_tokens={max_tokens}")
        for i, msg in enumerate(messages):
            print(f"[DEBUG LLM CALL] messages[{i}] role={msg['role']} content={msg['content'][:200]}...")
        print("=" * 60)

        try:
            response = self.llm_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            raw = response.choices[0].message.content
            answer = raw.strip() if raw else "[Model returned empty response]"
            logger.debug(f"Generated answer: {len(answer)} chars")
            return answer

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (
                f"[Generation failed: {str(e)}]\n\n"
                f"Retrieved {len(context_chunks)} relevant passages:\n\n"
                f"{context[:500]}..."
            )
