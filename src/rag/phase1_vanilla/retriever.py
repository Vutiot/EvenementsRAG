"""
Phase 1: Vanilla RAG Retriever

Simple semantic search using vector similarity:
1. Embed the query
2. Search Qdrant for top-K most similar chunks
3. Generate answer using LLM with retrieved context

No filters, no reranking, no hybrid search - just pure semantic similarity.

Usage:
    from src.rag.phase1_vanilla.retriever import VanillaRetriever

    retriever = VanillaRetriever(collection_name="ww2_events")
    response = retriever.query("What was D-Day?", top_k=5)
    print(response.answer)
"""

from typing import Dict, List, Optional

import openai

from config.settings import settings
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.rag.base_rag import BaseRAG, RAGResponse, RetrievedChunk
from src.utils.logger import get_logger
from src.vector_store.qdrant_manager import QdrantManager

logger = get_logger(__name__)


# Default prompt template for answer generation
DEFAULT_PROMPT_TEMPLATE = """You are a knowledgeable historian assistant. Answer the question based ONLY on the provided context from historical documents. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer: Provide a clear, concise answer based on the context above. Include specific dates, names, and events when relevant."""


class VanillaRetriever(BaseRAG):
    """Phase 1: Simple semantic search RAG."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        qdrant_manager: Optional[QdrantManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        llm_client: Optional[openai.OpenAI] = None,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize vanilla RAG retriever.

        Args:
            collection_name: Qdrant collection name
            qdrant_manager: Qdrant manager instance
            embedding_generator: Embedding generator instance
            llm_client: OpenAI-compatible LLM client
            prompt_template: Custom prompt template
        """
        super().__init__(name="Phase1_Vanilla_RAG")

        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.qdrant = qdrant_manager or QdrantManager()
        self.embedding_gen = embedding_generator or EmbeddingGenerator()
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            # Use OpenRouter by default
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

        # Verify collection exists
        if not self.qdrant.collection_exists(self.collection_name):
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist. "
                "Run indexing first: python -m src.vector_store.indexer"
            )

        collection_info = self.qdrant.get_collection_info(self.collection_name)
        logger.info(
            f"{self.name} initialized with collection '{self.collection_name}' "
            f"({collection_info['points_count']} chunks)"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks using semantic search.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters (not used in vanilla RAG)

        Returns:
            List of retrieved chunks with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_gen.generate_embedding(query)

        # Search Qdrant
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=None,  # No filtering
            filter_conditions=filters,
        )

        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for result in results:
            chunk = RetrievedChunk(
                chunk_id=result["payload"].get("chunk_id", result["id"]),
                content=result["payload"].get("content", ""),
                score=result["score"],
                metadata=result["payload"],
            )
            retrieved_chunks.append(chunk)

        logger.debug(
            f"Retrieved {len(retrieved_chunks)} chunks for query: '{query[:50]}...'"
        )

        return retrieved_chunks

    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """
        Generate answer using LLM with retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional LLM parameters

        Returns:
            Generated answer
        """
        # Format context
        context = self.format_context(context_chunks)

        # Build prompt
        prompt = self.prompt_template.format(
            context=context,
            query=query,
        )

        llm_model = kwargs.pop("model", None) or settings.CURRENT_LLM_MODEL

        logger.debug(f"Generating answer with {len(context_chunks)} context chunks")

        try:
            # Call LLM
            response = self.llm_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable historian assistant."},
                    {"role": "user", "content": prompt},
                ],
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
            # Fallback: return context summary
            return (
                f"[Generation failed: {str(e)}]\n\n"
                f"Retrieved {len(context_chunks)} relevant passages:\n\n"
                f"{context[:500]}..."
            )

    def get_statistics(self) -> Dict:
        """Get statistics about the retriever."""
        stats = super().get_statistics()
        stats.update({
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_gen.model_name,
            "embedding_dimension": self.embedding_gen.embedding_dim,
            "llm_model": settings.CURRENT_LLM_MODEL,
        })

        collection_info = self.qdrant.get_collection_info(self.collection_name)
        stats.update(collection_info)

        return stats


if __name__ == "__main__":
    # Test vanilla retriever
    import sys

    print("=" * 70)
    print("Phase 1: Vanilla RAG Retriever Test")
    print("=" * 70)

    # Check if collection exists
    try:
        qdrant = QdrantManager()
        collection_name = settings.QDRANT_COLLECTION_NAME

        if not qdrant.collection_exists(collection_name):
            print(f"\n❌ Collection '{collection_name}' not found")
            print("\nIndex documents first:")
            print("  python -m src.vector_store.indexer")
            sys.exit(1)

        # Initialize retriever
        print("\nInitializing vanilla retriever...")
        retriever = VanillaRetriever()

        # Show statistics
        stats = retriever.get_statistics()
        print("\nRetriever Statistics:")
        print(f"  Collection: {stats['collection_name']}")
        print(f"  Chunks indexed: {stats['points_count']}")
        print(f"  Embedding model: {stats['embedding_model']}")
        print(f"  LLM model: {stats['llm_model']}")

        # Test retrieval only
        print("\n" + "=" * 70)
        print("Test Query (Retrieval Only)")
        print("=" * 70)

        test_query = "What was the significance of the D-Day landings?"
        print(f"\nQuery: {test_query}")

        chunks = retriever.retrieve(test_query, top_k=3)
        print(f"\nRetrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  [{i}] Score: {chunk.score:.3f}")
            print(f"      Article: {chunk.article_title}")
            print(f"      Content: {chunk.content[:150]}...")

        # Test full RAG if API key is set
        if settings.OPENROUTER_API_KEY:
            print("\n" + "=" * 70)
            print("Test Query (Full RAG with Generation)")
            print("=" * 70)

            response = retriever.query(test_query, top_k=3)

            print(f"\nQuery: {response.query}")
            print(f"\nAnswer:")
            print("-" * 70)
            print(response.answer)
            print("-" * 70)
            print(f"\nSources: {', '.join(response.sources)}")
            print(f"Retrieval time: {response.retrieval_time_ms:.1f}ms")
            print(f"Generation time: {response.generation_time_ms:.1f}ms")
            print(f"Total time: {response.total_time_ms:.1f}ms")

        else:
            print("\n⚠ OpenRouter API key not set - skipping generation test")
            print("Set OPENROUTER_API_KEY in .env to test full RAG pipeline")

        print("\n" + "=" * 70)
        print("✓ Vanilla RAG retriever working!")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)
