"""
Abstract base class for RAG (Retrieval-Augmented Generation) systems.

Defines the interface that all RAG phases must implement:
- Phase 1: Vanilla RAG (simple semantic search)
- Phase 2: Temporal RAG (date-aware filtering)
- Phase 3: Hybrid RAG (semantic + keyword + reranking)
- Phase 4: Graph RAG (knowledge graph traversal)

Usage:
    from src.rag.base_rag import BaseRAG

    class MyRAG(BaseRAG):
        def retrieve(self, query, top_k=5):
            # Implementation
            ...

        def generate(self, query, context):
            # Implementation
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with metadata."""

    chunk_id: str
    content: str
    score: float
    metadata: Dict = field(default_factory=dict)

    # Metadata fields commonly used
    @property
    def article_title(self) -> str:
        return self.metadata.get("article_title", "")

    @property
    def source_url(self) -> str:
        return self.metadata.get("source_url", "")

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(id={self.chunk_id}, score={self.score:.3f}, "
            f"article='{self.article_title}', preview='{self.content[:50]}...')"
        )


@dataclass
class RAGResponse:
    """Response from RAG system including answer and provenance."""

    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    metadata: Dict = field(default_factory=dict)

    # Timing information
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0

    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.generation_time_ms

    @property
    def sources(self) -> List[str]:
        """Extract unique source articles."""
        return list(set(chunk.article_title for chunk in self.retrieved_chunks))

    def __repr__(self) -> str:
        return (
            f"RAGResponse(query='{self.query[:50]}...', "
            f"answer_length={len(self.answer)}, "
            f"chunks={len(self.retrieved_chunks)}, "
            f"time={self.total_time_ms:.1f}ms)"
        )


class BaseRAG(ABC):
    """Abstract base class for RAG systems."""

    def __init__(self, name: str = "BaseRAG"):
        """
        Initialize RAG system.

        Args:
            name: Name of the RAG implementation
        """
        self.name = name
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters

        Returns:
            List of retrieved chunks with scores
        """
        pass

    @abstractmethod
    def generate(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        **kwargs,
    ) -> str:
        """
        Generate answer from query and retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            **kwargs: Additional generation parameters

        Returns:
            Generated answer
        """
        pass

    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        return_chunks_only: bool = False,
        **generation_kwargs,
    ) -> RAGResponse:
        """
        Complete RAG query: retrieve + generate.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filters: Optional metadata filters
            return_chunks_only: If True, skip generation step
            **generation_kwargs: Parameters for generation

        Returns:
            RAGResponse with answer and provenance
        """
        logger.info(f"Processing query: '{query[:100]}...'")

        import time

        # Retrieve
        start_time = time.time()
        retrieved_chunks = self.retrieve(query, top_k=top_k, filters=filters)
        retrieval_time_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time_ms:.1f}ms"
        )

        # Generate (unless skipped)
        answer = ""
        generation_time_ms = 0.0

        if not return_chunks_only:
            start_time = time.time()
            answer = self.generate(query, retrieved_chunks, **generation_kwargs)
            generation_time_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Generated answer ({len(answer)} chars) in {generation_time_ms:.1f}ms"
            )

        # Create response
        response = RAGResponse(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            metadata={
                "rag_phase": self.name,
                "top_k": top_k,
                "filters": filters,
            },
        )

        logger.info(
            f"Query complete: {len(retrieved_chunks)} chunks, "
            f"{response.total_time_ms:.1f}ms total"
        )

        return response

    def format_context(self, chunks: List[RetrievedChunk], max_length: Optional[int] = None) -> str:
        """
        Format retrieved chunks into context string for LLM.

        Args:
            chunks: Retrieved chunks
            max_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            part = f"[Source {i}: {chunk.article_title}]\n{chunk.content}\n"
            context_parts.append(part)

            # Check length limit
            if max_length:
                current_length = sum(len(p) for p in context_parts)
                if current_length > max_length:
                    context_parts.pop()  # Remove last chunk
                    logger.debug(
                        f"Context truncated to {len(context_parts)} chunks "
                        f"({current_length} chars)"
                    )
                    break

        return "\n".join(context_parts)

    def get_statistics(self) -> Dict:
        """
        Get statistics about the RAG system.

        Returns:
            Dictionary with statistics
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


if __name__ == "__main__":
    # Test base class with a simple implementation
    print("=" * 70)
    print("Testing BaseRAG Interface")
    print("=" * 70)

    class DummyRAG(BaseRAG):
        """Dummy implementation for testing."""

        def __init__(self):
            super().__init__(name="DummyRAG")

        def retrieve(self, query, top_k=5, filters=None):
            """Return dummy chunks."""
            return [
                RetrievedChunk(
                    chunk_id=f"chunk_{i}",
                    content=f"This is test content for chunk {i} about {query}.",
                    score=1.0 / (i + 1),
                    metadata={
                        "article_title": f"Test Article {i}",
                        "chunk_index": i,
                        "source_url": f"https://example.com/article_{i}",
                    },
                )
                for i in range(top_k)
            ]

        def generate(self, query, context_chunks, **kwargs):
            """Return dummy answer."""
            return f"This is a test answer to: {query}. Based on {len(context_chunks)} sources."

    # Create dummy RAG
    rag = DummyRAG()
    print(f"\nCreated: {rag}")

    # Test retrieval only
    print("\n--- Testing Retrieval ---")
    chunks = rag.retrieve("What is D-Day?", top_k=3)
    print(f"Retrieved {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - {chunk}")

    # Test full query
    print("\n--- Testing Full Query ---")
    response = rag.query("What is D-Day?", top_k=3)
    print(f"\nResponse:")
    print(f"  Query: {response.query}")
    print(f"  Answer: {response.answer}")
    print(f"  Sources: {response.sources}")
    print(f"  Retrieval time: {response.retrieval_time_ms:.1f}ms")
    print(f"  Generation time: {response.generation_time_ms:.1f}ms")
    print(f"  Total time: {response.total_time_ms:.1f}ms")

    # Test context formatting
    print("\n--- Testing Context Formatting ---")
    context = rag.format_context(chunks, max_length=200)
    print(f"Formatted context ({len(context)} chars):")
    print(context[:200] + "..." if len(context) > 200 else context)

    print("\n" + "=" * 70)
    print("✓ BaseRAG interface working correctly!")
    print("=" * 70)
