"""
Reranker abstraction for the hybrid RAG pipeline.

Provides a common BaseReranker interface and four concrete implementations:
  - NoOpReranker   : passthrough (no reranking)
  - CohereReranker : Cohere rerank API v3
  - BGEReranker    : BAAI/bge-reranker-* via sentence-transformers CrossEncoder
  - CrossEncoderReranker : generic cross-encoder via sentence-transformers

Heavy dependencies (cohere, sentence_transformers) are lazy-imported so that
the module can be imported without them installed.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.rag.base_rag import RetrievedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Rerank *chunks* for *query* and return the top *top_k*.

        Args:
            query: The user query.
            chunks: Candidate chunks to rerank.
            top_k: Number of chunks to return.

        Returns:
            Reranked list of at most *top_k* RetrievedChunk objects.
        """


class NoOpReranker(BaseReranker):
    """Passthrough reranker — returns the first *top_k* chunks unchanged."""

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        return chunks[:top_k]


class CohereReranker(BaseReranker):
    """Reranker that calls the Cohere rerank API."""

    def __init__(self, model_name: str = "rerank-english-v3.0") -> None:
        self.model_name = model_name
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            import cohere  # lazy import
            import os
            self._client = cohere.Client(os.environ.get("COHERE_API_KEY", ""))
        return self._client

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        client = self._get_client()
        docs = [c.content for c in chunks]
        response = client.rerank(
            model=self.model_name,
            query=query,
            documents=docs,
            top_n=min(top_k, len(chunks)),
        )
        reranked = []
        for result in response.results:
            chunk = chunks[result.index]
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=result.relevance_score,
                    metadata=chunk.metadata,
                )
            )
        logger.debug(f"CohereReranker: reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked


class BGEReranker(BaseReranker):
    """Reranker using a BGE cross-encoder model via sentence-transformers."""

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3"
    ) -> None:
        self.model_name = model_name
        self._model = None  # lazy init

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        model = self._get_model()
        pairs = [(query, c.content) for c in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        reranked = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                content=c.content,
                score=float(s),
                metadata=c.metadata,
            )
            for s, c in ranked[:top_k]
        ]
        logger.debug(f"BGEReranker: reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked


class CrossEncoderReranker(BaseReranker):
    """Generic cross-encoder reranker via sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # lazy init

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy import
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        model = self._get_model()
        pairs = [(query, c.content) for c in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        reranked = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                content=c.content,
                score=float(s),
                metadata=c.metadata,
            )
            for s, c in ranked[:top_k]
        ]
        logger.debug(
            f"CrossEncoderReranker({self.model_name}): "
            f"reranked {len(chunks)} → {len(reranked)} chunks"
        )
        return reranked


class FlashRankReranker(BaseReranker):
    """Lightweight CPU reranker using FlashRank (ONNX-optimized)."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        self.model_name = model_name
        self._ranker = None  # lazy init

    def _get_ranker(self):
        if self._ranker is None:
            from flashrank import Ranker  # lazy import
            self._ranker = Ranker(model_name=self.model_name)
        return self._ranker

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        from flashrank import RerankRequest  # lazy import

        ranker = self._get_ranker()
        passages = [
            {"id": i, "text": c.content}
            for i, c in enumerate(chunks)
        ]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        reranked = []
        for result in results[:top_k]:
            idx = result["id"]
            chunk = chunks[idx]
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=float(result["score"]),
                    metadata=chunk.metadata,
                )
            )
        logger.debug(f"FlashRankReranker: reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked
