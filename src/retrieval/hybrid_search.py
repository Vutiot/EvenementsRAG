"""
Hybrid Search Implementation

Combines semantic (vector) search with BM25 or TF-IDF (keyword) search using
Reciprocal Rank Fusion (RRF) to merge results.
"""

from typing import List, Dict, Literal, Optional, Tuple
from dataclasses import dataclass
import math
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Unified search result from hybrid search."""
    chunk_id: str
    score: float
    payload: Dict
    rank: int


class BM25:
    """
    BM25 (Best Matching 25) implementation for keyword-based search.

    BM25 is a ranking function used for information retrieval that considers:
    - Term frequency (TF): How often terms appear in a document
    - Inverse document frequency (IDF): Rarity of terms across collection
    - Document length normalization: Prevents bias toward longer documents

    Parameters:
        k1: Controls term frequency saturation (default: 1.5)
        b: Controls document length normalization (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = defaultdict(int)  # How many docs contain each term
        self.idf = {}  # IDF scores for each term
        self.doc_lengths = []  # Length of each document
        self.avgdl = 0.0  # Average document length
        self.N = 0  # Total number of documents

    def fit(self, corpus: List[str]):
        """
        Build BM25 index from corpus.

        Args:
            corpus: List of document texts to index
        """
        self.corpus = [doc.lower().split() for doc in corpus]
        self.N = len(self.corpus)

        # Calculate document frequencies and lengths
        for doc in self.corpus:
            self.doc_lengths.append(len(doc))
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] += 1

        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # Calculate IDF for each term
        for term, freq in self.doc_freqs.items():
            # IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

        logger.info(f"BM25 index built: {self.N} documents, {len(self.idf)} unique terms")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search corpus using BM25 scoring.

        Args:
            query: Search query text
            top_k: Number of top results to return

        Returns:
            List of (doc_idx, score) tuples sorted by score descending
        """
        query_terms = query.lower().split()
        scores = [0.0] * self.N

        for doc_idx, doc in enumerate(self.corpus):
            doc_len = self.doc_lengths[doc_idx]

            for term in query_terms:
                if term not in self.idf:
                    continue

                # Calculate term frequency in document
                tf = doc.count(term)

                # BM25 score formula
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += idf * (numerator / denominator)

        # Get top-k results
        scored_docs = [(i, score) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]


class HybridSearcher:
    """
    Hybrid search combining semantic (vector) and keyword (BM25) search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both approaches:
    - Semantic search: Good for conceptual/semantic similarity
    - BM25 search: Good for exact keyword matches
    - RRF: Combines rankings without needing to normalize scores
    """

    def __init__(
        self,
        qdrant_manager,
        embedding_generator,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
        sparse_type: Literal["bm25", "tfidf"] = "bm25",
    ):
        """
        Initialize hybrid searcher.

        Args:
            qdrant_manager: QdrantManager instance for vector search
            embedding_generator: EmbeddingGenerator for query embeddings
            bm25_weight: Weight for sparse vs semantic (0.0 = all semantic, 1.0 = all sparse)
            rrf_k: RRF constant (default 60, standard value from literature)
            sparse_type: Sparse index type, either "bm25" or "tfidf"
        """
        self.qdrant = qdrant_manager
        self.embedding_gen = embedding_generator
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.sparse_type = sparse_type

        if sparse_type == "tfidf":
            from src.retrieval.tfidf_search import TFIDFIndex
            self._sparse_index = TFIDFIndex()
        else:
            self._sparse_index = BM25()

        # Keep self.bm25 as alias for backward compatibility
        self.bm25 = self._sparse_index
        self.chunk_ids = []  # Map sparse doc index to chunk UUID

        logger.info(
            f"HybridSearcher initialized: "
            f"bm25_weight={bm25_weight}, rrf_k={rrf_k}, sparse_type={sparse_type}"
        )

    def index_collection(self, collection_name: str):
        """
        Build BM25 index for a Qdrant collection.

        Args:
            collection_name: Name of collection to index
        """
        logger.info(f"Building BM25 index for collection '{collection_name}'")

        # Scroll through entire collection
        chunks = []
        chunk_ids = []
        offset = None

        while True:
            results, offset = self.qdrant.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not results:
                break

            for point in results:
                chunk_text = point.payload.get("chunk_text", "")
                chunks.append(chunk_text)
                chunk_ids.append(str(point.id))

            if offset is None:
                break

        # Build sparse index (BM25 or TF-IDF)
        self._sparse_index.fit(chunks)
        self.chunk_ids = chunk_ids

        logger.info(
            f"{self.sparse_type.upper()} index ready: {len(chunks)} chunks indexed"
        )

    def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_conditions: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25.

        Args:
            query: Search query text
            collection_name: Collection to search
            top_k: Number of results to return
            filter_conditions: Optional Qdrant filter (applied to vector search only)

        Returns:
            List of SearchResult objects sorted by hybrid score
        """
        # 1. Semantic (vector) search
        query_embedding = self.embedding_gen.generate_embeddings([query])[0]

        vector_results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k * 2,  # Get more to ensure overlap after fusion
            filter_conditions=filter_conditions,
        )

        # 2. Sparse (keyword) search — BM25 or TF-IDF
        bm25_results = self._sparse_index.search(query, top_k=top_k * 2)

        # 3. Apply RRF to merge results
        merged = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            top_k=top_k,
        )

        return merged

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Tuple[int, float]],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = Σ 1 / (k + rank)
        where k is a constant (default 60) and rank starts at 1.

        Args:
            vector_results: Results from semantic search
            bm25_results: Results from BM25 search
            top_k: Number of results to return

        Returns:
            Merged and sorted results
        """
        rrf_scores = defaultdict(float)
        chunk_data = {}  # Store payload for each chunk

        # Add vector search scores (weighted)
        semantic_weight = 1.0 - self.bm25_weight
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["id"]
            rrf_scores[chunk_id] += semantic_weight * (1.0 / (self.rrf_k + rank))
            chunk_data[chunk_id] = result["payload"]

        # Add BM25 scores (weighted)
        for rank, (doc_idx, bm25_score) in enumerate(bm25_results, start=1):
            chunk_id = self.chunk_ids[doc_idx]
            rrf_scores[chunk_id] += self.bm25_weight * (1.0 / (self.rrf_k + rank))

            # Store payload if not already from vector search
            if chunk_id not in chunk_data:
                # Need to fetch payload from Qdrant
                try:
                    point = self.qdrant.client.retrieve(
                        collection_name=self.qdrant.collection_name,
                        ids=[chunk_id],
                        with_payload=True,
                    )[0]
                    chunk_data[chunk_id] = point.payload
                except:
                    chunk_data[chunk_id] = {}

        # Sort by RRF score and create results
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for rank, (chunk_id, score) in enumerate(sorted_chunks, start=1):
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    payload=chunk_data.get(chunk_id, {}),
                    rank=rank,
                )
            )

        return results
