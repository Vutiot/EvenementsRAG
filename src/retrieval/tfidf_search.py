"""
TF-IDF sparse search implementation.

Provides TFIDFIndex with the same interface as BM25, so HybridSearcher can
dispatch to either based on sparse_type configuration.

Smoothed IDF formula: log((1 + N) / (1 + df)) + 1
(avoids zero IDF for universal terms, same as sklearn's default)
"""

import math
from collections import defaultdict
from typing import List, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TFIDFIndex:
    """
    TF-IDF index for keyword-based search.

    Uses raw term frequency (TF) multiplied by smoothed IDF:
        IDF(t) = log((1 + N) / (1 + df(t))) + 1

    Scores for a query are the sum of TF-IDF weights for each query term,
    then L2-normalised so scores are comparable across documents of varying
    length.

    Interface is identical to BM25 so HybridSearcher can swap them freely.
    """

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._corpus: list[list[str]] = []   # tokenised documents
        self._avg_len: float = 0.0
        self._N: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, corpus: List[str]) -> None:
        """Build TF-IDF index from *corpus*.

        Args:
            corpus: List of raw document strings.
        """
        self._corpus = [doc.lower().split() for doc in corpus]
        self._N = len(self._corpus)

        if self._N == 0:
            logger.info("TFIDFIndex: empty corpus, nothing to index")
            return

        # Document frequency per term
        doc_freq: dict[str, int] = defaultdict(int)
        total_len = 0
        for tokens in self._corpus:
            total_len += len(tokens)
            for term in set(tokens):
                doc_freq[term] += 1

        self._avg_len = total_len / self._N

        # Smoothed IDF: log((1 + N) / (1 + df)) + 1
        self._idf = {
            term: math.log((1 + self._N) / (1 + df)) + 1
            for term, df in doc_freq.items()
        }

        logger.info(
            f"TFIDFIndex built: {self._N} documents, {len(self._idf)} unique terms"
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Score all documents against *query* using TF-IDF, return top *top_k*.

        Scores are L2-normalised (divided by the L2-norm of the score vector),
        so the maximum possible value is 1.0.

        Args:
            query: Raw query string.
            top_k: Maximum number of results to return.

        Returns:
            List of ``(doc_idx, score)`` tuples sorted by score descending.
        """
        if self._N == 0:
            return []

        query_terms = query.lower().split()
        scores = [0.0] * self._N

        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_idx, tokens in enumerate(self._corpus):
                tf = tokens.count(term)
                if tf > 0:
                    scores[doc_idx] += tf * idf

        # L2-normalise
        norm = math.sqrt(sum(s * s for s in scores))
        if norm > 0:
            scores = [s / norm for s in scores]

        # Sort and truncate
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
