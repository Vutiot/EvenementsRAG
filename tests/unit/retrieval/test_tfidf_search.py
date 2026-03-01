"""Unit tests for TFIDFIndex."""

import math

import pytest

from src.retrieval.tfidf_search import TFIDFIndex


class TestTFIDFIndex:
    """8 tests covering fit, search, correctness, and edge cases."""

    def _make_index(self, corpus):
        idx = TFIDFIndex()
        idx.fit(corpus)
        return idx

    # 1. fit stores correct number of documents
    def test_fit_sets_N(self):
        corpus = ["hello world", "foo bar", "baz qux"]
        idx = self._make_index(corpus)
        assert idx._N == 3

    # 2. search returns at most top_k results
    def test_search_top_k(self):
        corpus = [f"document about topic {i}" for i in range(20)]
        idx = self._make_index(corpus)
        results = idx.search("topic", top_k=5)
        assert len(results) <= 5

    # 3. search result format is (int, float) tuples
    def test_search_returns_tuples(self):
        idx = self._make_index(["alpha beta", "gamma delta"])
        results = idx.search("alpha", top_k=2)
        assert len(results) > 0
        for doc_idx, score in results:
            assert isinstance(doc_idx, int)
            assert isinstance(score, float)

    # 4. term with high frequency in one document ranks that document first
    def test_term_ranking(self):
        corpus = [
            "paris is a great city in france france france france",
            "london is a great city in england",
            "berlin is a city in germany",
        ]
        idx = self._make_index(corpus)
        results = idx.search("france", top_k=3)
        top_doc_idx = results[0][0]
        assert top_doc_idx == 0  # 'france' appears 4 times in doc 0

    # 5. empty corpus returns empty results
    def test_empty_corpus(self):
        idx = TFIDFIndex()
        idx.fit([])
        assert idx.search("query", top_k=5) == []

    # 6. smoothed IDF: rare terms have higher IDF than frequent terms
    def test_smoothed_idf_rare_higher(self):
        corpus = [
            "cat cat cat cat cat",
            "cat dog",
            "dog fish",
        ]
        idx = self._make_index(corpus)
        idf_cat = idx._idf.get("cat", 0)
        idf_fish = idx._idf.get("fish", 0)
        # 'fish' appears in only 1/3 docs → higher IDF than 'cat' (2/3 docs)
        assert idf_fish > idf_cat

    # 7. frequent term scores lower than rare term (all else equal)
    def test_frequent_less_than_rare(self):
        corpus = [
            "apple apple apple",
            "apple banana",
            "banana cherry",
        ]
        idx = self._make_index(corpus)
        # 'apple' is in 2 docs, 'cherry' is in 1 doc
        results_apple = idx.search("apple apple apple", top_k=3)
        results_cherry = idx.search("cherry", top_k=3)
        # The top score for cherry-query should come from a less common term
        # Verify IDF ordering
        assert idx._idf["cherry"] > idx._idf["apple"]

    # 8. L2-normalised scores are in [0, 1]
    def test_l2_norm_scores_bounded(self):
        corpus = ["war in europe", "peace in europe", "war and peace globally"]
        idx = self._make_index(corpus)
        results = idx.search("war", top_k=3)
        for _, score in results:
            assert 0.0 <= score <= 1.0 + 1e-9
