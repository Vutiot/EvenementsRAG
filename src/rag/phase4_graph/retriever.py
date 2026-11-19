"""
Phase 4: LazyGraphRAG retriever (E5-F1-T1).

Graph-augmented retrieval without a knowledge-graph database and without any
LLM call at index time. A concept graph (named-entity-like phrases + article
titles, linked by co-occurrence in chunks) is built lazily on first use and
cached on disk (``src/graph_store/concept_graph.py``).

Per query:
1. dense search for ``rerank_k`` candidates (same as vanilla);
2. seed concepts = concepts named in the query (weight 1) + concepts of the
   top 3 dense hits (weight 0.5);
3. expand the seeds ``graph_hops`` times through the strongest co-occurring
   concepts (weight decays by half per hop);
4. graph score of a chunk = sum over its concepts of weight × idf;
5. weighted RRF of the dense ranking and the graph ranking
   (``graph_weight``), then the configured reranker keeps ``top_k``.

Graph-only candidates (not in the dense list) bring chunks that share the
query's entities and their neighbours, which is where it should beat dense
and hybrid retrieval on entity- and relation-heavy questions.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from src.graph_store.concept_graph import ConceptGraph, load_or_build
from src.rag.base_rag import RetrievedChunk
from src.rag.phase1_vanilla.retriever import VanillaRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

RRF_K = 60
SEED_CHUNKS = 3
SEED_CHUNK_WEIGHT = 0.5


class LazyGraphRetriever(VanillaRetriever):
    """Dense retrieval fused with a lazily built concept graph."""

    def __init__(self, *args, config=None, graph: Optional[ConceptGraph] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Phase4_LazyGraph_RAG"
        r = config.retrieval if config is not None else None
        self._rerank_k = r.rerank_k if r else 20
        self._graph_weight = r.graph_weight if r else 0.5
        self._hops = r.graph_hops if r else 1
        self._max_neighbors = r.graph_max_neighbors if r else 10
        self._method = r.graph_concepts if r else "heuristic"
        self._graph = graph

        reranker_cfg = config.reranker if config is not None else None
        if reranker_cfg is not None:
            from src.retrieval.reranker_factory import RerankerFactory

            self._reranker = RerankerFactory.from_config(reranker_cfg)
        else:
            from src.retrieval.reranker import NoOpReranker

            self._reranker = NoOpReranker()

    @property
    def graph(self) -> ConceptGraph:
        if self._graph is None:
            self._graph = load_or_build(self.qdrant, self.collection_name, self._method)
        return self._graph

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        n_cand = max(self._rerank_k, top_k)  # never fewer candidates than asked for
        dense = super().retrieve(query, top_k=n_cand, filters=filters)
        graph = self.graph

        seeds: Dict[str, float] = {c: 1.0 for c in graph.match_query(query)}
        for chunk in dense[:SEED_CHUNKS]:
            for c in graph.chunk_concepts.get(self._point_id(chunk), ()):
                seeds.setdefault(c, SEED_CHUNK_WEIGHT)

        graph_scores = (
            graph.score_chunks(seeds, hops=self._hops, max_neighbors=self._max_neighbors)
            if seeds
            else {}
        )
        if filters:
            # Graph-only hits cannot honour store-side filters: keep dense ones only
            allowed = {self._point_id(c) for c in dense}
            graph_scores = {k: v for k, v in graph_scores.items() if k in allowed}
        graph_ranked = sorted(graph_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n_cand]

        fused: Dict[str, float] = defaultdict(float)
        chunks: Dict[str, RetrievedChunk] = {}
        for rank, chunk in enumerate(dense, start=1):
            pid = self._point_id(chunk)
            fused[pid] += (1 - self._graph_weight) / (RRF_K + rank)
            chunks[pid] = chunk
        for rank, (pid, _score) in enumerate(graph_ranked, start=1):
            fused[pid] += self._graph_weight / (RRF_K + rank)
            if pid not in chunks:
                payload = graph.payloads.get(pid, {})
                chunks[pid] = RetrievedChunk(
                    chunk_id=payload.get("chunk_id", pid),
                    content=payload.get("content", ""),
                    score=0.0,
                    metadata=payload,
                    point_id=pid,
                )

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:n_cand]
        candidates = []
        for pid, score in ordered:
            c = chunks[pid]
            candidates.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    score=score,
                    metadata=c.metadata,
                    point_id=pid,
                )
            )

        logger.debug(
            f"LazyGraph: {len(seeds)} seed concepts, {len(dense)} dense + "
            f"{len(graph_ranked)} graph candidates -> {len(candidates)} fused"
        )
        return self._reranker.rerank(query, candidates, top_k=top_k)

    @staticmethod
    def _point_id(chunk: RetrievedChunk) -> str:
        """Vector-store point id of a chunk (the graph's keys)."""
        return str(chunk.point_id or chunk.chunk_id)
