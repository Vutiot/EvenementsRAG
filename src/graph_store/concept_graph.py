"""Lightweight concept graph for LazyGraphRAG (E5-F1-T1).

Built once per collection without any LLM call (the "lazy" part: no
summaries or community reports at index time):

* concepts of a chunk = named-entity-like capitalised phrases found in its text
  (spaCy NER instead when ``method="spacy"`` and a model is installed), plus
  the article title;
* an inverted index concept → chunks, and a co-occurrence graph
  concept ↔ concept weighted by the number of chunks they share;
* concepts present in too many chunks are dropped (they link everything).

The graph and the chunk payloads are cached with pickle in
``data/graph/<collection>__<method>.pkl``; the cache is rebuilt when the
number of points of the collection changes.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

GRAPH_DIR = Path(__file__).resolve().parents[2] / "data" / "graph"
GRAPH_VERSION = 1
MAX_CONCEPTS_PER_CHUNK = 25
MAX_NGRAM = 5

ConceptMethod = Literal["heuristic", "spacy"]

_PROPER_RE = re.compile(
    r"\b([A-Z][\w'’\-]+(?:\s+(?:of|de|the|von|van|du|la|and|for)?\s*[A-Z][\w'’\-]+)*)\b"
)
_TOKEN_RE = re.compile(r"[\w'’\-]+")
_GENERIC = {
    "the", "a", "an", "in", "on", "at", "of", "and", "or", "but", "he", "she", "it",
    "they", "we", "his", "her", "its", "their", "this", "that", "these", "those",
    "after", "before", "during", "when", "while", "however", "although", "january",
    "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "there", "then", "following", "according",
}


# Honorifics and ranks dropped from the front of a phrase so that
# "Admiral Chester Nimitz" and "Chester Nimitz" are the same concept.
_TITLES = {
    "admiral", "general", "president", "king", "queen", "prince", "princess", "sir",
    "dr", "mr", "mrs", "captain", "colonel", "marshal", "field", "lieutenant",
    "emperor", "empress", "pope", "major", "commander", "chancellor", "prime", "minister",
    "premier", "lord", "lady", "saint", "st", "sergeant", "vice", "rear", "grand",
}


def normalize_concept(text: str) -> str:
    words = re.sub(r"\s+", " ", text.strip().lower()).strip(" .,'’-").split(" ")
    while len(words) > 1 and (words[0] in _GENERIC or words[0] in _TITLES):
        words = words[1:]
    while len(words) > 1 and words[-1] in _GENERIC:
        words = words[:-1]
    return " ".join(words).strip(" .,'’-")


def heuristic_concepts(text: str) -> List[str]:
    """Capitalised phrases, skipping a lone capitalised word that starts a sentence."""
    found: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for m in _PROPER_RE.finditer(sentence):
            raw = m.group(1)
            if m.start() == 0 and " " not in raw:
                continue
            c = normalize_concept(raw)
            if len(c) < 3 or c in _GENERIC or c.isdigit():
                continue
            found.append(c)
    return found


def _spacy_concepts(texts: Sequence[str]) -> Optional[List[List[str]]]:
    from src.assistant.dataset_profiler import _SPACY_LABELS, _load_spacy

    nlp = _load_spacy()
    if nlp is None:
        return None
    out = []
    for doc in nlp.pipe(texts, batch_size=64):
        out.append(
            [
                c
                for ent in doc.ents
                if ent.label_ in _SPACY_LABELS
                and len(c := normalize_concept(ent.text)) >= 3
                and c not in _GENERIC
            ]
        )
    return out


@dataclass
class ConceptGraph:
    n_points: int = 0
    method: str = "heuristic"
    version: int = GRAPH_VERSION
    chunk_concepts: Dict[str, List[str]] = field(default_factory=dict)
    concept_chunks: Dict[str, List[str]] = field(default_factory=dict)
    cooccurrence: Dict[str, Dict[str, int]] = field(default_factory=dict)
    payloads: Dict[str, dict] = field(default_factory=dict)

    # -- building ----------------------------------------------------------

    @classmethod
    def build(
        cls,
        records: Iterable[dict],
        method: ConceptMethod = "heuristic",
        max_df_ratio: float = 0.05,
        min_max_df: int = 50,
    ) -> "ConceptGraph":
        """Build from scrolled records ``{"id", "payload"}``."""
        records = list(records)
        ids = [str(r["id"]) for r in records]
        payloads = {cid: (r.get("payload") or {}) for cid, r in zip(ids, records)}
        texts = [p.get("content", "") for p in payloads.values()]

        per_chunk = _spacy_concepts(texts) if method == "spacy" else None
        used = "spacy" if per_chunk is not None else "heuristic"
        if per_chunk is None:
            per_chunk = [heuristic_concepts(t) for t in texts]

        raw: Dict[str, List[str]] = {}
        for cid, concepts in zip(ids, per_chunk):
            title = normalize_concept(payloads[cid].get("article_title", "") or "")
            ordered = list(dict.fromkeys(([title] if title else []) + concepts))
            raw[cid] = ordered

        df = Counter(c for cs in raw.values() for c in set(cs))
        max_df = max(min_max_df, int(max_df_ratio * max(len(ids), 1)))
        keep = {c for c, n in df.items() if n <= max_df}

        g = cls(n_points=len(ids), method=used, payloads=payloads)
        concept_chunks: Dict[str, List[str]] = defaultdict(list)
        cooc: Dict[str, Counter] = defaultdict(Counter)
        for cid, concepts in raw.items():
            cs = [c for c in concepts if c in keep][:MAX_CONCEPTS_PER_CHUNK]
            g.chunk_concepts[cid] = cs
            for c in cs:
                concept_chunks[c].append(cid)
            for a, b in combinations(cs, 2):
                cooc[a][b] += 1
                cooc[b][a] += 1
        g.concept_chunks = dict(concept_chunks)
        g.cooccurrence = {c: dict(n) for c, n in cooc.items()}
        logger.info(
            f"Concept graph ({used}): {len(ids)} chunks, {len(g.concept_chunks)} concepts, "
            f"{sum(len(n) for n in g.cooccurrence.values()) // 2} edges, "
            f"{len(df) - len(keep)} over-frequent concepts dropped (df > {max_df})"
        )
        return g

    # -- querying ----------------------------------------------------------

    def idf(self, concept: str) -> float:
        n = len(self.concept_chunks.get(concept, ()))
        return math.log(1 + self.n_points / n) if n else 0.0

    def match_query(self, query: str) -> List[str]:
        """Known concepts appearing in the query (longest n-grams first, no overlap)."""
        tokens = [normalize_concept(t) for t in _TOKEN_RE.findall(query.lower())]
        tokens = [t for t in tokens if t]
        matched: List[str] = []
        used = [False] * len(tokens)
        for n in range(min(MAX_NGRAM, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                if any(used[i : i + n]):
                    continue
                phrase = " ".join(tokens[i : i + n])
                if phrase in self.concept_chunks and phrase not in _GENERIC:
                    matched.append(phrase)
                    for j in range(i, i + n):
                        used[j] = True
        return matched

    def neighbors(self, concept: str, limit: int) -> List[Tuple[str, float]]:
        """Strongest co-occurring concepts, weight normalised to (0, 1]."""
        nb = self.cooccurrence.get(concept) or {}
        if not nb:
            return []
        top = sorted(nb.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        best = top[0][1]
        return [(c, w / best) for c, w in top]

    def score_chunks(
        self,
        weighted_concepts: Dict[str, float],
        hops: int = 1,
        max_neighbors: int = 10,
        hop_decay: float = 0.5,
    ) -> Dict[str, float]:
        """Graph relevance of chunks for a set of weighted seed concepts."""
        weights = dict(weighted_concepts)
        frontier = dict(weighted_concepts)
        for _ in range(max(0, hops)):
            nxt: Dict[str, float] = {}
            for c, w in frontier.items():
                for nb, strength in self.neighbors(c, max_neighbors):
                    v = w * hop_decay * strength
                    if v > weights.get(nb, 0.0) and v > nxt.get(nb, 0.0):
                        nxt[nb] = v
            for c, v in nxt.items():
                weights[c] = max(weights.get(c, 0.0), v)
            frontier = nxt
        scores: Dict[str, float] = defaultdict(float)
        for c, w in weights.items():
            idf = self.idf(c)
            for cid in self.concept_chunks.get(c, ()):
                scores[cid] += w * idf
        return dict(scores)

    # -- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Optional["ConceptGraph"]:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)  # local cache written by save()
        except (OSError, pickle.UnpicklingError, EOFError):
            return None
        g = cls(**data)
        return g if g.version == GRAPH_VERSION else None


def graph_path(collection: str, method: str, graph_dir: Optional[Path] = None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", collection)
    return (graph_dir or GRAPH_DIR) / f"{safe}__{method}.pkl"


def scroll_all(store, collection: str, batch: int = 256) -> List[dict]:
    """Every point (payload only) of a collection, through BaseVectorStore.scroll."""
    out: List[dict] = []
    offset = None
    while True:
        page, offset = store.scroll(
            collection, limit=batch, offset=offset, with_payload=True, with_vectors=False
        )
        out.extend(page)
        if offset is None:
            break
    return out


def load_or_build(
    store,
    collection: str,
    method: ConceptMethod = "heuristic",
    graph_dir: Optional[Path] = None,
) -> ConceptGraph:
    """Cached concept graph of *collection*, rebuilt when its size changed."""
    path = graph_path(collection, method, graph_dir)
    try:
        n_points = int(store.count_vectors(collection))
    except Exception:
        n_points = None
    cached = ConceptGraph.load(path) if path.exists() else None
    if cached is not None and (n_points is None or cached.n_points == n_points):
        return cached
    graph = ConceptGraph.build(scroll_all(store, collection), method=method)
    graph.save(path)
    return graph
