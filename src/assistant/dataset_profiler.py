"""Deterministic dataset profiling harness (E8-F1-T1).

Explores a collection without any LLM call and produces a compact, LLM-ready
``DatasetProfile``: size, chunk length distribution, language, period covered,
top entities, top categories and thematic clusters (k-means on the stored
embeddings, with representative chunks and top terms per cluster).

The profile grounds the eval-dataset assistant (E8) and the AutoML assistant
(E9). It is cached on disk per collection (``data/profiles/<collection>.json``)
because loading every chunk and its vector takes a while on large collections.

    profile = get_profile("wiki_10k_qdrant_cs512_co50_minilm_l6_cosine")
    prompt_block = profile.to_prompt_json()   # < 4k tokens
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

PROFILE_VERSION = 1
PROFILES_DIR = Path(__file__).resolve().parents[2] / "data" / "profiles"

# Caps that keep profiling fast and deterministic on large collections
MAX_CLUSTER_POINTS = 20_000
MAX_NER_CHUNKS = 1_500
SNIPPET_CHARS = 220

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]{2,}")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
# Capitalised sequences, e.g. "Winston Churchill", "Battle of Midway"
_PROPER_RE = re.compile(
    r"\b([A-Z][a-zà-ÿ]+(?:\s+(?:of|de|the|von|van|du|la)?\s*[A-Z][a-zà-ÿ]+)*)\b"
)

_STOPWORDS: Dict[str, set] = {
    "en": set(
        "the and of to in a is was for on that with as by at from it an be this which "
        "were are or his her their its had has have not but also after into they he she "
        "who during first one two all been other more than when would there these over "
        "under between about such some most only them then its while where any may".split()
    ),
    "fr": set(
        "le la les de des du et en un une est que qui dans pour par sur au aux avec ce "
        "ces son sa ses il elle ils sont été pas plus ont fut lors".split()
    ),
    "de": set(
        "der die das und in den von zu mit ist im dem nicht ein eine als auch es an "
        "auf für sich des wurde war".split()
    ),
    "es": set(
        "el la los las de del y en un una que es por con para al se su sus como fue "
        "más pero".split()
    ),
}
_ALL_STOPWORDS = set().union(*_STOPWORDS.values())


# ---------------------------------------------------------------------------
# Profile model
# ---------------------------------------------------------------------------


class Distribution(BaseModel):
    min: float = 0
    p25: float = 0
    median: float = 0
    p75: float = 0
    max: float = 0
    mean: float = 0


class DateRange(BaseModel):
    """Years mentioned in the text. p5–p95 is the robust period covered."""

    min_year: int
    max_year: int
    p5_year: int
    p95_year: int
    top_decades: List[str] = Field(default_factory=list)
    chunks_with_year: float = Field(0, description="Share of chunks mentioning a year")


class Entity(BaseModel):
    text: str
    count: int
    label: Optional[str] = None  # spaCy label (PERSON, GPE, ...) when available


class RepresentativeChunk(BaseModel):
    chunk_id: str
    article_title: str
    snippet: str


class Cluster(BaseModel):
    cluster_id: int
    size: int
    share: float
    top_terms: List[str]
    top_titles: List[str]
    representatives: List[RepresentativeChunk]


class DatasetProfile(BaseModel):
    profile_version: int = PROFILE_VERSION
    collection_name: str
    created_at: str
    n_documents: int
    n_chunks: int
    chunk_tokens: Distribution
    chunks_per_document: Distribution
    language: str
    date_range: Optional[DateRange] = None
    top_entities: List[Entity] = Field(default_factory=list)
    entity_method: str = "none"  # "spacy:<model>" | "heuristic" | "none"
    top_categories: List[Entity] = Field(default_factory=list)
    sample_titles: List[str] = Field(default_factory=list)
    clusters: List[Cluster] = Field(default_factory=list)

    def to_prompt_json(self) -> str:
        """Compact JSON for an LLM prompt."""
        return json.dumps(
            self.model_dump(exclude={"profile_version", "created_at"}),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def approx_tokens(self) -> int:
        """Rough token count of :meth:`to_prompt_json` (~4 characters per token)."""
        return len(self.to_prompt_json()) // 4


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _distribution(values: Sequence[float]) -> Distribution:
    if not len(values):
        return Distribution()
    a = np.asarray(values, dtype=float)
    p25, med, p75 = np.percentile(a, [25, 50, 75])
    return Distribution(
        min=float(a.min()),
        p25=round(float(p25), 1),
        median=round(float(med), 1),
        p75=round(float(p75), 1),
        max=float(a.max()),
        mean=round(float(a.mean()), 1),
    )


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def detect_language(texts: Iterable[str], max_words: int = 50_000) -> str:
    """Language with the highest stopword hit rate (en/fr/de/es), else "unknown"."""
    counts: Counter = Counter()
    n = 0
    for text in texts:
        for w in _tokens(text):
            for lang, words in _STOPWORDS.items():
                if w in words:
                    counts[lang] += 1
            n += 1
            if n >= max_words:
                break
        if n >= max_words:
            break
    if not counts or counts.most_common(1)[0][1] < 0.02 * max(n, 1):
        return "unknown"
    return counts.most_common(1)[0][0]


def extract_date_range(texts: Sequence[str]) -> Optional[DateRange]:
    years: List[int] = []
    with_year = 0
    for text in texts:
        found = [int(y) for y in _YEAR_RE.findall(text)]
        if found:
            with_year += 1
            years.extend(found)
    if not years:
        return None
    a = np.asarray(years)
    decades = Counter(f"{(y // 10) * 10}s" for y in years)
    return DateRange(
        min_year=int(a.min()),
        max_year=int(a.max()),
        p5_year=int(np.percentile(a, 5)),
        p95_year=int(np.percentile(a, 95)),
        top_decades=[d for d, _ in decades.most_common(5)],
        chunks_with_year=round(with_year / len(texts), 3),
    )


def _heuristic_entities(texts: Sequence[str], top_n: int) -> List[Entity]:
    counts: Counter = Counter()
    for text in texts:
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            for m in _PROPER_RE.finditer(sentence):
                ent = m.group(1).strip()
                # Skip a lone capitalised word at the start of a sentence
                if m.start() == 0 and " " not in ent:
                    continue
                if ent.lower() in _ALL_STOPWORDS or len(ent) < 3:
                    continue
                counts[ent] += 1
    return [Entity(text=t, count=c) for t, c in counts.most_common(top_n)]


def _load_spacy():
    try:
        import spacy
    except ImportError:
        return None
    for name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
        try:
            return spacy.load(name, disable=["parser", "lemmatizer", "tagger"])
        except (OSError, ValueError):
            continue
    return None


_SPACY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "FAC", "WORK_OF_ART", "LAW"}


def extract_entities(
    texts: Sequence[str], top_n: int = 30, use_spacy: bool = True
) -> tuple[List[Entity], str]:
    """Top named entities over (a deterministic sample of) the chunks."""
    if not texts:
        return [], "none"
    sample = list(texts)
    if len(sample) > MAX_NER_CHUNKS:
        idx = np.linspace(0, len(sample) - 1, MAX_NER_CHUNKS).astype(int)
        sample = [sample[i] for i in idx]
    nlp = _load_spacy() if use_spacy else None
    if nlp is None:
        return _heuristic_entities(sample, top_n), "heuristic"
    counts: Counter = Counter()
    labels: Dict[str, Counter] = defaultdict(Counter)
    for doc in nlp.pipe(sample, batch_size=64):
        for ent in doc.ents:
            if ent.label_ in _SPACY_LABELS:
                key = re.sub(r"^the\s+", "", ent.text.strip(), flags=re.IGNORECASE)
                counts[key] += 1
                labels[key][ent.label_] += 1
    ents = [
        Entity(text=t, count=c, label=labels[t].most_common(1)[0][0])
        for t, c in counts.most_common(top_n)
    ]
    return ents, f"spacy:{nlp.meta.get('name', 'model')}"


def kmeans(x: np.ndarray, k: int, seed: int = 0, n_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means++ / Lloyd on L2-normalised rows. Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = max(1, min(k, n))
    centroids = np.empty((k, x.shape[1]), dtype=x.dtype)
    centroids[0] = x[rng.integers(n)]
    d2 = ((x - centroids[0]) ** 2).sum(1)
    for i in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1 / n)
        centroids[i] = x[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, ((x - centroids[i]) ** 2).sum(1))
    labels = np.zeros(n, dtype=int)
    for it in range(n_iter):
        sims = x @ centroids.T
        new = sims.argmax(1)
        if it > 0 and np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = x[labels == c]
            if len(members):
                v = members.mean(0)
                norm = np.linalg.norm(v)
                centroids[c] = v / norm if norm > 0 else v
    return labels, centroids


def _top_terms(cluster_texts: Sequence[str], doc_freq: Counter, n_docs: int, n: int = 8) -> List[str]:
    """Terms frequent in the cluster and rare overall (tf × idf)."""
    tf: Counter = Counter()
    for t in cluster_texts:
        tf.update(w for w in _tokens(t) if w not in _ALL_STOPWORDS)
    scored = {
        w: c * np.log((1 + n_docs) / (1 + doc_freq[w]))
        for w, c in tf.items()
        if doc_freq[w] >= 2
    }
    return [w for w, _ in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1].rstrip() + "…"


def _token_count(payload: Dict[str, Any]) -> int:
    tc = payload.get("token_count")
    if tc:
        return int(tc)
    return max(1, len(payload.get("content", "")) // 4)


def build_profile(
    collection_name: str,
    records: Sequence[Dict[str, Any]],
    n_clusters: int = 8,
    seed: int = 0,
    use_spacy: bool = True,
) -> DatasetProfile:
    """Build a profile from scrolled records (``{"id", "payload", "vector"?}``)."""
    # Stable order so the profile does not depend on scroll order
    records = sorted(records, key=lambda r: str(r.get("id")))
    payloads = [r.get("payload") or {} for r in records]
    texts = [p.get("content", "") for p in payloads]
    titles = [p.get("article_title") or p.get("title") or "" for p in payloads]

    per_doc = Counter(t for t in titles if t)
    categories: Counter = Counter()
    for p in payloads:
        categories.update(c for c in (p.get("categories") or []) if isinstance(c, str))

    entities, entity_method = extract_entities(texts, use_spacy=use_spacy)

    profile = DatasetProfile(
        collection_name=collection_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_documents=len(per_doc),
        n_chunks=len(records),
        chunk_tokens=_distribution([_token_count(p) for p in payloads]),
        chunks_per_document=_distribution(list(per_doc.values())),
        language=detect_language(texts),
        date_range=extract_date_range(texts) if texts else None,
        top_entities=entities,
        entity_method=entity_method,
        top_categories=[Entity(text=c, count=n) for c, n in categories.most_common(15)],
        sample_titles=[t for t, _ in per_doc.most_common(15)],
    )

    # Thematic clusters on the stored embeddings
    with_vec = [i for i, r in enumerate(records) if r.get("vector") is not None]
    if len(with_vec) >= 2:
        if len(with_vec) > MAX_CLUSTER_POINTS:
            rng = np.random.default_rng(seed)
            with_vec = sorted(rng.choice(with_vec, MAX_CLUSTER_POINTS, replace=False).tolist())
        x = np.asarray([records[i]["vector"] for i in with_vec], dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.where(norms > 0, norms, 1)
        labels, centroids = kmeans(x, n_clusters, seed=seed)

        doc_freq: Counter = Counter()
        for t in texts:
            doc_freq.update(set(_tokens(t)))

        clusters = []
        for c in range(centroids.shape[0]):
            member_pos = np.flatnonzero(labels == c)
            if not len(member_pos):
                continue
            sims = x[member_pos] @ centroids[c]
            order = member_pos[np.argsort(-sims, kind="stable")]
            members = [with_vec[i] for i in member_pos]
            reps = [with_vec[i] for i in order[:3]]
            clusters.append(
                Cluster(
                    cluster_id=c,
                    size=len(members),
                    share=round(len(members) / len(with_vec), 3),
                    top_terms=_top_terms([texts[i] for i in members], doc_freq, len(texts)),
                    top_titles=[
                        t for t, _ in Counter(titles[i] for i in members if titles[i]).most_common(3)
                    ],
                    representatives=[
                        RepresentativeChunk(
                            chunk_id=str(payloads[i].get("chunk_id") or records[i].get("id")),
                            article_title=titles[i],
                            snippet=_snippet(texts[i]),
                        )
                        for i in reps
                    ],
                )
            )
        clusters.sort(key=lambda cl: -cl.size)
        for new_id, cl in enumerate(clusters):
            cl.cluster_id = new_id
        profile.clusters = clusters

    return profile


# ---------------------------------------------------------------------------
# Loading and caching
# ---------------------------------------------------------------------------


def _backend_from_name(collection_name: str) -> str:
    for backend in ("qdrant", "faiss", "pgvector"):
        if f"_{backend}_" in collection_name:
            return backend
    return "qdrant"


def load_records(collection_name: str, store=None, batch: int = 256) -> List[Dict[str, Any]]:
    """Scroll every point (payload + vector) of a collection."""
    if store is None:
        from src.api.collection_service import CollectionService

        store = CollectionService()._store_for(_backend_from_name(collection_name))
        if store is None:
            raise ValueError(f"No vector store available for '{collection_name}'")
    if not store.collection_exists(collection_name):
        raise KeyError(collection_name)
    records: List[Dict[str, Any]] = []
    offset = None
    while True:
        page, offset = store.scroll(
            collection_name, limit=batch, offset=offset, with_payload=True, with_vectors=True
        )
        records.extend(page)
        if offset is None:
            break
    return records


def get_profile(
    collection_name: str,
    refresh: bool = False,
    store=None,
    profiles_dir: Optional[Path] = None,
    **kwargs: Any,
) -> DatasetProfile:
    """Cached profile of a collection; ``refresh=True`` recomputes it."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", collection_name):
        raise ValueError(f"Invalid collection name: {collection_name!r}")
    cache_dir = Path(profiles_dir) if profiles_dir else PROFILES_DIR
    path = cache_dir / f"{collection_name}.json"
    if not refresh and path.exists():
        try:
            cached = DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))
            if cached.profile_version == PROFILE_VERSION:
                return cached
        except ValueError:
            logger.warning(f"Unreadable profile cache {path}, recomputing")

    records = load_records(collection_name, store=store)
    logger.info(f"Profiling {collection_name}: {len(records)} chunks")
    profile = build_profile(collection_name, records, **kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=1), encoding="utf-8")
    return profile
