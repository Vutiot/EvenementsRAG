"""
FAISSStore — BaseVectorStore implementation backed by FAISS.

FAISS has no native payload support, so metadata is stored in a sidecar dict.
Each collection is a ``_FAISSCollection`` holding the FAISS index, id↔position
mappings, and payloads.  Persistence writes ``.faiss`` + ``.meta.pkl`` files.

Usage:
    from src.vector_store.faiss_store import FAISSStore

    store = FAISSStore(persist_dir="/tmp/faiss_data")
    store.create_collection("test", vector_size=384)
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _FAISSCollection:
    """In-memory state for a single FAISS-backed collection."""

    name: str
    vector_size: int
    distance: DistanceMetric
    index: Any  # faiss.Index — typed as Any to keep import lazy
    # Bidirectional mappings between string IDs and internal int positions
    id_to_pos: Dict[str, int] = field(default_factory=dict)
    pos_to_id: Dict[int, str] = field(default_factory=dict)
    payloads: Dict[str, Dict] = field(default_factory=dict)
    _next_pos: int = 0


class FAISSStore(BaseVectorStore):
    """FAISS-based vector store with sidecar metadata."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        default_distance: DistanceMetric = DistanceMetric.COSINE,
    ):
        """
        Args:
            persist_dir: Directory for ``.faiss`` / ``.meta.pkl`` files.
                         ``None`` means purely in-memory (no persistence).
            default_distance: Metric used when ``create_collection`` is called
                              without an explicit ``distance``.
        """
        super().__init__(default_distance=default_distance)
        self._collections: Dict[str, _FAISSCollection] = {}
        self._persist_dir: Optional[Path] = Path(persist_dir) if persist_dir else None
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index(vector_size: int, distance: DistanceMetric):
        """Create the appropriate flat FAISS index."""
        import faiss

        if distance in (DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT):
            return faiss.IndexFlatIP(vector_size)
        elif distance == DistanceMetric.EUCLIDEAN:
            return faiss.IndexFlatL2(vector_size)
        else:
            raise ValueError(f"FAISS does not support distance metric '{distance.value}'")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise rows in-place (for cosine similarity via inner product)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def _get(self, name: str) -> _FAISSCollection:
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")
        return self._collections[name]

    def _match_filter(self, payload: Dict, filter_conditions: Dict) -> bool:
        """Return True if *payload* satisfies all *filter_conditions*."""
        for fld, value in filter_conditions.items():
            pval = payload.get(fld)
            if isinstance(value, dict):
                # Range filter
                if pval is None:
                    return False
                if "gte" in value and pval < value["gte"]:
                    return False
                if "lte" in value and pval > value["lte"]:
                    return False
                if "gt" in value and pval <= value["gt"]:
                    return False
                if "lt" in value and pval >= value["lt"]:
                    return False
            elif isinstance(value, list):
                if pval not in value:
                    return False
            else:
                if pval != value:
                    return False
        return True

    def _persist_collection(self, coll: _FAISSCollection) -> None:
        if self._persist_dir is None:
            return
        import faiss

        faiss.write_index(coll.index, str(self._persist_dir / f"{coll.name}.faiss"))
        meta = {
            "name": coll.name,
            "vector_size": coll.vector_size,
            "distance": coll.distance,
            "id_to_pos": coll.id_to_pos,
            "pos_to_id": coll.pos_to_id,
            "payloads": coll.payloads,
            "_next_pos": coll._next_pos,
        }
        with open(self._persist_dir / f"{coll.name}.meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    def _load_collection(self, name: str) -> Optional[_FAISSCollection]:
        if self._persist_dir is None:
            return None
        index_path = self._persist_dir / f"{name}.faiss"
        meta_path = self._persist_dir / f"{name}.meta.pkl"
        if not index_path.exists() or not meta_path.exists():
            return None
        import faiss

        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        coll = _FAISSCollection(
            name=meta["name"],
            vector_size=meta["vector_size"],
            distance=meta["distance"],
            index=index,
            id_to_pos=meta["id_to_pos"],
            pos_to_id=meta["pos_to_id"],
            payloads=meta["payloads"],
            _next_pos=meta["_next_pos"],
        )
        return coll

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Optional[DistanceMetric] = None,
        recreate: bool = False,
    ) -> bool:
        distance = distance or self._default_distance
        if collection_name in self._collections:
            if recreate:
                self.delete_collection(collection_name)
            else:
                return True

        index = self._build_index(vector_size, distance)
        self._collections[collection_name] = _FAISSCollection(
            name=collection_name,
            vector_size=vector_size,
            distance=distance,
            index=index,
        )
        logger.info(
            f"Created FAISS collection '{collection_name}' "
            f"(dim={vector_size}, distance={distance.value})"
        )
        return True

    def collection_exists(self, collection_name: str) -> bool:
        if collection_name in self._collections:
            return True
        # Try loading from disk
        coll = self._load_collection(collection_name)
        if coll is not None:
            self._collections[collection_name] = coll
            return True
        return False

    def delete_collection(self, collection_name: str) -> bool:
        self._collections.pop(collection_name, None)
        if self._persist_dir is not None:
            for suffix in (".faiss", ".meta.pkl"):
                path = self._persist_dir / f"{collection_name}{suffix}"
                if path.exists():
                    path.unlink()
        logger.info(f"Deleted FAISS collection '{collection_name}'")
        return True

    def list_collections(self) -> list:
        # Start from in-memory collections
        names = set(self._collections.keys())
        # Also discover persisted collections on disk
        if self._persist_dir is not None and self._persist_dir.exists():
            for meta_file in self._persist_dir.glob("*.meta.pkl"):
                name = meta_file.stem.replace(".meta", "")
                if name not in names:
                    # Load into memory so get_collection_info works
                    coll = self._load_collection(name)
                    if coll is not None:
                        self._collections[name] = coll
                        names.add(name)
        result = []
        for name in sorted(names):
            try:
                result.append(self.get_collection_info(name))
            except KeyError:
                pass
        return result

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def upsert_vectors(
        self,
        collection_name: str,
        vectors: Union[List[List[float]], np.ndarray],
        payloads: List[Dict],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> int:
        coll = self._get(collection_name)
        vecs = np.array(vectors, dtype=np.float32)
        if len(vecs) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")

        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vecs))]

        if coll.distance == DistanceMetric.COSINE:
            vecs = self._normalize(vecs)

        for i, (vid, payload) in enumerate(zip(ids, payloads)):
            if vid in coll.id_to_pos:
                # Update existing: replace vector at position
                pos = coll.id_to_pos[vid]
                vec = vecs[i : i + 1]
                # FAISS flat indexes store vectors contiguously — reconstruct
                # is cheaper than remove+add; we just overwrite via
                # direct memory access on flat indexes.
                import faiss

                faiss.copy_array_to_vector(
                    vec.ravel(),
                    coll.index.get_xb(),
                    pos * coll.vector_size,
                )
            else:
                pos = coll._next_pos
                coll.id_to_pos[vid] = pos
                coll.pos_to_id[pos] = vid
                coll._next_pos += 1
                coll.index.add(vecs[i : i + 1])
            coll.payloads[vid] = payload

        self._persist_collection(coll)
        logger.info(f"Upserted {len(ids)} vectors to FAISS collection '{collection_name}'")
        return len(ids)

    def search(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict]:
        coll = self._get(collection_name)
        qvec = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        if coll.distance == DistanceMetric.COSINE:
            qvec = self._normalize(qvec)

        # Over-fetch when filtering to compensate for post-filter losses
        fetch_limit = limit * 3 if filter_conditions else limit
        fetch_limit = min(fetch_limit, coll.index.ntotal)
        if fetch_limit == 0:
            return []

        distances, indices = coll.index.search(qvec, fetch_limit)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            vid = coll.pos_to_id.get(int(idx))
            if vid is None:
                continue

            # Convert FAISS distance → score
            if coll.distance == DistanceMetric.EUCLIDEAN:
                score = -float(dist)  # lower L2 = better → negate
            else:
                score = float(dist)  # IP-based: higher = better

            if score_threshold is not None and score < score_threshold:
                continue

            payload = coll.payloads.get(vid, {})
            if filter_conditions and not self._match_filter(payload, filter_conditions):
                continue

            results.append({"id": vid, "score": score, "payload": payload})
            if len(results) >= limit:
                break

        return results

    def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        filter_conditions: Optional[Dict] = None,
    ) -> Tuple[List[Dict], Optional[str]]:
        coll = self._get(collection_name)

        # Sort IDs for deterministic pagination
        all_ids = sorted(coll.payloads.keys())

        # Find start position via offset (offset = last returned id)
        start_idx = 0
        if offset is not None:
            for i, vid in enumerate(all_ids):
                if vid > offset:
                    start_idx = i
                    break
            else:
                return [], None

        records = []
        for vid in all_ids[start_idx:]:
            if len(records) >= limit:
                break
            payload = coll.payloads.get(vid, {})
            if filter_conditions and not self._match_filter(payload, filter_conditions):
                continue
            record: Dict = {"id": vid}
            if with_payload:
                record["payload"] = payload
            if with_vectors:
                pos = coll.id_to_pos.get(vid)
                if pos is not None:
                    record["vector"] = coll.index.reconstruct(pos).tolist()
            records.append(record)

        next_offset = records[-1]["id"] if records else None
        # If we've exhausted all ids past the last record, signal end
        if records:
            last_idx = all_ids.index(records[-1]["id"])
            if last_idx >= len(all_ids) - 1:
                next_offset = None

        return records, next_offset

    # ------------------------------------------------------------------
    # Metadata / statistics
    # ------------------------------------------------------------------

    def get_collection_info(self, collection_name: str) -> Dict:
        coll = self._get(collection_name)
        return {
            "name": coll.name,
            "vector_size": coll.vector_size,
            "distance": coll.distance.value,
            "points_count": coll.index.ntotal,
        }

    def count_vectors(
        self,
        collection_name: str,
        filter_conditions: Optional[Dict] = None,
    ) -> int:
        coll = self._get(collection_name)
        if not filter_conditions:
            return coll.index.ntotal
        return sum(
            1
            for payload in coll.payloads.values()
            if self._match_filter(payload, filter_conditions)
        )

    def delete_vectors(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> bool:
        coll = self._get(collection_name)
        if ids is None and filter_conditions is None:
            raise ValueError("Must provide either ids or filter_conditions")

        targets = set()
        if ids:
            targets.update(ids)
        if filter_conditions:
            for vid, payload in coll.payloads.items():
                if self._match_filter(payload, filter_conditions):
                    targets.add(vid)

        if not targets:
            return True

        # Rebuild index without deleted vectors
        remaining_ids = [vid for vid in coll.id_to_pos if vid not in targets]
        remaining_vecs = []
        for vid in remaining_ids:
            pos = coll.id_to_pos[vid]
            remaining_vecs.append(coll.index.reconstruct(pos))

        # Reset index
        new_index = self._build_index(coll.vector_size, coll.distance)
        coll.index = new_index
        coll.id_to_pos.clear()
        coll.pos_to_id.clear()
        coll._next_pos = 0

        if remaining_vecs:
            vecs_array = np.array(remaining_vecs, dtype=np.float32)
            # Already normalised if cosine (they were normalised at insert time)
            coll.index.add(vecs_array)
            for i, vid in enumerate(remaining_ids):
                coll.id_to_pos[vid] = i
                coll.pos_to_id[i] = vid
            coll._next_pos = len(remaining_ids)

        for vid in targets:
            coll.payloads.pop(vid, None)

        self._persist_collection(coll)
        logger.info(f"Deleted {len(targets)} vectors from FAISS collection '{collection_name}'")
        return True

    def get_statistics(self) -> Dict:
        stats: Dict = {
            "backend": "faiss",
            "persist_dir": str(self._persist_dir) if self._persist_dir else None,
            "total_collections": len(self._collections),
            "collections": {},
        }
        for name in self._collections:
            stats["collections"][name] = self.get_collection_info(name)
        return stats
