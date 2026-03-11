"""
PgVectorStore — BaseVectorStore implementation backed by PostgreSQL + pgvector.

Each collection maps to a table: ``id TEXT PK, vector vector(dim), payload JSONB``.
Filtering uses native JSONB queries.  Pagination (``scroll()``) uses keyset
pagination on ``id``.

Usage:
    from src.vector_store.pgvector_store import PgVectorStore

    store = PgVectorStore(connection_params={"dbname": "rag", "user": "postgres"})
    store.create_collection("test", vector_size=384)

Requires:
    - ``psycopg2`` (or ``psycopg2-binary``)
    - PostgreSQL with the ``vector`` extension enabled
"""

import json
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

from src.vector_store.base import BaseVectorStore, DistanceMetric
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DISTANCE_OPS: Dict[DistanceMetric, str] = {
    DistanceMetric.COSINE: "<=>",
    DistanceMetric.EUCLIDEAN: "<->",
    DistanceMetric.DOT_PRODUCT: "<#>",
}

# pgvector operators where lower = better (for score negation)
_NEGATE_SCORE: Dict[DistanceMetric, bool] = {
    DistanceMetric.COSINE: True,       # cosine distance: 0 = identical
    DistanceMetric.EUCLIDEAN: True,     # L2 distance: 0 = identical
    DistanceMetric.DOT_PRODUCT: True,   # <#> returns negative inner product
}


def _sanitize_table_name(name: str) -> str:
    """Only allow alphanumeric and underscores to prevent SQL injection."""
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return f"vec_{sanitized}"


class PgVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector vector store."""

    def __init__(
        self,
        connection_params: Optional[Dict] = None,
        default_distance: DistanceMetric = DistanceMetric.COSINE,
    ):
        """
        Args:
            connection_params: kwargs passed to ``psycopg2.connect()``.
                               Example: ``{"dbname": "rag", "user": "postgres",
                               "host": "localhost", "port": 5432}``
            default_distance: Metric used when ``create_collection`` is called
                              without an explicit ``distance``.
        """
        super().__init__(default_distance=default_distance)
        self._conn_params = connection_params or {}
        self._conn = None
        # Cache collection metadata: name → {vector_size, distance}
        self._meta: Dict[str, Dict] = {}

    def _get_conn(self):
        """Lazy connection."""
        if self._conn is None or self._conn.closed:
            import psycopg2

            self._conn = psycopg2.connect(**self._conn_params)
            self._conn.autocommit = True
            # Ensure pgvector extension
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        return self._conn

    def _table(self, collection_name: str) -> str:
        return _sanitize_table_name(collection_name)

    def _build_where(self, filter_conditions: Dict) -> Tuple[str, list]:
        """Build WHERE clause fragments + params from filter dict."""
        clauses = []
        params: list = []
        for fld, value in filter_conditions.items():
            if isinstance(value, dict):
                for op_name, op_sql in [("gte", ">="), ("lte", "<="), ("gt", ">"), ("lt", "<")]:
                    if op_name in value:
                        clauses.append(f"(payload->>%s)::float {op_sql} %s")
                        params.extend([fld, value[op_name]])
            elif isinstance(value, list):
                placeholders = ",".join(["%s"] * len(value))
                clauses.append(f"payload->>%s IN ({placeholders})")
                params.append(fld)
                params.extend([str(v) for v in value])
            else:
                clauses.append("payload->>%s = %s")
                params.extend([fld, str(value)])
        return " AND ".join(clauses), params

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
        if distance not in _DISTANCE_OPS:
            raise ValueError(
                f"pgvector does not support distance metric '{distance.value}'. "
                f"Supported: {list(_DISTANCE_OPS.keys())}"
            )

        conn = self._get_conn()
        table = self._table(collection_name)

        if recreate:
            self.delete_collection(collection_name)

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT to_regclass(%s)",
                (table,),
            )
            if cur.fetchone()[0] is not None:
                self._meta[collection_name] = {"vector_size": vector_size, "distance": distance}
                return True

            cur.execute(
                f"CREATE TABLE {table} ("
                f"  id TEXT PRIMARY KEY,"
                f"  vector vector({vector_size}),"
                f"  payload JSONB DEFAULT '{{}}'"
                f")"
            )
        self._meta[collection_name] = {"vector_size": vector_size, "distance": distance}
        logger.info(f"Created pgvector table '{table}' (dim={vector_size})")
        return True

    def collection_exists(self, collection_name: str) -> bool:
        conn = self._get_conn()
        table = self._table(collection_name)
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table,))
            return cur.fetchone()[0] is not None

    def delete_collection(self, collection_name: str) -> bool:
        conn = self._get_conn()
        table = self._table(collection_name)
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        self._meta.pop(collection_name, None)
        logger.info(f"Dropped pgvector table '{table}'")
        return True

    def list_collections(self) -> list:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'vec_%'"
            )
            tables = [row[0] for row in cur.fetchall()]
        result = []
        for table in tables:
            # Reverse the vec_ prefix to get collection name
            collection_name = table[4:]  # strip "vec_"
            try:
                info = self.get_collection_info(collection_name)
                result.append(info)
            except Exception:
                result.append({"name": collection_name, "vector_size": None, "distance": None, "points_count": None})
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
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads must have the same length")
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]

        conn = self._get_conn()
        table = self._table(collection_name)
        total = 0

        from psycopg2.extras import execute_values

        for i in range(0, len(vectors), batch_size):
            batch = [
                (vid, str(vec), json.dumps(payload))
                for vid, vec, payload in zip(
                    ids[i : i + batch_size],
                    vectors[i : i + batch_size],
                    payloads[i : i + batch_size],
                )
            ]
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO {table} (id, vector, payload) "
                    f"VALUES %s "
                    f"ON CONFLICT (id) DO UPDATE SET vector=EXCLUDED.vector, payload=EXCLUDED.payload",
                    batch,
                )
            total += len(batch)

        logger.info(f"Upserted {total} vectors to pgvector table '{table}'")
        return total

    def search(
        self,
        collection_name: str,
        query_vector: Union[List[float], np.ndarray],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict]:
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        meta = self._meta.get(collection_name, {})
        distance = meta.get("distance", DistanceMetric.COSINE)
        op = _DISTANCE_OPS[distance]

        conn = self._get_conn()
        table = self._table(collection_name)
        vec_str = str(query_vector)

        where_clause = ""
        params: list = [vec_str]
        if filter_conditions:
            frag, fparams = self._build_where(filter_conditions)
            if frag:
                where_clause = f"WHERE {frag}"
                params.extend(fparams)

        params.append(limit)

        sql = (
            f"SELECT id, payload, vector {op} %s AS dist "
            f"FROM {table} {where_clause} "
            f"ORDER BY vector {op} %s "
            f"LIMIT %s"
        )
        # We need to pass vec_str twice (once for dist, once for ORDER BY)
        all_params = [vec_str]
        if filter_conditions:
            _, fparams = self._build_where(filter_conditions)
            all_params.extend(fparams)
        all_params.extend([vec_str, limit])

        with conn.cursor() as cur:
            # Rebuild SQL to avoid double-param confusion
            where_clause_inner = ""
            inner_params: list = []
            if filter_conditions:
                frag, fparams = self._build_where(filter_conditions)
                if frag:
                    where_clause_inner = f"WHERE {frag}"
                    inner_params = fparams

            final_sql = (
                f"SELECT id, payload, vector {op} %s::vector AS dist "
                f"FROM {table} {where_clause_inner} "
                f"ORDER BY vector {op} %s::vector "
                f"LIMIT %s"
            )
            final_params = [vec_str] + inner_params + [vec_str, limit]
            cur.execute(final_sql, final_params)
            rows = cur.fetchall()

        results = []
        for row_id, payload_json, dist in rows:
            # Convert distance → score (higher = better for all metrics)
            if _NEGATE_SCORE.get(distance, False):
                score = -float(dist)
            else:
                score = float(dist)

            if score_threshold is not None and score < score_threshold:
                continue

            payload = payload_json if isinstance(payload_json, dict) else json.loads(payload_json)
            results.append({"id": row_id, "score": score, "payload": payload})

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
        conn = self._get_conn()
        table = self._table(collection_name)

        conditions = []
        params: list = []
        if offset is not None:
            conditions.append("id > %s")
            params.append(offset)
        if filter_conditions:
            frag, fparams = self._build_where(filter_conditions)
            if frag:
                conditions.append(frag)
                params.extend(fparams)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        cols = ["id"]
        if with_payload:
            cols.append("payload")
        if with_vectors:
            cols.append("vector::text")

        params.append(limit + 1)  # fetch one extra to detect more pages

        sql = f"SELECT {', '.join(cols)} FROM {table} {where} ORDER BY id LIMIT %s"

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        has_more = len(rows) > limit
        rows = rows[:limit]

        records = []
        for row in rows:
            idx = 0
            record: Dict = {"id": row[idx]}
            idx += 1
            if with_payload:
                pval = row[idx]
                record["payload"] = pval if isinstance(pval, dict) else json.loads(pval)
                idx += 1
            if with_vectors:
                record["vector"] = row[idx]
                idx += 1
            records.append(record)

        next_offset = records[-1]["id"] if records and has_more else None
        return records, next_offset

    # ------------------------------------------------------------------
    # Metadata / statistics
    # ------------------------------------------------------------------

    def get_collection_info(self, collection_name: str) -> Dict:
        conn = self._get_conn()
        table = self._table(collection_name)
        meta = self._meta.get(collection_name, {})

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]

        return {
            "name": collection_name,
            "vector_size": meta.get("vector_size"),
            "distance": meta.get("distance", DistanceMetric.COSINE).value,
            "points_count": count,
        }

    def count_vectors(
        self,
        collection_name: str,
        filter_conditions: Optional[Dict] = None,
    ) -> int:
        conn = self._get_conn()
        table = self._table(collection_name)

        where = ""
        params: list = []
        if filter_conditions:
            frag, fparams = self._build_where(filter_conditions)
            if frag:
                where = f"WHERE {frag}"
                params = fparams

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table} {where}", params)
            return cur.fetchone()[0]

    def delete_vectors(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict] = None,
    ) -> bool:
        if ids is None and filter_conditions is None:
            raise ValueError("Must provide either ids or filter_conditions")

        conn = self._get_conn()
        table = self._table(collection_name)

        with conn.cursor() as cur:
            if ids:
                placeholders = ",".join(["%s"] * len(ids))
                cur.execute(f"DELETE FROM {table} WHERE id IN ({placeholders})", ids)
            if filter_conditions:
                frag, fparams = self._build_where(filter_conditions)
                if frag:
                    cur.execute(f"DELETE FROM {table} WHERE {frag}", fparams)

        return True

    def get_statistics(self) -> Dict:
        return {
            "backend": "pgvector",
            "connection_params": {
                k: v for k, v in self._conn_params.items() if k != "password"
            },
            "total_collections": len(self._meta),
            "collections": {
                name: self.get_collection_info(name) for name in self._meta
            },
        }
