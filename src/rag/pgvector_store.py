"""
src/rag/pgvector_store.py
-------------------------
A PostgreSQL + pgvector backed vector store replacing FAISSStore.

Stores dense embeddings and JSON metadata in a shared ``mediflow_vectors``
table, scoped by *namespace* (which maps to the old FAISS directory path).

The public interface mirrors the old FAISSStore so all existing callers only
need an import-level change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from src.rag.common import flatten_record
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Two dedicated tables with different retention policies:
#   mediflow_knowledge      — permanent clinical knowledge base (written once)
#   mediflow_patient_vectors — ephemeral per-session patient data (TTL-deleted)
TABLE_KNOWLEDGE = "mediflow_knowledge"
TABLE_PATIENT = "mediflow_patient_vectors"

# Kept for backwards-compat imports only; no longer used internally.
TABLE_NAME = TABLE_KNOWLEDGE


class _IndexProxy:
    """Provides a FAISS-compatible ``.index.ntotal`` property backed by the DB."""

    __slots__ = ("_count_fn",)

    def __init__(self, count_fn) -> None:
        self._count_fn = count_fn

    @property
    def ntotal(self) -> int:
        return self._count_fn()


class PGVectorStore:
    """
    PostgreSQL pgvector-backed replacement for FAISSStore.

    Each instance is scoped to one *namespace* (a free-form string derived
    from the old FAISS directory path).  Use ``table_name=TABLE_KNOWLEDGE``
    for the permanent clinical knowledge base and ``table_name=TABLE_PATIENT``
    for ephemeral per-session patient data.

    Key behavioural notes
    ----------------------
    * ``save_local()`` is a no-op — data is already persisted in PostgreSQL.
    * ``load_local()`` opens (or creates) a store for the given namespace.
    * ``delete_all()`` removes every row for this namespace (replaces rmtree).
    * ``.index.ntotal`` is a live DB count wrapped in ``_IndexProxy``.
    """

    def __init__(
        self,
        dimension: int,
        namespace: str,
        db_url: str,
        table_name: str = TABLE_KNOWLEDGE,
        required_metadata_keys: Optional[Set[str]] = None,
    ) -> None:
        self.dimension = dimension
        self.namespace = namespace
        self._db_url = db_url
        self._table_name = table_name
        self.required_metadata_keys = required_metadata_keys or set()
        self._conn = self._connect()
        self._ensure_schema()
        self.index = _IndexProxy(self._count)

    # ── Connection helpers ────────────────────────────────────────────────────

    def _connect(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(self._db_url)
        conn.autocommit = True
        register_vector(conn)
        return conn

    def _get_cursor(self) -> psycopg2.extensions.cursor:
        """Return a RealDictCursor, reconnecting if the connection was dropped."""
        if self._conn.closed:
            self._conn = self._connect()
        return self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _ensure_schema(self) -> None:
        """Create the pgvector extension and the appropriate table if absent."""
        tbl = self._table_name
        with self._conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except psycopg2.errors.InsufficientPrivilege:
                logger.warning(
                    "pgvector_extension_privilege_error",
                    msg=(
                        "pgvector extension must be installed by a superuser. "
                        "Run `CREATE EXTENSION vector;` manually and retry."
                    ),
                )
            # Patient table gets a created_at column for scheduled TTL cleanup.
            # Knowledge table omits it — that data is permanent.
            extra_col = (
                ",\n                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"
                if tbl == TABLE_PATIENT
                else ""
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                    id        BIGSERIAL PRIMARY KEY,
                    namespace TEXT    NOT NULL,
                    chunk_id  TEXT,
                    embedding vector({self.dimension}),
                    metadata  JSONB   NOT NULL DEFAULT '{{}}'
                    {extra_col}
                );
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{tbl}_ns "
                f"ON {tbl}(namespace);"
            )
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{tbl}_ns_chunk "
                f"ON {tbl}(namespace, chunk_id) "
                f"WHERE chunk_id IS NOT NULL;"
            )
            # Index created_at on the patient table so the scheduled cleanup
            # query (DELETE WHERE created_at < NOW() - INTERVAL '24 hours')
            # uses an index scan instead of a sequential scan.
            if tbl == TABLE_PATIENT:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{tbl}_created_at "
                    f"ON {tbl}(created_at);"
                )

    def _count(self) -> int:
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) AS cnt FROM {self._table_name} WHERE namespace = %s",
                (self.namespace,),
            )
            row = cur.fetchone()
            return int(row["cnt"]) if row else 0

    # ── Validation + metadata helpers ─────────────────────────────────────────

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            raise ValueError("Each metadata entry must be a dictionary.")
        if not self.required_metadata_keys:
            return
        missing = [k for k in self.required_metadata_keys if k not in metadata]
        if missing:
            raise ValueError(f"Metadata is missing required keys: {missing}")

    @staticmethod
    def _extract_chunk_id(metadata: Dict[str, Any]) -> Optional[str]:
        normalized = flatten_record(metadata)
        chunk_id = normalized.get("chunk_id")
        return str(chunk_id) if chunk_id else None

    # ── Public interface ──────────────────────────────────────────────────────

    def existing_chunk_ids(self) -> Set[str]:
        with self._get_cursor() as cur:
            cur.execute(
                f"SELECT chunk_id FROM {self._table_name} "
                f"WHERE namespace = %s AND chunk_id IS NOT NULL",
                (self.namespace,),
            )
            return {row["chunk_id"] for row in cur.fetchall()}

    def add(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        dedupe_by_chunk_id: bool = False,
    ) -> int:
        """
        Insert a batch of dense vectors and their metadata.

        Parameters
        ----------
        embeddings:
            2-D float32 array of shape ``(n, dimension)``.
        metadatas:
            List of ``n`` metadata dicts to store alongside each vector.
        dedupe_by_chunk_id:
            When ``True``, rows whose ``chunk_id`` is already present in this
            namespace are skipped before insertion.

        Returns
        -------
        int
            Number of rows inserted (excluding skipped duplicates).
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dimension}, "
                f"received {embeddings.shape[1]}."
            )
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Size mismatch between embeddings and metadatas.")
        if embeddings.shape[0] == 0:
            return 0

        for meta in metadatas:
            self._validate_metadata(meta)

        existing_ids = self.existing_chunk_ids() if dedupe_by_chunk_id else set()
        skipped = 0
        rows: List[tuple] = []

        for i, meta in enumerate(metadatas):
            chunk_id = self._extract_chunk_id(meta)
            if dedupe_by_chunk_id and chunk_id and chunk_id in existing_ids:
                skipped += 1
                continue
            if dedupe_by_chunk_id and chunk_id:
                existing_ids.add(chunk_id)
            rows.append(
                (
                    self.namespace,
                    chunk_id,
                    embeddings[i].astype(np.float32),
                    psycopg2.extras.Json(meta),
                )
            )

        if not rows:
            logger.debug("pgvector_store_add_skipped_all", duplicates=skipped)
            return 0

        with self._get_cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO {self._table_name} (namespace, chunk_id, embedding, metadata)
                VALUES %s
                ON CONFLICT (namespace, chunk_id) WHERE chunk_id IS NOT NULL DO NOTHING
                """,
                rows,
                template="(%s, %s, %s, %s)",
            )

        added = len(rows)
        logger.debug(
            "pgvector_store_add",
            added=added,
            duplicates_skipped=skipped,
            total=self._count(),
        )
        return added

    @staticmethod
    def _metadata_matches_filter(
        metadata: Dict[str, Any], metadata_filter: Dict[str, Any]
    ) -> bool:
        normalized = flatten_record(metadata)
        for key, value in metadata_filter.items():
            if normalized.get(key) != value:
                return False
        return True

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search for the nearest ``k`` matches using L2 (Euclidean) distance.

        Parameters
        ----------
        query_vector:
            1-D or 2-D float32 array whose last axis has length ``dimension``.
        k:
            Maximum number of results to return.
        metadata_filter:
            Optional flat key/value filter applied in Python after retrieval.

        Returns
        -------
        List of ``(l2_distance, metadata_dict)`` tuples sorted ascending.
        """
        if k <= 0 or self._count() == 0:
            return []

        q = query_vector.flatten().astype(np.float32)
        if q.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self.dimension}, "
                f"received {q.shape[0]}."
            )

        # Over-fetch when post-filtering to ensure we can return k results.
        fetch_k = k if not metadata_filter else min(k * 20, self._count())

        with self._get_cursor() as cur:
            cur.execute(
                f"""
                SELECT metadata, embedding <-> %s AS distance
                FROM   {self._table_name}
                WHERE  namespace = %s
                ORDER  BY embedding <-> %s
                LIMIT  %s
                """,
                (q, self.namespace, q, fetch_k),
            )
            rows = cur.fetchall()

        results: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            meta = row["metadata"]
            if metadata_filter and not self._metadata_matches_filter(
                meta, metadata_filter
            ):
                continue
            results.append((float(row["distance"]), meta))
            if len(results) >= k:
                break

        return results

    def delete_all(self) -> int:
        """
        Delete every row in this namespace from the database.

        Returns the number of rows deleted.  Use this in place of the old
        ``shutil.rmtree(store_path)`` calls.
        """
        with self._get_cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table_name} WHERE namespace = %s",
                (self.namespace,),
            )
            count = cur.rowcount
        logger.info("pgvector_store_deleted", namespace=self.namespace, count=count)
        return count

    def save_local(self, directory: str) -> None:
        """
        No-op compatibility shim — data is already persisted in PostgreSQL.
        Kept so existing ``store.save_local(path)`` calls do not break.
        """
        logger.info(
            "pgvector_store_save_noop",
            namespace=self.namespace,
            note="Vectors are persisted in PostgreSQL — no local files written.",
        )

    @classmethod
    def load_local(
        cls,
        directory: str,
        dimension: int = 384,
        table_name: str = TABLE_KNOWLEDGE,
        required_metadata_keys: Optional[Set[str]] = None,
        db_url: Optional[str] = None,
    ) -> "PGVectorStore":
        """
        Open (or create) a ``PGVectorStore`` scoped to ``directory``.

        The ``directory`` string is used verbatim as the PostgreSQL namespace,
        preserving full compatibility with callers that previously passed
        filesystem paths.

        Parameters
        ----------
        directory:
            Namespace key (historically a FAISS directory path).
        dimension:
            Expected embedding dimension.
        table_name:
            Which DB table to use.  Pass ``TABLE_KNOWLEDGE`` for the permanent
            clinical knowledge base or ``TABLE_PATIENT`` for ephemeral session
            data.  Defaults to ``TABLE_KNOWLEDGE``.
        required_metadata_keys:
            Optional set of keys that every metadata dict must contain.
        db_url:
            PostgreSQL DSN.  Defaults to ``settings.pgvector_database_url``.
        """
        from src.core.config import settings  # local import avoids circular deps

        resolved_url = db_url or settings.pgvector_database_url
        namespace = str(directory)
        store = cls(
            dimension=dimension,
            namespace=namespace,
            db_url=resolved_url,
            table_name=table_name,
            required_metadata_keys=required_metadata_keys,
        )
        logger.info(
            "pgvector_store_loaded",
            namespace=namespace,
            total_documents=store._count(),
        )
        return store
