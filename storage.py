"""
Phase 2 — storage & serving layer (local-first, production-shaped).

- Blob store: local disk by default; optional MinIO (S3-compatible) via env
- Metadata: SQLite by default; set DATABASE_URL for Postgres later
- Ingest queue: SQLite job table; optional Redis when REDIS_URL is set
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data/rag"))
BLOB_DIR = DATA_DIR / "blobs"
SQLITE_PATH = os.getenv("SQLITE_PATH", str(DATA_DIR / "metadata.db"))
DATABASE_URL = os.getenv("DATABASE_URL", "")  # postgres://... when ready

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "rag-documents")

REDIS_URL = os.getenv("REDIS_URL", "")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BlobStore:
    """Persist uploaded files (local disk or MinIO)."""

    def __init__(self) -> None:
        BLOB_DIR.mkdir(parents=True, exist_ok=True)
        self._use_minio = bool(MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY)
        self._minio = None
        if self._use_minio:
            try:
                from minio import Minio
                endpoint = MINIO_ENDPOINT.replace("https://", "").replace("http://", "")
                secure = MINIO_ENDPOINT.startswith("https")
                self._minio = Minio(
                    endpoint,
                    access_key=MINIO_ACCESS_KEY,
                    secret_key=MINIO_SECRET_KEY,
                    secure=secure,
                )
                if not self._minio.bucket_exists(MINIO_BUCKET):
                    self._minio.make_bucket(MINIO_BUCKET)
                log.info("BlobStore: MinIO @ %s/%s", MINIO_ENDPOINT, MINIO_BUCKET)
            except Exception as e:
                log.warning("MinIO unavailable, using local blobs: %s", e)
                self._use_minio = False

    def put(self, data: bytes, file_name: str) -> str:
        blob_id = hashlib.sha256(data).hexdigest()[:16]
        if self._use_minio and self._minio:
            object_name = f"{blob_id}/{file_name}"
            import io
            self._minio.put_object(
                MINIO_BUCKET,
                object_name,
                io.BytesIO(data),
                length=len(data),
                content_type="application/pdf",
            )
            return blob_id

        dest_dir = BLOB_DIR / blob_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / file_name
        dest.write_bytes(data)
        return blob_id

    def local_path(self, blob_id: str, file_name: str) -> Path:
        return BLOB_DIR / blob_id / file_name


class MetadataStore:
    """Ingestion jobs, document versions, and chunk lineage."""

    def __init__(self, db_path: str = SQLITE_PATH) -> None:
        if DATABASE_URL.startswith("postgres"):
            raise NotImplementedError(
                "Postgres DATABASE_URL detected — use SQLite locally for now "
                "or add psycopg2 integration in a follow-up."
            )
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    blob_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    segment_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    blob_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    version_num INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES ingest_jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS chunk_lineage (
                    chunk_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    doc_id TEXT,
                    page_start INTEGER,
                    page_end INTEGER,
                    char_count INTEGER,
                    preview TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (version_id) REFERENCES document_versions(version_id)
                );
            """)

    def create_job(self, blob_id: str, file_name: str) -> str:
        job_id = str(uuid.uuid4())
        now = _utc_now()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO ingest_jobs
                   (job_id, blob_id, file_name, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'queued', ?, ?)""",
                (job_id, blob_id, file_name, now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        status: str,
        *,
        error: Optional[str] = None,
        segment_count: Optional[int] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE ingest_jobs
                   SET status = ?, error = COALESCE(?, error),
                       segment_count = COALESCE(?, segment_count),
                       updated_at = ?
                   WHERE job_id = ?""",
                (status, error, segment_count, _utc_now(), job_id),
            )

    def create_version(
        self, job_id: str, blob_id: str, file_name: str, file_bytes: bytes
    ) -> str:
        version_id = str(uuid.uuid4())
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version_num), 0) FROM document_versions WHERE blob_id = ?",
                (blob_id,),
            ).fetchone()
            version_num = int(row[0]) + 1
            conn.execute(
                """INSERT INTO document_versions
                   (version_id, job_id, blob_id, file_name, file_hash, version_num, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (version_id, job_id, blob_id, file_name, file_hash, version_num, _utc_now()),
            )
        return version_id

    def record_chunks(self, version_id: str, nodes: list) -> int:
        rows = []
        now = _utc_now()
        for node in nodes:
            text = node.get_content() if hasattr(node, "get_content") else str(node.text or "")
            chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            m = node.metadata or {}
            rows.append((
                str(uuid.uuid4()),
                version_id,
                chunk_hash,
                m.get("doc_id"),
                m.get("page_start"),
                m.get("page_end"),
                len(text),
                text[:120].replace("\n", " "),
                now,
            ))
        with self._conn() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO chunk_lineage
                   (chunk_id, version_id, chunk_hash, doc_id, page_start, page_end,
                    char_count, preview, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None


class IngestQueue:
    """Async ingestion queue — SQLite by default, Redis when REDIS_URL is set."""

    def __init__(self, metadata: MetadataStore) -> None:
        self.metadata = metadata
        self._redis = None
        if REDIS_URL:
            try:
                import redis
                self._redis = redis.from_url(REDIS_URL)
                log.info("IngestQueue: Redis @ %s", REDIS_URL.split("@")[-1])
            except Exception as e:
                log.warning("Redis unavailable, using SQLite queue: %s", e)

    def enqueue(self, blob_id: str, file_name: str) -> str:
        job_id = self.metadata.create_job(blob_id, file_name)
        if self._redis:
            self._redis.lpush("rag:ingest:queue", json.dumps({
                "job_id": job_id, "blob_id": blob_id, "file_name": file_name,
            }))
        return job_id

    def dequeue(self) -> Optional[Dict[str, Any]]:
        if self._redis:
            raw = self._redis.rpop("rag:ingest:queue")
            if raw:
                return json.loads(raw)
            return None
        with self.metadata._conn() as conn:
            row = conn.execute(
                """SELECT * FROM ingest_jobs
                   WHERE status = 'queued'
                   ORDER BY created_at ASC LIMIT 1"""
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE ingest_jobs SET status = 'processing', updated_at = ? WHERE job_id = ?",
                (_utc_now(), row["job_id"]),
            )
            return dict(row)
