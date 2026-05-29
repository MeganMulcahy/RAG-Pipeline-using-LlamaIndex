"""
Local Chroma persistence: chunk text + embeddings, file-hash dedup, filename versions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

log = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data/rag"))
BLOB_DIR = DATA_DIR / "blobs"
CHROMA_DIR = DATA_DIR / "chroma"
VERSIONS_PATH = DATA_DIR / "versions.json"
INGESTED_PATH = DATA_DIR / "ingested_hashes.json"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "rag_chunks")


def file_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_id_from_hash(file_hash: str) -> str:
    return file_hash[:16]


class PersistStore:
    """Chroma vector store + local blobs + file-hash / version registry."""

    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        BLOB_DIR.mkdir(parents=True, exist_ok=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        import chromadb
        self._chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = self._chroma.get_or_create_collection(COLLECTION_NAME)
        self._vector_store = ChromaVectorStore(chroma_collection=self._collection)

        self._versions = self._load_json(VERSIONS_PATH, default={})
        self._ingested = set(self._load_json(INGESTED_PATH, default=[]))

    @staticmethod
    def _load_json(path: Path, default):
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Corrupt %s — resetting", path.name)
        return default

    def _save_versions(self) -> None:
        VERSIONS_PATH.write_text(json.dumps(self._versions, indent=2), encoding="utf-8")

    def _save_ingested(self) -> None:
        INGESTED_PATH.write_text(json.dumps(sorted(self._ingested)), encoding="utf-8")

    def save_blob(self, data: bytes, file_name: str) -> Tuple[str, str]:
        file_hash = file_hash_bytes(data)
        blob_id = blob_id_from_hash(file_hash)
        dest_dir = BLOB_DIR / blob_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / file_name
        if not dest.exists():
            dest.write_bytes(data)
        return file_hash, blob_id

    def is_ingested(self, file_hash: str) -> bool:
        if file_hash in self._ingested:
            return True
        try:
            res = self._collection.get(
                where={"file_hash": file_hash},
                limit=1,
                include=[],
            )
            if res.get("ids"):
                self._ingested.add(file_hash)
                self._save_ingested()
                return True
        except Exception:
            pass
        return False

    def get_or_assign_version(self, file_name: str, file_hash: str) -> int:
        per_file = self._versions.setdefault(file_name, {})
        if file_hash in per_file:
            return int(per_file[file_hash])
        version_num = len(per_file) + 1
        per_file[file_hash] = version_num
        self._save_versions()
        return version_num

    def mark_ingested(
        self,
        file_hash: str,
        file_name: str,
        blob_id: str,
        version_num: int,
        chunk_count: int,
    ) -> None:
        self._ingested.add(file_hash)
        self._save_ingested()
        log.info(
            "Indexed %s v%d (%d chunks, hash %s…)",
            file_name, version_num, chunk_count, blob_id,
        )

    def has_index(self) -> bool:
        return self._collection.count() > 0

    def chunk_count(self) -> int:
        return self._collection.count()

    def _storage_context(self) -> StorageContext:
        return StorageContext.from_defaults(
            vector_store=self._vector_store,
            docstore=SimpleDocumentStore(),
        )

    def _sanitize_metadata(self, meta: dict) -> dict:
        out = {}
        for k, v in (meta or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def build_index(self, nodes: List[TextNode]) -> Tuple[VectorStoreIndex, List[TextNode]]:
        for node in nodes:
            node.metadata = self._sanitize_metadata(node.metadata)

        ctx = self._storage_context()
        if self.has_index():
            index = VectorStoreIndex.from_vector_store(
                self._vector_store,
                storage_context=ctx,
            )
            index.insert_nodes(nodes)
        else:
            index = VectorStoreIndex(nodes, storage_context=ctx)
        return index, nodes

    def load_index(self) -> Tuple[VectorStoreIndex, List[TextNode]]:
        ctx = self._storage_context()
        index = VectorStoreIndex.from_vector_store(
            self._vector_store,
            storage_context=ctx,
        )
        nodes = self._all_nodes()
        for node in nodes:
            try:
                index.docstore.add_documents([node], allow_update=True)
            except TypeError:
                index.docstore.add_documents([node])
        return index, nodes

    def _all_nodes(self) -> List[TextNode]:
        res = self._collection.get(include=["documents", "metadatas"])
        nodes: List[TextNode] = []
        for doc_id, text, meta in zip(
            res.get("ids") or [],
            res.get("documents") or [],
            res.get("metadatas") or [],
        ):
            if not text:
                continue
            nodes.append(
                TextNode(text=text, id_=doc_id, metadata=self._sanitize_metadata(meta or {}))
            )
        return nodes
