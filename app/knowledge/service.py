"""Chroma-backed retrieval over explicitly configured repository Markdown.

The corpus boundary is intentionally narrow: only repository-relative paths listed in
settings are eligible. In particular, Cursor's global skills are not project knowledge
and cannot be indexed by accident.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.config import Settings
from app.observability.instrumentation import track_operation

_COLLECTION = "k8s_ops_knowledge"
_SCHEMA_VERSION = "1"


class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class KnowledgeHit(BaseModel):
    content: str
    source_path: str
    heading_path: str | None = None
    chunk_index: int
    score: float | None = None


class RefreshResult(BaseModel):
    added_or_updated: int
    unchanged: int
    removed: int
    sources: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Chunk:
    id: str
    document: Document


class KnowledgeService:
    """Owns one persistent Chroma collection and its repository corpus."""

    def __init__(
        self,
        settings: Settings,
        *,
        repository_root: Path,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._settings = settings
        self._root = repository_root.resolve()
        self._embeddings = embeddings
        self._store: Chroma | None = None

    def refresh(self) -> RefreshResult:
        """Upsert changed chunks and delete chunks no longer in the configured corpus."""
        chunks = list(self._load_chunks())
        store = self._vector_store()
        existing = store.get(include=["metadatas"])
        existing_ids = set(existing["ids"])
        desired_ids = {chunk.id for chunk in chunks}

        with track_operation("knowledge.refresh", chunk_count=len(chunks)) as log:
            changed = [chunk for chunk in chunks if chunk.id not in existing_ids]
            unchanged = len(chunks) - len(changed)
            if changed:
                store.add_documents(
                    [chunk.document for chunk in changed],
                    ids=[chunk.id for chunk in changed],
                )
            removed_ids = existing_ids - desired_ids
            if removed_ids:
                store.delete(ids=sorted(removed_ids))
            log["added_or_updated"] = len(changed)
            log["unchanged"] = unchanged
            log["removed"] = len(removed_ids)

        return RefreshResult(
            added_or_updated=len(changed),
            unchanged=unchanged,
            removed=len(removed_ids),
            sources=sorted({chunk.document.metadata["source_path"] for chunk in chunks}),
        )

    def search(self, query: str) -> list[KnowledgeHit]:
        """Return cited documentation chunks; never return arbitrary file contents."""
        if not self._settings.rag_enabled:
            return []

        # `refresh` is incremental, so making it the first-use path means a checkout
        # has a usable index without embedding work during application startup.
        self.refresh()
        with track_operation("knowledge.search", query_length=len(query)) as log:
            results = self._vector_store().similarity_search_with_relevance_scores(
                query,
                k=self._settings.rag_top_k,
            )
            log["result_count"] = len(results)

        return [
            KnowledgeHit(
                content=document.page_content,
                source_path=str(document.metadata["source_path"]),
                heading_path=document.metadata.get("heading_path"),
                chunk_index=int(document.metadata["chunk_index"]),
                score=score,
            )
            for document, score in results
        ]

    def _vector_store(self) -> Chroma:
        if self._store is None:
            persist = self._resolve_storage_path(self._settings.rag_persist_directory)
            persist.mkdir(parents=True, exist_ok=True)
            self._store = Chroma(
                collection_name=_COLLECTION,
                persist_directory=str(persist),
                embedding_function=cast(LangChainEmbeddings, self._embedding_function()),
            )
        return self._store

    def _embedding_function(self) -> Embeddings:
        if self._embeddings is None:
            # Imported only when retrieval/indexing is used; `make run` remains fast
            # and does not download a model merely by starting FastAPI.
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self._settings.rag_embedding_model,
            )
        return self._embeddings

    def _load_chunks(self) -> Iterable[_Chunk]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.rag_chunk_size,
            chunk_overlap=self._settings.rag_chunk_overlap,
        )
        for configured_path in self._settings.rag_corpus_paths:
            source = self._resolve_corpus_path(configured_path)
            if not source.is_file() or source.suffix.lower() not in {".md", ".mdx"}:
                continue
            source_path = source.relative_to(self._root).as_posix()
            text = source.read_text(encoding="utf-8")
            source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            heading = _heading_for(text)
            parts = splitter.split_text(text)
            for index, content in enumerate(parts):
                digest = hashlib.sha256(
                    f"{source_path}:{source_hash}:{index}:{content}".encode()
                ).hexdigest()
                metadata = {
                    "source_path": source_path,
                    "source_type": "markdown",
                    "heading_path": heading,
                    "chunk_index": index,
                    "content_hash": source_hash,
                    "ingested_at": datetime.now(UTC).isoformat(),
                    "schema_version": _SCHEMA_VERSION,
                }
                yield _Chunk(
                    id=f"{source_path}:{digest}:{index}",
                    document=Document(page_content=content, metadata=metadata),
                )

    def _resolve_corpus_path(self, path: Path) -> Path:
        candidate = (self._root / path).resolve()
        if not candidate.is_relative_to(self._root):
            raise ValueError(f"RAG corpus path escapes the repository: {path}")
        return candidate

    def _resolve_storage_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self._root / path


def _heading_for(markdown: str) -> str | None:
    """Best-effort primary heading retained on every chunk for source citation."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None
