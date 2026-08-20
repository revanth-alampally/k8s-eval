from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.knowledge.service import KnowledgeService


class TinyEmbeddings:
    """Small deterministic embedding implementation for hermetic vector-store tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        text = text.lower()
        return [float(text.count("imagepull")), float(text.count("restart")), 1.0]


def _service(root: Path) -> KnowledgeService:
    settings = Settings(
        rag_corpus_paths=[Path("runbook.md")],
        rag_persist_directory=Path(".test-chroma"),
        rag_chunk_size=200,
        rag_chunk_overlap=20,
    )
    return KnowledgeService(settings, repository_root=root, embeddings=TinyEmbeddings())


def test_refresh_adds_cited_markdown_chunks_and_is_incremental(tmp_path: Path) -> None:
    (tmp_path / "runbook.md").write_text(
        "# Image Pull Runbook\n\nImagePullBackOff means Kubernetes could not retrieve the image.\n"
    )
    service = _service(tmp_path)

    first = service.refresh()
    second = service.refresh()
    hits = service.search("How do I troubleshoot ImagePullBackOff?")

    assert first.added_or_updated == 1
    assert second.added_or_updated == 0
    assert hits[0].source_path == "runbook.md"
    assert hits[0].chunk_index == 0
    assert "ImagePullBackOff" in hits[0].content


def test_refresh_removes_stale_chunks_after_source_changes(tmp_path: Path) -> None:
    source = tmp_path / "runbook.md"
    source.write_text("# Restart\n\nRestart the deployment after a rollout is stuck.")
    service = _service(tmp_path)
    service.refresh()

    source.write_text("# Images\n\nUse ImagePullBackOff events to inspect image failures.")
    result = service.refresh()

    assert result.added_or_updated == 1
    assert result.removed == 1


def test_corpus_paths_cannot_escape_repository(tmp_path: Path) -> None:
    settings = Settings(
        rag_corpus_paths=[Path("../outside.md")],
        rag_persist_directory=Path(".test-chroma"),
    )
    service = KnowledgeService(settings, repository_root=tmp_path, embeddings=TinyEmbeddings())

    with pytest.raises(ValueError, match="escapes the repository"):
        service.refresh()


def test_disabled_knowledge_returns_no_results(tmp_path: Path) -> None:
    settings = Settings(rag_enabled=False, rag_corpus_paths=[Path("runbook.md")])
    service = KnowledgeService(settings, repository_root=tmp_path, embeddings=TinyEmbeddings())

    assert service.search("anything") == []
