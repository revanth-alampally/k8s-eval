"""CLI entry point for explicit knowledge-index refreshes."""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.knowledge.service import KnowledgeService


def main() -> None:
    settings = get_settings()
    result = KnowledgeService(settings, repository_root=Path.cwd()).refresh()
    print(result.model_dump_json())


if __name__ == "__main__":
    main()
