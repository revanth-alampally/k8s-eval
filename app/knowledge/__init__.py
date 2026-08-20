"""Repository-owned documentation retrieval.

Knowledge retrieval supplies static guidance (runbooks, design notes, troubleshooting
docs). It is deliberately separate from Kubernetes tools, which establish live state.
"""

from app.knowledge.service import KnowledgeHit, KnowledgeService, RefreshResult

__all__ = ["KnowledgeHit", "KnowledgeService", "RefreshResult"]
