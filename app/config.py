"""Application configuration.

Settings are read from environment variables (prefix ``KAGENT_``) and an optional
``.env`` file. ``get_settings()`` is cached so the rest of the codebase can treat
configuration as an immutable singleton, while tests can override it through the
FastAPI dependency system or by clearing the cache.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


class LogFormat(StrEnum):
    JSON = "json"
    CONSOLE = "console"


class LLMProviderName(StrEnum):
    # Keyword-driven stand-in: runs the real tools, needs no API key.
    FAKE = "fake"
    HEURISTIC = "heuristic"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KAGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    service_name: str = "k8s-ops-agent"
    environment: Environment = Environment.LOCAL

    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.CONSOLE

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- Kubernetes -------------------------------------------------------
    # None means "use the default resolution order": in-cluster config first,
    # then $KUBECONFIG, then ~/.kube/config.
    kubeconfig_path: Path | None = None
    kube_context: str | None = None
    # Hard boundary enforced before any tool touches the cluster.
    allowed_namespaces: list[str] = Field(default_factory=lambda: ["ai-agent-demo"])
    kube_request_timeout_seconds: int = 10

    # --- Safety -----------------------------------------------------------
    # Global kill switch: when true, mutating tools are not even registered.
    read_only_mode: bool = False
    require_confirmation: bool = True
    confirmation_ttl_seconds: int = 300

    # --- LLM / agent ------------------------------------------------------
    llm_provider: LLMProviderName = LLMProviderName.FAKE
    llm_model: str = "gpt-4o-mini"
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: int = 30
    llm_temperature: float = 0.0
    # Bounds the agent loop so a confused model cannot spin against the cluster.
    max_tool_calls_per_request: int = 5
    max_log_lines: int = 200

    # --- Knowledge / RAG --------------------------------------------------
    rag_enabled: bool = True
    # Repository-relative Markdown roots. System and global Cursor skills are not
    # indexed unless explicitly added here.
    rag_corpus_paths: list[Path] = Field(default_factory=lambda: [Path("README.md")])
    rag_persist_directory: Path = Path(".data/chroma")
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_chunk_size: int = 1800
    rag_chunk_overlap: int = 240
    rag_top_k: int = 4

    @field_validator("allowed_namespaces", mode="before")
    @classmethod
    def _split_namespaces(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("rag_corpus_paths", mode="before")
    @classmethod
    def _split_corpus_paths(cls, value: object) -> object:
        if isinstance(value, str):
            return [Path(item.strip()) for item in value.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {value}")
        return level

    def is_namespace_allowed(self, namespace: str) -> bool:
        return namespace in self.allowed_namespaces

    @property
    def default_namespace(self) -> str:
        return self.allowed_namespaces[0] if self.allowed_namespaces else "default"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
