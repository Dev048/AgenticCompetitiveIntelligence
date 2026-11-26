"""System configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProductionConfigurationError(ValueError):
    """Raised when production environment is missing required services or API keys."""


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General & Network
    environment: Literal["development", "staging", "production", "test", "demo"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    project_name: str = "Competitive Intelligence Agent"
    api_v1_str: str = "/api/v1"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    metrics_enabled: bool = True
    metrics_token: str | None = None

    # LLM Settings
    llm_provider: Literal["openai", "anthropic", "mock"] = "mock"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    default_llm_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    reasoning_llm_model: str = "gpt-4o"
    embedding_model: str = "BAAI/bge-m3"
    embedding_provider: Literal["bge_m3", "mock"] = "mock"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    llm_timeout_seconds: float = Field(default=30.0, gt=0.0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Task Budgets
    max_llm_calls_per_task: int = Field(default=10, ge=1, le=50)
    max_retrieval_hops: int = Field(default=3, ge=1, le=10)
    task_token_budget: int = Field(default=30000, ge=1000)

    # Vector Storage (Qdrant)
    qdrant_url: str | None = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "ci_knowledge_chunks"
    qdrant_timeout: int = Field(default=10, ge=1)
    qdrant_location: str | None = None
    # Embedded (on-disk, in-process) Qdrant storage folder. Used by the single-container demo.
    qdrant_path: str | None = None

    # Acquisition
    crawl4ai_headless: bool = True
    agent_reach_api_key: str | None = None

    # Persistence
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/competitive_intelligence"
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=20, ge=0, le=100)
    db_pool_timeout: int = Field(default=30, ge=1)
    db_echo: bool = False
    redis_url: str | None = None

    # Continuous Monitoring & Scheduler
    monitoring_enabled: bool = True
    monitoring_mode: Literal["in_process", "worker", "disabled"] = "in_process"
    monitoring_poll_interval_seconds: float = Field(default=15.0, ge=1.0)
    monitoring_batch_size: int = Field(default=10, ge=1, le=100)
    monitoring_max_concurrency: int = Field(default=5, ge=1, le=50)
    monitoring_lease_timeout_seconds: int = Field(default=300, ge=10)
    monitoring_heartbeat_interval_seconds: int = Field(default=60, ge=5)

    # Public read-only demo (see DEMO_PLAN.md). DEMO_MODE blocks writes and rate-limits queries.
    demo_mode: bool = False
    demo_allow_recommendation_review: bool = True
    demo_query_per_ip_per_minute: int = Field(default=5, ge=1, le=1000)
    demo_query_daily_cap: int = Field(default=200, ge=1)
    # Reverse proxies in front of the app that append to X-Forwarded-For (Spaces: 1).
    demo_trusted_proxy_hops: int = Field(default=1, ge=0, le=5)
    # Seed the demo workspace at startup from recorded LLM fixtures (never the live LLM).
    demo_seed_on_startup: bool = True
    demo_llm_fixtures_path: str | None = None

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def normalize_openai_base_url(cls, value: str | None) -> str | None:
        """Strip whitespace and trailing slashes from OpenAI-compatible base URL."""
        if value is None:
            return None
        cleaned = str(value).strip().rstrip("/")
        return cleaned if cleaned else None

    @property
    def effective_qdrant_url(self) -> str:
        """Resolve effective Qdrant HTTP URL."""
        if self.qdrant_url and self.qdrant_url.strip():
            return self.qdrant_url.strip()
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    def _validate_real_model_providers(self, errors: list[str]) -> None:
        """Reject mock LLM/embedding providers and require credentials for the active LLM."""
        if self.llm_provider == "mock":
            errors.append(
                f"Mock LLM provider is not permitted in {self.environment}. "
                "Configure 'openai' or 'anthropic' with valid credentials."
            )
        elif self.llm_provider == "openai":
            if not self.openai_api_key or not self.openai_api_key.strip():
                errors.append("OPENAI_API_KEY must be provided when llm_provider is 'openai'.")
        elif self.llm_provider == "anthropic":
            if not self.anthropic_api_key or not self.anthropic_api_key.strip():
                errors.append("ANTHROPIC_API_KEY must be provided when llm_provider is 'anthropic'.")

        if self.embedding_provider == "mock":
            errors.append(
                f"Mock embedding provider is not permitted in {self.environment}. "
                "Configure 'bge_m3' for production embeddings."
            )

    def validate_production_configuration(self, raise_exception: bool = True) -> list[str]:
        """Validate deployment requirements for production, staging and the public demo.

        Production/staging enforce:
        - Debug mode is disabled.
        - Mock LLM provider is rejected.
        - API keys for active LLM provider are supplied.
        - Mock embedding provider is rejected.
        - SQLite or in-memory database is rejected.
        - In-memory vector store is rejected.
        - Collection name is non-empty.

        Demo (single container, public) enforces:
        - Debug mode is disabled and DEMO_MODE is on (public deployments must be read-only).
        - Real LLM and embedding providers (same rules as production).
        - SQLite and embedded/in-memory Qdrant ARE permitted.
        """
        errors: list[str] = []

        if self.environment == "demo":
            if self.debug:
                errors.append("DEBUG mode must be disabled in demo.")
            if not self.demo_mode:
                errors.append("DEMO_MODE must be enabled when ENVIRONMENT=demo (public read-only).")
            self._validate_real_model_providers(errors)
            if not self.qdrant_collection_name or not self.qdrant_collection_name.strip():
                errors.append("QDRANT_COLLECTION_NAME must be specified.")

        if self.environment in ("production", "staging"):
            if self.debug:
                errors.append("DEBUG mode must be disabled in production/staging.")

            # Persistence validation
            db_lower = self.database_url.lower()
            if "sqlite" in db_lower or ":memory:" in db_lower:
                errors.append(
                    f"SQLite database is not permitted in {self.environment}. "
                    "A production PostgreSQL URL must be configured."
                )

            # LLM and embedding provider validation
            self._validate_real_model_providers(errors)

            # Qdrant validation
            if self.qdrant_location == ":memory:":
                errors.append(
                    f"In-memory Qdrant (:memory:) is not permitted in {self.environment}. "
                    "Configure external QDRANT_URL or QDRANT_HOST."
                )
            if self.qdrant_path:
                errors.append(
                    f"Embedded Qdrant (QDRANT_PATH) is not permitted in {self.environment}. "
                    "Configure external QDRANT_URL or QDRANT_HOST."
                )
            if not self.qdrant_collection_name or not self.qdrant_collection_name.strip():
                errors.append("QDRANT_COLLECTION_NAME must be specified.")

        if errors and raise_exception:
            formatted_errors = "\n  - " + "\n  - ".join(errors)
            raise ProductionConfigurationError(
                f"Production configuration validation failed for environment '{self.environment}':{formatted_errors}"
            )

        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of Settings."""
    return Settings()

