"""Tests for Production Configuration, Deployment artifacts, and Application Lifecycle."""

import os
from pathlib import Path
import pytest
from sqlalchemy import text
import yaml

from competitive_intelligence.application.api.deps import (
    get_embedding_service,
    get_vector_store,
)
from competitive_intelligence.core.config import (
    ProductionConfigurationError,
    Settings,
)
from competitive_intelligence.knowledge_pipeline.vector_store.qdrant_store import (
    QdrantVectorStore,
)


# ==============================================================================
# 1. Production Configuration Validation Tests
# ==============================================================================


def test_production_settings_validation_rejects_mock_llm() -> None:
    """Production environment must strictly reject mock LLM provider."""
    settings = Settings(
        environment="production",
        llm_provider="mock",
        embedding_provider="bge_m3",
        database_url="postgresql+asyncpg://postgres:postgres@postgres:5432/ci",
        qdrant_url="http://qdrant:6333",
    )
    with pytest.raises(ProductionConfigurationError) as exc_info:
        settings.validate_production_configuration(raise_exception=True)
    assert "Mock LLM provider is not permitted in production" in str(exc_info.value)


def test_production_settings_validation_rejects_missing_openai_key() -> None:
    """Production environment must reject openai provider when OPENAI_API_KEY is missing."""
    settings = Settings(
        environment="production",
        llm_provider="openai",
        openai_api_key=None,
        embedding_provider="bge_m3",
        database_url="postgresql+asyncpg://postgres:postgres@postgres:5432/ci",
        qdrant_url="http://qdrant:6333",
    )
    with pytest.raises(ProductionConfigurationError) as exc_info:
        settings.validate_production_configuration(raise_exception=True)
    assert "OPENAI_API_KEY must be provided" in str(exc_info.value)


def test_production_settings_validation_rejects_missing_anthropic_key() -> None:
    """Production environment must reject anthropic provider when ANTHROPIC_API_KEY is missing."""
    settings = Settings(
        environment="production",
        llm_provider="anthropic",
        anthropic_api_key=None,
        embedding_provider="bge_m3",
        database_url="postgresql+asyncpg://postgres:postgres@postgres:5432/ci",
        qdrant_url="http://qdrant:6333",
    )
    with pytest.raises(ProductionConfigurationError) as exc_info:
        settings.validate_production_configuration(raise_exception=True)
    assert "ANTHROPIC_API_KEY must be provided" in str(exc_info.value)


def test_production_settings_validation_rejects_mock_embeddings() -> None:
    """Production environment must reject mock embedding provider."""
    settings = Settings(
        environment="production",
        llm_provider="openai",
        openai_api_key="sk-test-mock-key-12345",
        embedding_provider="mock",
        database_url="postgresql+asyncpg://postgres:postgres@postgres:5432/ci",
        qdrant_url="http://qdrant:6333",
    )
    with pytest.raises(ProductionConfigurationError) as exc_info:
        settings.validate_production_configuration(raise_exception=True)
    assert "Mock embedding provider is not permitted in production" in str(exc_info.value)


def test_production_settings_validation_rejects_sqlite_database() -> None:
    """Production environment must reject sqlite or in-memory database."""
    settings = Settings(
        environment="production",
        llm_provider="openai",
        openai_api_key="sk-test-mock-key-12345",
        embedding_provider="bge_m3",
        database_url="sqlite+aiosqlite:///./test.db",
        qdrant_url="http://qdrant:6333",
    )
    with pytest.raises(ProductionConfigurationError) as exc_info:
        settings.validate_production_configuration(raise_exception=True)
    assert "SQLite database is not permitted in production" in str(exc_info.value)


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_production_settings_validation_rejects_embedded_qdrant(environment: str) -> None:
    """Production and staging must use an external Qdrant, never QDRANT_PATH."""
    settings = Settings(
        environment=environment,
        llm_provider="openai",
        openai_api_key="sk-test-mock-key-12345",
        embedding_provider="bge_m3",
        database_url="postgresql+asyncpg://u:p@db:5432/ci",
        qdrant_url="http://qdrant:6333",
        qdrant_path="/tmp/qdrant",
    )
    with pytest.raises(ProductionConfigurationError) as exc_info:
        settings.validate_production_configuration(raise_exception=True)
    assert "Embedded Qdrant (QDRANT_PATH) is not permitted" in str(exc_info.value)


def test_production_settings_validation_rejects_memory_qdrant() -> None:
    """Production environment must reject in-memory vector storage."""
    settings = Settings(
        environment="production",
        llm_provider="openai",
        openai_api_key="sk-test-mock-key-12345",
        embedding_provider="bge_m3",
        database_url="postgresql+asyncpg://postgres:postgres@postgres:5432/ci",
        qdrant_location=":memory:",
    )
    with pytest.raises(ProductionConfigurationError) as exc_info:
        settings.validate_production_configuration(raise_exception=True)
    assert "In-memory Qdrant (:memory:) is not permitted" in str(exc_info.value)


def test_production_settings_validation_rejects_debug_mode() -> None:
    """Production environment must have debug disabled."""
    settings = Settings(
        environment="production",
        debug=True,
        llm_provider="openai",
        openai_api_key="sk-test-mock-key-12345",
        embedding_provider="bge_m3",
        database_url="postgresql+asyncpg://postgres:postgres@postgres:5432/ci",
        qdrant_url="http://qdrant:6333",
    )
    with pytest.raises(ProductionConfigurationError) as exc_info:
        settings.validate_production_configuration(raise_exception=True)
    assert "DEBUG mode must be disabled in production" in str(exc_info.value)


def test_production_settings_validation_accepts_valid_config() -> None:
    """Production configuration with all valid parameters passes cleanly."""
    settings = Settings(
        environment="production",
        debug=False,
        llm_provider="openai",
        openai_api_key="sk-test-production-key-valid",
        embedding_provider="bge_m3",
        database_url="postgresql+asyncpg://postgres:postgres@postgres:5432/competitive_intelligence",
        qdrant_url="http://qdrant:6333",
        qdrant_collection_name="ci_knowledge_chunks",
    )
    errors = settings.validate_production_configuration(raise_exception=True)
    assert len(errors) == 0


def test_test_environment_permits_mock_providers_and_sqlite() -> None:
    """Test and development environments permit mock LLM, mock embeddings, and sqlite."""
    settings = Settings(
        environment="test",
        debug=True,
        llm_provider="mock",
        embedding_provider="mock",
        database_url="sqlite+aiosqlite:///:memory:",
        qdrant_location=":memory:",
    )
    errors = settings.validate_production_configuration(raise_exception=False)
    assert len(errors) == 0


# ==============================================================================
# 2. Dependency Wiring Tests
# ==============================================================================


def test_get_vector_store_production_wiring() -> None:
    """get_vector_store instantiates real Qdrant adapter with production URL."""
    settings = Settings(
        environment="production",
        qdrant_url="http://qdrant.infra:6333",
        qdrant_collection_name="ci_knowledge_prod",
    )
    store = get_vector_store(settings)
    assert isinstance(store, QdrantVectorStore)
    assert store.config.url == "http://qdrant.infra:6333"
    assert store.config.collection_name == "ci_knowledge_prod"
    assert store.config.location is None


def test_get_embedding_service_rejects_mock_in_production() -> None:
    """get_embedding_service raises ValueError if mock embedding is requested in production."""
    settings = Settings(
        environment="production",
        embedding_provider="mock",
    )
    with pytest.raises(ValueError) as exc:
        get_embedding_service(settings)
    assert "Mock embedding provider is not permitted in production" in str(exc.value)


# ==============================================================================
# 3. Docker and Compose Artifact Verification Tests
# ==============================================================================


def test_dockerfile_exists_and_contains_invariants() -> None:
    """Verify Dockerfile specifies non-root user, entrypoint, and healthcheck."""
    dockerfile_path = Path(__file__).resolve().parent.parent.parent / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist at repository root"
    content = dockerfile_path.read_text(encoding="utf-8")

    assert "python:3.11-slim" in content
    assert "useradd" in content and "appuser" in content
    assert "USER appuser" in content
    assert "ENTRYPOINT" in content
    assert "HEALTHCHECK" in content
    assert "EXPOSE 8000" in content


def test_docker_compose_structure() -> None:
    """Verify docker-compose.yml defines postgres, qdrant, api, worker, and volumes."""
    compose_path = Path(__file__).resolve().parent.parent.parent / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist at repository root"
    content = compose_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    services = data.get("services", {})
    assert "postgres" in services, "postgres service must be defined"
    assert "qdrant" in services, "qdrant service must be defined"
    assert "api" in services, "api service must be defined"
    assert "worker" in services, "worker service must be defined"

    # Verify persistent volumes
    volumes = data.get("volumes", {})
    assert "postgres_data" in volumes
    assert "qdrant_data" in volumes

    # Verify dependency order
    api_depends = services["api"].get("depends_on", {})
    assert "postgres" in api_depends
    assert "qdrant" in api_depends


def test_env_example_contains_all_production_variables() -> None:
    """Verify .env.example defines all required configuration variables."""
    env_example_path = Path(__file__).resolve().parent.parent.parent / ".env.example"
    assert env_example_path.exists()
    content = env_example_path.read_text(encoding="utf-8")

    required_vars = [
        "ENVIRONMENT",
        "DATABASE_URL",
        "QDRANT_URL",
        "QDRANT_COLLECTION_NAME",
        "EMBEDDING_PROVIDER",
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "MONITORING_MODE",
    ]
    for var in required_vars:
        assert var in content, f"{var} must be documented in .env.example"
