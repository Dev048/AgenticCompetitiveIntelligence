"""Global pytest fixtures."""

import pytest

from competitive_intelligence.core.config import Settings, get_settings

# Hermetic test runs: never read the developer's local `.env`. Without this, a local
# `.env` pointing at a real LLM provider or a running Qdrant/Postgres leaks into tests,
# causing real (billed, rate-limited) LLM calls and non-deterministic failures.
Settings.model_config["env_file"] = None
get_settings.cache_clear()


@pytest.fixture
def test_settings() -> Settings:
    """Provide isolated test settings instance."""
    return Settings(
        environment="test",
        debug=True,
        max_llm_calls_per_task=5,
        max_retrieval_hops=2,
        task_token_budget=5000,
    )
