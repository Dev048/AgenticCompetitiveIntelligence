"""Global pytest fixtures."""

import pytest

from competitive_intelligence.core.config import Settings


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
