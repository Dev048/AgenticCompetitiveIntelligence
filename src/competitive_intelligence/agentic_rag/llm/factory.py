"""Factory for instantiating configured LLM clients."""

from competitive_intelligence.agentic_rag.llm.anthropic_client import AnthropicLLMClient
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.llm.models import LLMConfig, LLMProvider
from competitive_intelligence.agentic_rag.llm.openai_client import OpenAILLMClient
from competitive_intelligence.core.config import get_settings


def llm_config_from_settings(max_retries: int | None = None) -> LLMConfig:
    """Build the LLMConfig for the configured provider (optionally overriding retries)."""
    settings = get_settings()
    provider_val = getattr(settings, "llm_provider", "openai")
    try:
        provider = LLMProvider(str(provider_val).lower())
    except ValueError:
        provider = LLMProvider.OPENAI

    if provider == LLMProvider.ANTHROPIC:
        model_name = settings.anthropic_model
        api_key = settings.anthropic_api_key
        base_url = None
    else:
        model_name = settings.default_llm_model
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url

    return LLMConfig(
        provider=provider,
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=getattr(settings, "llm_timeout_seconds", 30.0),
        max_retries=(
            max_retries if max_retries is not None else getattr(settings, "llm_max_retries", 3)
        ),
        temperature=getattr(settings, "llm_temperature", 0.0),
    )


def create_llm_client(config: LLMConfig | None = None) -> BaseLLMClient:
    """Create a configured BaseLLMClient instance based on settings or supplied config.

    Args:
        config: Optional LLMConfig instance. If None, builds from application settings.

    Returns:
        Instance conforming to BaseLLMClient.
    """
    settings = get_settings()

    if config is None:
        config = llm_config_from_settings()

    if config.provider == LLMProvider.MOCK:
        if settings.environment == "production":
            raise ValueError("Mock LLM provider is not permitted in production environment")
        return MockLLMClient(model=config.model)
    elif config.provider == LLMProvider.ANTHROPIC:
        return AnthropicLLMClient(
            api_key=config.api_key or settings.anthropic_api_key,
            base_url=config.base_url or "https://api.anthropic.com/v1",
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    elif config.provider == LLMProvider.OPENAI:
        return OpenAILLMClient(
            api_key=config.api_key or settings.openai_api_key,
            base_url=config.base_url or settings.openai_base_url or "https://api.openai.com/v1",
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
