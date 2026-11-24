"""Data models and configuration schemas for the LLM abstraction subsystem."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class LLMConfig(BaseModel):
    """Configuration for LLM client instances."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)

    @field_validator("model")
    @classmethod
    def validate_non_empty_model(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Model name cannot be empty")
        return value.strip()
