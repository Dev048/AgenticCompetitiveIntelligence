"""Anthropic Claude-compatible LLM client implementing BaseLLMClient."""

import asyncio
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from competitive_intelligence.agentic_rag.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMRateLimitError,
    LLMResponseMalformedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.config import get_settings
from competitive_intelligence.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class AnthropicLLMClient(BaseLLMClient):
    """Anthropic Messages API client for structured output and text generation."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        model: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.anthropic_api_key
        self.base_url = base_url.rstrip("/")
        self.model: str = str(model or getattr(settings, "anthropic_model", "claude-3-5-sonnet-20241022"))
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self.model

    async def _post_with_retries(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send POST request to Anthropic messages endpoint with bounded retries."""
        if not self.api_key:
            raise LLMAuthError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable or pass api_key."
            )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        url = f"{self.base_url}/messages"
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(url, json=payload, headers=headers)

                if response.status_code == 200:
                    try:
                        return response.json()  # type: ignore[no-any-return]
                    except Exception as exc:
                        raise LLMResponseMalformedError(f"Malformed JSON response: {response.text}") from exc

                if response.status_code in (401, 403):
                    raise LLMAuthError(
                        f"Anthropic authentication failed ({response.status_code}): {response.text}"
                    )
                if response.status_code == 400:
                    raise LLMBadRequestError(
                        f"Anthropic rejected request (400): {response.text}"
                    )
                if response.status_code == 429:
                    if attempt <= self.max_retries:
                        backoff = 0.5 * (2 ** (attempt - 1))
                        logger.warning("Anthropic rate limited, retrying", attempt=attempt, backoff=backoff)
                        await asyncio.sleep(backoff)
                        continue
                    raise LLMRateLimitError(
                        f"Anthropic rate limit exceeded (429) after {self.max_retries} retries: {response.text}"
                    )

                if response.status_code in (500, 502, 503, 504):
                    if attempt <= self.max_retries:
                        backoff = 0.5 * (2 ** (attempt - 1))
                        logger.warning("Anthropic server error, retrying", status=response.status_code, attempt=attempt)
                        await asyncio.sleep(backoff)
                        continue
                    raise LLMUnavailableError(
                        f"Anthropic server error ({response.status_code}): {response.text}"
                    )

                raise LLMUnavailableError(
                    f"Anthropic API returned error status {response.status_code}: {response.text}"
                )

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                raise LLMTimeoutError(f"Anthropic request timed out after {self.timeout_seconds}s") from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                raise LLMUnavailableError(f"Failed to connect to Anthropic provider: {exc}") from exc

        raise LLMUnavailableError(f"Anthropic request failed after {self.max_retries} retries: {last_error}")

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        """Call Anthropic with forced tool use to return strictly validated structured output."""
        tool_name = "record_output"
        tools = [
            {
                "name": tool_name,
                "description": f"Output structured response adhering to {response_schema.__name__}",
                "input_schema": response_schema.model_json_schema(),
            }
        ]

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
            "tool_choice": {"type": "tool", "name": tool_name},
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        data = await self._post_with_retries(payload)

        # Record token budget
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        if budget_tracker:
            budget_tracker.record_llm_call(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        content_blocks = data.get("content", [])
        tool_input: dict[str, Any] | None = None
        for block in content_blocks:
            if block.get("type") == "tool_use" and block.get("name") == tool_name:
                tool_input = block.get("input")
                break

        if tool_input is None:
            raise LLMResponseMalformedError(
                f"Anthropic response did not contain expected tool_use block '{tool_name}': {data}"
            )

        try:
            return response_schema.model_validate(tool_input)
        except ValidationError as exc:
            logger.warning("Anthropic response schema validation failed", raw_input=tool_input, error=str(exc))
            raise LLMResponseMalformedError(
                f"Anthropic output failed validation against {response_schema.__name__}: {exc}"
            ) from exc

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        """Call Anthropic for text generation."""
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        data = await self._post_with_retries(payload)

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        if budget_tracker:
            budget_tracker.record_llm_call(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        content_blocks = data.get("content", [])
        text_parts: list[str] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return "".join(text_parts)
