"""OpenAI-compatible LLM client implementing BaseLLMClient."""

import asyncio
import json
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


class OpenAILLMClient(BaseLLMClient):
    """OpenAI API-compatible client for structured JSON generation with budget tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url.rstrip("/")
        self.model = model or settings.default_llm_model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send POST request to OpenAI completions endpoint with bounded retries."""
        if not self.api_key:
            raise LLMAuthError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable or pass api_key."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"

        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers=headers,
                    )

                if response.status_code == 200:
                    try:
                        return response.json()  # type: ignore[no-any-return]
                    except Exception as exc:
                        raise LLMResponseMalformedError(f"Malformed JSON response: {response.text}") from exc

                if response.status_code in (401, 403):
                    raise LLMAuthError(
                        f"OpenAI authentication failed ({response.status_code}): {response.text}"
                    )
                if response.status_code == 400:
                    raise LLMBadRequestError(
                        f"OpenAI rejected request (400): {response.text}"
                    )
                if response.status_code == 429:
                    if attempt <= self.max_retries:
                        backoff = 0.5 * (2 ** (attempt - 1))
                        logger.warning("OpenAI rate limited, retrying", attempt=attempt, backoff=backoff)
                        await asyncio.sleep(backoff)
                        continue
                    raise LLMRateLimitError(
                        f"OpenAI rate limit exceeded (429) after {self.max_retries} retries: {response.text}"
                    )

                if response.status_code in (500, 502, 503, 504):
                    if attempt <= self.max_retries:
                        backoff = 0.5 * (2 ** (attempt - 1))
                        logger.warning("OpenAI server error, retrying", status=response.status_code, attempt=attempt)
                        await asyncio.sleep(backoff)
                        continue
                    raise LLMUnavailableError(
                        f"OpenAI server error ({response.status_code}): {response.text}"
                    )

                raise LLMUnavailableError(
                    f"LLM API returned error status {response.status_code}: {response.text}"
                )

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                raise LLMTimeoutError(f"LLM request timed out after {self.timeout_seconds}s") from exc
            except httpx.RequestError as exc:
                last_error = exc
                if attempt <= self.max_retries:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    await asyncio.sleep(backoff)
                    continue
                raise LLMUnavailableError(f"Failed to connect to LLM provider: {exc}") from exc

        raise LLMUnavailableError(f"OpenAI request failed after {self.max_retries} retries: {last_error}")

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        """Call OpenAI chat completions with structured JSON output, parsing into response_schema."""
        messages: list[dict[str, str]] = []
        schema_json = json.dumps(response_schema.model_json_schema())
        sys_msg = (
            (system_prompt or "You are a competitive intelligence assistant.")
            + f"\nYou MUST respond strictly in valid JSON adhering to the following JSON schema:\n{schema_json}"
        )
        messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        data = await self._post_with_retries(payload)

        try:
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        except (KeyError, IndexError) as exc:
            raise LLMResponseMalformedError(f"Malformed API response envelope: {data}") from exc

        if budget_tracker:
            budget_tracker.record_llm_call(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        try:
            return response_schema.model_validate_json(content)
        except ValidationError as exc:
            logger.warning("LLM response validation failed", raw_content=content, error=str(exc))
            raise LLMResponseMalformedError(
                f"LLM response failed validation against {response_schema.__name__}: {exc}"
            ) from exc

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        """Call OpenAI chat completions for raw text output."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        data = await self._post_with_retries(payload)

        try:
            content: str = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        except (KeyError, IndexError) as exc:
            raise LLMResponseMalformedError(f"Malformed API response envelope: {data}") from exc

        if budget_tracker:
            budget_tracker.record_llm_call(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return content
