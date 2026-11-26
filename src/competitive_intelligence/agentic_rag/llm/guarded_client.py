"""Concurrency- and rate-limit-aware wrapper around a BaseLLMClient.

Live LLM calls from the API share one process-wide guard:
- A semaphore caps concurrent provider calls (free tiers have tight per-minute token
  limits, and a burst of parallel calls just converts into 429s).
- Waiting for a slot is bounded; a caller that cannot get one in time fails fast with
  ``LLMRateLimitError`` instead of queueing indefinitely.
- After the provider returns a 429, calls fail fast for a cool-down period instead of
  repeatedly hitting the provider.

Failing with ``LLMRateLimitError`` lets the callers' existing fallbacks take over (the
agentic retriever degrades to deterministic retrieval; analyzers use rule-based paths).
"""

import asyncio
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel

from competitive_intelligence.agentic_rag.llm.errors import LLMRateLimitError
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")


class LLMCallGuard:
    """Shared concurrency limit and 429 cool-down state."""

    def __init__(
        self,
        max_concurrency: int,
        queue_timeout_seconds: float,
        cooldown_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self.queue_timeout_seconds = queue_timeout_seconds
        self.cooldown_seconds = cooldown_seconds
        self._monotonic = monotonic
        self._cooldown_until = 0.0
        self._lock = threading.Lock()
        # asyncio primitives bind to one event loop; keep one semaphore per loop.
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = weakref.WeakKeyDictionary()

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._lock:
            sem = self._semaphores.get(loop)
            if sem is None:
                sem = asyncio.Semaphore(self.max_concurrency)
                self._semaphores[loop] = sem
            return sem

    @property
    def cooling_down_seconds(self) -> float:
        return max(0.0, self._cooldown_until - self._monotonic())

    def start_cooldown(self) -> None:
        with self._lock:
            self._cooldown_until = max(
                self._cooldown_until, self._monotonic() + self.cooldown_seconds
            )

    def check_cooldown(self) -> None:
        remaining = self.cooling_down_seconds
        if remaining > 0:
            raise LLMRateLimitError(
                f"LLM provider is rate limited; cooling down for {remaining:.0f}s"
            )

    async def acquire(self) -> asyncio.Semaphore:
        sem = self._semaphore()
        try:
            await asyncio.wait_for(sem.acquire(), timeout=self.queue_timeout_seconds)
        except TimeoutError as err:
            raise LLMRateLimitError(
                f"LLM is busy: no free slot within {self.queue_timeout_seconds:.0f}s"
            ) from err
        return sem


class GuardedLLMClient(BaseLLMClient):
    """Delegates to ``inner`` under an ``LLMCallGuard``."""

    def __init__(self, inner: BaseLLMClient, guard: LLMCallGuard) -> None:
        self.inner = inner
        self.guard = guard

    @property
    def provider_name(self) -> str:
        return self.inner.provider_name

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    async def _guarded(self, call: Callable[[], Awaitable[R]]) -> R:
        self.guard.check_cooldown()
        sem = await self.guard.acquire()
        try:
            # A peer may have hit a 429 while this call waited for a slot. Rejecting here
            # must not extend the cool-down, so it happens before the provider call.
            self.guard.check_cooldown()
            try:
                return await call()
            except LLMRateLimitError:
                self.guard.start_cooldown()
                logger.warning(
                    "LLM provider rate limited; cooling down",
                    cooldown_seconds=self.guard.cooldown_seconds,
                )
                raise
        finally:
            sem.release()

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        return await self._guarded(
            lambda: self.inner.generate_structured(
                prompt=prompt,
                response_schema=response_schema,
                system_prompt=system_prompt,
                temperature=temperature,
                budget_tracker=budget_tracker,
            )
        )

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        return await self._guarded(
            lambda: self.inner.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                budget_tracker=budget_tracker,
            )
        )


_guard: LLMCallGuard | None = None
_guard_lock = threading.Lock()


def get_llm_call_guard() -> LLMCallGuard:
    """Process-wide guard built from settings on first use."""
    from competitive_intelligence.core.config import get_settings

    global _guard
    with _guard_lock:
        if _guard is None:
            settings = get_settings()
            _guard = LLMCallGuard(
                max_concurrency=settings.llm_max_concurrency,
                queue_timeout_seconds=settings.llm_queue_timeout_seconds,
                cooldown_seconds=settings.llm_rate_limit_cooldown_seconds,
            )
        return _guard


def reset_llm_call_guard() -> None:
    global _guard
    with _guard_lock:
        _guard = None
