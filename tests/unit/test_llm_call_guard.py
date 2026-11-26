"""LLMCallGuard / GuardedLLMClient: concurrency cap, bounded queueing, 429 cool-down."""

import asyncio

import pytest
from pydantic import BaseModel

from competitive_intelligence.agentic_rag.llm.errors import LLMRateLimitError
from competitive_intelligence.agentic_rag.llm.guarded_client import GuardedLLMClient, LLMCallGuard
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient, T
from competitive_intelligence.core.budgets import LLMBudgetTracker


class _Answer(BaseModel):
    text: str = "ok"


class _FakeLLM(BaseLLMClient):
    def __init__(self, delay: float = 0.0, fail_with: Exception | None = None) -> None:
        self.delay = delay
        self.fail_with = fail_with
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if self.fail_with is not None:
                raise self.fail_with
            return response_schema()
        finally:
            self.in_flight -= 1

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        return "ok"


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _guard(**kwargs: object) -> LLMCallGuard:
    values: dict[str, object] = {
        "max_concurrency": 2,
        "queue_timeout_seconds": 5.0,
        "cooldown_seconds": 30.0,
    }
    values.update(kwargs)
    return LLMCallGuard(**values)  # type: ignore[arg-type]


async def test_concurrency_is_capped() -> None:
    inner = _FakeLLM(delay=0.05)
    client = GuardedLLMClient(inner, _guard(max_concurrency=2))

    await asyncio.gather(*(client.generate_structured("q", _Answer) for _ in range(6)))

    assert inner.calls == 6
    assert inner.max_in_flight == 2


async def test_waiting_for_a_slot_is_bounded() -> None:
    inner = _FakeLLM(delay=0.5)
    client = GuardedLLMClient(inner, _guard(max_concurrency=1, queue_timeout_seconds=0.05))

    results = await asyncio.gather(
        client.generate_structured("a", _Answer),
        client.generate_structured("b", _Answer),
        return_exceptions=True,
    )

    assert isinstance(results[0], _Answer)
    assert isinstance(results[1], LLMRateLimitError)
    assert "busy" in str(results[1])


async def test_provider_429_starts_cooldown_and_fails_fast() -> None:
    clock = _Clock()
    inner = _FakeLLM(fail_with=LLMRateLimitError("429 from provider"))
    guard = _guard(cooldown_seconds=30.0, monotonic=clock)
    client = GuardedLLMClient(inner, guard)

    with pytest.raises(LLMRateLimitError):
        await client.generate_structured("q", _Answer)
    assert inner.calls == 1

    # During the cool-down the provider is not called at all.
    clock.now += 10
    with pytest.raises(LLMRateLimitError, match="cooling down"):
        await client.generate_structured("q", _Answer)
    assert inner.calls == 1
    # Rejected calls do not extend the cool-down.
    assert guard.cooling_down_seconds == pytest.approx(20.0)

    inner.fail_with = None
    clock.now += 21
    assert isinstance(await client.generate_structured("q", _Answer), _Answer)
    assert inner.calls == 2


async def test_slot_is_released_after_failures() -> None:
    inner = _FakeLLM(fail_with=RuntimeError("boom"))
    client = GuardedLLMClient(inner, _guard(max_concurrency=1, queue_timeout_seconds=0.1))

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await client.generate_structured("q", _Answer)
    assert inner.calls == 3


def test_guard_works_across_event_loops() -> None:
    guard = _guard(max_concurrency=1)

    async def _once() -> _Answer:
        return await GuardedLLMClient(_FakeLLM(), guard).generate_structured("q", _Answer)

    assert isinstance(asyncio.run(_once()), _Answer)
    assert isinstance(asyncio.run(_once()), _Answer)
