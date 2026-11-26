"""Record/replay LLM client used to seed the public demo without spending LLM quota.

The demo workspace is seeded when the demo image is built (and again at boot only if
the baked data is missing). Calling the real LLM each time would burn the free-tier quota and make the demo depend on provider uptime, so
LLM responses are recorded once against the real provider and committed as fixtures.

Prompts are not byte-identical between boots: version/chunk IDs are random UUIDs and
retrieval scores can drift slightly across CPUs. Before hashing, every generated ID in the
prompt is replaced by an ordered placeholder (``<ID1>``, ``<ID2>``...) and scores are
masked. Recorded responses are stored in the same placeholder form and, on replay, the
placeholders are mapped back to the IDs present in the *current* prompt. Citations
therefore stay grounded in the current boot's real chunk/event/insight IDs and still pass
the pipeline's citation validators.

Lookup order on replay: exact normalized-prompt hash, then (seed step, schema, call
ordinal) as a fallback for prompts whose retrieved evidence differs slightly. A miss raises
``LLMUnavailableError`` so every stage takes its existing deterministic fallback path.
"""

import asyncio
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.llm.errors import (
    LLMRateLimitError,
    LLMUnavailableError,
)
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

ReplayMode = Literal["replay", "record"]

FIXTURE_FORMAT_VERSION = 1

_ID_PATTERN = re.compile(
    r"\b(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"|(?:ev|sig|ins|opt|rec|chunk|ver|doc)_[0-9a-f]{8,64})\b"
)
_SCORE_PATTERN = re.compile(r"(score[=:]\s*)-?\d+(?:\.\d+)?")
_PLACEHOLDER_PATTERN = re.compile(r"<ID(\d+)>")


def normalize_prompt(text: str) -> tuple[str, list[str]]:
    """Replace generated IDs with ordered placeholders and mask scores.

    Returns the normalized text and the IDs in order of first appearance
    (``ids[0]`` is ``<ID1>``).
    """
    ids: list[str] = []
    index: dict[str, int] = {}

    def _sub(match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in index:
            ids.append(value)
            index[value] = len(ids)
        return f"<ID{index[value]}>"

    normalized = _ID_PATTERN.sub(_sub, text)
    normalized = _SCORE_PATTERN.sub(r"\1<S>", normalized)
    return normalized, ids


def to_placeholders(text: str, ids: list[str]) -> str:
    """Rewrite known IDs in a recorded response as placeholders (longest first)."""
    for position, value in sorted(enumerate(ids, start=1), key=lambda p: -len(p[1])):
        text = text.replace(value, f"<ID{position}>")
    return text


def from_placeholders(text: str, ids: list[str]) -> str:
    """Map placeholders back to the current prompt's IDs.

    A placeholder with no counterpart in the current prompt is left as-is: it is not a
    valid ID, so the pipeline's citation validators reject it instead of fabricating one.
    """

    def _sub(match: re.Match[str]) -> str:
        position = int(match.group(1))
        return ids[position - 1] if 0 < position <= len(ids) else match.group(0)

    return _PLACEHOLDER_PATTERN.sub(_sub, text)


class FixtureEntry(BaseModel):
    schema_name: str
    step: str
    ordinal: int
    response: str  # JSON in placeholder form (or raw text for generate_text)
    kind: Literal["structured", "text"] = "structured"


class FixtureFile(BaseModel):
    format_version: int = FIXTURE_FORMAT_VERSION
    provider: str | None = None
    model: str | None = None
    entries: dict[str, FixtureEntry] = Field(default_factory=dict)


class ReplayStats(BaseModel):
    exact_hits: int = 0
    ordinal_hits: int = 0
    misses: int = 0
    recorded: int = 0


class LLMFixtureStore:
    """JSON-backed store of recorded LLM responses."""

    def __init__(self, path: Path, data: FixtureFile | None = None) -> None:
        self.path = path
        self.data = data or FixtureFile()
        self._by_ordinal: dict[str, str] = {}
        self._reindex()

    @classmethod
    def load(cls, path: Path) -> "LLMFixtureStore":
        if not path.exists():
            return cls(path)
        data = FixtureFile.model_validate_json(path.read_text(encoding="utf-8"))
        if data.format_version != FIXTURE_FORMAT_VERSION:
            raise ValueError(f"Unsupported LLM fixture format version {data.format_version}")
        return cls(path, data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.data.model_dump(mode="json")
        payload["entries"] = dict(sorted(payload["entries"].items()))
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")

    def get(self, key: str, ordinal_key: str) -> tuple[FixtureEntry | None, bool]:
        """Return (entry, exact_match)."""
        entry = self.data.entries.get(key)
        if entry is not None:
            return entry, True
        fallback = self._by_ordinal.get(ordinal_key)
        return (self.data.entries[fallback], False) if fallback else (None, False)

    def put(self, key: str, entry: FixtureEntry) -> None:
        self.data.entries[key] = entry
        self._by_ordinal[_ordinal_key(entry.step, entry.schema_name, entry.ordinal)] = key

    def _reindex(self) -> None:
        self._by_ordinal = {
            _ordinal_key(e.step, e.schema_name, e.ordinal): k for k, e in self.data.entries.items()
        }


def _ordinal_key(step: str, schema_name: str, ordinal: int) -> str:
    return f"{step}::{schema_name}::{ordinal}"


def fixture_key(schema_name: str, normalized_system: str, normalized_prompt: str) -> str:
    digest = hashlib.sha256()
    for part in (schema_name, normalized_system, normalized_prompt):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class ReplayLLMClient(BaseLLMClient):
    """BaseLLMClient that replays recorded responses (or records them from ``inner``)."""

    def __init__(
        self,
        store: LLMFixtureStore,
        mode: ReplayMode = "replay",
        inner: BaseLLMClient | None = None,
        record_rate_limit_retries: int = 6,
        record_rate_limit_backoff_seconds: float = 20.0,
    ) -> None:
        if mode == "record" and inner is None:
            raise ValueError("Record mode requires an inner LLM client")
        self.store = store
        self.mode = mode
        self.inner = inner
        self.record_rate_limit_retries = record_rate_limit_retries
        self.record_rate_limit_backoff_seconds = record_rate_limit_backoff_seconds
        self.stats = ReplayStats()
        self._step = "default"
        self._ordinals: dict[str, int] = defaultdict(int)
        if mode == "record" and inner is not None:
            self.store.data.provider = inner.provider_name
            self.store.data.model = inner.model_name

    @property
    def provider_name(self) -> str:
        return f"replay:{self.store.data.provider or 'unknown'}"

    @property
    def model_name(self) -> str:
        return self.store.data.model or "recorded"

    async def _patiently(self, call: Callable[[], Awaitable[R]]) -> R:
        """Recording only: wait out free-tier rate limits so the fixture set is complete."""
        for attempt in range(self.record_rate_limit_retries + 1):
            try:
                return await call()
            except LLMRateLimitError:
                if attempt == self.record_rate_limit_retries:
                    raise
                logger.info("Rate limited while recording; backing off", attempt=attempt + 1)
                await asyncio.sleep(self.record_rate_limit_backoff_seconds)
        raise AssertionError("unreachable")

    def begin_step(self, step: str) -> None:
        """Scope subsequent calls to a seed step (used for the ordinal fallback key)."""
        self._step = step
        self._ordinals = defaultdict(int)

    def _next_ordinal(self, schema_name: str) -> int:
        self._ordinals[schema_name] += 1
        return self._ordinals[schema_name]

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        schema_name = response_schema.__name__
        norm_system, _ = normalize_prompt(system_prompt or "")
        norm_prompt, ids = _renormalize(system_prompt or "", prompt)
        key = fixture_key(schema_name, norm_system, norm_prompt)
        ordinal = self._next_ordinal(schema_name)

        if self.mode == "record":
            assert self.inner is not None
            inner = self.inner
            response = await self._patiently(
                lambda: inner.generate_structured(
                    prompt=prompt,
                    response_schema=response_schema,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    budget_tracker=budget_tracker,
                )
            )
            self.store.put(
                key,
                FixtureEntry(
                    schema_name=schema_name,
                    step=self._step,
                    ordinal=ordinal,
                    response=to_placeholders(response.model_dump_json(), ids),
                ),
            )
            self.stats.recorded += 1
            return response

        entry, exact = self.store.get(key, _ordinal_key(self._step, schema_name, ordinal))
        if entry is None or entry.schema_name != schema_name:
            self.stats.misses += 1
            logger.warning(
                "No recorded LLM fixture; stage will use its deterministic fallback",
                step=self._step,
                schema=schema_name,
                ordinal=ordinal,
            )
            raise LLMUnavailableError(
                f"No recorded LLM response for {schema_name} (step={self._step}, call={ordinal})"
            )
        if exact:
            self.stats.exact_hits += 1
        else:
            self.stats.ordinal_hits += 1
            logger.info(
                "Replayed LLM fixture by call order (prompt differed from recording)",
                step=self._step,
                schema=schema_name,
                ordinal=ordinal,
                prompt_ids=len(ids),
            )
        return response_schema.model_validate_json(from_placeholders(entry.response, ids))

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        norm_prompt, ids = _renormalize(system_prompt or "", prompt)
        norm_system, _ = normalize_prompt(system_prompt or "")
        key = fixture_key("__text__", norm_system, norm_prompt)
        ordinal = self._next_ordinal("__text__")

        if self.mode == "record":
            assert self.inner is not None
            inner = self.inner
            text = await self._patiently(
                lambda: inner.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    budget_tracker=budget_tracker,
                )
            )
            self.store.put(
                key,
                FixtureEntry(
                    schema_name="__text__",
                    step=self._step,
                    ordinal=ordinal,
                    response=to_placeholders(text, ids),
                    kind="text",
                ),
            )
            self.stats.recorded += 1
            return text

        entry, exact = self.store.get(key, _ordinal_key(self._step, "__text__", ordinal))
        if entry is None:
            self.stats.misses += 1
            raise LLMUnavailableError(f"No recorded LLM text response (step={self._step})")
        if exact:
            self.stats.exact_hits += 1
        else:
            self.stats.ordinal_hits += 1
        return from_placeholders(entry.response, ids)


def _renormalize(system_prompt: str, prompt: str) -> tuple[str, list[str]]:
    """Normalize the user prompt with placeholder numbering shared with the system prompt."""
    combined, ids = normalize_prompt(system_prompt + "\n\x00\n" + prompt)
    _, _, normalized_user = combined.partition("\n\x00\n")
    return normalized_user, ids

