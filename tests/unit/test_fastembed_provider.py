"""fastembed provider: wiring, vector size, environment rules (fastembed itself is faked)."""

import sys
import types
from collections.abc import Iterable, Iterator
from typing import Any

import pytest

from competitive_intelligence.application.api import deps
from competitive_intelligence.core.config import ProductionConfigurationError, Settings
from competitive_intelligence.knowledge_pipeline.embeddings.fastembed_provider import (
    FastEmbedEmbeddingProvider,
    FastEmbedProviderError,
    fastembed_dimensions,
)


class _FakeTextEmbedding:
    calls: list[tuple[str, Any]] = []

    def __init__(self, model_name: str, threads: int | None = None) -> None:
        self.model_name = model_name
        _FakeTextEmbedding.calls.append(("init", (model_name, threads)))

    @staticmethod
    def list_supported_models() -> list[dict[str, Any]]:
        return [{"model": "BAAI/bge-small-en-v1.5", "dim": 384}]

    def passage_embed(self, texts: Iterable[str], batch_size: int = 256) -> Iterator[list[float]]:
        for text in texts:
            _FakeTextEmbedding.calls.append(("passage", text))
            yield [float(len(text))] * 384

    def query_embed(self, query: str) -> Iterator[list[float]]:
        _FakeTextEmbedding.calls.append(("query", query))
        yield [0.5] * 384


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_FakeTextEmbedding]]:
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    _FakeTextEmbedding.calls = []
    deps.reset_resource_caches()
    yield _FakeTextEmbedding
    deps.reset_resource_caches()


def test_provider_embeds_passages_and_queries(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    provider = FastEmbedEmbeddingProvider("BAAI/bge-small-en-v1.5", threads=1)

    assert provider.dimensions == 384
    assert provider.embed_texts([]) == []
    vectors = provider.embed_texts(["ab", "abcd"])
    assert [v[0] for v in vectors] == [2.0, 4.0] and all(len(v) == 384 for v in vectors)
    assert provider.embed_query("  price  ") == [0.5] * 384
    assert ("query", "price") in fake_fastembed.calls
    with pytest.raises(ValueError):
        provider.embed_query("   ")


def test_unknown_model_is_rejected(fake_fastembed: type[_FakeTextEmbedding]) -> None:
    with pytest.raises(FastEmbedProviderError, match="Unknown fastembed model"):
        fastembed_dimensions("not/a-model")


def test_vector_store_size_follows_the_embedding_provider(
    fake_fastembed: type[_FakeTextEmbedding], tmp_path: Any
) -> None:
    fe = Settings(environment="development", embedding_provider="fastembed",
                  qdrant_path=str(tmp_path / "q"))
    assert deps.embedding_dimensions(fe) == 384
    assert deps.get_vector_store(fe).config.vector_dimensions == 384  # type: ignore[attr-defined]
    assert deps.get_embedding_service(fe).provider.dimensions == 384

    assert deps.embedding_dimensions(Settings(environment="development")) == 1024  # mock
    assert deps.embedding_dimensions(
        Settings(environment="development", embedding_provider="bge_m3")
    ) == 1024


def test_embedding_service_reuses_one_fastembed_model(
    fake_fastembed: type[_FakeTextEmbedding],
) -> None:
    cfg = Settings(environment="development", embedding_provider="fastembed")
    first = deps.get_embedding_service(cfg).provider
    second = deps.get_embedding_service(cfg).provider
    assert first is second
    assert [c for c in fake_fastembed.calls if c[0] == "init"] == [
        ("init", ("BAAI/bge-small-en-v1.5", None))
    ]


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_production_and_staging_require_bge_m3(environment: str) -> None:
    settings = Settings(
        environment=environment,
        llm_provider="openai",
        openai_api_key="sk-test-mock-key-12345",
        embedding_provider="fastembed",
        database_url="postgresql+asyncpg://u:p@db:5432/ci",
        qdrant_url="http://qdrant:6333",
    )
    with pytest.raises(ProductionConfigurationError, match="fastembed embedding provider"):
        settings.validate_production_configuration(raise_exception=True)


def test_demo_and_development_allow_fastembed() -> None:
    demo = Settings(
        environment="demo",
        demo_mode=True,
        llm_provider="openai",
        openai_api_key="sk-test-mock-key-12345",
        embedding_provider="fastembed",
        database_url="sqlite+aiosqlite:///./demo.db",
        qdrant_path="/tmp/qdrant",
    )
    assert demo.validate_production_configuration(raise_exception=False) == []
    dev = Settings(environment="development", embedding_provider="fastembed")
    assert dev.validate_production_configuration(raise_exception=False) == []
