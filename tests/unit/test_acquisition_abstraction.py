"""Unit tests for the Source Acquisition abstraction layer.

Tests AcquisitionClient, AcquisitionResult, AcquisitionOrchestrator, DefaultFallbackPolicy,
and ProviderRegistry without concrete network/tool dependencies.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
import pytest

from competitive_intelligence.source_acquisition import (
    AccessPattern,
    AcquisitionClient,
    AcquisitionConfig,
    AcquisitionMethod,
    AcquisitionOrchestrator,
    AcquisitionResult,
    AcquisitionStatus,
    ContentType,
    DefaultFallbackPolicy,
    FallbackPolicy,
    PlatformType,
    ProviderNotRegisteredError,
    ProviderRegistry,
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)


# ==============================================================================
# Test Doubles / Fake Acquisition Clients
# ==============================================================================


class FakeAcquisitionClient(AcquisitionClient):
    """Configurable test double for AcquisitionClient."""

    def __init__(self, result: AcquisitionResult) -> None:
        self.result = result
        self.call_count = 0
        self.acquired_sources: list[Source] = []

    async def acquire(self, source: Source) -> AcquisitionResult:
        self.call_count += 1
        self.acquired_sources.append(source)
        return self.result


class ExceptionRaisingClient(AcquisitionClient):
    """Test double that simulates an unexpected runtime exception during acquire."""

    def __init__(self, error_message: str = "Connection reset by peer") -> None:
        self.error_message = error_message
        self.call_count = 0

    async def acquire(self, source: Source) -> AcquisitionResult:
        self.call_count += 1
        raise RuntimeError(self.error_message)


# ==============================================================================
# Helper Fixtures & Factories
# ==============================================================================


def create_test_source(
    primary_method: AcquisitionMethod = AcquisitionMethod.CRAWL4AI,
    fallbacks: list[AcquisitionMethod] | None = None,
    url: str = "https://competitor.com/pricing",
) -> Source:
    """Helper to build a valid test Source."""
    fallback_list = fallbacks if fallbacks is not None else []
    return Source(
        identity=SourceIdentity(
            company_id="comp_test",
            workspace_id="ws_test",
            display_name="Test Competitor Pricing",
        ),
        classification=SourceClassification(
            category=SourceCategory.PRICING,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier=url,
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=primary_method,
            tool_chain=ToolChain(
                primary=primary_method,
                fallbacks=fallback_list,
            ),
        ),
    )


# ==============================================================================
# AcquisitionResult Domain Model Tests
# ==============================================================================


def test_acquisition_result_successful_with_content_and_metadata() -> None:
    """Test creating a valid successful AcquisitionResult with content and metadata."""
    now = datetime.now(timezone.utc)
    result = AcquisitionResult(
        status=AcquisitionStatus.SUCCESS,
        content="<html><body><h1>Pricing: $49/mo</h1></body></html>",
        metadata={"response_time_ms": 142, "content_length": 45},
        acquired_at=now,
        method=AcquisitionMethod.CRAWL4AI,
        provider_name="Crawl4AIFakeClient",
        status_code=200,
    )

    assert result.status == AcquisitionStatus.SUCCESS
    assert result.is_success is True
    assert result.is_failure is False
    assert result.has_content is True
    assert result.content == "<html><body><h1>Pricing: $49/mo</h1></body></html>"
    assert result.metadata["response_time_ms"] == 142
    assert result.acquired_at == now
    assert result.method == AcquisitionMethod.CRAWL4AI
    assert result.provider_name == "Crawl4AIFakeClient"
    assert result.status_code == 200
    assert result.error_message is None


def test_acquisition_result_failed_with_no_content() -> None:
    """Test creating a failed AcquisitionResult containing no content and detailed error info."""
    result = AcquisitionResult(
        status=AcquisitionStatus.NOT_FOUND,
        content=None,
        status_code=404,
        error_message="HTTP 404 Not Found",
        error_code="RESOURCE_NOT_FOUND",
        error_details={"url": "https://competitor.com/old-pricing"},
        method=AcquisitionMethod.DIRECT_HTTP,
    )

    assert result.status == AcquisitionStatus.NOT_FOUND
    assert result.is_success is False
    assert result.is_failure is True
    assert result.has_content is False
    assert result.content is None
    assert result.status_code == 404
    assert result.error_message == "HTTP 404 Not Found"
    assert result.error_code == "RESOURCE_NOT_FOUND"
    assert result.error_details == {"url": "https://competitor.com/old-pricing"}


def test_acquisition_result_requires_timezone_aware_acquired_at() -> None:
    """Test that AcquisitionResult rejects timezone-naive datetime."""
    naive_dt = datetime(2026, 8, 22, 12, 0, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        AcquisitionResult(
            status=AcquisitionStatus.SUCCESS,
            content="sample content",
            acquired_at=naive_dt,
        )


def test_acquisition_result_strict_method_validation() -> None:
    """Test that AcquisitionResult method field strictly accepts AcquisitionMethod or None."""
    # Valid None
    res_none = AcquisitionResult(status=AcquisitionStatus.SUCCESS, method=None)
    assert res_none.method is None

    # Valid enum
    res_enum = AcquisitionResult(
        status=AcquisitionStatus.SUCCESS, method=AcquisitionMethod.REST_API
    )
    assert res_enum.method == AcquisitionMethod.REST_API

    # Invalid method string not in AcquisitionMethod enum
    with pytest.raises(ValidationError):
        AcquisitionResult(status=AcquisitionStatus.SUCCESS, method="unsupported_method")  # type: ignore[arg-type]


# ==============================================================================
# Fallback Policy Unit Tests
# ==============================================================================


@pytest.mark.parametrize(
    ("status", "expected_recoverable"),
    [
        (AcquisitionStatus.RATE_LIMITED, True),
        (AcquisitionStatus.BLOCKED, True),
        (AcquisitionStatus.TIMEOUT, True),
        (AcquisitionStatus.NETWORK_ERROR, True),
        (AcquisitionStatus.PARSER_ERROR, True),
        (AcquisitionStatus.SERVER_ERROR, True),
        (AcquisitionStatus.FAILURE, True),
        (AcquisitionStatus.NOT_FOUND, False),
        (AcquisitionStatus.AUTH_REQUIRED, False),
        (AcquisitionStatus.SUCCESS, False),
    ],
)
def test_default_fallback_policy_recoverability(
    status: AcquisitionStatus, expected_recoverable: bool
) -> None:
    """Verify DefaultFallbackPolicy classifies statuses accurately."""
    policy = DefaultFallbackPolicy()
    result = AcquisitionResult(status=status)

    assert policy.is_recoverable(result) == expected_recoverable
    assert policy.should_fallback(result) == expected_recoverable


# ==============================================================================
# Provider Registry Unit Tests
# ==============================================================================


def test_provider_registry_registration_and_lookup() -> None:
    """Test registering and retrieving providers from ProviderRegistry."""
    registry = ProviderRegistry()
    client = FakeAcquisitionClient(AcquisitionResult(status=AcquisitionStatus.SUCCESS))

    assert registry.has_provider(AcquisitionMethod.CRAWL4AI) is False
    assert registry.get(AcquisitionMethod.CRAWL4AI) is None

    registry.register(AcquisitionMethod.CRAWL4AI, client)

    assert registry.has_provider(AcquisitionMethod.CRAWL4AI) is True
    assert AcquisitionMethod.CRAWL4AI in registry
    assert registry.get(AcquisitionMethod.CRAWL4AI) is client
    assert registry[AcquisitionMethod.CRAWL4AI] is client


def test_provider_registry_getitem_raises_for_unregistered() -> None:
    """Test that indexing an unregistered method raises ProviderNotRegisteredError."""
    registry = ProviderRegistry()
    with pytest.raises(ProviderNotRegisteredError, match="crawl4ai"):
        _ = registry[AcquisitionMethod.CRAWL4AI]


# ==============================================================================
# Acquisition Orchestrator Unit Tests
# ==============================================================================


@pytest.mark.anyio
async def test_orchestrator_successful_primary_acquisition() -> None:
    """Test successful acquisition on primary provider returns immediately without calling fallback."""
    primary_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.SUCCESS,
            content="<h1>Official Plan</h1>",
            method=AcquisitionMethod.CRAWL4AI,
            status_code=200,
        )
    )
    fallback_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.SUCCESS,
            content="<h1>Fallback Plan</h1>",
            method=AcquisitionMethod.DIRECT_HTTP,
        )
    )

    orchestrator = AcquisitionOrchestrator(
        providers={
            AcquisitionMethod.CRAWL4AI: primary_client,
            AcquisitionMethod.DIRECT_HTTP: fallback_client,
        }
    )

    source = create_test_source(
        primary_method=AcquisitionMethod.CRAWL4AI,
        fallbacks=[AcquisitionMethod.DIRECT_HTTP],
    )

    result = await orchestrator.acquire(source)

    assert result.is_success is True
    assert result.status == AcquisitionStatus.SUCCESS
    assert result.content == "<h1>Official Plan</h1>"
    assert result.method == AcquisitionMethod.CRAWL4AI
    assert primary_client.call_count == 1
    assert fallback_client.call_count == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "recoverable_status",
    [
        AcquisitionStatus.RATE_LIMITED,
        AcquisitionStatus.BLOCKED,
        AcquisitionStatus.TIMEOUT,
        AcquisitionStatus.NETWORK_ERROR,
        AcquisitionStatus.PARSER_ERROR,
        AcquisitionStatus.SERVER_ERROR,
        AcquisitionStatus.FAILURE,
    ],
)
async def test_orchestrator_recoverable_failure_falls_back_successfully(
    recoverable_status: AcquisitionStatus,
) -> None:
    """Test that all recoverable failure statuses trigger fallback to secondary provider."""
    primary_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=recoverable_status,
            error_message="Primary provider encountered recoverable error",
            method=AcquisitionMethod.CRAWL4AI,
        )
    )
    fallback_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.SUCCESS,
            content="<h1>Recovered Content</h1>",
            method=AcquisitionMethod.DIRECT_HTTP,
            status_code=200,
        )
    )

    orchestrator = AcquisitionOrchestrator(
        providers={
            AcquisitionMethod.CRAWL4AI: primary_client,
            AcquisitionMethod.DIRECT_HTTP: fallback_client,
        }
    )

    source = create_test_source(
        primary_method=AcquisitionMethod.CRAWL4AI,
        fallbacks=[AcquisitionMethod.DIRECT_HTTP],
    )

    result = await orchestrator.acquire(source)

    assert result.is_success is True
    assert result.status == AcquisitionStatus.SUCCESS
    assert result.content == "<h1>Recovered Content</h1>"
    assert result.method == AcquisitionMethod.DIRECT_HTTP
    assert primary_client.call_count == 1
    assert fallback_client.call_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "non_recoverable_status",
    [
        AcquisitionStatus.NOT_FOUND,
        AcquisitionStatus.AUTH_REQUIRED,
    ],
)
async def test_orchestrator_non_recoverable_failure_stops_chain(
    non_recoverable_status: AcquisitionStatus,
) -> None:
    """Test that non-recoverable failures halt the tool chain immediately without trying fallbacks."""
    primary_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=non_recoverable_status,
            error_message="Non-recoverable error encountered",
            status_code=404 if non_recoverable_status == AcquisitionStatus.NOT_FOUND else 401,
            method=AcquisitionMethod.CRAWL4AI,
        )
    )
    fallback_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.SUCCESS,
            content="<h1>Should Never Be Called</h1>",
            method=AcquisitionMethod.DIRECT_HTTP,
        )
    )

    orchestrator = AcquisitionOrchestrator(
        providers={
            AcquisitionMethod.CRAWL4AI: primary_client,
            AcquisitionMethod.DIRECT_HTTP: fallback_client,
        }
    )

    source = create_test_source(
        primary_method=AcquisitionMethod.CRAWL4AI,
        fallbacks=[AcquisitionMethod.DIRECT_HTTP],
    )

    result = await orchestrator.acquire(source)

    assert result.is_success is False
    assert result.status == non_recoverable_status
    assert result.method == AcquisitionMethod.CRAWL4AI
    assert primary_client.call_count == 1
    assert fallback_client.call_count == 0


@pytest.mark.anyio
async def test_orchestrator_all_providers_fail_recoverably() -> None:
    """Test that when all primary and fallback providers fail recoverably, the final failure result is returned."""
    client1 = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.RATE_LIMITED,
            error_message="Primary rate limited",
            method=AcquisitionMethod.CRAWL4AI,
        )
    )
    client2 = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.BLOCKED,
            error_message="Fallback 1 blocked by Cloudflare",
            method=AcquisitionMethod.DIRECT_HTTP,
        )
    )
    client3 = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.TIMEOUT,
            error_message="Fallback 2 request timed out after 30s",
            method=AcquisitionMethod.REST_API,
        )
    )

    orchestrator = AcquisitionOrchestrator(
        providers={
            AcquisitionMethod.CRAWL4AI: client1,
            AcquisitionMethod.DIRECT_HTTP: client2,
            AcquisitionMethod.REST_API: client3,
        }
    )

    source = create_test_source(
        primary_method=AcquisitionMethod.CRAWL4AI,
        fallbacks=[AcquisitionMethod.DIRECT_HTTP, AcquisitionMethod.REST_API],
    )

    result = await orchestrator.acquire(source)

    assert result.is_success is False
    assert result.status == AcquisitionStatus.TIMEOUT
    assert result.method == AcquisitionMethod.REST_API
    assert result.error_message == "Fallback 2 request timed out after 30s"
    assert client1.call_count == 1
    assert client2.call_count == 1
    assert client3.call_count == 1


@pytest.mark.anyio
async def test_orchestrator_provider_ordering_execution() -> None:
    """Test that orchestrator executes providers strictly in the order configured by Source.tool_chain."""
    call_order: list[AcquisitionMethod] = []

    class OrderTrackingClient(AcquisitionClient):
        def __init__(self, method: AcquisitionMethod, should_succeed: bool = False) -> None:
            self.method = method
            self.should_succeed = should_succeed

        async def acquire(self, source: Source) -> AcquisitionResult:
            call_order.append(self.method)
            if self.should_succeed:
                return AcquisitionResult(
                    status=AcquisitionStatus.SUCCESS,
                    content="Success",
                    method=self.method,
                )
            return AcquisitionResult(
                status=AcquisitionStatus.NETWORK_ERROR,
                error_message="Network glitch",
                method=self.method,
            )

    client_crawl = OrderTrackingClient(AcquisitionMethod.CRAWL4AI, should_succeed=False)
    client_reach = OrderTrackingClient(AcquisitionMethod.AGENT_REACH, should_succeed=False)
    client_http = OrderTrackingClient(AcquisitionMethod.DIRECT_HTTP, should_succeed=True)

    orchestrator = AcquisitionOrchestrator(
        providers={
            AcquisitionMethod.CRAWL4AI: client_crawl,
            AcquisitionMethod.AGENT_REACH: client_reach,
            AcquisitionMethod.DIRECT_HTTP: client_http,
        }
    )

    source = create_test_source(
        primary_method=AcquisitionMethod.CRAWL4AI,
        fallbacks=[AcquisitionMethod.AGENT_REACH, AcquisitionMethod.DIRECT_HTTP],
    )

    result = await orchestrator.acquire(source)

    assert result.is_success is True
    assert result.method == AcquisitionMethod.DIRECT_HTTP
    assert call_order == [
        AcquisitionMethod.CRAWL4AI,
        AcquisitionMethod.AGENT_REACH,
        AcquisitionMethod.DIRECT_HTTP,
    ]


@pytest.mark.anyio
async def test_orchestrator_raises_on_unregistered_provider() -> None:
    """Test that missing provider in registry raises ProviderNotRegisteredError rather than masking configuration issue."""
    # Only CRAWL4AI is registered, but DIRECT_HTTP is required as fallback
    orchestrator = AcquisitionOrchestrator(
        providers={
            AcquisitionMethod.CRAWL4AI: FakeAcquisitionClient(
                AcquisitionResult(
                    status=AcquisitionStatus.TIMEOUT,
                    method=AcquisitionMethod.CRAWL4AI,
                )
            )
        }
    )

    source = create_test_source(
        primary_method=AcquisitionMethod.CRAWL4AI,
        fallbacks=[AcquisitionMethod.DIRECT_HTTP],
    )

    with pytest.raises(ProviderNotRegisteredError) as exc_info:
        await orchestrator.acquire(source)

    assert exc_info.value.method == AcquisitionMethod.DIRECT_HTTP
    assert "direct_http" in str(exc_info.value)


@pytest.mark.anyio
async def test_orchestrator_handles_client_exception_gracefully() -> None:
    """Test that unexpected client exceptions are caught, converted to failure AcquisitionResult, and trigger fallback."""
    failing_client = ExceptionRaisingClient(error_message="Unexpected connection abort")
    fallback_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.SUCCESS,
            content="Fallback content",
            method=AcquisitionMethod.DIRECT_HTTP,
        )
    )

    orchestrator = AcquisitionOrchestrator(
        providers={
            AcquisitionMethod.CRAWL4AI: failing_client,
            AcquisitionMethod.DIRECT_HTTP: fallback_client,
        }
    )

    source = create_test_source(
        primary_method=AcquisitionMethod.CRAWL4AI,
        fallbacks=[AcquisitionMethod.DIRECT_HTTP],
    )

    result = await orchestrator.acquire(source)

    assert result.is_success is True
    assert result.content == "Fallback content"
    assert failing_client.call_count == 1
    assert fallback_client.call_count == 1


@pytest.mark.anyio
async def test_orchestrator_with_custom_fallback_policy() -> None:
    """Test injecting a custom FallbackPolicy into AcquisitionOrchestrator."""

    class StrictNoFallbackPolicy(FallbackPolicy):
        """Custom policy that never allows fallback under any circumstance."""

        def is_recoverable(self, result: AcquisitionResult) -> bool:
            return False

        def should_fallback(self, result: AcquisitionResult) -> bool:
            return False

    primary_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.RATE_LIMITED,
            method=AcquisitionMethod.CRAWL4AI,
        )
    )
    fallback_client = FakeAcquisitionClient(
        AcquisitionResult(
            status=AcquisitionStatus.SUCCESS,
            content="Never called",
            method=AcquisitionMethod.DIRECT_HTTP,
        )
    )

    orchestrator = AcquisitionOrchestrator(
        providers={
            AcquisitionMethod.CRAWL4AI: primary_client,
            AcquisitionMethod.DIRECT_HTTP: fallback_client,
        },
        fallback_policy=StrictNoFallbackPolicy(),
    )

    source = create_test_source(
        primary_method=AcquisitionMethod.CRAWL4AI,
        fallbacks=[AcquisitionMethod.DIRECT_HTTP],
    )

    result = await orchestrator.acquire(source)

    assert result.status == AcquisitionStatus.RATE_LIMITED
    assert primary_client.call_count == 1
    assert fallback_client.call_count == 0
