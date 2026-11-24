"""Unit tests for Crawl4AIClient concrete acquisition provider.

Verifies Crawl4AI wrapping, deterministic content precedence, metadata allowlisting,
JavaScript configuration, and error/status classifications using mocks/fakes.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from competitive_intelligence.source_acquisition import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    AcquisitionResult,
    AcquisitionStatus,
    ContentType,
    Crawl4AIClient,
    PlatformType,
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)


# ==============================================================================
# Helper Factories
# ==============================================================================


def create_web_source(
    url: str = "https://competitor.com/pricing",
    requires_javascript: bool = False,
) -> Source:
    """Create a valid web-based test Source."""
    return Source(
        identity=SourceIdentity(
            company_id="comp_test",
            workspace_id="ws_test",
            display_name="Test Pricing",
        ),
        classification=SourceClassification(
            category=SourceCategory.PRICING,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier=url,
            access_pattern=AccessPattern.PUBLIC_WEB,
            requires_javascript=requires_javascript,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
        ),
    )


class MockCrawlResult:
    """Mock object simulating Crawl4AI CrawlResult with customizable properties."""

    def __init__(
        self,
        url: str = "https://competitor.com/pricing",
        success: bool = True,
        status_code: int | None = 200,
        markdown: str | None = "# Pricing\nPlan A: $10/mo",
        cleaned_html: str | None = "<p>Plan A: $10/mo</p>",
        html: str | None = "<html><body><p>Plan A: $10/mo</p></body></html>",
        metadata: dict[str, Any] | None = None,
        links: dict[str, Any] | None = None,
        media: dict[str, Any] | None = None,
        tables: list[dict[str, Any]] | None = None,
        response_headers: dict[str, Any] | None = None,
        redirected_url: str | None = None,
        redirected_status_code: int | None = None,
        crawl_stats: dict[str, Any] | None = None,
        error_message: str | None = None,
        screenshot: str | None = "base64_screenshot_data_to_ignore",
        pdf: bytes | None = b"%PDF-1.4 binary data to ignore",
    ) -> None:
        self.url = url
        self.success = success
        self.status_code = status_code
        self.markdown = markdown
        self.cleaned_html = cleaned_html
        self.html = html
        self.metadata = metadata if metadata is not None else {"title": "Competitor Pricing"}
        self.links = links if links is not None else {"internal": [{"href": "/features"}]}
        self.media = media if media is not None else {"images": [{"src": "logo.png"}]}
        self.tables = tables if tables is not None else [{"rows": [["Plan", "Price"]]}]
        self.response_headers = response_headers if response_headers is not None else {"Content-Type": "text/html"}
        self.redirected_url = redirected_url
        self.redirected_status_code = redirected_status_code
        self.crawl_stats = crawl_stats if crawl_stats is not None else {"timing": 1.2}
        self.error_message = error_message
        self.screenshot = screenshot
        self.pdf = pdf


# ==============================================================================
# Crawl4AIClient Unit Tests
# ==============================================================================


@pytest.mark.anyio
async def test_crawl4ai_successful_acquisition() -> None:
    """Test successful acquisition converts CrawlResult to AcquisitionResult correctly."""
    mock_crawl_result = MockCrawlResult(
        url="https://competitor.com/pricing",
        success=True,
        status_code=200,
        markdown="# Official Pricing\n- Pro: $49/mo",
    )

    mock_crawler = MagicMock()
    mock_crawler.arun = AsyncMock(return_value=mock_crawl_result)

    client = Crawl4AIClient(crawler=mock_crawler)
    source = create_web_source(url="https://competitor.com/pricing", requires_javascript=False)

    result = await client.acquire(source)

    assert result.status == AcquisitionStatus.SUCCESS
    assert result.is_success is True
    assert result.is_failure is False
    assert result.content == "# Official Pricing\n- Pro: $49/mo"
    assert result.method == AcquisitionMethod.CRAWL4AI
    assert result.provider_name == "Crawl4AIClient"
    assert result.status_code == 200
    assert result.error_message is None
    assert result.acquired_at.tzinfo is not None

    # Verify URL was passed to crawler
    mock_crawler.arun.assert_awaited_once()
    call_args = mock_crawler.arun.call_args
    assert call_args.kwargs["url"] == "https://competitor.com/pricing"


@pytest.mark.anyio
async def test_crawl4ai_content_resolution_precedence() -> None:
    """Test deterministic content preference: markdown -> cleaned_html -> raw_html -> None."""
    mock_crawler = MagicMock()
    client = Crawl4AIClient(crawler=mock_crawler)
    source = create_web_source()

    # 1. Markdown is present -> chosen over cleaned_html and html
    mock_crawler.arun = AsyncMock(
        return_value=MockCrawlResult(
            markdown="# Markdown Content",
            cleaned_html="<div>Cleaned HTML</div>",
            html="<html>Raw HTML</html>",
        )
    )
    res1 = await client.acquire(source)
    assert res1.content == "# Markdown Content"

    # 2. Markdown is None -> cleaned_html chosen over raw html
    mock_crawler.arun = AsyncMock(
        return_value=MockCrawlResult(
            markdown=None,
            cleaned_html="<div>Cleaned HTML</div>",
            html="<html>Raw HTML</html>",
        )
    )
    res2 = await client.acquire(source)
    assert res2.content == "<div>Cleaned HTML</div>"

    # 3. Markdown and Cleaned HTML are None -> raw html chosen
    mock_crawler.arun = AsyncMock(
        return_value=MockCrawlResult(
            markdown=None,
            cleaned_html=None,
            html="<html>Raw HTML</html>",
        )
    )
    res3 = await client.acquire(source)
    assert res3.content == "<html>Raw HTML</html>"

    # 4. All are None -> content is None
    mock_crawler.arun = AsyncMock(
        return_value=MockCrawlResult(
            markdown=None,
            cleaned_html=None,
            html=None,
        )
    )
    res4 = await client.acquire(source)
    assert res4.content is None


@pytest.mark.anyio
async def test_crawl4ai_metadata_allowlist_excludes_binaries() -> None:
    """Test that metadata explicitly preserves allowlisted fields and excludes large binaries."""
    mock_crawl_result = MockCrawlResult(
        url="https://competitor.com/pricing",
        metadata={"title": "Pricing Page"},
        links={"internal": [{"href": "/sub"}]},
        media={"images": [{"src": "img.png"}]},
        tables=[{"rows": [["A", "B"]]}],
        response_headers={"Server": "nginx"},
        redirected_url="https://competitor.com/pricing-v2",
        redirected_status_code=301,
        crawl_stats={"execution_time": 0.45},
        screenshot="HUGE_BASE64_STRING_THAT_SHOULD_NOT_BE_IN_METADATA",
        pdf=b"%PDF_BINARY_DATA_THAT_SHOULD_NOT_BE_IN_METADATA",
    )

    mock_crawler = MagicMock()
    mock_crawler.arun = AsyncMock(return_value=mock_crawl_result)
    client = Crawl4AIClient(crawler=mock_crawler)
    source = create_web_source()

    result = await client.acquire(source)

    meta = result.metadata
    assert meta["url"] == "https://competitor.com/pricing"
    assert meta["title"] == "Pricing Page"
    assert meta["links"] == {"internal": [{"href": "/sub"}]}
    assert meta["media"] == {"images": [{"src": "img.png"}]}
    assert meta["tables"] == [{"rows": [["A", "B"]]}]
    assert meta["response_headers"] == {"Server": "nginx"}
    assert meta["redirected_url"] == "https://competitor.com/pricing-v2"
    assert meta["redirected_status_code"] == 301
    assert meta["crawl_stats"] == {"execution_time": 0.45}

    # Verify binaries are excluded
    assert "screenshot" not in meta
    assert "pdf" not in meta


def test_crawl4ai_javascript_browser_config() -> None:
    """Test that JavaScript configuration is properly reflected in BrowserConfig."""
    client = Crawl4AIClient()

    # requires_javascript=True
    cfg_js = client._build_browser_config(requires_js=True)
    assert cfg_js.java_script_enabled is True

    # requires_javascript=False
    cfg_no_js = client._build_browser_config(requires_js=False)
    assert cfg_no_js.java_script_enabled is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [
        (404, AcquisitionStatus.NOT_FOUND),
        (429, AcquisitionStatus.RATE_LIMITED),
        (403, AcquisitionStatus.BLOCKED),
        (451, AcquisitionStatus.BLOCKED),
        (401, AcquisitionStatus.AUTH_REQUIRED),
        (408, AcquisitionStatus.TIMEOUT),
        (504, AcquisitionStatus.TIMEOUT),
        (500, AcquisitionStatus.SERVER_ERROR),
        (502, AcquisitionStatus.SERVER_ERROR),
        (503, AcquisitionStatus.SERVER_ERROR),
    ],
)
async def test_crawl4ai_http_status_code_mappings(
    status_code: int, expected_status: AcquisitionStatus
) -> None:
    """Test classifying failures based on reliable HTTP status codes."""
    mock_crawl_result = MockCrawlResult(
        success=False,
        status_code=status_code,
        error_message=f"HTTP Error {status_code}",
        markdown=None,
    )
    mock_crawler = MagicMock()
    mock_crawler.arun = AsyncMock(return_value=mock_crawl_result)

    client = Crawl4AIClient(crawler=mock_crawler)
    source = create_web_source()

    result = await client.acquire(source)

    assert result.status == expected_status
    assert result.is_success is False
    assert result.status_code == status_code
    assert result.content is None


@pytest.mark.anyio
async def test_crawl4ai_exception_mapping() -> None:
    """Test classifying unhandled runtime exceptions during crawl execution."""
    # 1. TimeoutError
    mock_crawler = MagicMock()
    mock_crawler.arun = AsyncMock(side_effect=TimeoutError("Navigation timed out"))
    client = Crawl4AIClient(crawler=mock_crawler)
    source = create_web_source()

    res_timeout = await client.acquire(source)
    assert res_timeout.status == AcquisitionStatus.TIMEOUT
    assert "Navigation timed out" in (res_timeout.error_message or "")

    # 2. ConnectionResetError (Network Error)
    mock_crawler.arun = AsyncMock(side_effect=ConnectionResetError("Connection reset by peer"))
    res_network = await client.acquire(source)
    assert res_network.status == AcquisitionStatus.NETWORK_ERROR
    assert "Connection reset" in (res_network.error_message or "")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_msg", "expected_status"),
    [
        ("403 Forbidden - Cloudflare bot challenge", AcquisitionStatus.BLOCKED),
        ("Rate limit exceeded. Please wait.", AcquisitionStatus.RATE_LIMITED),
        ("DNS resolution failed: net::ERR_NAME_NOT_RESOLVED", AcquisitionStatus.NETWORK_ERROR),
        ("Extraction failed while parsing DOM structure", AcquisitionStatus.PARSER_ERROR),
        ("502 Bad Gateway from upstream host", AcquisitionStatus.SERVER_ERROR),
        ("Some unknown error occurred", AcquisitionStatus.FAILURE),
    ],
)
async def test_crawl4ai_error_message_fallback_mappings(
    error_msg: str, expected_status: AcquisitionStatus
) -> None:
    """Test error-message pattern matching fallback when status code is None."""
    mock_crawl_result = MockCrawlResult(
        success=False,
        status_code=None,
        error_message=error_msg,
        markdown=None,
    )
    mock_crawler = MagicMock()
    mock_crawler.arun = AsyncMock(return_value=mock_crawl_result)

    client = Crawl4AIClient(crawler=mock_crawler)
    source = create_web_source()

    result = await client.acquire(source)

    assert result.status == expected_status
    assert result.is_success is False


@pytest.mark.anyio
async def test_crawl4ai_resilience_to_minimal_crawl_result() -> None:
    """Test that Crawl4AIClient operates safely with a bare minimal CrawlResult object missing optional fields."""

    class MinimalCrawlResult:
        def __init__(self) -> None:
            self.success = True
            self.html = "<html>Bare minimum</html>"

    mock_crawler = MagicMock()
    mock_crawler.arun = AsyncMock(return_value=MinimalCrawlResult())

    client = Crawl4AIClient(crawler=mock_crawler)
    source = create_web_source()

    result = await client.acquire(source)

    assert result.status == AcquisitionStatus.SUCCESS
    assert result.content == "<html>Bare minimum</html>"
    assert result.metadata == {}
