"""Crawl4AI acquisition provider implementation.

Wraps Crawl4AI AsyncWebCrawler behind the AcquisitionClient interface, translating
crawled web outputs and errors into the AcquisitionResult domain model.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

from competitive_intelligence.source_acquisition.interfaces import AcquisitionClient
from competitive_intelligence.source_acquisition.models import (
    AcquisitionMethod,
    AcquisitionResult,
    AcquisitionStatus,
    Source,
)


class Crawl4AIClient(AcquisitionClient):
    """Concrete acquisition client using Crawl4AI for web content extraction.

    Responsibilities:
    - Extracts target URL from Source.location.url_or_identifier.
    - Configures JavaScript execution based on Source.location.requires_javascript.
    - Executes async crawl requests via AsyncWebCrawler.
    - Translates responses into AcquisitionResult using deterministic content precedence
      (markdown -> cleaned_html -> raw_html -> None).
    - Preserves lightweight structured metadata using an explicit allowlist (omitting large binaries like screenshots/PDFs).
    - Classifies HTTP status codes, exception types, and error messages into AcquisitionStatus.
    """

    PROVIDER_NAME: str = "Crawl4AIClient"

    def __init__(
        self,
        crawler: AsyncWebCrawler | None = None,
        browser_config: BrowserConfig | None = None,
        crawler_run_config: CrawlerRunConfig | None = None,
    ) -> None:
        """Initialize Crawl4AIClient with optional crawler or configuration overrides.

        Args:
            crawler: Optional pre-initialized or mocked AsyncWebCrawler instance.
            browser_config: Optional custom BrowserConfig.
            crawler_run_config: Optional custom CrawlerRunConfig.
        """
        self.crawler = crawler
        self.browser_config = browser_config
        self.crawler_run_config = crawler_run_config

    async def acquire(self, source: Source) -> AcquisitionResult:
        """Acquire web content snapshot for the given source using Crawl4AI.

        Args:
            source: The Source domain model defining target URL and constraints.

        Returns:
            AcquisitionResult populated with acquired content, metadata, and status.
        """
        target_url = source.location.url_or_identifier
        requires_js = source.location.requires_javascript
        acquired_at = datetime.now(timezone.utc)

        # Build configurations respecting source requirements
        browser_cfg = self._build_browser_config(requires_js)
        run_cfg = self._build_run_config(requires_js)

        try:
            if self.crawler is not None:
                crawl_result = await self.crawler.arun(url=target_url, config=run_cfg)
            else:
                async with AsyncWebCrawler(config=browser_cfg) as crawler:
                    crawl_result = await crawler.arun(url=target_url, config=run_cfg)

            return self._build_result_from_crawl(
                crawl_result=crawl_result,
                acquired_at=acquired_at,
            )

        except Exception as exc:
            return self._build_result_from_exception(
                exc=exc,
                acquired_at=acquired_at,
            )

    def _build_browser_config(self, requires_js: bool) -> BrowserConfig:
        """Construct or clone BrowserConfig respecting JavaScript requirement."""
        if self.browser_config is not None:
            return self.browser_config
        return BrowserConfig(
            java_script_enabled=requires_js,
            headless=True,
        )

    def _build_run_config(self, requires_js: bool) -> CrawlerRunConfig:
        """Construct or clone CrawlerRunConfig with appropriate wait conditions."""
        if self.crawler_run_config is not None:
            return self.crawler_run_config
        return CrawlerRunConfig(
            wait_until="networkidle" if requires_js else "domcontentloaded",
        )

    def _build_result_from_crawl(
        self, crawl_result: Any, acquired_at: datetime
    ) -> AcquisitionResult:
        """Convert a Crawl4AI CrawlResult into an AcquisitionResult."""
        success = bool(getattr(crawl_result, "success", False))
        status_code = getattr(crawl_result, "status_code", None)
        error_msg = getattr(crawl_result, "error_message", None)

        status = self._classify_status(
            status_code=status_code,
            error_message=error_msg,
            exc=None,
            success=success,
        )

        content = self._resolve_content(crawl_result) if status == AcquisitionStatus.SUCCESS else None
        metadata = self._extract_metadata(crawl_result)

        return AcquisitionResult(
            status=status,
            content=content,
            metadata=metadata,
            acquired_at=acquired_at,
            method=AcquisitionMethod.CRAWL4AI,
            provider_name=self.PROVIDER_NAME,
            status_code=status_code,
            error_message=error_msg,
        )

    def _build_result_from_exception(
        self, exc: Exception, acquired_at: datetime
    ) -> AcquisitionResult:
        """Convert an unhandled runtime exception during crawl into an AcquisitionResult."""
        error_msg = str(exc)
        status = self._classify_status(
            status_code=None,
            error_message=error_msg,
            exc=exc,
            success=False,
        )

        return AcquisitionResult(
            status=status,
            content=None,
            metadata={},
            acquired_at=acquired_at,
            method=AcquisitionMethod.CRAWL4AI,
            provider_name=self.PROVIDER_NAME,
            error_message=error_msg,
            error_details={"exception_type": type(exc).__name__},
        )

    def _resolve_content(self, result: Any) -> str | None:
        """Extract content using deterministic precedence: markdown -> cleaned_html -> raw_html -> None."""
        # 1. Markdown
        markdown = getattr(result, "markdown", None)
        if markdown is not None:
            md_str = str(markdown).strip()
            if md_str:
                return md_str

        # 2. Cleaned HTML
        cleaned_html = getattr(result, "cleaned_html", None)
        if cleaned_html is not None and isinstance(cleaned_html, str):
            cleaned_str = cleaned_html.strip()
            if cleaned_str:
                return cleaned_str

        # 3. Raw HTML
        html = getattr(result, "html", None)
        if html is not None and isinstance(html, str):
            html_str = html.strip()
            if html_str:
                return html_str

        return None

    def _extract_metadata(self, result: Any) -> dict[str, Any]:
        """Extract lightweight structured metadata according to explicit allowlist."""
        meta: dict[str, Any] = {}

        url = getattr(result, "url", None)
        if url is not None:
            meta["url"] = url

        raw_meta = getattr(result, "metadata", None)
        if isinstance(raw_meta, dict) and raw_meta.get("title"):
            meta["title"] = raw_meta["title"]

        links = getattr(result, "links", None)
        if links:
            meta["links"] = links

        media = getattr(result, "media", None)
        if media:
            meta["media"] = media

        tables = getattr(result, "tables", None)
        if tables:
            meta["tables"] = tables

        response_headers = getattr(result, "response_headers", None)
        if response_headers:
            meta["response_headers"] = response_headers

        redirected_url = getattr(result, "redirected_url", None)
        if redirected_url is not None:
            meta["redirected_url"] = redirected_url

        redirected_status_code = getattr(result, "redirected_status_code", None)
        if redirected_status_code is not None:
            meta["redirected_status_code"] = redirected_status_code

        crawl_stats = getattr(result, "crawl_stats", None)
        if crawl_stats:
            meta["crawl_stats"] = crawl_stats

        return meta

    def _classify_status(
        self,
        status_code: int | None,
        error_message: str | None,
        exc: Exception | None,
        success: bool,
    ) -> AcquisitionStatus:
        """Classify failure and success states prioritizing status codes and exceptions over string matching."""
        # 1. Successful outcome
        if success and (status_code is None or (200 <= status_code < 400)):
            return AcquisitionStatus.SUCCESS

        # 2. HTTP Status Code Mapping
        if status_code is not None:
            if status_code == 404:
                return AcquisitionStatus.NOT_FOUND
            if status_code == 429:
                return AcquisitionStatus.RATE_LIMITED
            if status_code in (403, 451):
                return AcquisitionStatus.BLOCKED
            if status_code == 401:
                return AcquisitionStatus.AUTH_REQUIRED
            if status_code in (408, 504):
                return AcquisitionStatus.TIMEOUT
            if 500 <= status_code <= 599:
                return AcquisitionStatus.SERVER_ERROR

        # 3. Exception Type Mapping
        if exc is not None:
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                return AcquisitionStatus.TIMEOUT
            if isinstance(exc, (ConnectionError, OSError)):
                return AcquisitionStatus.NETWORK_ERROR

        # 4. Error Message Matching (Fallback)
        if error_message:
            msg_lower = error_message.lower()

            if any(term in msg_lower for term in ("timeout", "timed out", "navigation timeout")):
                return AcquisitionStatus.TIMEOUT
            if any(term in msg_lower for term in ("not found", "404", "net::err_http_response_code_failure")):
                return AcquisitionStatus.NOT_FOUND
            if any(term in msg_lower for term in ("rate limit", "too many requests", "429")):
                return AcquisitionStatus.RATE_LIMITED
            if any(term in msg_lower for term in ("403", "forbidden", "blocked", "cloudflare", "captcha", "access denied", "bot detected")):
                return AcquisitionStatus.BLOCKED
            if any(term in msg_lower for term in ("401", "unauthorized", "auth required", "authentication required")):
                return AcquisitionStatus.AUTH_REQUIRED
            if any(term in msg_lower for term in ("connection", "net::err_", "dns", "name not resolved", "failed to connect", "host unreachable", "network")):
                return AcquisitionStatus.NETWORK_ERROR
            if any(term in msg_lower for term in ("parse", "parsing", "extraction", "extracting", "invalid html", "lxml")):
                return AcquisitionStatus.PARSER_ERROR
            if any(term in msg_lower for term in ("500", "502", "503", "server error", "bad gateway", "service unavailable")):
                return AcquisitionStatus.SERVER_ERROR

        return AcquisitionStatus.FAILURE
