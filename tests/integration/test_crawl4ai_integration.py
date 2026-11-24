"""End-to-end integration test for the Source Acquisition subsystem using real Crawl4AI.

Executes the full pipeline:
Source -> ProviderRegistry -> AcquisitionOrchestrator -> Crawl4AIClient -> Crawl4AI -> real public webpage -> AcquisitionResult

Marked with @pytest.mark.integration as it requires live internet access and real browser automation.
"""

from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from typing import Any

import pytest

from competitive_intelligence.source_acquisition import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    AcquisitionOrchestrator,
    AcquisitionResult,
    AcquisitionStatus,
    ContentType,
    Crawl4AIClient,
    PlatformType,
    ProviderRegistry,
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)


class _MockPricingPageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = """<!DOCTYPE html>
<html>
<head><title>Acme Pricing Intelligence</title></head>
<body>
<h1>Acme Corp Enterprise Offerings</h1>
<p>Acme Corp announced Enterprise Flagship Tier priced at $1,299 per month with dedicated support and unlimited vector search.</p>
<a href="/pricing/enterprise">Enterprise Details</a>
</body>
</html>"""
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def local_pricing_page() -> Generator[str, None, None]:
    """Provide a controlled local HTTP server serving known page content for Crawl4AI."""
    server = HTTPServer(("127.0.0.1", 0), _MockPricingPageHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/pricing"
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.integration
@pytest.mark.anyio
async def test_crawl4ai_real_web_acquisition_end_to_end(local_pricing_page: str) -> None:
    """Verify live acquisition of a controlled webpage through AcquisitionOrchestrator and Crawl4AI."""
    test_url = local_pricing_page

    # 1. Construct real Source domain model
    source = Source(
        identity=SourceIdentity(
            company_id="company_public_benchmark",
            workspace_id="ws_integration_test",
            display_name="Acme Benchmark Public Page",
        ),
        classification=SourceClassification(
            category=SourceCategory.WEBSITE,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier=test_url,
            access_pattern=AccessPattern.PUBLIC_WEB,
            requires_javascript=False,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(
                primary=AcquisitionMethod.CRAWL4AI,
                fallbacks=[],
            ),
        ),
    )

    # 2. Instantiate real components
    client = Crawl4AIClient()
    registry = ProviderRegistry()
    registry.register(AcquisitionMethod.CRAWL4AI, client)
    orchestrator = AcquisitionOrchestrator(providers=registry)

    # 3. Execute real acquisition
    result: AcquisitionResult = await orchestrator.acquire(source)

    # 4. Verify outcome contracts
    assert result.status == AcquisitionStatus.SUCCESS
    assert result.is_success is True
    assert result.is_failure is False
    assert result.method == AcquisitionMethod.CRAWL4AI
    assert result.provider_name == "Crawl4AIClient"
    assert result.status_code == 200
    assert result.acquired_at is not None
    assert result.acquired_at.tzinfo is not None

    # 5. Verify acquired content matches known page content
    assert result.has_content is True
    assert result.content is not None
    assert "Acme Corp Enterprise Offerings" in result.content
    assert "Enterprise Flagship Tier priced at $1,299 per month" in result.content

    # 6. Verify native metadata preservation
    assert "url" in result.metadata
    assert test_url in result.metadata["url"]
    assert result.metadata.get("title") == "Acme Pricing Intelligence"
    assert "links" in result.metadata
