"""Unit tests for Knowledge Pipeline DocumentNormalizer.

Verifies single-document normalization, deterministic document identity derivation,
version creation, SHA-256 content hashing, metadata extraction, error handling,
and provider agnosticism.
"""

from datetime import datetime, timezone
import hashlib
import sys
from typing import Any

import pytest

from competitive_intelligence.knowledge_pipeline import (
    DocumentNormalizer,
    MissingContentError,
    NormalizationError,
    UnsuccessfulAcquisitionError,
    compute_content_hash,
    derive_single_document_id,
)
from competitive_intelligence.source_acquisition import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    AcquisitionResult,
    AcquisitionStatus,
    ContentType,
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


def create_test_source(
    source_id: str = "src_anthropic_pricing",
    company_id: str = "comp_anthropic",
    workspace_id: str = "ws_ci_default",
    url: str = "https://www.anthropic.com/pricing",
) -> Source:
    """Build a valid test Source domain model."""
    return Source(
        identity=SourceIdentity(
            source_id=source_id,
            company_id=company_id,
            workspace_id=workspace_id,
            display_name="Anthropic Pricing Page",
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
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
        ),
    )


def create_test_result(
    content: str | None = "# Pricing\nClaude Pro: $20/mo\nTeam: $25/seat/mo",
    status: AcquisitionStatus = AcquisitionStatus.SUCCESS,
    method: AcquisitionMethod | None = AcquisitionMethod.CRAWL4AI,
    metadata: dict[str, Any] | None = None,
    acquired_at: datetime | None = None,
) -> AcquisitionResult:
    """Build a valid test AcquisitionResult domain model."""
    now = acquired_at or datetime.now(timezone.utc)
    meta = metadata if metadata is not None else {
        "title": "Anthropic Pricing",
        "links": {"internal": [{"href": "/contact"}]},
        "tables": [{"rows": [["Plan", "Price"]]}],
    }
    return AcquisitionResult(
        status=status,
        content=content,
        metadata=meta,
        acquired_at=now,
        method=method,
        provider_name="Crawl4AIClient",
        status_code=200 if status == AcquisitionStatus.SUCCESS else 500,
    )


# ==============================================================================
# Normalization Unit Tests
# ==============================================================================


def test_successful_single_document_normalization() -> None:
    """Test successful normalization produces Document and DocumentVersion with correct fields."""
    normalizer = DocumentNormalizer()
    source = create_test_source()
    result = create_test_result()

    doc, version = normalizer.normalize(source, result)

    # Validate Document
    assert doc.document_id is not None
    assert doc.source_id == source.identity.source_id
    assert doc.company_id == source.identity.company_id
    assert doc.workspace_id == source.identity.workspace_id
    assert doc.canonical_url == source.location.url_or_identifier

    # Validate DocumentVersion
    assert version.version_id is not None
    assert version.document_id == doc.document_id
    assert version.content == result.content
    assert version.title == "Anthropic Pricing"
    assert version.content_hash == compute_content_hash(result.content or "")
    assert version.acquired_at == result.acquired_at
    assert version.metadata["source_url"] == source.location.url_or_identifier
    assert version.metadata["source_category"] == "pricing"
    assert version.metadata["acquisition_method"] == "crawl4ai"
    assert version.metadata["provider_name"] == "Crawl4AIClient"
    assert version.metadata["links"] == {"internal": [{"href": "/contact"}]}


def test_stable_document_identity_across_multiple_normalizations() -> None:
    """Test that normalizing the same Source produces the identical deterministic document_id."""
    normalizer = DocumentNormalizer()
    source = create_test_source(source_id="src_stable_test_123")

    result1 = create_test_result(content="Initial Content Snapshot")
    result2 = create_test_result(content="Updated Content Snapshot")

    doc1, version1 = normalizer.normalize(source, result1)
    doc2, version2 = normalizer.normalize(source, result2)

    # Document ID must be strictly identical
    assert doc1.document_id == doc2.document_id
    assert doc1.document_id == derive_single_document_id("src_stable_test_123")

    # DocumentVersions reference the same document but have unique version IDs
    assert version1.document_id == doc1.document_id
    assert version2.document_id == doc1.document_id
    assert version1.version_id != version2.version_id


def test_content_hash_determinism_and_uniqueness() -> None:
    """Test that identical content generates identical SHA-256 hashes, and changes generate different hashes."""
    content_a = "# Pricing Tier: $50"
    content_b = "# Pricing Tier: $60"

    expected_hash_a = hashlib.sha256(content_a.encode("utf-8")).hexdigest()
    expected_hash_b = hashlib.sha256(content_b.encode("utf-8")).hexdigest()

    assert compute_content_hash(content_a) == expected_hash_a
    assert compute_content_hash(content_b) == expected_hash_b
    assert compute_content_hash(content_a) != compute_content_hash(content_b)

    normalizer = DocumentNormalizer()
    source = create_test_source()

    _, v1 = normalizer.normalize(source, create_test_result(content=content_a))
    _, v2 = normalizer.normalize(source, create_test_result(content=content_a))
    _, v3 = normalizer.normalize(source, create_test_result(content=content_b))

    assert v1.content_hash == v2.content_hash
    assert v1.content_hash == expected_hash_a
    assert v3.content_hash == expected_hash_b
    assert v1.content_hash != v3.content_hash


def test_metadata_and_title_extraction() -> None:
    """Test title extraction when present and fallback to None when omitted."""
    normalizer = DocumentNormalizer()
    source = create_test_source()

    # Case 1: Title in metadata
    result_with_title = create_test_result(metadata={"title": "  Custom Extracted Title  "})
    _, version_with_title = normalizer.normalize(source, result_with_title)
    assert version_with_title.title == "Custom Extracted Title"

    # Case 2: No title in metadata
    result_no_title = create_test_result(metadata={"links": {}})
    _, version_no_title = normalizer.normalize(source, result_no_title)
    assert version_no_title.title is None

    # Case 3: Empty string title in metadata
    result_empty_title = create_test_result(metadata={"title": "   "})
    _, version_empty_title = normalizer.normalize(source, result_empty_title)
    assert version_empty_title.title is None


def test_provenance_fields_preservation() -> None:
    """Verify that source_id, company_id, workspace_id, and canonical_url are accurately preserved."""
    normalizer = DocumentNormalizer()
    source = create_test_source(
        source_id="src_unique_99",
        company_id="comp_special",
        workspace_id="ws_enterprise",
        url="https://competitor.com/enterprise/pricing",
    )
    result = create_test_result()

    doc, version = normalizer.normalize(source, result)

    assert doc.source_id == "src_unique_99"
    assert doc.company_id == "comp_special"
    assert doc.workspace_id == "ws_enterprise"
    assert doc.canonical_url == "https://competitor.com/enterprise/pricing"
    assert version.metadata["source_url"] == "https://competitor.com/enterprise/pricing"


@pytest.mark.parametrize(
    "failed_status",
    [
        AcquisitionStatus.RATE_LIMITED,
        AcquisitionStatus.NOT_FOUND,
        AcquisitionStatus.BLOCKED,
        AcquisitionStatus.TIMEOUT,
        AcquisitionStatus.NETWORK_ERROR,
        AcquisitionStatus.PARSER_ERROR,
        AcquisitionStatus.SERVER_ERROR,
        AcquisitionStatus.FAILURE,
    ],
)
def test_unsuccessful_acquisition_rejected(failed_status: AcquisitionStatus) -> None:
    """Test that any unsuccessful acquisition attempt raises UnsuccessfulAcquisitionError."""
    normalizer = DocumentNormalizer()
    source = create_test_source()
    failed_result = create_test_result(
        status=failed_status,
        content=None,
    )

    with pytest.raises(UnsuccessfulAcquisitionError) as exc_info:
        normalizer.normalize(source, failed_result)

    assert exc_info.value.status == failed_status
    assert issubclass(UnsuccessfulAcquisitionError, NormalizationError)


@pytest.mark.parametrize("empty_content", [None, "", "   ", "\n\t  \n"])
def test_missing_or_empty_content_rejected(empty_content: str | None) -> None:
    """Test that successful acquisitions with missing or whitespace-only content raise MissingContentError."""
    normalizer = DocumentNormalizer()
    source = create_test_source()

    result = AcquisitionResult(
        status=AcquisitionStatus.SUCCESS,
        content=empty_content,
        status_code=200,
    )

    with pytest.raises(MissingContentError, match="missing or empty content"):
        normalizer.normalize(source, result)

    assert issubclass(MissingContentError, NormalizationError)


def test_timezone_aware_acquired_at_preserved() -> None:
    """Test that timezone-aware acquired_at timestamp is preserved on DocumentVersion."""
    normalizer = DocumentNormalizer()
    source = create_test_source()
    custom_time = datetime(2026, 8, 22, 14, 30, 0, tzinfo=timezone.utc)
    result = create_test_result(acquired_at=custom_time)

    _, version = normalizer.normalize(source, result)

    assert version.acquired_at == custom_time
    assert version.acquired_at.tzinfo == timezone.utc


def test_normalizer_has_no_dependency_on_crawl4ai() -> None:
    """Verify that the normalizer module operates purely on domain contracts with zero crawl4ai imports."""
    import competitive_intelligence.knowledge_pipeline.normalizer as norm_mod

    module_file = getattr(norm_mod, "__file__", "")
    assert module_file != ""

    with open(module_file, "r", encoding="utf-8") as f:
        code_text = f.read()

    assert "crawl4ai" not in code_text
    assert "AsyncWebCrawler" not in code_text

    # Verify normalizer works seamlessly with non-Crawl4AI acquisition methods
    normalizer = DocumentNormalizer()
    source = create_test_source()
    direct_http_result = create_test_result(
        method=AcquisitionMethod.DIRECT_HTTP,
        metadata={"title": "HTTP Source"},
    )
    doc, version = normalizer.normalize(source, direct_http_result)
    assert version.metadata["acquisition_method"] == "direct_http"

    agent_reach_result = create_test_result(
        method=AcquisitionMethod.AGENT_REACH,
        metadata={"title": "Social Source"},
    )
    doc_reach, version_reach = normalizer.normalize(source, agent_reach_result)
    assert version_reach.metadata["acquisition_method"] == "agent_reach"
