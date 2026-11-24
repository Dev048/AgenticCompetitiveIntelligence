"""Unit tests for Knowledge Pipeline canonical document domain models (Document, DocumentVersion)."""

from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
import pytest

from competitive_intelligence.knowledge_pipeline import (
    Document,
    DocumentVersion,
    SourceDocument,
)


# ==============================================================================
# Document Domain Model Tests
# ==============================================================================


def test_valid_document_instantiation() -> None:
    """Test creating a valid Document with generated and explicit IDs."""
    # Auto-generated document_id
    doc = Document(
        source_id="src_anthropic_pricing",
        company_id="comp_anthropic",
        workspace_id="ws_default",
        canonical_url="https://www.anthropic.com/pricing",
    )

    assert doc.document_id is not None
    assert len(doc.document_id) > 0
    assert doc.source_id == "src_anthropic_pricing"
    assert doc.company_id == "comp_anthropic"
    assert doc.workspace_id == "ws_default"
    assert doc.canonical_url == "https://www.anthropic.com/pricing"

    # Explicit document_id
    doc_explicit = Document(
        document_id="doc_custom_123",
        source_id="src_1",
        company_id="comp_1",
        workspace_id="ws_1",
        canonical_url="https://example.com/page",
    )
    assert doc_explicit.document_id == "doc_custom_123"


def test_source_document_alias() -> None:
    """Test that SourceDocument alias is identical to Document."""
    assert SourceDocument is Document
    doc = SourceDocument(
        source_id="src_1",
        company_id="comp_1",
        workspace_id="ws_1",
        canonical_url="https://example.com",
    )
    assert isinstance(doc, Document)


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"document_id": ""},
        {"document_id": "   "},
        {"source_id": ""},
        {"source_id": "   "},
        {"company_id": ""},
        {"company_id": "   "},
        {"workspace_id": ""},
        {"workspace_id": "   "},
    ],
)
def test_invalid_document_identity_fields(invalid_field: dict[str, str]) -> None:
    """Test that empty or whitespace-only identity fields raise ValidationError."""
    base_kwargs: dict[str, Any] = {
        "document_id": "doc_1",
        "source_id": "src_1",
        "company_id": "comp_1",
        "workspace_id": "ws_1",
        "canonical_url": "https://example.com",
    }
    base_kwargs.update(invalid_field)

    with pytest.raises(ValidationError, match="Identifier cannot be empty"):
        Document(**base_kwargs)


@pytest.mark.parametrize(
    "invalid_url",
    [
        "",
        "   ",
        "not-a-valid-url",
        "ftp://example.com/file",
        "https://",
        "http://",
        "/relative/path/page.html",
    ],
)
def test_invalid_document_canonical_url(invalid_url: str) -> None:
    """Test that malformed or non-HTTP/HTTPS canonical URLs raise ValidationError."""
    with pytest.raises(ValidationError, match="Invalid URL format|canonical_url cannot be empty"):
        Document(
            source_id="src_1",
            company_id="comp_1",
            workspace_id="ws_1",
            canonical_url=invalid_url,
        )


# ==============================================================================
# DocumentVersion Domain Model Tests
# ==============================================================================


def test_valid_document_version_instantiation() -> None:
    """Test creating a valid DocumentVersion with defaults and custom metadata."""
    now = datetime.now(timezone.utc)
    version = DocumentVersion(
        document_id="doc_123",
        content="# Pricing\nTier 1: $20/mo",
        title="Pricing Tier 1",
        content_hash="sha256_hash_abc123",
        acquired_at=now,
        metadata={"author": "Marketing Team", "http_status": 200},
    )

    assert version.version_id is not None
    assert len(version.version_id) > 0
    assert version.document_id == "doc_123"
    assert version.content == "# Pricing\nTier 1: $20/mo"
    assert version.title == "Pricing Tier 1"
    assert version.content_hash == "sha256_hash_abc123"
    assert version.acquired_at == now
    assert version.metadata == {"author": "Marketing Team", "http_status": 200}


def test_document_version_metadata_defaults_to_empty_dict() -> None:
    """Test that metadata defaults to an independent empty dictionary."""
    v1 = DocumentVersion(
        document_id="doc_1",
        content="Content 1",
        content_hash="hash_1",
    )
    v2 = DocumentVersion(
        document_id="doc_1",
        content="Content 2",
        content_hash="hash_2",
    )

    assert v1.metadata == {}
    assert v2.metadata == {}

    # Verify mutating one doesn't affect the other
    v1.metadata["custom"] = "value"
    assert v2.metadata == {}


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"version_id": ""},
        {"version_id": "   "},
        {"document_id": ""},
        {"document_id": "   "},
    ],
)
def test_invalid_document_version_identifiers(invalid_field: dict[str, str]) -> None:
    """Test that empty or whitespace-only version identifiers raise ValidationError."""
    base_kwargs: dict[str, Any] = {
        "version_id": "ver_1",
        "document_id": "doc_1",
        "content": "Valid Content",
        "content_hash": "hash_123",
    }
    base_kwargs.update(invalid_field)

    with pytest.raises(ValidationError, match="Field cannot be empty"):
        DocumentVersion(**base_kwargs)


@pytest.mark.parametrize("empty_content", ["", "   "])
def test_empty_document_version_content(empty_content: str) -> None:
    """Test that empty or whitespace-only content raises ValidationError."""
    with pytest.raises(ValidationError, match="Document content cannot be empty"):
        DocumentVersion(
            document_id="doc_1",
            content=empty_content,
            content_hash="hash_123",
        )


@pytest.mark.parametrize("empty_hash", ["", "   "])
def test_empty_document_version_content_hash(empty_hash: str) -> None:
    """Test that empty or whitespace-only content_hash raises ValidationError."""
    with pytest.raises(ValidationError, match="Field cannot be empty"):
        DocumentVersion(
            document_id="doc_1",
            content="Valid content",
            content_hash=empty_hash,
        )


def test_document_version_timezone_naive_acquired_at() -> None:
    """Test that timezone-naive acquired_at timestamps raise ValidationError."""
    naive_dt = datetime(2026, 8, 22, 12, 0, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        DocumentVersion(
            document_id="doc_1",
            content="Valid content",
            content_hash="hash_123",
            acquired_at=naive_dt,
        )


def test_multiple_versions_referencing_same_document_id() -> None:
    """Test that multiple DocumentVersion instances can track the evolution of one Document."""
    doc = Document(
        source_id="src_pricing",
        company_id="comp_1",
        workspace_id="ws_1",
        canonical_url="https://competitor.com/pricing",
    )

    t1 = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
    v1 = DocumentVersion(
        document_id=doc.document_id,
        content="# Pricing\nBasic: $10/mo\nPro: $20/mo",
        title="Pricing August 20",
        content_hash="hash_v1",
        acquired_at=t1,
    )

    t2 = datetime(2026, 8, 22, 10, 0, 0, tzinfo=timezone.utc)
    v2 = DocumentVersion(
        document_id=doc.document_id,
        content="# Pricing\nBasic: $15/mo\nPro: $25/mo\nEnterprise: Custom",
        title="Pricing August 22",
        content_hash="hash_v2",
        acquired_at=t2,
    )

    assert v1.document_id == doc.document_id
    assert v2.document_id == doc.document_id
    assert v1.version_id != v2.version_id
    assert v1.content_hash != v2.content_hash
    assert v1.acquired_at < v2.acquired_at
