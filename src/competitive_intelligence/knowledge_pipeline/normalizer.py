"""Document normalization transforms for Knowledge Pipeline subsystem.

Bridges Source and AcquisitionResult into canonical Document and DocumentVersion domain entities.
"""

from datetime import datetime, timezone
import hashlib
from typing import Any
import uuid

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseDocumentNormalizer,
)
from competitive_intelligence.knowledge_pipeline.models import Document, DocumentVersion
from competitive_intelligence.source_acquisition.models import (
    AcquisitionResult,
    AcquisitionStatus,
    Source,
)


class NormalizationError(Exception):
    """Base exception for document normalization failures."""

    pass


class UnsuccessfulAcquisitionError(NormalizationError):
    """Raised when attempting to normalize an AcquisitionResult that failed."""

    def __init__(self, status: AcquisitionStatus, error_message: str | None = None) -> None:
        self.status = status
        self.error_message = error_message
        super().__init__(
            f"Cannot normalize unsuccessful acquisition result with status '{status.value}': "
            f"{error_message or 'No error details'}"
        )


class MissingContentError(NormalizationError):
    """Raised when an AcquisitionResult contains missing or whitespace-only content."""

    def __init__(self, message: str = "AcquisitionResult contains missing or empty content") -> None:
        super().__init__(message)


def compute_content_hash(content: str) -> str:
    """Compute deterministic SHA-256 hex digest of normalized content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def derive_single_document_id(source_id: str) -> str:
    """Derive deterministic document UUID for a single-document source using UUID v5."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:ci:source:{source_id}"))


class DocumentNormalizer(BaseDocumentNormalizer):
    """Normalizes acquired content from a Source into canonical Document and DocumentVersion entities.

    Scope: Single-document sources.
    Responsibilities:
    - Validates acquisition success and content availability.
    - Deterministically derives logical Document identity from the Source.
    - Generates immutable DocumentVersion snapshots with SHA-256 content hashes.
    - Preserves provenance (source_id, company_id, workspace_id, canonical_url) and structured metadata.
    """

    def normalize(
        self, source: Source, result: AcquisitionResult
    ) -> tuple[Document, DocumentVersion]:
        """Normalize an AcquisitionResult into a Document and DocumentVersion pair.

        Args:
            source: The monitored Source domain model.
            result: Successful AcquisitionResult obtained from acquisition providers.

        Returns:
            A tuple of (Document, DocumentVersion).

        Raises:
            UnsuccessfulAcquisitionError: If the acquisition attempt was not successful.
            MissingContentError: If the acquisition result content is missing or empty.
        """
        # 1. Validate acquisition outcome
        if not result.is_success or result.status != AcquisitionStatus.SUCCESS:
            raise UnsuccessfulAcquisitionError(
                status=result.status,
                error_message=result.error_message,
            )

        # 2. Validate content presence
        if result.content is None or not result.content.strip():
            raise MissingContentError("AcquisitionResult contains missing or empty content")

        normalized_content = result.content.strip()

        # 3. Compute deterministic content hash
        content_hash = compute_content_hash(normalized_content)

        # 4. Construct canonical logical Document
        doc_id = derive_single_document_id(source.identity.source_id)
        canonical_url = source.location.url_or_identifier

        document = Document(
            document_id=doc_id,
            source_id=source.identity.source_id,
            company_id=source.identity.company_id,
            workspace_id=source.identity.workspace_id,
            canonical_url=canonical_url,
        )

        # 5. Extract title and structured metadata
        title = self._extract_title(result)
        metadata = self._build_version_metadata(source, result)

        # 6. Construct DocumentVersion snapshot
        version = DocumentVersion(
            version_id=str(uuid.uuid4()),
            document_id=document.document_id,
            content=normalized_content,
            title=title,
            content_hash=content_hash,
            acquired_at=result.acquired_at,
            metadata=metadata,
        )

        return document, version

    def _extract_title(self, result: AcquisitionResult) -> str | None:
        """Extract title from result metadata if available and non-empty."""
        if isinstance(result.metadata, dict):
            raw_title = result.metadata.get("title")
            if isinstance(raw_title, str) and raw_title.strip():
                return raw_title.strip()
        return None

    def _build_version_metadata(
        self, source: Source, result: AcquisitionResult
    ) -> dict[str, Any]:
        """Construct normalized version metadata with provenance and acquisition details."""
        meta: dict[str, Any] = {
            "source_url": source.location.url_or_identifier,
            "source_category": source.classification.category.value,
            "platform": source.classification.platform.value,
        }

        if result.method is not None:
            meta["acquisition_method"] = result.method.value
        if result.provider_name:
            meta["provider_name"] = result.provider_name
        if result.status_code is not None:
            meta["status_code"] = result.status_code

        # Preserve useful native metadata if present
        for key in (
            "links",
            "media",
            "tables",
            "response_headers",
            "redirected_url",
            "redirected_status_code",
            "crawl_stats",
        ):
            if key in result.metadata and result.metadata[key]:
                meta[key] = result.metadata[key]

        return meta
