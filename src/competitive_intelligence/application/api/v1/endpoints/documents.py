"""Documents, Versions, and Chunks API endpoints."""

from fastapi import APIRouter, Depends, Query

from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_document_service,
)
from competitive_intelligence.application.api.v1.schemas.document import (
    DocumentChunkResponse,
    DocumentResponse,
    DocumentVersionResponse,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.document_service import DocumentService

router = APIRouter(prefix="/workspaces/{workspace_id}/documents", tags=["Documents"])


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    workspace_id: str,
    company_id: str | None = Query(default=None, description="Filter by competitor company ID"),
    source_id: str | None = Query(default=None, description="Filter by source ID"),
    _ws: Workspace = Depends(get_current_workspace),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentResponse]:
    """List canonical documents in workspace."""
    docs = await service.list_documents(
        workspace_id=workspace_id, company_id=company_id, source_id=source_id
    )
    return [
        DocumentResponse(
            document_id=d.document_id,
            workspace_id=d.workspace_id,
            company_id=d.company_id,
            source_id=d.source_id,
            canonical_url=d.canonical_url,
        )
        for d in docs
    ]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    workspace_id: str,
    document_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Get canonical document metadata."""
    d = await service.get_document(workspace_id=workspace_id, document_id=document_id)
    return DocumentResponse(
        document_id=d.document_id,
        workspace_id=d.workspace_id,
        company_id=d.company_id,
        source_id=d.source_id,
        canonical_url=d.canonical_url,
    )


@router.get("/{document_id}/versions", response_model=list[DocumentVersionResponse])
async def list_document_versions(
    workspace_id: str,
    document_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentVersionResponse]:
    """List immutable version history snapshots for a document."""
    versions = await service.list_document_versions(
        workspace_id=workspace_id, document_id=document_id
    )
    return [
        DocumentVersionResponse(
            version_id=v.version_id,
            document_id=v.document_id,
            title=v.title,
            content=v.content,
            content_hash=v.content_hash,
            acquired_at=v.acquired_at,
            metadata=v.metadata,
        )
        for v in versions
    ]


@router.get(
    "/{document_id}/versions/{version_id}", response_model=DocumentVersionResponse
)
async def get_document_version(
    workspace_id: str,
    document_id: str,
    version_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DocumentService = Depends(get_document_service),
) -> DocumentVersionResponse:
    """Get specific immutable version snapshot."""
    v = await service.get_document_version(
        workspace_id=workspace_id, document_id=document_id, version_id=version_id
    )
    return DocumentVersionResponse(
        version_id=v.version_id,
        document_id=v.document_id,
        title=v.title,
        content=v.content,
        content_hash=v.content_hash,
        acquired_at=v.acquired_at,
        metadata=v.metadata,
    )


@router.get(
    "/{document_id}/versions/{version_id}/chunks",
    response_model=list[DocumentChunkResponse],
)
async def list_version_chunks(
    workspace_id: str,
    document_id: str,
    version_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentChunkResponse]:
    """Get structured chunks for an immutable document version."""
    chunks = await service.list_version_chunks(
        workspace_id=workspace_id, document_id=document_id, version_id=version_id
    )
    return [
        DocumentChunkResponse(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            version_id=c.version_id,
            chunk_index=c.metadata.chunk_index,
            content=c.content,
            content_hash=c.content_hash,
            parent_chunk_id=c.parent_chunk_id,
            chunk_level=c.chunk_level,
            start_char=c.metadata.start_char,
            end_char=c.metadata.end_char,
            total_chunks=c.metadata.total_chunks,
            language=c.metadata.language,
            custom_tags=c.metadata.custom_tags,
            created_at=c.created_at,
        )
        for c in chunks
    ]
