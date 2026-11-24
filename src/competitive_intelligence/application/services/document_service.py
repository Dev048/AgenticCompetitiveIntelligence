"""Application use-case service for Canonical Documents, Versions, and Chunks."""

from competitive_intelligence.application.interfaces import (
    BaseDocumentRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.core.errors import (
    EntityNotFoundError,
    WorkspaceIsolationError,
)
from competitive_intelligence.knowledge_pipeline.models import (
    Document,
    DocumentChunk,
    DocumentVersion,
)


class DocumentService:
    """Manages document metadata, version lineage, and chunk queries with workspace isolation."""

    def __init__(
        self,
        document_repo: BaseDocumentRepository,
        workspace_repo: BaseWorkspaceRepository | None = None,
    ) -> None:
        self._document_repo = document_repo
        self._workspace_repo = workspace_repo

    async def list_documents(
        self,
        workspace_id: str,
        company_id: str | None = None,
        source_id: str | None = None,
    ) -> list[Document]:
        """List canonical documents in workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        return await self._document_repo.list_documents(
            workspace_id=workspace_id, company_id=company_id, source_id=source_id
        )

    async def get_document(self, workspace_id: str, document_id: str) -> Document:
        """Get canonical document by ID and verify workspace isolation."""
        doc = await self._document_repo.get_document(document_id)
        if doc is None:
            raise EntityNotFoundError(f"Document '{document_id}' not found")
        if doc.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Document '{document_id}' does not belong to workspace '{workspace_id}'"
            )
        return doc

    async def list_document_versions(
        self, workspace_id: str, document_id: str
    ) -> list[DocumentVersion]:
        """List immutable version history snapshots for a document."""
        # Verify document exists and belongs to workspace
        await self.get_document(workspace_id, document_id)
        return await self._document_repo.list_versions(document_id)

    async def get_document_version(
        self, workspace_id: str, document_id: str, version_id: str
    ) -> DocumentVersion:
        """Get specific immutable version snapshot."""
        # Verify document belongs to workspace
        await self.get_document(workspace_id, document_id)
        ver = await self._document_repo.get_version(version_id)
        if ver is None or ver.document_id != document_id:
            raise EntityNotFoundError(
                f"Version '{version_id}' not found for document '{document_id}'"
            )
        return ver

    async def list_version_chunks(
        self, workspace_id: str, document_id: str, version_id: str
    ) -> list[DocumentChunk]:
        """Get structured chunks for an immutable document version."""
        # Verify version and workspace ownership
        await self.get_document_version(workspace_id, document_id, version_id)
        return await self._document_repo.get_chunks_for_version(version_id)
