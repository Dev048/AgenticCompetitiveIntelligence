"""In-memory Document Repository implementation for testing and standalone orchestration."""

from competitive_intelligence.application.interfaces import BaseDocumentRepository
from competitive_intelligence.knowledge_pipeline.models import (
    Document,
    DocumentChunk,
    DocumentVersion,
)


class InMemoryDocumentRepository(BaseDocumentRepository):
    """In-memory implementation of BaseDocumentRepository."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.versions: dict[str, DocumentVersion] = {}
        self.chunks: dict[str, DocumentChunk] = {}

    async def create_document(self, document: Document) -> Document:
        self.documents[document.document_id] = document
        return document

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def get_document_by_id(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def get_document_by_canonical_url(
        self, source_id: str, canonical_url: str
    ) -> Document | None:
        for doc in self.documents.values():
            if doc.source_id == source_id and doc.canonical_url == canonical_url:
                return doc
        return None

    async def list_documents(
        self,
        workspace_id: str,
        company_id: str | None = None,
        source_id: str | None = None,
    ) -> list[Document]:
        return [
            d for d in self.documents.values()
            if d.workspace_id == workspace_id
            and (company_id is None or d.company_id == company_id)
            and (source_id is None or d.source_id == source_id)
        ]

    async def list_documents_by_company(self, company_id: str) -> list[Document]:
        return [d for d in self.documents.values() if d.company_id == company_id]

    async def create_version(self, version: DocumentVersion) -> DocumentVersion:
        # Check uniqueness constraint on (document_id, content_hash)
        for existing in self.versions.values():
            if (
                existing.document_id == version.document_id
                and existing.content_hash == version.content_hash
            ):
                return existing
        self.versions[version.version_id] = version
        return version

    async def get_version(self, version_id: str) -> DocumentVersion | None:
        return self.versions.get(version_id)

    async def get_version_by_id(self, version_id: str) -> DocumentVersion | None:
        return self.versions.get(version_id)

    async def get_version_by_hash(
        self, document_id: str, content_hash: str
    ) -> DocumentVersion | None:
        for ver in self.versions.values():
            if ver.document_id == document_id and ver.content_hash == content_hash:
                return ver
        return None

    async def get_latest_version(self, document_id: str) -> DocumentVersion | None:
        doc_vers = [v for v in self.versions.values() if v.document_id == document_id]
        if not doc_vers:
            return None
        doc_vers.sort(key=lambda x: x.acquired_at, reverse=True)
        return doc_vers[0]

    async def list_versions(self, document_id: str) -> list[DocumentVersion]:
        doc_vers = [v for v in self.versions.values() if v.document_id == document_id]
        doc_vers.sort(key=lambda x: x.acquired_at)
        return doc_vers

    async def list_versions_for_document(self, document_id: str) -> list[DocumentVersion]:
        return await self.list_versions(document_id)

    async def create_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        self.chunks[chunk.chunk_id] = chunk
        return chunk

    async def create_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        for c in chunks:
            self.chunks[c.chunk_id] = c
        return chunks

    async def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        return self.chunks.get(chunk_id)

    async def get_chunks_for_version(self, version_id: str) -> list[DocumentChunk]:
        ver_chunks = [c for c in self.chunks.values() if c.version_id == version_id]
        ver_chunks.sort(key=lambda x: x.chunk_index)
        return ver_chunks

    async def list_chunks_for_version(self, version_id: str) -> list[DocumentChunk]:
        return await self.get_chunks_for_version(version_id)

    async def get_child_chunks(self, parent_chunk_id: str) -> list[DocumentChunk]:
        children = [c for c in self.chunks.values() if c.parent_chunk_id == parent_chunk_id]
        children.sort(key=lambda x: x.chunk_index)
        return children
