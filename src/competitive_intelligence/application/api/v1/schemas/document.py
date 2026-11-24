"""API schemas for Documents, Versions, and Chunks."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Canonical document response DTO."""

    document_id: str
    workspace_id: str
    company_id: str
    source_id: str
    canonical_url: str


class DocumentVersionResponse(BaseModel):
    """Immutable document version snapshot DTO."""

    version_id: str
    document_id: str
    title: str | None = None
    content: str
    content_hash: str
    acquired_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunkResponse(BaseModel):
    """Document chunk response DTO."""

    chunk_id: str
    document_id: str
    version_id: str
    chunk_index: int
    content: str
    content_hash: str
    parent_chunk_id: str | None = None
    chunk_level: int
    start_char: int | None = None
    end_char: int | None = None
    total_chunks: int
    language: str
    custom_tags: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

