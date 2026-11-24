"""Qdrant vector store adapter for Knowledge Pipeline and Retrieval subsystem.

Manages collection lifecycle, point mapping, vector dimension validation,
similarity search, and payload filtering for 1024-dimensional embeddings (e.g. BAAI/bge-m3).
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient, models as qdrant_models

from competitive_intelligence.knowledge_pipeline.interfaces import BaseVectorStore
from competitive_intelligence.knowledge_pipeline.models import (
    DistanceMetric,
    SearchConfiguration,
    VectorSearchFilter,
    VectorSearchResult,
    VectorStoreConfig,
    VectorStoreEntry,
)


class VectorStoreError(Exception):
    """Exception raised for failures in vector store operations."""


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed implementation of BaseVectorStore.

    Responsibilities:
    - Ensures the Qdrant collection exists with configured dimensions (1024) and distance (COSINE).
    - Upserts VectorStoreEntry points with deterministic chunk_id UUID point IDs.
    - Preserves rich provenance payload (workspace_id, company_id, version_id, parent_chunk_id).
    - Supports multiple historical document versions in the same collection.
    - Executes vector similarity searches and returns clean domain VectorSearchResult models.
    - Applies metadata filters on payload fields without exposing Qdrant types.
    - Supports in-memory mode (:memory:) for local testing and standalone Qdrant endpoints.
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        client: Any = None,
    ) -> None:
        """Initialize QdrantVectorStore.

        Args:
            config: Optional VectorStoreConfig. Defaults to 1024-dim Cosine BGE-M3 config.
            client: Optional pre-configured QdrantClient for testing / dependency injection.
        """
        self.config = config or VectorStoreConfig()

        if client is not None:
            self._client = client
        else:
            if self.config.url:
                self._client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                )
            elif self.config.path:
                self._client = QdrantClient(
                    path=self.config.path,
                    timeout=self.config.timeout,
                )
            else:
                self._client = QdrantClient(
                    location=self.config.location or ":memory:",
                    timeout=self.config.timeout,
                )

    def ensure_collection(self) -> None:
        """Create or verify the vector collection exists with expected dimensions and metric."""
        try:
            exists = self._client.collection_exists(self.config.collection_name)
            if not exists:
                distance_map = {
                    DistanceMetric.COSINE: qdrant_models.Distance.COSINE,
                    DistanceMetric.DOT: qdrant_models.Distance.DOT,
                    DistanceMetric.EUCLID: qdrant_models.Distance.EUCLID,
                }
                distance = distance_map.get(
                    self.config.distance_metric, qdrant_models.Distance.COSINE
                )

                self._client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.config.vector_dimensions,
                        distance=distance,
                    ),
                )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to ensure collection '{self.config.collection_name}': {exc}"
            ) from exc

    def upsert_entries(self, entries: Sequence[VectorStoreEntry]) -> None:
        """Upsert a batch of VectorStoreEntry points into the collection.

        Args:
            entries: Sequence of VectorStoreEntry points to store.

        Raises:
            ValueError: If vector dimension does not match collection configuration.
            VectorStoreError: If Qdrant client upsert fails.
        """
        if not entries:
            return

        points: list[qdrant_models.PointStruct] = []

        for entry in entries:
            if len(entry.vector) != self.config.vector_dimensions:
                raise ValueError(
                    f"Vector dimension mismatch for chunk '{entry.chunk_id}': "
                    f"expected {self.config.vector_dimensions}, got {len(entry.vector)}"
                )

            payload: dict[str, Any] = {
                "text": entry.text,
                "chunk_id": entry.chunk_id,
                "document_id": entry.document_id,
                "version_id": entry.version_id,
                "source_id": entry.source_id,
                "company_id": entry.company_id,
                "workspace_id": entry.workspace_id,
                "source_category": entry.source_category,
                "content_type": entry.content_type,
                "canonical_url": entry.canonical_url,
                "chunk_index": entry.chunk_index,
                "parent_chunk_id": entry.parent_chunk_id,
                "chunk_level": entry.chunk_level,
                "collected_at": (
                    entry.collected_at.isoformat() if entry.collected_at else None
                ),
                "title": entry.title,
                "metadata": entry.metadata,
            }

            points.append(
                qdrant_models.PointStruct(
                    id=entry.chunk_id,
                    vector=entry.vector,
                    payload=payload,
                )
            )

        try:
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=points,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert {len(points)} points into '{self.config.collection_name}': {exc}"
            ) from exc

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        filters: VectorSearchFilter | None = None,
        search_config: SearchConfiguration | None = None,
    ) -> list[VectorSearchResult]:
        """Perform vector similarity search against the collection.

        Args:
            query_vector: Query dense vector (must match configured dimensions).
            top_k: Maximum number of ranked results to return (used if search_config is not provided).
            filters: Optional metadata payload filter conditions.
            search_config: Optional SearchConfiguration controlling top_k, hnsw_ef, exact_search.

        Returns:
            A list of VectorSearchResult instances ordered by descending similarity score.

        Raises:
            ValueError: If query vector dimensions mismatch or top_k is non-positive.
            VectorStoreError: If Qdrant query fails.
        """
        if len(query_vector) != self.config.vector_dimensions:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.config.vector_dimensions}, got {len(query_vector)}"
            )

        effective_top_k = search_config.top_k if search_config is not None else top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        qdrant_filter = self._build_qdrant_filter(filters)

        qdrant_search_params: qdrant_models.SearchParams | None = None
        if search_config is not None:
            if search_config.hnsw_ef is not None or search_config.exact_search:
                qdrant_search_params = qdrant_models.SearchParams(
                    hnsw_ef=search_config.hnsw_ef,
                    exact=search_config.exact_search,
                )

        score_threshold = (
            search_config.score_threshold if search_config is not None else None
        )

        try:
            results = self._client.query_points(
                collection_name=self.config.collection_name,
                query=list(query_vector),
                limit=effective_top_k,
                query_filter=qdrant_filter,
                search_params=qdrant_search_params,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Search query failed in collection '{self.config.collection_name}': {exc}"
            ) from exc

        return [self._map_scored_point_to_result(pt) for pt in results.points]

    def get_by_id(self, chunk_id: str) -> VectorSearchResult | None:
        """Retrieve a single vector entry by its chunk_id."""
        if not chunk_id or not chunk_id.strip():
            raise ValueError("chunk_id cannot be empty or whitespace-only")

        try:
            points = self._client.retrieve(
                collection_name=self.config.collection_name,
                ids=[chunk_id.strip()],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to retrieve point '{chunk_id}' from '{self.config.collection_name}': {exc}"
            ) from exc

        if not points:
            return None

        pt = points[0]
        payload = pt.payload or {}
        collected_at_str = payload.get("collected_at")
        collected_at = (
            datetime.fromisoformat(collected_at_str) if collected_at_str else None
        )

        return VectorSearchResult(
            chunk_id=str(payload.get("chunk_id", pt.id)),
            score=1.0,
            text=str(payload.get("text", "")),
            document_id=str(payload.get("document_id", "")),
            version_id=str(payload.get("version_id", "")),
            source_id=payload.get("source_id"),
            company_id=payload.get("company_id"),
            workspace_id=payload.get("workspace_id"),
            source_category=payload.get("source_category"),
            content_type=payload.get("content_type"),
            canonical_url=payload.get("canonical_url"),
            chunk_index=int(payload.get("chunk_index", 0)),
            parent_chunk_id=payload.get("parent_chunk_id"),
            chunk_level=int(payload.get("chunk_level", 0)),
            collected_at=collected_at,
            title=payload.get("title"),
            metadata=dict(payload.get("metadata", {})),
        )

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> None:
        """Delete specific points by their chunk_ids."""
        if not chunk_ids:
            return

        try:
            self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=qdrant_models.PointIdsList(points=list(chunk_ids)),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete {len(chunk_ids)} points from '{self.config.collection_name}': {exc}"
            ) from exc

    def delete_by_filter(self, filters: VectorSearchFilter) -> None:
        """Delete points matching payload filter conditions."""
        qdrant_filter = self._build_qdrant_filter(filters)
        if qdrant_filter is None:
            raise ValueError("Must provide at least one filter condition to delete by filter")

        try:
            self._client.delete(
                collection_name=self.config.collection_name,
                points_selector=qdrant_models.FilterSelector(filter=qdrant_filter),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete points by filter from '{self.config.collection_name}': {exc}"
            ) from exc

    def _build_qdrant_filter(
        self, filters: VectorSearchFilter | None
    ) -> qdrant_models.Filter | None:
        """Translate VectorSearchFilter domain model into Qdrant Filter."""
        if filters is None:
            return None

        must_conditions: list[qdrant_models.FieldCondition] = []

        fields_to_check = [
            ("workspace_id", filters.workspace_id),
            ("company_id", filters.company_id),
            ("source_id", filters.source_id),
            ("source_category", filters.source_category),
            ("content_type", filters.content_type),
            ("version_id", filters.version_id),
            ("document_id", filters.document_id),
            ("parent_chunk_id", filters.parent_chunk_id),
            ("chunk_level", filters.chunk_level),
        ]

        for key, value in fields_to_check:
            if value is not None:
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                )

        if not must_conditions:
            return None

        return qdrant_models.Filter(must=must_conditions)  # type: ignore[arg-type]

    def _map_scored_point_to_result(
        self, pt: qdrant_models.ScoredPoint
    ) -> VectorSearchResult:
        """Map a Qdrant ScoredPoint into our canonical VectorSearchResult domain model."""
        payload = pt.payload or {}
        collected_at_str = payload.get("collected_at")
        collected_at = (
            datetime.fromisoformat(collected_at_str) if collected_at_str else None
        )

        return VectorSearchResult(
            chunk_id=str(payload.get("chunk_id", pt.id)),
            score=float(pt.score),
            text=str(payload.get("text", "")),
            document_id=str(payload.get("document_id", "")),
            version_id=str(payload.get("version_id", "")),
            source_id=payload.get("source_id"),
            company_id=payload.get("company_id"),
            workspace_id=payload.get("workspace_id"),
            source_category=payload.get("source_category"),
            content_type=payload.get("content_type"),
            canonical_url=payload.get("canonical_url"),
            chunk_index=int(payload.get("chunk_index", 0)),
            parent_chunk_id=payload.get("parent_chunk_id"),
            chunk_level=int(payload.get("chunk_level", 0)),
            collected_at=collected_at,
            title=payload.get("title"),
            metadata=dict(payload.get("metadata", {})),
        )
