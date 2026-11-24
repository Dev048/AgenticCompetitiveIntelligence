"""Domain models for continuous source monitoring and lease lifecycle."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class SourceMonitoringLease(BaseModel):
    """Represents an active or historical lease on a monitored source to prevent concurrency conflicts."""

    source_id: str
    lease_owner: str
    lease_acquired_at: datetime
    lease_expires_at: datetime
    heartbeat_at: datetime
    attempt_count: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source_id", "lease_owner")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Ensure required identifiers are not empty or whitespace."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()

    @field_validator(
        "lease_acquired_at",
        "lease_expires_at",
        "heartbeat_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure all timestamps are timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value

    @property
    def is_expired(self) -> bool:
        """Whether the lease has expired relative to current UTC time."""
        return datetime.now(timezone.utc) >= self.lease_expires_at
