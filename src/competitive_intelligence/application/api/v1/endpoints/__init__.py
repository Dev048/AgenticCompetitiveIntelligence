"""API v1 router endpoints."""

from competitive_intelligence.application.api.v1.endpoints import (
    competitors,
    decisions,
    documents,
    events,
    health,
    insights,
    recommendations,
    signals,
    sources,
    workflows,
    workspaces,
)

__all__ = [
    "health",
    "workspaces",
    "competitors",
    "sources",
    "documents",
    "events",
    "signals",
    "insights",
    "decisions",
    "recommendations",
    "workflows",
]
