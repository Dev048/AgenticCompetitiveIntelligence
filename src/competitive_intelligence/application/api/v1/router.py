"""Aggregated API Version 1 Router."""

from fastapi import APIRouter

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

api_v1_router = APIRouter()

api_v1_router.include_router(health.router)
api_v1_router.include_router(workspaces.router)
api_v1_router.include_router(competitors.router)
api_v1_router.include_router(sources.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(events.router)
api_v1_router.include_router(signals.router)
api_v1_router.include_router(insights.router)
api_v1_router.include_router(decisions.router)
api_v1_router.include_router(recommendations.router)
api_v1_router.include_router(workflows.router)
