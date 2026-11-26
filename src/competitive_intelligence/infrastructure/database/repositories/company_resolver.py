"""SQL-backed competitor directory used to resolve LLM-proposed company filters."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from competitive_intelligence.agentic_rag.filters import BaseCompanyResolver, CompanyRef
from competitive_intelligence.infrastructure.database.repositories.competitor_repo import (
    SqlCompetitorRepository,
)


class SqlCompanyResolver(BaseCompanyResolver):
    """Lists a workspace's tracked competitors (ID, name, domain)."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def known_companies(self, workspace_id: str) -> list[CompanyRef]:
        async with self._sessionmaker() as session:
            competitors = await SqlCompetitorRepository(session).list_by_workspace(workspace_id)
        return [
            CompanyRef(company_id=c.competitor_id, name=c.name, domain=c.domain)
            for c in competitors
        ]
