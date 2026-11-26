"""SQL repository for Decision Support, candidate options, recommendations, and end-to-end provenance graphs."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from competitive_intelligence.application.interfaces import BaseDecisionRepository
from competitive_intelligence.decision_support.models import (
    ActionType,
    ConstraintType,
    DecisionConstraint,
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    DecisionOption,
    DecisionRisk,
    DecisionUrgency,
    EvaluatedAlternative,
    ObjectiveDimension,
    PredictedConsequence,
    Recommendation,
    RecommendationConfidenceAssessment,
    RecommendationStatus,
    RiskBasisType,
    RiskTolerance,
    TimeHorizon,
)
from competitive_intelligence.infrastructure.database.models import (
    DecisionContextCompetitorOrm,
    DecisionContextOrm,
    DecisionOptionEventLinkOrm,
    DecisionOptionInsightLinkOrm,
    DecisionOptionOrm,
    DocumentChunkOrm,
    DocumentOrm,
    DocumentVersionOrm,
    EventEvidenceLinkOrm,
    EvidenceReferenceOrm,
    InferredSignalOrm,
    InsightEventLinkOrm,
    InsightOrm,
    InsightSignalLinkOrm,
    IntelligenceEventOrm,
    RecommendationEventLinkOrm,
    RecommendationInsightLinkOrm,
    RecommendationOrm,
    SignalEventLinkOrm,
    SourceOrm,
)
from competitive_intelligence.infrastructure.database.session import (
    DatabaseSessionManager,
    get_db_manager,
)
from competitive_intelligence.core.errors import EntityNotFoundError
from competitive_intelligence.intelligence.models import ConfidenceLevel, SeverityLevel


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlDecisionRepository(BaseDecisionRepository):
    """Repository managing DecisionContexts, DecisionOptions, Recommendations, and complete provenance."""

    def __init__(
        self,
        session: AsyncSession | async_sessionmaker[AsyncSession] | DatabaseSessionManager | None = None,
    ) -> None:
        self._session: AsyncSession | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

        if isinstance(session, AsyncSession):
            self._session = session
        elif isinstance(session, DatabaseSessionManager):
            self._sessionmaker = session.sessionmaker
        elif session is not None:
            self._sessionmaker = session
        else:
            self._sessionmaker = get_db_manager().sessionmaker

    @asynccontextmanager
    async def _get_session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._session is not None:
            yield self._session
        elif self._sessionmaker is not None:
            async with self._sessionmaker() as session:
                yield session
                await session.commit()
        else:
            raise RuntimeError("SqlDecisionRepository has no active session or sessionmaker")

    # --------------------------------------------------------------------------
    # DecisionContext Operations
    # --------------------------------------------------------------------------

    async def create_context(self, context: DecisionContext) -> DecisionContext:
        """Persist a DecisionContext and associate tracked competitor IDs."""
        async with self._get_session() as session:
            existing_orm = await session.get(DecisionContextOrm, context.decision_id)
            if existing_orm is not None:
                return context

            objectives_data = [
                {
                    "objective_id": o.objective_id,
                    "name": o.name,
                    "dimension": o.dimension.value,
                    "target_description": o.target_description,
                    "priority_weight": o.priority_weight,
                }
                for o in context.business_objectives
            ]

            constraints_data = [
                {
                    "constraint_id": c.constraint_id,
                    "constraint_type": c.constraint_type.value,
                    "description": c.description,
                    "is_hard_constraint": c.is_hard_constraint,
                    "threshold_value": c.threshold_value,
                }
                for c in context.constraints
            ]

            orm_ctx = DecisionContextOrm(
                decision_context_id=context.decision_id,
                workspace_id=context.workspace_id,
                organization_id=context.organization_id,
                title=context.title,
                decision_domain=context.decision_domain.value,
                business_objectives=objectives_data,
                constraints=constraints_data,
                risk_tolerance=context.risk_tolerance.value,
                time_horizon=context.time_horizon.value,
                deadline=context.deadline,
                current_state_summary=context.current_state_summary,
                metadata_=context.metadata,
            )

            session.add(orm_ctx)
            await session.flush()

            # Link competitors M:N idempotently
            for comp_id in context.competitor_ids:
                existing_link = await session.get(
                    DecisionContextCompetitorOrm, (context.decision_id, comp_id)
                )
                if not existing_link:
                    link = DecisionContextCompetitorOrm(
                        decision_context_id=context.decision_id, competitor_id=comp_id
                    )
                    session.add(link)
            await session.flush()

        return context

    async def get_context_by_id(self, decision_id: str) -> DecisionContext | None:
        """Fetch a DecisionContext by primary decision_id."""
        async with self._get_session() as session:
            stmt = select(DecisionContextOrm).where(
                DecisionContextOrm.decision_context_id == decision_id
            )
            result = await session.execute(stmt)
            orm_ctx = result.scalar_one_or_none()
            if orm_ctx is None:
                return None

            # Fetch competitor links
            comp_stmt = select(DecisionContextCompetitorOrm.competitor_id).where(
                DecisionContextCompetitorOrm.decision_context_id == decision_id
            )
            comp_res = await session.execute(comp_stmt)
            competitor_ids = list(comp_res.scalars().all())

            return self._orm_to_context(orm_ctx, competitor_ids)

    # --------------------------------------------------------------------------
    # DecisionOption Operations
    # --------------------------------------------------------------------------

    async def create_option(self, option: DecisionOption) -> DecisionOption:
        """Persist a candidate DecisionOption and link supporting insights and events."""
        async with self._get_session() as session:
            existing_orm = await session.get(DecisionOptionOrm, option.option_id)
            if existing_orm is not None:
                return option

            effects_data = [
                {
                    "dimension": e.dimension.value,
                    "description": e.description,
                    "is_positive": e.is_positive,
                    "confidence_score": e.confidence_score,
                    "underlying_assumptions": e.underlying_assumptions,
                }
                for e in option.expected_effects
            ]

            risks_data = [
                {
                    "risk_id": r.risk_id,
                    "description": r.description,
                    "likelihood": r.likelihood.value,
                    "severity": r.severity.value,
                    "basis_type": r.basis_type.value,
                    "mitigation_strategy": r.mitigation_strategy,
                }
                for r in option.risks
            ]

            orm_opt = DecisionOptionOrm(
                option_id=option.option_id,
                decision_context_id=option.decision_id,
                title=option.title,
                description=option.description,
                action_type=option.action_type.value,
                expected_effects=effects_data,
                required_resources=option.required_resources,
                estimated_cost_usd=option.estimated_cost_usd,
                risks=risks_data,
                tradeoffs=option.tradeoffs,
                assumptions=option.assumptions,
            )

            session.add(orm_opt)
            await session.flush()

            # Link supporting insights M:N idempotently
            for ins_id in dict.fromkeys(option.supporting_insight_ids):
                link_ins = await session.get(DecisionOptionInsightLinkOrm, (option.option_id, ins_id))
                if not link_ins:
                    session.add(DecisionOptionInsightLinkOrm(option_id=option.option_id, insight_id=ins_id))

            # Link supporting events M:N idempotently
            for ev_id in dict.fromkeys(option.supporting_event_ids):
                link_ev = await session.get(DecisionOptionEventLinkOrm, (option.option_id, ev_id))
                if not link_ev:
                    session.add(DecisionOptionEventLinkOrm(option_id=option.option_id, event_id=ev_id))

            await session.flush()
        return option

    async def get_option_by_id(self, option_id: str) -> DecisionOption | None:
        """Fetch a DecisionOption with associated insight and event links."""
        async with self._get_session() as session:
            stmt = select(DecisionOptionOrm).where(DecisionOptionOrm.option_id == option_id)
            result = await session.execute(stmt)
            orm_opt = result.scalar_one_or_none()
            if orm_opt is None:
                return None

            # Fetch insight links
            ins_stmt = select(DecisionOptionInsightLinkOrm.insight_id).where(
                DecisionOptionInsightLinkOrm.option_id == option_id
            )
            ins_res = await session.execute(ins_stmt)
            insight_ids = list(ins_res.scalars().all())

            # Fetch event links
            ev_stmt = select(DecisionOptionEventLinkOrm.event_id).where(
                DecisionOptionEventLinkOrm.option_id == option_id
            )
            ev_res = await session.execute(ev_stmt)
            event_ids = list(ev_res.scalars().all())

            return self._orm_to_option(orm_opt, insight_ids, event_ids)

    async def list_options_for_context(self, decision_id: str) -> list[DecisionOption]:
        """List all candidate options associated with a DecisionContext."""
        async with self._get_session() as session:
            stmt = (
                select(DecisionOptionOrm)
                .where(DecisionOptionOrm.decision_context_id == decision_id)
                .order_by(DecisionOptionOrm.created_at.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            options: list[DecisionOption] = []
            for r in rows:
                ins_stmt = select(DecisionOptionInsightLinkOrm.insight_id).where(
                    DecisionOptionInsightLinkOrm.option_id == r.option_id
                )
                ins_res = await session.execute(ins_stmt)
                insight_ids = list(ins_res.scalars().all())

                ev_stmt = select(DecisionOptionEventLinkOrm.event_id).where(
                    DecisionOptionEventLinkOrm.option_id == r.option_id
                )
                ev_res = await session.execute(ev_stmt)
                event_ids = list(ev_res.scalars().all())

                options.append(self._orm_to_option(r, insight_ids, event_ids))
            return options

    # --------------------------------------------------------------------------
    # Recommendation Operations
    # --------------------------------------------------------------------------

    async def create_recommendation(self, recommendation: Recommendation) -> Recommendation:
        """Persist a Recommendation with evaluated alternatives and provenance links."""
        async with self._get_session() as session:
            existing_orm = await session.get(RecommendationOrm, recommendation.recommendation_id)
            if existing_orm is not None:
                return recommendation

            eval_alts_data = [
                {
                    "option_id": a.option_id,
                    "title": a.title,
                    "is_selected": a.is_selected,
                    "is_eliminated": a.is_eliminated,
                    "elimination_reason": a.elimination_reason,
                    "tradeoff_vs_selected": a.tradeoff_vs_selected,
                }
                for a in recommendation.evaluated_alternatives
            ]

            conf_data = {
                "score": recommendation.confidence.score,
                "level": recommendation.confidence.level.value,
                "rationale": recommendation.confidence.rationale,
                "option_analysis_confidence": recommendation.confidence.option_analysis_confidence,
            }

            orm_rec = RecommendationOrm(
                recommendation_id=recommendation.recommendation_id,
                decision_context_id=recommendation.decision_id,
                workspace_id=recommendation.workspace_id,
                selected_option_id=recommendation.selected_option_id,
                urgency=recommendation.urgency.value,
                status=recommendation.status.value,
                title=recommendation.title,
                summary=recommendation.summary,
                rationale=recommendation.rationale,
                evaluated_alternatives=eval_alts_data,
                expected_benefits=recommendation.expected_benefits,
                tradeoffs=recommendation.tradeoffs,
                critical_assumptions=recommendation.critical_assumptions,
                unresolved_uncertainties=recommendation.unresolved_uncertainties,
                confidence=conf_data,
                supersedes_recommendation_id=recommendation.supersedes_recommendation_id,
                metadata_=recommendation.metadata,
                generated_at=recommendation.generated_at,
            )

            session.add(orm_rec)
            await session.flush()

            # Link supporting insights M:N idempotently
            for ins_id in recommendation.supporting_insight_ids:
                link_ins = await session.get(
                    RecommendationInsightLinkOrm, (recommendation.recommendation_id, ins_id)
                )
                if not link_ins:
                    session.add(
                        RecommendationInsightLinkOrm(
                            recommendation_id=recommendation.recommendation_id, insight_id=ins_id
                        )
                    )

            # Link supporting events M:N idempotently
            for ev_id in recommendation.supporting_event_ids:
                link_ev = await session.get(
                    RecommendationEventLinkOrm, (recommendation.recommendation_id, ev_id)
                )
                if not link_ev:
                    session.add(
                        RecommendationEventLinkOrm(
                            recommendation_id=recommendation.recommendation_id, event_id=ev_id
                        )
                    )

            await session.flush()
        return recommendation

    async def get_recommendation_by_id(self, recommendation_id: str) -> Recommendation | None:
        """Fetch a Recommendation with joined insight and event links."""
        async with self._get_session() as session:
            stmt = select(RecommendationOrm).where(
                RecommendationOrm.recommendation_id == recommendation_id
            )
            result = await session.execute(stmt)
            orm_rec = result.scalar_one_or_none()
            if orm_rec is None:
                return None

            # Fetch insight links
            ins_stmt = select(RecommendationInsightLinkOrm.insight_id).where(
                RecommendationInsightLinkOrm.recommendation_id == recommendation_id
            )
            ins_res = await session.execute(ins_stmt)
            insight_ids = list(ins_res.scalars().all())

            # Fetch event links
            ev_stmt = select(RecommendationEventLinkOrm.event_id).where(
                RecommendationEventLinkOrm.recommendation_id == recommendation_id
            )
            ev_res = await session.execute(ev_stmt)
            event_ids = list(ev_res.scalars().all())

            return self._orm_to_recommendation(orm_rec, insight_ids, event_ids)

    async def list_recommendations_for_context(
        self, context_id: str
    ) -> list[Recommendation]:
        """List all recommendations associated with a decision context."""
        async with self._get_session() as session:
            stmt = (
                select(RecommendationOrm)
                .where(RecommendationOrm.decision_context_id == context_id)
                .order_by(RecommendationOrm.generated_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            recommendations: list[Recommendation] = []
            for r in rows:
                ins_stmt = select(RecommendationInsightLinkOrm.insight_id).where(
                    RecommendationInsightLinkOrm.recommendation_id == r.recommendation_id
                )
                ins_res = await session.execute(ins_stmt)
                insight_ids = list(ins_res.scalars().all())

                ev_stmt = select(RecommendationEventLinkOrm.event_id).where(
                    RecommendationEventLinkOrm.recommendation_id == r.recommendation_id
                )
                ev_res = await session.execute(ev_stmt)
                event_ids = list(ev_res.scalars().all())

                recommendations.append(self._orm_to_recommendation(r, insight_ids, event_ids))
            return recommendations

    async def list_contexts_by_workspace(
        self, workspace_id: str
    ) -> list[DecisionContext]:
        """List all decision contexts in a workspace."""
        async with self._get_session() as session:
            stmt = (
                select(DecisionContextOrm)
                .where(DecisionContextOrm.workspace_id == workspace_id)
                .order_by(DecisionContextOrm.created_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            contexts: list[DecisionContext] = []
            for r in rows:
                comp_stmt = select(DecisionContextCompetitorOrm.competitor_id).where(
                    DecisionContextCompetitorOrm.decision_context_id == r.decision_context_id
                )
                comp_res = await session.execute(comp_stmt)
                competitor_ids = list(comp_res.scalars().all())
                contexts.append(self._orm_to_context(r, competitor_ids))
            return contexts

    async def list_recommendations_by_workspace(
        self, workspace_id: str
    ) -> list[Recommendation]:
        """List all recommendations in a workspace."""
        async with self._get_session() as session:
            stmt = (
                select(RecommendationOrm)
                .where(RecommendationOrm.workspace_id == workspace_id)
                .order_by(RecommendationOrm.generated_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            recommendations: list[Recommendation] = []
            for r in rows:
                ins_stmt = select(RecommendationInsightLinkOrm.insight_id).where(
                    RecommendationInsightLinkOrm.recommendation_id == r.recommendation_id
                )
                ins_res = await session.execute(ins_stmt)
                insight_ids = list(ins_res.scalars().all())

                ev_stmt = select(RecommendationEventLinkOrm.event_id).where(
                    RecommendationEventLinkOrm.recommendation_id == r.recommendation_id
                )
                ev_res = await session.execute(ev_stmt)
                event_ids = list(ev_res.scalars().all())

                recommendations.append(self._orm_to_recommendation(r, insight_ids, event_ids))
            return recommendations

    async def update_recommendation_status(
        self, recommendation_id: str, status: RecommendationStatus
    ) -> Recommendation:
        """Update recommendation status in place."""
        async with self._get_session() as session:
            stmt = select(RecommendationOrm).where(
                RecommendationOrm.recommendation_id == recommendation_id
            )
            result = await session.execute(stmt)
            orm_rec = result.scalar_one_or_none()
            if orm_rec is None:
                raise EntityNotFoundError(f"Recommendation '{recommendation_id}' not found")
            orm_rec.status = status.value
            await session.flush()

            ins_stmt = select(RecommendationInsightLinkOrm.insight_id).where(
                RecommendationInsightLinkOrm.recommendation_id == recommendation_id
            )
            ins_res = await session.execute(ins_stmt)
            insight_ids = list(ins_res.scalars().all())

            ev_stmt = select(RecommendationEventLinkOrm.event_id).where(
                RecommendationEventLinkOrm.recommendation_id == recommendation_id
            )
            ev_res = await session.execute(ev_stmt)
            event_ids = list(ev_res.scalars().all())

            return self._orm_to_recommendation(orm_rec, insight_ids, event_ids)

    # --------------------------------------------------------------------------
    # Full Transitive Provenance Reconstruction
    # --------------------------------------------------------------------------

    async def reconstruct_recommendation_provenance(
        self, recommendation_id: str
    ) -> dict[str, Any]:
        """Traverse and reconstruct the complete 10-layer provenance lineage graph."""
        rec = await self.get_recommendation_by_id(recommendation_id)
        if rec is None:
            return {}

        # 1. Selected Option
        option: DecisionOption | None = None
        if rec.selected_option_id:
            option = await self.get_option_by_id(rec.selected_option_id)

        # 2. Insights linked from Recommendation or Option
        insight_ids = set(rec.supporting_insight_ids)
        if option:
            insight_ids.update(option.supporting_insight_ids)

        context = await self.get_context_by_id(rec.decision_id)

        async with self._get_session() as session:
            ins_orms = []
            if insight_ids:
                insights_stmt = select(InsightOrm).where(InsightOrm.insight_id.in_(insight_ids))
                ins_res = await session.execute(insights_stmt)
                ins_orms = list(ins_res.scalars().all())

            # 3. Signals linked from Insights
            signal_ids: set[str] = set()
            sig_orms = []
            if insight_ids:
                signal_ids_stmt = select(InsightSignalLinkOrm.signal_id).where(
                    InsightSignalLinkOrm.insight_id.in_(insight_ids)
                )
                sig_ids_res = await session.execute(signal_ids_stmt)
                signal_ids = set(sig_ids_res.scalars().all())

                if signal_ids:
                    signals_stmt = select(InferredSignalOrm).where(InferredSignalOrm.signal_id.in_(signal_ids))
                    sig_res = await session.execute(signals_stmt)
                    sig_orms = list(sig_res.scalars().all())

            # 4. Events linked from Recommendation, Option, Insights, and Signals
            event_ids = set(rec.supporting_event_ids)
            if option:
                event_ids.update(option.supporting_event_ids)

            # Events from insights
            if insight_ids:
                ins_ev_stmt = select(InsightEventLinkOrm.event_id).where(
                    InsightEventLinkOrm.insight_id.in_(insight_ids)
                )
                ins_ev_res = await session.execute(ins_ev_stmt)
                event_ids.update(ins_ev_res.scalars().all())

            # Events from signals
            if signal_ids:
                sig_ev_stmt = select(SignalEventLinkOrm.event_id).where(
                    SignalEventLinkOrm.signal_id.in_(signal_ids)
                )
                sig_ev_res = await session.execute(sig_ev_stmt)
                event_ids.update(sig_ev_res.scalars().all())

            event_orms = []
            if event_ids:
                events_stmt = select(IntelligenceEventOrm).where(
                    IntelligenceEventOrm.event_id.in_(event_ids)
                )
                ev_res = await session.execute(events_stmt)
                event_orms = list(ev_res.scalars().all())

            # 5. Evidence linked from Events
            evidence_ids: set[str] = set()
            evidence_orms = []
            if event_ids:
                ev_ids_stmt = select(EventEvidenceLinkOrm.evidence_id).where(
                    EventEvidenceLinkOrm.event_id.in_(event_ids)
                )
                ev_ids_res = await session.execute(ev_ids_stmt)
                evidence_ids = set(ev_ids_res.scalars().all())

                if evidence_ids:
                    evidence_stmt = select(EvidenceReferenceOrm).where(
                        EvidenceReferenceOrm.evidence_id.in_(evidence_ids)
                    )
                    evidence_res = await session.execute(evidence_stmt)
                    evidence_orms = list(evidence_res.scalars().all())

            # 6. Chunks, Versions, Documents, Sources
            chunk_ids = {e.chunk_id for e in evidence_orms if e.chunk_id is not None}
            version_ids = {e.version_id for e in evidence_orms}
            document_ids = {e.document_id for e in evidence_orms}
            source_ids = {e.source_id for e in evidence_orms}

            chunk_orms = []
            if chunk_ids:
                chunks_stmt = select(DocumentChunkOrm).where(DocumentChunkOrm.chunk_id.in_(chunk_ids))
                chunks_res = await session.execute(chunks_stmt)
                chunk_orms = list(chunks_res.scalars().all())

            version_orms = []
            if version_ids:
                versions_stmt = select(DocumentVersionOrm).where(
                    DocumentVersionOrm.version_id.in_(version_ids)
                )
                versions_res = await session.execute(versions_stmt)
                version_orms = list(versions_res.scalars().all())

            document_orms = []
            if document_ids:
                documents_stmt = select(DocumentOrm).where(DocumentOrm.document_id.in_(document_ids))
                documents_res = await session.execute(documents_stmt)
                document_orms = list(documents_res.scalars().all())

            source_orms = []
            if source_ids:
                sources_stmt = select(SourceOrm).where(SourceOrm.source_id.in_(source_ids))
                sources_res = await session.execute(sources_stmt)
                source_orms = list(sources_res.scalars().all())

        return {
            "recommendation_id": rec.recommendation_id,
            "decision_id": rec.decision_id,
            "selected_option": option.model_dump() if option else None,
            "insight_count": len(ins_orms),
            "signal_count": len(sig_orms),
            "event_count": len(event_orms),
            "evidence_count": len(evidence_orms),
            "chunk_count": len(chunk_orms),
            "version_count": len(version_orms),
            "document_count": len(document_orms),
            "source_count": len(source_orms),
            "source_ids": list(source_ids),
            "document_ids": list(document_ids),
            "version_ids": list(version_ids),
            "chunk_ids": list(chunk_ids),
            "evidence_ids": list(evidence_ids),
            "event_ids": list(event_ids),
            "signal_ids": list(signal_ids),
            "insight_ids": list(insight_ids),
            "layers": {
                "recommendation": {
                    "recommendation_id": rec.recommendation_id,
                    "title": rec.title,
                    "summary": rec.summary,
                    "rationale": rec.rationale,
                    "status": _plain(rec.status),
                    "urgency": _plain(rec.urgency),
                },
                "decision_context": (
                    {
                        "decision_id": context.decision_id,
                        "title": context.title,
                        "decision_domain": _plain(context.decision_domain),
                    }
                    if context
                    else None
                ),
                "insights": [
                    {
                        "insight_id": o.insight_id,
                        "insight_type": _plain(o.insight_type),
                        "title": o.title,
                        "statement": o.statement,
                    }
                    for o in sorted(ins_orms, key=lambda o: o.insight_id)
                ],
                "signals": [
                    {
                        "signal_id": o.signal_id,
                        "signal_type": _plain(o.signal_type),
                        "title": o.title,
                        "summary": o.summary,
                    }
                    for o in sorted(sig_orms, key=lambda o: o.signal_id)
                ],
                "events": [
                    {
                        "event_id": o.event_id,
                        "event_type": _plain(o.event_type),
                        "title": o.title,
                        "summary": o.summary,
                        "detected_at": _iso(o.detected_at),
                    }
                    for o in sorted(event_orms, key=lambda o: o.event_id)
                ],
                "evidence": [
                    {
                        "evidence_id": o.evidence_id,
                        "quote": _truncate(o.extracted_text, _QUOTE_CHARS),
                        "evidence_role": _plain(o.evidence_role),
                        "retrieval_score": o.retrieval_score,
                        "chunk_id": o.chunk_id,
                        "version_id": o.version_id,
                        "document_id": o.document_id,
                        "source_id": o.source_id,
                        "source_url": o.source_url,
                    }
                    for o in sorted(evidence_orms, key=lambda o: o.evidence_id)
                ],
                "chunks": [
                    {
                        "chunk_id": o.chunk_id,
                        "version_id": o.version_id,
                        "chunk_index": o.chunk_index,
                        "total_chunks": o.total_chunks,
                        "start_char": o.start_char,
                        "end_char": o.end_char,
                        "content_hash": o.content_hash,
                        "preview": _truncate(o.content, _PREVIEW_CHARS),
                    }
                    for o in sorted(chunk_orms, key=lambda o: (o.version_id, o.chunk_index))
                ],
                "versions": [
                    {
                        "version_id": o.version_id,
                        "document_id": o.document_id,
                        "content_hash": o.content_hash,
                        "acquired_at": _iso(o.acquired_at),
                    }
                    for o in sorted(version_orms, key=lambda o: o.version_id)
                ],
                "documents": [
                    {
                        "document_id": o.document_id,
                        "source_id": o.source_id,
                        "canonical_url": o.canonical_url,
                    }
                    for o in sorted(document_orms, key=lambda o: o.document_id)
                ],
                "sources": [
                    {
                        "source_id": o.source_id,
                        "display_name": o.display_name,
                        "category": _plain(o.category),
                        "url": o.url_or_identifier,
                    }
                    for o in sorted(source_orms, key=lambda o: o.source_id)
                ],
            },
        }

    # --------------------------------------------------------------------------
    # Helper Mappers
    # --------------------------------------------------------------------------

    @staticmethod
    def _orm_to_context(
        orm: DecisionContextOrm, competitor_ids: list[str]
    ) -> DecisionContext:
        objectives = [
            DecisionObjective(
                objective_id=o["objective_id"],
                name=o["name"],
                dimension=ObjectiveDimension(o["dimension"]),
                target_description=o["target_description"],
                priority_weight=o.get("priority_weight"),
            )
            for o in orm.business_objectives
        ]

        constraints = [
            DecisionConstraint(
                constraint_id=c["constraint_id"],
                constraint_type=ConstraintType(c["constraint_type"]),
                description=c["description"],
                is_hard_constraint=c.get("is_hard_constraint", True),
                threshold_value=c.get("threshold_value"),
            )
            for c in orm.constraints
        ]

        return DecisionContext(
            decision_id=orm.decision_context_id,
            workspace_id=orm.workspace_id,
            organization_id=orm.organization_id,
            title=orm.title,
            decision_domain=DecisionDomain(orm.decision_domain),
            business_objectives=objectives,
            constraints=constraints,
            risk_tolerance=RiskTolerance(orm.risk_tolerance),
            time_horizon=TimeHorizon(orm.time_horizon),
            deadline=_ensure_utc(orm.deadline),
            current_state_summary=orm.current_state_summary,
            competitor_ids=competitor_ids,
            metadata=orm.metadata_,
        )

    @staticmethod
    def _orm_to_option(
        orm: DecisionOptionOrm, insight_ids: list[str], event_ids: list[str]
    ) -> DecisionOption:
        effects = [
            PredictedConsequence(
                dimension=ObjectiveDimension(e["dimension"]),
                description=e["description"],
                is_positive=e["is_positive"],
                confidence_score=e.get("confidence_score"),
                underlying_assumptions=e.get("underlying_assumptions", []),
            )
            for e in orm.expected_effects
        ]

        risks = [
            DecisionRisk(
                risk_id=r["risk_id"],
                description=r["description"],
                likelihood=ConfidenceLevel(r.get("likelihood", "medium")),
                severity=SeverityLevel(r.get("severity", "medium")),
                basis_type=RiskBasisType(r.get("basis_type", "model_predicted")),
                mitigation_strategy=r.get("mitigation_strategy"),
            )
            for r in orm.risks
        ]

        return DecisionOption(
            option_id=orm.option_id,
            decision_id=orm.decision_context_id,
            title=orm.title,
            description=orm.description,
            action_type=ActionType(orm.action_type),
            expected_effects=effects,
            required_resources=orm.required_resources,
            estimated_cost_usd=orm.estimated_cost_usd,
            risks=risks,
            tradeoffs=orm.tradeoffs,
            assumptions=orm.assumptions,
            supporting_insight_ids=insight_ids,
            supporting_event_ids=event_ids,
        )

    @staticmethod
    def _orm_to_recommendation(
        orm: RecommendationOrm, insight_ids: list[str], event_ids: list[str]
    ) -> Recommendation:
        alternatives = [
            EvaluatedAlternative(
                option_id=a["option_id"],
                title=a["title"],
                is_selected=a["is_selected"],
                is_eliminated=a.get("is_eliminated", False),
                elimination_reason=a.get("elimination_reason"),
                tradeoff_vs_selected=a.get("tradeoff_vs_selected"),
            )
            for a in orm.evaluated_alternatives
        ]

        conf_data = orm.confidence
        confidence = RecommendationConfidenceAssessment(
            score=conf_data.get("score", 0.8),
            level=ConfidenceLevel(conf_data.get("level", "high")),
            rationale=conf_data.get("rationale", ""),
            option_analysis_confidence=conf_data.get("option_analysis_confidence"),
        )

        gen_at = _ensure_utc(orm.generated_at)
        assert gen_at is not None

        return Recommendation(
            recommendation_id=orm.recommendation_id,
            decision_id=orm.decision_context_id,
            workspace_id=orm.workspace_id,
            selected_option_id=orm.selected_option_id or "",
            urgency=DecisionUrgency(orm.urgency),
            status=RecommendationStatus(orm.status),
            title=orm.title,
            summary=orm.summary,
            rationale=orm.rationale,
            evaluated_alternatives=alternatives,
            expected_benefits=orm.expected_benefits,
            tradeoffs=orm.tradeoffs,
            critical_assumptions=orm.critical_assumptions,
            unresolved_uncertainties=orm.unresolved_uncertainties,
            confidence=confidence,
            supporting_insight_ids=insight_ids,
            supporting_event_ids=event_ids if event_ids else ["unknown_event"],
            generated_at=gen_at,
            supersedes_recommendation_id=orm.supersedes_recommendation_id,
            metadata=orm.metadata_,
        )


_QUOTE_CHARS = 600
_PREVIEW_CHARS = 300


def _plain(value: Any) -> str | None:
    """Enum/str/None -> JSON-friendly string."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"
