"""Fictional demo corpus: a product-analytics market with three invented competitors.

Every company, product, person, price and URL here is fictional. Domains use the reserved
``.example`` TLD, so nothing resolves and nothing is ever crawled. Each document is
supplied as pre-acquired text; documents with two versions let change detection see a
real before/after diff.
"""

from typing import Literal

from pydantic import BaseModel, Field

from competitive_intelligence.decision_support.models import (
    ConstraintType,
    DecisionConstraint,
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    ObjectiveDimension,
    RiskTolerance,
    TimeHorizon,
)
from competitive_intelligence.source_acquisition.models import SourceCategory

DEMO_WORKSPACE_ID = "demo-workspace"
DEMO_WORKSPACE_NAME = "Demo: Product Analytics Market (fictional data)"
DEMO_OWNER_USER_ID = "demo-operator"
DEMO_ORGANIZATION_ID = "demo-org"

DEMO_DATA_NOTICE = "[Fictional demo data. Companies, products, people and prices are invented.]"


class DemoCompetitor(BaseModel):
    competitor_id: str
    name: str
    domain: str
    description: str


class DemoDocument(BaseModel):
    """One monitored source and the acquired text of each of its versions (oldest first)."""

    source_id: str
    competitor_id: str
    display_name: str
    category: SourceCategory
    url: str
    versions: list[str] = Field(min_length=1)


class DemoPipelineStep(BaseModel):
    """One workflow run executed by the seeder (fixed run_id => idempotent)."""

    run_id: str
    source_id: str
    version_index: int
    target_stop_stage: Literal["knowledge_processing", "insight_generation", "decision_support"]
    query_text: str | None = None
    decision_context: DecisionContext | None = None


COMPETITORS: list[DemoCompetitor] = [
    DemoCompetitor(
        competitor_id="comp-northwind",
        name="Northwind Metrics",
        domain="northwind-metrics.example",
        description="Self-serve product analytics for SMB SaaS teams (fictional).",
    ),
    DemoCompetitor(
        competitor_id="comp-helios",
        name="Helios Insight",
        domain="helios-insight.example",
        description="Mid-market BI and anomaly monitoring platform (fictional).",
    ),
    DemoCompetitor(
        competitor_id="comp-quarry",
        name="Quarry Labs",
        domain="quarry-labs.example",
        description="Session replay and journey analytics vendor (fictional).",
    ),
]


DOCUMENTS: list[DemoDocument] = [
    DemoDocument(
        source_id="src-northwind-pricing",
        competitor_id="comp-northwind",
        display_name="Northwind Metrics pricing page",
        category=SourceCategory.PRICING,
        url="https://northwind-metrics.example/pricing",
        versions=[
            f"""{DEMO_DATA_NOTICE}
Northwind Metrics Pricing.
Starter plan: $49 per month, billed monthly. Includes 2 seats and 50,000 tracked events.
Growth plan: $199 per month. Includes 10 seats, 1 million tracked events and funnels.
Enterprise plan: contact sales. SAML single sign-on available as a paid add-on.
There is no free plan; a 14-day trial is available on Starter and Growth.""",
            f"""{DEMO_DATA_NOTICE}
Northwind Metrics Pricing.
New Free plan: $0 forever. Includes up to 3 seats and 10,000 tracked events per month.
Starter plan: now $29 per month (previously $49), billed monthly. Includes 5 seats and
100,000 tracked events.
Growth plan: $199 per month. Includes 10 seats, 1 million tracked events and funnels.
Enterprise plan: contact sales. SAML single sign-on is now included at no extra cost.""",
        ],
    ),
    DemoDocument(
        source_id="src-northwind-review",
        competitor_id="comp-northwind",
        display_name="SaaS Review Hub: Northwind Metrics reviews",
        category=SourceCategory.NEWS,
        url="https://saas-review-hub.example/products/northwind-metrics",
        versions=[
            f"""{DEMO_DATA_NOTICE}
Review of Northwind Metrics by a verified user (Head of Growth, 40-person startup).
We moved to Northwind's new Free plan after the price change. In our workspace the Free
plan only allows 1 seat, not the 3 seats shown on the pricing page, so we upgraded to
Starter at $29. Onboarding was fast, but event limits are reached quickly. 4 out of 5."""
        ],
    ),
    DemoDocument(
        source_id="src-helios-changelog",
        competitor_id="comp-helios",
        display_name="Helios Insight changelog",
        category=SourceCategory.DOCUMENTATION,
        url="https://helios-insight.example/changelog",
        versions=[
            f"""{DEMO_DATA_NOTICE}
Helios Insight Changelog.
Release 7.2: faster dashboard rendering and new chart types.
Release 7.1: CSV export for all reports. Anomaly detection remains in private beta.""",
            f"""{DEMO_DATA_NOTICE}
Helios Insight Changelog.
Release 8.0: Anomaly Detection is now generally available on all paid plans.
Release 8.0: Helios Insight achieved SOC 2 Type II certification.
Release 8.0: Legacy dashboards are deprecated and will be removed in Q3.
Release 7.2: faster dashboard rendering and new chart types.""",
        ],
    ),
    DemoDocument(
        source_id="src-helios-blog",
        competitor_id="comp-helios",
        display_name="Helios Insight blog",
        category=SourceCategory.BLOG,
        url="https://helios-insight.example/blog/stratus-partnership",
        versions=[
            f"""{DEMO_DATA_NOTICE}
Helios Insight announces a strategic partnership with Stratus Cloud. Helios Insight is now
available in the Stratus Cloud Marketplace, and Stratus customers can apply committed
cloud spend to Helios subscriptions. The two companies will co-sell to mid-market and
enterprise accounts starting next quarter."""
        ],
    ),
    DemoDocument(
        source_id="src-quarry-news",
        competitor_id="comp-quarry",
        display_name="Quarry Labs newsroom",
        category=SourceCategory.NEWS,
        url="https://quarry-labs.example/newsroom",
        versions=[
            f"""{DEMO_DATA_NOTICE}
Quarry Labs Newsroom.
Quarry Labs releases version 3.6 with improved heatmaps and mobile session replay.""",
            f"""{DEMO_DATA_NOTICE}
Quarry Labs Newsroom.
Quarry Labs has acquired Beacon Replay, a real-time session analytics startup, for
$40 million. The Beacon team joins Quarry's platform group.
Quarry Labs releases version 4.0 with enterprise SSO, audit logs and data residency in
the EU.
Quarry Labs releases version 3.6 with improved heatmaps and mobile session replay.""",
        ],
    ),
    DemoDocument(
        source_id="src-quarry-careers",
        competitor_id="comp-quarry",
        display_name="Quarry Labs careers page",
        category=SourceCategory.CAREERS,
        url="https://quarry-labs.example/careers",
        versions=[
            f"""{DEMO_DATA_NOTICE}
Quarry Labs careers. Open roles: VP of Enterprise Sales (London), six Enterprise Account
Executives for EMEA, two Solutions Engineers focused on security reviews, and a
Compliance Lead to own ISO 27001 certification."""
        ],
    ),
]


def _pricing_context() -> DecisionContext:
    return DecisionContext(
        decision_id="dc-northwind-pricing",
        workspace_id=DEMO_WORKSPACE_ID,
        organization_id=DEMO_ORGANIZATION_ID,
        title="Respond to Northwind Metrics' entry-tier price cut and new Free plan",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(
                objective_id="obj-smb-acquisition",
                name="Protect SMB self-serve acquisition",
                dimension=ObjectiveDimension.GROWTH,
                target_description="Keep monthly self-serve signups within 10% of last quarter",
                priority_weight=0.6,
            ),
            DecisionObjective(
                objective_id="obj-gross-margin",
                name="Hold blended gross margin",
                dimension=ObjectiveDimension.REVENUE,
                target_description="Blended gross margin stays above 75%",
                priority_weight=0.4,
            ),
        ],
        constraints=[
            DecisionConstraint(
                constraint_id="con-no-free-tier-before-q3",
                constraint_type=ConstraintType.TECHNICAL_FEASIBILITY,
                description="Usage metering for a free tier cannot ship before Q3",
                is_hard_constraint=False,
            )
        ],
        risk_tolerance=RiskTolerance.MODERATE,
        time_horizon=TimeHorizon.IMMEDIATE_SHORT_TERM,
        competitor_ids=["comp-northwind"],
    )


def _product_context() -> DecisionContext:
    return DecisionContext(
        decision_id="dc-helios-anomaly",
        workspace_id=DEMO_WORKSPACE_ID,
        organization_id=DEMO_ORGANIZATION_ID,
        title="Decide how to answer Helios Insight's GA anomaly detection and SOC 2 Type II",
        decision_domain=DecisionDomain.PRODUCT,
        business_objectives=[
            DecisionObjective(
                objective_id="obj-midmarket-winrate",
                name="Defend mid-market win rate",
                dimension=ObjectiveDimension.MARKET_SHARE,
                target_description="Mid-market competitive win rate against Helios above 45%",
                priority_weight=0.7,
            ),
            DecisionObjective(
                objective_id="obj-security-reviews",
                name="Pass enterprise security reviews",
                dimension=ObjectiveDimension.RISK_REDUCTION,
                target_description="No deals lost on missing compliance attestations",
                priority_weight=0.3,
            ),
        ],
        constraints=[
            DecisionConstraint(
                constraint_id="con-eng-capacity",
                constraint_type=ConstraintType.STAFFING_CAPACITY,
                description="At most one product squad can be reassigned this quarter",
            )
        ],
        risk_tolerance=RiskTolerance.MODERATE,
        time_horizon=TimeHorizon.MEDIUM_TERM,
        competitor_ids=["comp-helios"],
    )


def _enterprise_context() -> DecisionContext:
    return DecisionContext(
        decision_id="dc-quarry-enterprise",
        workspace_id=DEMO_WORKSPACE_ID,
        organization_id=DEMO_ORGANIZATION_ID,
        title="Assess Quarry Labs' enterprise push (Beacon acquisition, SSO, EMEA hiring)",
        decision_domain=DecisionDomain.GTM_MARKETING,
        business_objectives=[
            DecisionObjective(
                objective_id="obj-emea-pipeline",
                name="Grow EMEA enterprise pipeline",
                dimension=ObjectiveDimension.REVENUE,
                target_description="EMEA enterprise pipeline up 30% within two quarters",
                priority_weight=0.5,
            ),
            DecisionObjective(
                objective_id="obj-positioning",
                name="Keep differentiated positioning",
                dimension=ObjectiveDimension.STRATEGIC_POSITIONING,
                target_description="Remain the default choice for privacy-sensitive buyers",
                priority_weight=0.5,
            ),
        ],
        constraints=[
            DecisionConstraint(
                constraint_id="con-budget",
                constraint_type=ConstraintType.BUDGET,
                description="No incremental headcount budget before next fiscal year",
                threshold_value=0,
            )
        ],
        risk_tolerance=RiskTolerance.VERY_LOW_CONSERVATIVE,
        time_horizon=TimeHorizon.LONG_TERM,
        competitor_ids=["comp-quarry"],
    )


def pipeline_steps() -> list[DemoPipelineStep]:
    """Seeding order: baselines (no LLM), single-version evidence, then the updates.

    Baselines only index version 1 so the later run sees a genuine change. Single-version
    sources run first so their events are available to signal detection when the
    decision-support runs execute.
    """
    return [
        DemoPipelineStep(
            run_id="demo-run-01-northwind-pricing-v1",
            source_id="src-northwind-pricing",
            version_index=0,
            target_stop_stage="knowledge_processing",
        ),
        DemoPipelineStep(
            run_id="demo-run-02-helios-changelog-v1",
            source_id="src-helios-changelog",
            version_index=0,
            target_stop_stage="knowledge_processing",
        ),
        DemoPipelineStep(
            run_id="demo-run-03-quarry-news-v1",
            source_id="src-quarry-news",
            version_index=0,
            target_stop_stage="knowledge_processing",
        ),
        DemoPipelineStep(
            run_id="demo-run-04-northwind-review",
            source_id="src-northwind-review",
            version_index=0,
            target_stop_stage="insight_generation",
            query_text="Northwind Metrics Free plan seat limit",
        ),
        DemoPipelineStep(
            run_id="demo-run-05-helios-blog",
            source_id="src-helios-blog",
            version_index=0,
            target_stop_stage="insight_generation",
            query_text="Helios Insight Stratus Cloud partnership",
        ),
        DemoPipelineStep(
            run_id="demo-run-06-quarry-careers",
            source_id="src-quarry-careers",
            version_index=0,
            target_stop_stage="insight_generation",
            query_text="Quarry Labs enterprise sales hiring",
        ),
        DemoPipelineStep(
            run_id="demo-run-07-northwind-pricing-v2",
            source_id="src-northwind-pricing",
            version_index=1,
            target_stop_stage="decision_support",
            query_text="Northwind Metrics pricing change Starter plan Free plan",
            decision_context=_pricing_context(),
        ),
        DemoPipelineStep(
            run_id="demo-run-08-helios-changelog-v2",
            source_id="src-helios-changelog",
            version_index=1,
            target_stop_stage="decision_support",
            query_text="Helios Insight anomaly detection general availability SOC 2",
            decision_context=_product_context(),
        ),
        DemoPipelineStep(
            run_id="demo-run-09-quarry-news-v2",
            source_id="src-quarry-news",
            version_index=1,
            target_stop_stage="decision_support",
            query_text="Quarry Labs acquisition Beacon Replay enterprise SSO",
            decision_context=_enterprise_context(),
        ),
    ]


def document_by_source_id(source_id: str) -> DemoDocument:
    for doc in DOCUMENTS:
        if doc.source_id == source_id:
            return doc
    raise KeyError(source_id)
