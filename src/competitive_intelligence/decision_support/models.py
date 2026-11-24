"""Domain models for Decision Support and Recommendation subsystem."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from competitive_intelligence.intelligence.models import (
    ConfidenceLevel,
    SeverityLevel,
)


class DecisionDomain(str, Enum):
    """Business domain of the decision."""

    PRICING = "pricing"
    PRODUCT = "product"
    HIRING_TALENT = "hiring_talent"
    GTM_MARKETING = "gtm_marketing"
    TECHNOLOGY_INFRASTRUCTURE = "technology_infrastructure"
    PARTNERSHIP_ECOSYSTEM = "partnership_ecosystem"
    RISK_COMPLIANCE = "risk_compliance"
    GENERAL_STRATEGY = "general_strategy"


class ObjectiveDimension(str, Enum):
    """Strategic dimensions that a business objective aims to optimize."""

    REVENUE = "revenue"
    GROWTH = "growth"
    RETENTION = "retention"
    MARKET_SHARE = "market_share"
    COST_EFFICIENCY = "cost_efficiency"
    RISK_REDUCTION = "risk_reduction"
    TIME_TO_MARKET = "time_to_market"
    STRATEGIC_POSITIONING = "strategic_positioning"
    CUSTOMER_EXPERIENCE = "customer_experience"
    OTHER = "other"


class ConstraintType(str, Enum):
    """Categories of operational or organizational constraints."""

    BUDGET = "budget"
    STAFFING_CAPACITY = "staffing_capacity"
    TIMELINE = "timeline"
    REGULATORY_LEGAL = "regulatory_legal"
    TECHNICAL_FEASIBILITY = "technical_feasibility"
    ORGANIZATIONAL_POLICY = "organizational_policy"
    MAX_ACCEPTABLE_RISK = "max_acceptable_risk"
    OTHER = "other"


class RiskTolerance(str, Enum):
    """Organizational appetite for risk."""

    VERY_LOW_CONSERVATIVE = "very_low_conservative"
    MODERATE = "moderate"
    HIGH_AGGRESSIVE = "high_aggressive"


class TimeHorizon(str, Enum):
    """Decision execution timeframe."""

    IMMEDIATE_SHORT_TERM = "immediate_short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class DecisionUrgency(str, Enum):
    """Context-derived necessity and timeframe for acting."""

    NO_ACTION_REQUIRED = "no_action_required"
    OPTIONAL = "optional"
    RECOMMENDED = "recommended"
    URGENT = "urgent"


class ActionType(str, Enum):
    """Taxonomy of candidate decision moves."""

    MAINTAIN_CURRENT_STANCE = "maintain_current_stance"
    MONITOR_AND_WAIT = "monitor_and_wait"
    DIRECT_COUNTER_MOVE = "direct_counter_move"
    ASYMMETRIC_RESPONSE = "asymmetric_response"
    DIFFERENTIATION = "differentiation"
    EXPERIMENTATION_TEST = "experimentation_test"
    RESOURCE_REALLOCATION = "resource_reallocation"
    OTHER = "other"


class RecommendationStatus(str, Enum):
    """Lifecycle status of a proposed recommendation (strictly starts at PROPOSED)."""

    DRAFT = "draft"
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class RiskBasisType(str, Enum):
    """Epistemic grounding of an identified risk."""

    EVIDENCE_SUPPORTED = "evidence_supported"
    MODEL_PREDICTED = "model_predicted"
    ASSUMPTION_DERIVED = "assumption_derived"


class DecisionObjective(BaseModel):
    """Explicit business objective supplied by the decision maker."""

    objective_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    dimension: ObjectiveDimension
    target_description: str
    priority_weight: float | None = Field(default=None, ge=0.0, le=1.0)


class DecisionConstraint(BaseModel):
    """Explicit operational constraint supplied by the decision maker."""

    constraint_id: str = Field(default_factory=lambda: str(uuid4()))
    constraint_type: ConstraintType
    description: str
    is_hard_constraint: bool = True
    threshold_value: Any | None = None


class DecisionContext(BaseModel):
    """Context in which a recommendation is evaluated (strictly user-provided)."""

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = "default_workspace"
    organization_id: str = "default_org"
    title: str
    decision_domain: DecisionDomain = DecisionDomain.GENERAL_STRATEGY
    business_objectives: list[DecisionObjective] = Field(default_factory=list)
    constraints: list[DecisionConstraint] = Field(default_factory=list)
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    deadline: datetime | None = None
    current_state_summary: str | None = None
    competitor_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictedConsequence(BaseModel):
    """Forecasted effect of an action (strictly labeled as a prediction, not an observed fact)."""

    dimension: ObjectiveDimension
    description: str
    is_positive: bool
    confidence_score: float | None = None  # None indicates unassessed prediction confidence
    underlying_assumptions: list[str] = Field(default_factory=list)


class DecisionRisk(BaseModel):
    """Structured assessment of potential risk associated with an option."""

    risk_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    likelihood: ConfidenceLevel = ConfidenceLevel.MEDIUM
    severity: SeverityLevel = SeverityLevel.MEDIUM
    basis_type: RiskBasisType = RiskBasisType.MODEL_PREDICTED
    mitigation_strategy: str | None = None


class DecisionOption(BaseModel):
    """A candidate course of action evaluated against the decision context."""

    option_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_id: str
    title: str
    description: str
    action_type: ActionType
    expected_effects: list[PredictedConsequence] = Field(default_factory=list)
    required_resources: list[str] = Field(default_factory=list)
    estimated_cost_usd: float | None = None
    risks: list[DecisionRisk] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    # Transitive Provenance
    supporting_insight_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class EvaluatedAlternative(BaseModel):
    """Structured audit trail record for an evaluated alternative option."""

    option_id: str
    title: str
    is_selected: bool
    is_eliminated: bool = False
    elimination_reason: str | None = None
    tradeoff_vs_selected: str | None = None


class RecommendationConfidenceAssessment(BaseModel):
    """Confidence in the prescription itself given objectives and candidate options."""

    score: float = Field(default=0.8, ge=0.0, le=1.0)
    level: ConfidenceLevel = ConfidenceLevel.HIGH
    rationale: str
    option_analysis_confidence: float | None = None


class Recommendation(BaseModel):
    """Actionable prescriptive proposal preferred under supplied objectives and constraints."""

    recommendation_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_id: str
    workspace_id: str = "default_workspace"
    selected_option_id: str
    urgency: DecisionUrgency = DecisionUrgency.RECOMMENDED
    status: RecommendationStatus = RecommendationStatus.PROPOSED

    title: str
    summary: str
    rationale: str  # Why this option was preferred over candidate alternatives
    evaluated_alternatives: list[EvaluatedAlternative] = Field(default_factory=list)
    expected_benefits: list[str] = Field(min_length=1)
    tradeoffs: list[str] = Field(default_factory=list)
    critical_assumptions: list[str] = Field(default_factory=list)
    unresolved_uncertainties: list[str] = Field(default_factory=list)

    # Multi-Dimensional Confidence & Provenance
    confidence: RecommendationConfidenceAssessment
    supporting_insight_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    supersedes_recommendation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawRecommendationPayload(BaseModel):
    """Intermediate raw payload produced by a recommendation interpreter before validation."""

    decision_id: str
    selected_option_id: str
    urgency: DecisionUrgency = DecisionUrgency.RECOMMENDED
    title: str
    summary: str
    rationale: str
    evaluated_alternatives: list[EvaluatedAlternative] = Field(min_length=1)
    expected_benefits: list[str] = Field(min_length=1)
    tradeoffs: list[str] = Field(default_factory=list)
    critical_assumptions: list[str] = Field(default_factory=list)
    unresolved_uncertainties: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    confidence_rationale: str
    supporting_insight_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionPolicy(BaseModel):
    """Configuration governing decision support and recommendation generation."""

    require_explicit_weights_for_multi_objective: bool = True
    enforce_hard_constraints: bool = True
    default_urgency: DecisionUrgency = DecisionUrgency.OPTIONAL
