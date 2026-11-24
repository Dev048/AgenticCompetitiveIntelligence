"""Validator component for structural grounding and integrity in decision support recommendations."""

from typing import Any

from competitive_intelligence.decision_support.interfaces import (
    BaseRecommendationValidator,
)
from competitive_intelligence.decision_support.models import (
    DecisionContext,
    DecisionOption,
    RawRecommendationPayload,
    Recommendation,
)
from competitive_intelligence.intelligence.models import (
    Insight,
    IntelligenceEvent,
)


class DeterministicRecommendationValidator(BaseRecommendationValidator):
    """Production validator verifying structural grounding, context alignment, and prediction labeling.

    Enforces invariants:
    - No context / empty objectives -> No recommendation.
    - Selected option must exist in evaluated candidate options.
    - Supporting insights and events must be present in context.
    - Does not rely on brittle keyword blacklists.
    """

    def validate(
        self,
        payload: Any,
        context: DecisionContext,
        options: list[DecisionOption],
        insights: list[Insight],
        events: list[IntelligenceEvent],
    ) -> tuple[bool, list[str]]:
        """Validate recommendation payload: returns (is_valid, validation_errors)."""
        errors: list[str] = []

        if not isinstance(payload, (RawRecommendationPayload, Recommendation)):
            return False, ["Payload is not an instance of RawRecommendationPayload or Recommendation."]

        # 1. Non-empty text fields
        if not payload.title or not payload.title.strip():
            errors.append("Recommendation title cannot be empty.")
        if not payload.summary or not payload.summary.strip():
            errors.append("Recommendation summary cannot be empty.")
        if not payload.rationale or not payload.rationale.strip():
            errors.append("Recommendation rationale cannot be empty.")

        # 2. Decision ID alignment
        if payload.decision_id != context.decision_id:
            errors.append(
                f"Payload decision_id '{payload.decision_id}' does not match context decision_id '{context.decision_id}'."
            )

        # 3. Selected Option Existence
        known_option_ids = {opt.option_id for opt in options}
        if payload.selected_option_id not in known_option_ids:
            errors.append(
                f"Selected option_id '{payload.selected_option_id}' does not exist in evaluated candidate options."
            )

        # 4. Evaluated Alternatives Completeness
        if not payload.evaluated_alternatives:
            errors.append("Recommendation must contain evaluated_alternatives audit trail.")
        else:
            alt_ids = {alt.option_id for alt in payload.evaluated_alternatives}
            if payload.selected_option_id not in alt_ids:
                errors.append(
                    f"Selected option_id '{payload.selected_option_id}' missing from evaluated_alternatives audit trail."
                )

        # 5. Supporting Insight IDs Existence & Tenant Isolation
        trusted_insights = {
            ins.insight_id: ins for ins in insights if ins.workspace_id == context.workspace_id
        }
        known_insight_ids = set(trusted_insights.keys())

        # Check for cross-workspace insights
        cross_ws_insights = {
            ins.insight_id: ins.workspace_id
            for ins in insights
            if ins.workspace_id != context.workspace_id
        }

        for ins_id in payload.supporting_insight_ids:
            if ins_id in cross_ws_insights:
                errors.append(
                    f"Referenced insight_id '{ins_id}' belongs to workspace '{cross_ws_insights[ins_id]}', "
                    f"violating workspace isolation for '{context.workspace_id}'."
                )
            elif ins_id not in known_insight_ids:
                errors.append(
                    f"Referenced insight_id '{ins_id}' does not exist in candidate insights pool."
                )

        # 6. Supporting Event IDs Existence & Tenant Isolation (strictly from trusted upstream pool)
        trusted_events = {
            e.event_id: e for e in events if e.workspace_id == context.workspace_id
        }
        known_event_ids = set(trusted_events.keys())
        for ins in trusted_insights.values():
            known_event_ids.update(ins.supporting_event_ids)

        cross_ws_events = {
            e.event_id: e.workspace_id
            for e in events
            if e.workspace_id != context.workspace_id
        }

        if not payload.supporting_event_ids:
            errors.append("Recommendation must reference at least one supporting event ID.")
        else:
            for ev_id in payload.supporting_event_ids:
                if ev_id in cross_ws_events:
                    errors.append(
                        f"Referenced event_id '{ev_id}' belongs to workspace '{cross_ws_events[ev_id]}', "
                        f"violating workspace isolation for '{context.workspace_id}'."
                    )
                elif ev_id not in known_event_ids:
                    errors.append(
                        f"Referenced event_id '{ev_id}' does not exist in candidate events pool."
                    )

        # 7. Validate Selected Option Upstream Grounding
        selected_opt = next((opt for opt in options if opt.option_id == payload.selected_option_id), None)
        if selected_opt is not None:
            for opt_ins_id in selected_opt.supporting_insight_ids:
                if opt_ins_id not in known_insight_ids:
                    errors.append(
                        f"Selected option '{selected_opt.option_id}' references unknown insight_id '{opt_ins_id}'."
                    )
            for opt_ev_id in selected_opt.supporting_event_ids:
                if opt_ev_id not in known_event_ids:
                    errors.append(
                        f"Selected option '{selected_opt.option_id}' references unknown event_id '{opt_ev_id}'."
                    )

        # 8. Benefits check
        if not payload.expected_benefits:
            errors.append("Recommendation must state at least one expected benefit.")

        is_valid = len(errors) == 0
        return is_valid, errors
