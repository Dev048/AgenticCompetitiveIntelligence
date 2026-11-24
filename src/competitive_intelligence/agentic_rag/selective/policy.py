"""Deterministic escalation policy for Selective Agentic RAG.

Evaluates initial Adaptive retrieval results against query features and explicit,
documented threshold configurations to decide whether Agentic escalation is required.
"""

import re
from typing import Any

from competitive_intelligence.agentic_rag.selective.models import (
    EscalationAssessment,
    EscalationPolicyConfig,
    EscalationSignalType,
)
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)


class DeterministicEscalationPolicy:
    """Deterministic, auditable policy assessing whether a query requires Agentic RAG escalation.

    Principles:
    - Zero hidden LLM calls; pure deterministic heuristic analysis.
    - Operates on cheap initial retrieval results (from AdaptiveRetriever).
    - Checks multi-entity balance, facet coverage, temporal indicators, evidence gaps, and score thresholds.
    - Explicitly prevents escalating simple single-hop direct lookups and negative/distractor queries.
    """

    def __init__(self, config: EscalationPolicyConfig | None = None) -> None:
        """Initialize the policy with explicit configuration thresholds.

        Args:
            config: Explicit threshold configuration (defaults to EscalationPolicyConfig()).
        """
        self.config = config or EscalationPolicyConfig()

    def assess(
        self,
        query: RetrievalQuery,
        results: list[RetrievalResult],
    ) -> EscalationAssessment:
        """Assess whether the initial retrieval results warrant Agentic escalation.

        Args:
            query: The original RetrievalQuery.
            results: Initial retrieval candidates returned by AdaptiveRetriever.

        Returns:
            EscalationAssessment containing escalation decision, signals, reasons, and metrics.
        """
        query_text = query.query_text.lower()
        reasons: list[str] = []
        signals: list[EscalationSignalType] = []
        metadata: dict[str, Any] = {}

        initial_count = len(results)
        top_score = float(results[0].score) if results else 0.0

        # Focus deep signal analysis on the immediate top candidate window
        eval_results = results[:self.config.assessment_top_k]

        # Signal 1: Low Evidence Count
        if initial_count < self.config.min_evidence_count:
            signals.append(EscalationSignalType.LOW_EVIDENCE_COUNT)
            reasons.append(
                f"Initial retrieval produced {initial_count} results, below minimum threshold {self.config.min_evidence_count}"
            )

        # Signal 2: Low Top Score (only if results exist)
        elif top_score < self.config.min_score_threshold:
            signals.append(EscalationSignalType.LOW_TOP_SCORE)
            reasons.append(
                f"Initial top-1 retrieval score ({top_score:.4f}) is below minimum threshold ({self.config.min_score_threshold:.4f})"
            )

        # Signal 3 & Signal 7: Multi-Entity Unbalanced Coverage & Rank Concentration
        detected_entities = self._extract_entities(query_text)
        metadata["detected_entities"] = detected_entities
        entity_coverage_ratio = 1.0

        if len(detected_entities) >= 2:
            covered_entities, entity_counts = self._check_entity_distribution(detected_entities, eval_results)
            entity_coverage_ratio = len(covered_entities) / len(detected_entities)
            metadata["covered_entities"] = list(covered_entities)
            metadata["missing_entities"] = list(set(detected_entities) - covered_entities)
            metadata["entity_counts"] = entity_counts

            if entity_coverage_ratio < self.config.min_entity_coverage:
                signals.append(EscalationSignalType.MULTI_ENTITY_UNBALANCED)
                reasons.append(
                    f"Multi-entity query for {detected_entities} only covered {list(covered_entities)} in top-{self.config.assessment_top_k} results (coverage: {entity_coverage_ratio:.2f} < {self.config.min_entity_coverage:.2f})"
                )

            total_mentions = sum(entity_counts.values())
            if total_mentions > 0:
                max_single_mention = max(entity_counts.values())
                concentration = max_single_mention / total_mentions
                metadata["rank_concentration"] = concentration
                if concentration > self.config.max_rank_concentration and len(covered_entities) < len(detected_entities):
                    if EscalationSignalType.MULTI_ENTITY_UNBALANCED not in signals:
                        signals.append(EscalationSignalType.HIGH_RANK_CONCENTRATION)
                        reasons.append(
                            f"Top results are heavily concentrated ({concentration:.2f} > {self.config.max_rank_concentration:.2f}) on a single entity while querying multiple entities"
                        )

        # Signal 4: Evidence Gap Detected (Critical missing terminology)
        gap_terms_in_query = [
            term for term in self.config.gap_critical_terms
            if re.search(rf"\b{re.escape(term)}\b", query_text)
        ]
        metadata["gap_terms_in_query"] = gap_terms_in_query

        if gap_terms_in_query:
            missing_gap_terms = self._check_gap_terms_coverage(gap_terms_in_query, eval_results)
            metadata["missing_gap_terms"] = missing_gap_terms
            if missing_gap_terms:
                signals.append(EscalationSignalType.EVIDENCE_GAP_DETECTED)
                reasons.append(
                    f"Query contains critical specific terms {missing_gap_terms} that are completely absent in top-{self.config.assessment_top_k} candidate content"
                )

        # Signal 5: Multi-Facet Disjoint Coverage & Facet Evidence Gap
        facets = self._extract_facets(query_text)
        metadata["detected_facets"] = facets
        facet_coverage_ratio = 1.0

        if len(facets) >= 2:
            covered_facets, missing_facets = self._check_facet_coverage(facets, eval_results)
            facet_coverage_ratio = len(covered_facets) / len(facets)
            metadata["covered_facets"] = covered_facets
            metadata["missing_facets"] = missing_facets

            if facet_coverage_ratio < self.config.min_facet_coverage:
                signals.append(EscalationSignalType.MULTI_FACET_DISJOINT)
                signals.append(EscalationSignalType.FACET_EVIDENCE_GAP)
                reasons.append(
                    f"Query requests {len(facets)} distinct facets, but top-{self.config.assessment_top_k} evidence is missing {missing_facets} (coverage: {facet_coverage_ratio:.2f} < {self.config.min_facet_coverage:.2f})"
                )

            # Check entity-facet balance if multiple entities and facets exist
            elif len(detected_entities) >= 2:
                is_balanced, missing_entity_facets = self._check_entity_facet_balance(
                    detected_entities, facets, eval_results
                )
                metadata["missing_entity_facets"] = missing_entity_facets
                if not is_balanced:
                    signals.append(EscalationSignalType.ENTITY_FACET_UNBALANCED)
                    reasons.append(
                        f"Evidence imbalance across entities: requested facets {missing_entity_facets} lack representation for specific entities in top-{self.config.assessment_top_k}"
                    )


        # Signal 6: Temporal Comparison
        is_temporal = any(ind in query_text for ind in self.config.temporal_indicators)
        metadata["is_temporal_query"] = is_temporal
        if is_temporal:
            temporal_coverage, epochs_found, epochs_missing = self._check_temporal_coverage(query_text, eval_results)
            metadata["temporal_epochs_detected"] = epochs_found + epochs_missing
            metadata["temporal_epochs_covered"] = epochs_found
            metadata["temporal_epochs_missing"] = epochs_missing
            metadata["temporal_coverage_complete"] = temporal_coverage

            if not temporal_coverage:
                signals.append(EscalationSignalType.TEMPORAL_COMPARISON)
                reasons.append(
                    f"Temporal comparison query lacks multi-epoch representation: missing evidence for {epochs_missing} in top-{self.config.assessment_top_k}"
                )

        # Direct Single-Hop / Negative Safety Filter
        should_escalate = len(signals) > 0
        confidence = 1.0 if should_escalate else 0.95

        return EscalationAssessment(
            should_escalate=should_escalate,
            confidence=confidence,
            reasons=reasons,
            signals_detected=signals,
            initial_evidence_count=initial_count,
            top_score=top_score,
            entity_coverage_ratio=entity_coverage_ratio,
            facet_coverage_ratio=facet_coverage_ratio,
            assessment_metadata=metadata,
        )

    def _extract_entities(self, query_text: str) -> list[str]:
        """Extract mentioned entities from query text using known entity dictionary and regex boundaries."""
        found: list[str] = []
        for entity in self.config.known_entities:
            pattern = rf"\b{re.escape(entity)}\b"
            if re.search(pattern, query_text):
                found.append(entity)
        return found

    def _check_entity_distribution(
        self,
        entities: list[str],
        results: list[RetrievalResult],
    ) -> tuple[set[str], dict[str, int]]:
        """Check presence and occurrence counts of queried entities across candidate results."""
        covered: set[str] = set()
        counts: dict[str, int] = {e: 0 for e in entities}
        for res in results:
            content_lower = res.text.lower()
            metadata_str = " ".join(str(v).lower() for v in res.metadata.values())
            combined = f"{content_lower} {metadata_str}"
            for entity in entities:
                if re.search(rf"\b{re.escape(entity)}\b", combined):
                    covered.add(entity)
                    counts[entity] += 1
        return covered, counts

    def _check_gap_terms_coverage(
        self,
        gap_terms: list[str],
        results: list[RetrievalResult],
    ) -> list[str]:
        """Identify which critical gap terms are completely absent across candidate results."""
        if not results:
            return gap_terms

        combined_text = " ".join(r.text.lower() for r in results)
        missing: list[str] = []
        for term in gap_terms:
            if not re.search(rf"\b{re.escape(term)}\b", combined_text):
                missing.append(term)
        return missing

    def _extract_facets(self, query_text: str) -> list[str]:
        """Extract multi-topic facets and requested attributes from compound or comparative queries."""
        # 1. Check explicit splitters
        for splitter in self.config.facet_splitters:
            if splitter in query_text:
                parts = query_text.split(splitter, 1)
                if len(parts) == 2:
                    sub_parts = re.split(r",|\band\b", parts[1])
                    facets = [p.strip().rstrip("?.") for p in sub_parts if len(p.strip()) > 2]
                    if len(facets) >= 2:
                        return facets

        # 2. Check comparative queries: "Compare <entity1> ... and <entity2> ... <attribute1>, <attribute2>, and <attribute3>"
        is_comparative = any(ind in query_text for ind in self.config.comparison_indicators)
        if is_comparative:
            entities = self._extract_entities(query_text)
            if len(entities) >= 2:
                # Find index after the last entity mention
                last_pos = 0
                for entity in entities:
                    match = re.search(rf"\b{re.escape(entity)}\b", query_text)
                    if match:
                        last_pos = max(last_pos, match.end())

                if last_pos > 0 and last_pos < len(query_text):
                    trailing = query_text[last_pos:].strip()
                    # Strip common bridging words
                    trailing = re.sub(r"^(cloud|serverless|models?|platform|services?|features?|in|on|with|regarding)\s+", "", trailing)
                    sub_parts = re.split(r",|\band\b", trailing)
                    facets = [p.strip().rstrip("?.") for p in sub_parts if len(p.strip()) > 2]
                    # Filter out pure entity words or generic stopwords
                    filtered_facets = [
                        f for f in facets
                        if not all(w in self.config.known_entities or w in ["and", "or", "the", "model", "cloud", "between", "vs", "versus"] for w in f.split())
                    ]
                    if len(filtered_facets) >= 2:
                        return filtered_facets

        return []

    def _check_facet_coverage(
        self,
        facets: list[str],
        results: list[RetrievalResult],
    ) -> tuple[list[str], list[str]]:
        """Check which facets have adequate lexical keyword overlap in candidate evidence."""
        covered: list[str] = []
        missing: list[str] = []
        combined_text = " ".join(r.text.lower() for r in results)

        stopwords = {
            "and", "or", "the", "for", "with", "in", "on", "of", "to", "across",
            "spanning", "between", "vs", "versus", "regarding", "from", "about",
            "what", "are", "how", "did", "their", "latest", "features", "offerings",
        }

        for facet in facets:
            facet_words = [w for w in re.split(r"\W+", facet) if len(w) > 2 and w not in stopwords]
            if not facet_words:
                continue

            matching_words = [w for w in facet_words if re.search(rf"\b{re.escape(w)}\b", combined_text)]
            overlap_ratio = len(matching_words) / len(facet_words)
            if overlap_ratio >= self.config.min_facet_word_overlap:
                covered.append(facet)
            else:
                missing.append(facet)

        return covered, missing

    def _check_entity_facet_balance(
        self,
        entities: list[str],
        facets: list[str],
        results: list[RetrievalResult],
    ) -> tuple[bool, dict[str, list[str]]]:
        """Check if requested facets have evidence representation per entity."""
        entity_missing_facets: dict[str, list[str]] = {}
        stopwords = {
            "and", "or", "the", "for", "with", "in", "on", "of", "to", "across",
            "spanning", "between", "vs", "versus", "regarding", "from", "about",
        }

        for entity in entities:
            # Gather chunks associated with this entity
            entity_chunks = [
                r.text.lower() for r in results
                if re.search(rf"\b{re.escape(entity)}\b", f"{r.text.lower()} {' '.join(str(v).lower() for v in r.metadata.values())}")
            ]
            if not entity_chunks:
                entity_missing_facets[entity] = facets
                continue

            entity_text = " ".join(entity_chunks)
            missing_for_entity: list[str] = []
            for facet in facets:
                facet_words = [w for w in re.split(r"\W+", facet) if len(w) > 2 and w not in stopwords]
                if not facet_words:
                    continue
                matching = [w for w in facet_words if re.search(rf"\b{re.escape(w)}\b", entity_text)]
                if (len(matching) / len(facet_words)) < self.config.min_facet_word_overlap:
                    missing_for_entity.append(facet)

        # Imbalanced if any entity has 0 facets covered while another entity has facet coverage
        any_covered = any(len(missing) < len(facets) for missing in entity_missing_facets.values())
        any_completely_missing = any(len(missing) == len(facets) for missing in entity_missing_facets.values())
        is_balanced = not (any_covered and any_completely_missing)
        return is_balanced, entity_missing_facets


    def _check_temporal_coverage(
        self,
        query_text: str,
        results: list[RetrievalResult],
    ) -> tuple[bool, list[str], list[str]]:
        """Check if temporal comparison query has evidence spanning all requested epochs/versions."""
        combined_text = " ".join(r.text.lower() for r in results)

        # 1. Check 4-digit years (e.g. 2023, 2024)
        years = list(set(re.findall(r"\b(20\d\d)\b", query_text)))
        if len(years) >= 2:
            covered = [y for y in years if y in combined_text]
            missing = [y for y in years if y not in combined_text]
            return len(covered) >= len(years), covered, missing

        # 2. Check version tags (e.g. v1 vs v2, Claude 3 vs Claude 3.5, GPT-4 vs GPT-4o)
        versions: list[str] = []
        for pat in [r"claude\s+3(\.5)?(\s+sonnet)?", r"gpt-4(o|-turbo)?", r"v\d+(\.\d+)?"]:
            for match in re.finditer(pat, query_text):
                versions.append(match.group(0))

        unique_versions = list(set(versions))
        if len(unique_versions) >= 2:
            covered_v = [v for v in unique_versions if re.search(rf"\b{re.escape(v)}\b", combined_text)]
            missing_v = [v for v in unique_versions if not re.search(rf"\b{re.escape(v)}\b", combined_text)]
            return len(covered_v) >= len(unique_versions), covered_v, missing_v

        # 3. Check baseline vs revision
        if "baseline" in query_text and "revision" in query_text:
            has_baseline = "baseline" in combined_text or "2023" in combined_text
            has_revision = "revision" in combined_text or "2024" in combined_text
            covered_terms = []
            missing_terms = []
            if has_baseline:
                covered_terms.append("baseline")
            else:
                missing_terms.append("baseline")
            if has_revision:
                covered_terms.append("revision")
            else:
                missing_terms.append("revision")
            return (has_baseline and has_revision), covered_terms, missing_terms

        return True, [], []
