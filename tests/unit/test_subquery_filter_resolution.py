"""LLM-proposed sub-query filters are resolved deterministically and never waste a hop."""

import pytest

from competitive_intelligence.agentic_rag.filters import (
    BaseCompanyResolver,
    CompanyRef,
    resolve_company_refs,
    sanitize_source_categories,
)
from competitive_intelligence.agentic_rag.models import SubQueryPlan
from competitive_intelligence.agentic_rag.nodes.execute_subqueries_node import (
    ExecuteSubqueriesNode,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)

KNOWN = [
    CompanyRef(company_id="comp-northwind", name="Northwind Metrics", domain="northwind-metrics.example"),
    CompanyRef(company_id="comp-helios", name="Helios Insight", domain="helios-insight.example"),
    CompanyRef(company_id="comp-quarry", name="Quarry Labs", domain="quarry-labs.example"),
]


@pytest.mark.parametrize(
    ("refs", "expected"),
    [
        (["comp-northwind"], ["comp-northwind"]),
        (["Northwind Metrics"], ["comp-northwind"]),
        (["northwind metrics, inc"], []),  # extra words are not guessed at
        (["Northwind"], ["comp-northwind"]),  # unambiguous prefix
        (["northwind-metrics.example"], ["comp-northwind"]),
        (["www.northwind-metrics.example"], ["comp-northwind"]),
        (["HELIOS insight", "Quarry Labs"], ["comp-helios", "comp-quarry"]),
        (["Acme Corp"], []),
        (["No"], []),  # too short to be a prefix
        (["Northwind", "comp-northwind"], ["comp-northwind"]),  # deduplicated
        ([""], []),
    ],
)
def test_resolve_company_refs(refs: list[str], expected: list[str]) -> None:
    assert resolve_company_refs(refs, KNOWN) == expected


def test_ambiguous_prefix_is_dropped() -> None:
    known = KNOWN + [CompanyRef(company_id="comp-north2", name="Northwind Labs")]
    assert resolve_company_refs(["Northwind"], known) == []
    assert resolve_company_refs(["Northwind Labs"], known) == ["comp-north2"]


def test_sanitize_source_categories() -> None:
    assert sanitize_source_categories(["Pricing", "press", "BLOG", "pricing"]) == ["pricing", "blog"]


class _FilterAwareRetriever(BaseRetriever):
    """Returns a chunk only when the filter admits its company/category."""

    def __init__(self, company_id: str, category: str) -> None:
        self.company_id = company_id
        self.category = category
        self.filters: list[MetadataFilter] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        flt = query.metadata_filter
        self.filters.append(flt)
        if flt.company_ids and self.company_id not in flt.company_ids:
            return []
        if flt.source_categories and self.category not in flt.source_categories:
            return []
        return [
            RetrievalResult(
                chunk_id="chunk-1",
                score=0.9,
                text="Free plan: up to 3 seats.",
                document_id="doc-1",
                version_id="ver-1",
                workspace_id=flt.workspace_id,
                company_id=self.company_id,
                source_category=self.category,
            )
        ]


class _StaticResolver(BaseCompanyResolver):
    def __init__(self) -> None:
        self.calls = 0

    async def known_companies(self, workspace_id: str) -> list[CompanyRef]:
        self.calls += 1
        return KNOWN


def _state(sub_query: SubQueryPlan) -> dict[str, object]:
    return {
        "workspace_id": "ws-1",
        "metadata_filter": MetadataFilter(workspace_id="ws-1"),
        "current_sub_queries": [sub_query],
        "accumulated_results": [],
        "hop_count": 1,
        "top_k": 5,
    }


async def test_llm_company_name_is_resolved_to_the_stored_id() -> None:
    retriever = _FilterAwareRetriever("comp-northwind", "website")
    resolver = _StaticResolver()
    node = ExecuteSubqueriesNode(retriever=retriever, company_resolver=resolver)

    update = await node(_state(SubQueryPlan(sub_query="free plan seats", target_company_ids=["Northwind"])))  # type: ignore[arg-type]

    assert [r.chunk_id for r in update["accumulated_results"]] == ["chunk-1"]
    assert retriever.filters[0].company_ids == ["comp-northwind"]
    assert len(retriever.filters) == 1  # no retry needed
    assert resolver.calls == 1


async def test_unresolvable_company_filter_is_ignored() -> None:
    retriever = _FilterAwareRetriever("comp-northwind", "website")
    node = ExecuteSubqueriesNode(retriever=retriever, company_resolver=_StaticResolver())

    update = await node(_state(SubQueryPlan(sub_query="seats", target_company_ids=["Acme Corp"])))  # type: ignore[arg-type]

    assert retriever.filters[0].company_ids is None
    assert update["accumulated_results"]


async def test_empty_result_with_llm_filters_retries_with_base_filter() -> None:
    # "pricing" is a valid category, but the chunk was indexed as "website".
    retriever = _FilterAwareRetriever("comp-northwind", "website")
    node = ExecuteSubqueriesNode(retriever=retriever, company_resolver=_StaticResolver())

    update = await node(
        _state(SubQueryPlan(sub_query="seats", target_source_categories=["pricing", "invented"]))  # type: ignore[arg-type]
    )

    assert retriever.filters[0].source_categories == ["pricing"]
    assert retriever.filters[1].source_categories is None
    assert retriever.filters[1].workspace_id == "ws-1"
    results = update["accumulated_results"]
    assert results and results[0].metadata["agentic_filter_relaxed"] is True


async def test_caller_filter_is_never_relaxed_or_overridden() -> None:
    retriever = _FilterAwareRetriever("comp-helios", "website")
    node = ExecuteSubqueriesNode(retriever=retriever, company_resolver=_StaticResolver())
    state = _state(SubQueryPlan(sub_query="seats", target_company_ids=["Helios Insight"]))
    state["metadata_filter"] = MetadataFilter(workspace_id="ws-1", company_ids=["comp-northwind"])

    update = await node(state)  # type: ignore[arg-type]

    assert [f.company_ids for f in retriever.filters] == [["comp-northwind"]]
    assert update["accumulated_results"] == []
