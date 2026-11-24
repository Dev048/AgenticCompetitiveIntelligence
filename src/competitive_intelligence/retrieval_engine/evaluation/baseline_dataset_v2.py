"""Retrieval evaluation dataset v2 with controlled hard-negatives."""

import uuid
from datetime import datetime, timezone

from competitive_intelligence.knowledge_pipeline.models import ChunkMetadata, DocumentChunk
from competitive_intelligence.knowledge_pipeline.normalizer import compute_content_hash
from competitive_intelligence.retrieval_engine.evaluation.models import EvaluationDataset, EvaluationQuery


def _create_chunk(content: str, source_category: str, company_id: str) -> DocumentChunk:
    cid = str(uuid.uuid4())
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    meta = ChunkMetadata(
        source_id="src_baseline_v2",
        document_id=doc_id,
        version_id="v1",
        company_id=company_id,
        workspace_id="ws_eval_v2",
        source_url=f"https://example.com/v2/{source_category}/{cid}",
        source_category=source_category,
        collected_at=datetime.now(timezone.utc),
        start_char=0,
        end_char=len(content),
        chunk_index=0,
        total_chunks=1,
    )
    return DocumentChunk(
        chunk_id=cid,
        document_id=doc_id,
        version_id="v1",
        content=content,
        content_hash=compute_content_hash(content),
        metadata=meta,
    )


def get_baseline_dataset_v2() -> EvaluationDataset:
    """
    Generates a deterministic dataset of 250 chunks and 50 queries.
    Uses templated patterns to inject hard negatives specifically targeting:
    - Lexical overlap (distractor has exact terms but wrong entity/event/version)
    - Semantic overlap (distractor has related meaning but misses specific query terms)
    """
    corpus: list[DocumentChunk] = []
    queries: list[EvaluationQuery] = []

    companies = [
        "OpenAI",
        "Anthropic",
        "Google",
        "Microsoft",
        "Meta",
        "Apple",
        "Amazon",
        "Netflix",
        "Tesla",
        "NVIDIA",
    ]

    # Deterministic generation
    for i, company in enumerate(companies):
        other_company = companies[(i + 1) % len(companies)]

        # Scenario 1: Pricing Change
        # Query: Did {company} increase their enterprise pricing?
        cat_price = "pricing"
        c1 = _create_chunk(f"{company} officially raised the Enterprise Tier from $20 to $30 per user.", cat_price, company)
        c2 = _create_chunk(f"According to {company}'s new pricing page, enterprise plans start at $30.", cat_price, company)
        c3 = _create_chunk(f"Reddit: I think {company} is going to increase enterprise pricing to $30 soon.", "reddit", company)
        # Hard Negatives
        c4_lex = _create_chunk(f"{other_company} officially raised the Enterprise Tier from $20 to $30 per user.", cat_price, other_company)
        c5_sem = _create_chunk(f"{company} officially expanded their Enterprise features, adding SSO and SLA guarantees.", cat_price, company)

        corpus.extend([c1, c2, c3, c4_lex, c5_sem])
        queries.append(
            EvaluationQuery(
                query_id=f"q_price_{company.lower()}",
                query_text=f"Did {company} increase their enterprise pricing?",
                relevant_chunk_ids={c1.chunk_id: 3, c2.chunk_id: 2, c3.chunk_id: 1},
            )
        )

        # Scenario 2: Product Feature Launch
        # Query: When did {company} launch the AI Copilot feature?
        cat_blog = "blogs"
        c1 = _create_chunk(f"{company} released AI Copilot to all users on October 15, 2023, as part of v2.4.", cat_blog, company)
        c2 = _create_chunk(f"v2.4 Release Notes ({company}): Introduces AI Copilot, our new generative assistant.", "github", company)
        c3 = _create_chunk(f"Blog post: We are excited to announce that AI Copilot is finally here for {company}.", cat_blog, company)
        # Hard Negatives
        c4_lex = _create_chunk(f"{other_company} released AI Copilot to all users on October 15, 2023.", cat_blog, other_company)
        c5_sem = _create_chunk(f"{company} updated their machine learning infrastructure on October 15, 2023.", cat_blog, company)

        corpus.extend([c1, c2, c3, c4_lex, c5_sem])
        queries.append(
            EvaluationQuery(
                query_id=f"q_launch_{company.lower()}",
                query_text=f"When did {company} launch the AI Copilot feature?",
                relevant_chunk_ids={c1.chunk_id: 3, c2.chunk_id: 2, c3.chunk_id: 1},
            )
        )

        # Scenario 3: Careers/Tech Stack
        # Query: Is {company} using Rust in their backend?
        cat_careers = "careers"
        c1 = _create_chunk(f"Job posting: {company} is hiring a Backend Engineer to rewrite our core microservices in Rust.", cat_careers, company)
        c2 = _create_chunk(f"We use Rust for our high-performance data processing pipelines at {company}.", cat_careers, company)
        c3 = _create_chunk(f"Reddit: I heard {company} might be migrating from Go to Rust for their backend.", "reddit", company)
        # Hard Negatives
        c4_lex = _create_chunk(f"Job posting: {other_company} is hiring a Backend Engineer to rewrite our core microservices in Rust.", cat_careers, other_company)
        c5_sem = _create_chunk(f"{company} is hiring a Frontend Engineer with experience in React and TypeScript.", cat_careers, company)

        corpus.extend([c1, c2, c3, c4_lex, c5_sem])
        queries.append(
            EvaluationQuery(
                query_id=f"q_tech_{company.lower()}",
                query_text=f"Is {company} using Rust in their backend?",
                relevant_chunk_ids={c1.chunk_id: 3, c2.chunk_id: 2, c3.chunk_id: 1},
            )
        )

        # Scenario 4: Version/Changelog
        # Query: What are the breaking changes in {company} v3.0?
        cat_github = "github"
        c1 = _create_chunk(f"{company} v3.0 Changelog: Breaking changes include removing the legacy REST API v1 and updating the authentication flow.", cat_github, company)
        c2 = _create_chunk(f"Migrating to {company} v3.0 requires updating your API keys due to the new authentication flow.", cat_github, company)
        c3 = _create_chunk(f"Blog: Getting ready for {company} v3.0! Watch out for the API deprecations.", "blogs", company)
        # Hard Negatives
        c4_lex = _create_chunk(f"{company} v2.0 Changelog: Breaking changes include removing the legacy XML export.", cat_github, company)
        c5_sem = _create_chunk(f"{company} v3.0 introduces a brand new dark mode and redesigned dashboard.", cat_github, company)

        corpus.extend([c1, c2, c3, c4_lex, c5_sem])
        queries.append(
            EvaluationQuery(
                query_id=f"q_version_{company.lower()}",
                query_text=f"What are the breaking changes in {company} v3.0?",
                relevant_chunk_ids={c1.chunk_id: 3, c2.chunk_id: 2, c3.chunk_id: 1},
            )
        )

        # Scenario 5: News/Acquisitions
        # Query: Did {company} acquire a startup recently?
        cat_news = "news"
        c1 = _create_chunk(f"News: {company} announced the acquisition of DataTech Inc for $50M to bolster their analytics.", cat_news, company)
        c2 = _create_chunk(f"{company} CEO confirms the purchase of DataTech, integrating their team next month.", cat_news, company)
        c3 = _create_chunk(f"Rumor: {company} is looking to buy a data analytics startup soon.", "reddit", company)
        # Hard Negatives
        c4_lex = _create_chunk(f"News: {other_company} announced the acquisition of DataTech Inc for $50M.", cat_news, other_company)
        c5_sem = _create_chunk(f"News: {company} announced a new strategic partnership with DataTech Inc to share data.", cat_news, company)

        corpus.extend([c1, c2, c3, c4_lex, c5_sem])
        queries.append(
            EvaluationQuery(
                query_id=f"q_acq_{company.lower()}",
                query_text=f"Did {company} acquire a startup recently?",
                relevant_chunk_ids={c1.chunk_id: 3, c2.chunk_id: 2, c3.chunk_id: 1},
            )
        )

    return EvaluationDataset(
        dataset_id="baseline_v2_hard_negatives",
        queries=queries,
        corpus=corpus,
    )
