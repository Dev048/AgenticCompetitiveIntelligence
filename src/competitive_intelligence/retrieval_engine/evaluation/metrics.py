import math

from competitive_intelligence.retrieval_engine.models import RetrievalResult


def get_binary_relevant_ids(relevant_chunk_ids: dict[str, int], threshold: int = 2) -> set[str]:
    """Return a set of chunk IDs that meet the minimum relevance grade threshold."""
    return {chunk_id for chunk_id, grade in relevant_chunk_ids.items() if grade >= threshold}


def calculate_recall_at_k(
    results: list[RetrievalResult], relevant_chunk_ids: dict[str, int], k: int
) -> float:
    """Calculate Recall@K using grades >= 2 as relevant."""
    binary_relevant = get_binary_relevant_ids(relevant_chunk_ids, threshold=2)
    if not binary_relevant:
        return 0.0

    top_k_ids = {r.chunk_id for r in results[:k]}
    hits = len(binary_relevant.intersection(top_k_ids))
    return hits / len(binary_relevant)


def calculate_mrr(results: list[RetrievalResult], relevant_chunk_ids: dict[str, int]) -> float:
    """Calculate Mean Reciprocal Rank using grades >= 2 as relevant."""
    binary_relevant = get_binary_relevant_ids(relevant_chunk_ids, threshold=2)
    if not binary_relevant:
        return 0.0

    for i, res in enumerate(results):
        if res.chunk_id in binary_relevant:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg_at_k(
    results: list[RetrievalResult], relevant_chunk_ids: dict[str, int], k: int
) -> float:
    """Calculate NDCG@K utilizing full 0-3 relevance grades."""
    if not relevant_chunk_ids:
        return 0.0

    dcg = 0.0
    for i, res in enumerate(results[:k]):
        grade = relevant_chunk_ids.get(res.chunk_id, 0)
        if grade > 0:
            dcg += grade / math.log2(i + 2)  # index is 0-based, formula wants log2(rank + 1) -> log2(i + 2)

    # Calculate Ideal DCG (IDCG)
    ideal_grades = sorted(relevant_chunk_ids.values(), reverse=True)
    idcg = 0.0
    for i, grade in enumerate(ideal_grades[:k]):
        if grade > 0:
            idcg += grade / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
