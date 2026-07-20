import math
from typing import List

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two numeric float vectors.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(x * x for x in v2))
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def get_vector_list(point) -> List[float]:
    """
    Safely retrieves the vector float list from a Qdrant point object,
    handling lists, dictionaries (named vectors), or missing attributes.
    """
    if not hasattr(point, "vector") or point.vector is None:
        return []
    if isinstance(point.vector, list):
        return point.vector
    if isinstance(point.vector, dict):
        # Named vectors case: fetch the first available vector list
        return next(iter(point.vector.values()), [])
    return []

def maximum_marginal_relevance(
    query_vector: List[float],
    candidate_points: List,
    top_k: int = 5,
    lambda_mult: float = 0.5,
) -> List:
    """
    Filters and ranks candidate points using Maximum Marginal Relevance (MMR).
    Optimizes for both query relevance and mutual diversity among selected items.

    Args:
        query_vector:     The query embedding float list.
        candidate_points: List of Qdrant ScoredPoint objects with vector attribute retrieved.
        top_k:            Number of final diversified chunks to select.
        lambda_mult:      Relevance weighting coefficient (0.0 to 1.0).
                          1.0 = pure similarity search, 0.0 = pure diversity.

    Returns:
        List of selected ScoredPoint objects, preserving original score and adding
        an `mmr_score` attribute for observability.
    """
    if not candidate_points:
        return []

    # Limit top_k to actual number of candidates
    top_k = min(top_k, len(candidate_points))

    selected = []
    remaining = list(candidate_points)

    # 1. First choice is always the highest-ranking candidate
    # (Since candidates from Qdrant are sorted by similarity, the first one is the most relevant)
    first_choice = remaining.pop(0)
    if first_choice.payload is None:
        first_choice.payload = {}
    # Store its MMR score (which is just lambda * relevance, since similarity to selected is 0)
    first_choice.payload["mmr_score"] = float(lambda_mult * first_choice.score)
    selected.append(first_choice)

    # 2. Iteratively select next points maximizing the MMR equation
    while len(selected) < top_k and remaining:
        best_score = -float("inf")
        best_idx = -1

        for idx, candidate in enumerate(remaining):
            cand_vector = get_vector_list(candidate)
            relevance = candidate.score

            # Compute maximum similarity between this candidate and already selected elements
            max_sim_to_selected = -float("inf")
            for sel_point in selected:
                sel_vector = get_vector_list(sel_point)
                sim = cosine_similarity(cand_vector, sel_vector)
                if sim > max_sim_to_selected:
                    max_sim_to_selected = sim

            # Handle case where selected list might be empty (should not happen here)
            if max_sim_to_selected == -float("inf"):
                max_sim_to_selected = 0.0

            # MMR formula
            mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * max_sim_to_selected)

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx != -1:
            chosen = remaining.pop(best_idx)
            if chosen.payload is None:
                chosen.payload = {}
            chosen.payload["mmr_score"] = float(best_score)
            selected.append(chosen)
        else:
            break

    return selected
