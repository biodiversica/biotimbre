"""Vector similarity search and metadata filtering."""

from biotimbre.search.filters import (
    SearchFilters,
    build_candidate_query,
    haversine_km,
    refine_by_distance,
    select_candidates,
)
from biotimbre.search.vector_search import (
    METRICS,
    RankedItem,
    WindowMatch,
    normalize_rows,
    rank,
    reduce_to_items,
    score_matrix,
    top_k,
)

__all__ = [
    "METRICS",
    "RankedItem",
    "SearchFilters",
    "WindowMatch",
    "build_candidate_query",
    "haversine_km",
    "normalize_rows",
    "rank",
    "reduce_to_items",
    "refine_by_distance",
    "score_matrix",
    "select_candidates",
    "top_k",
]
