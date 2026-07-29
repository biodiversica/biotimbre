"""Similarity scoring, per-item reduction, and cross-model fusion.

Three stages, each a separate function so each can be tested against
known-correct answers:

1. :func:`score_matrix` — pairwise similarity between query windows and library
   windows.
2. :func:`reduce_to_items` — collapse a window-level score matrix to one score
   per item by taking the best window pair, remembering which pair won.
3. :func:`fuse` — combine per-model item scores into one ranking.

The max-over-pairs reduction in stage 2 is what lets a long query match a short
library item on a single shared syllable, instead of being penalised for all the
material they do not share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biotimbre.domain.errors import ValidationError

#: Guards against dividing by zero when normalising a zero vector.
_EPSILON = 1e-8

METRICS = ("cosine", "dot", "euclidean")

#: How per-model scores combine into one ranking.
FUSIONS = ("max", "mean")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of a matrix.

    Args:
        matrix: ``(n, dim)`` float array.

    Returns:
        A new ``(n, dim)`` float32 array with unit-norm rows. A zero row stays
        zero rather than becoming NaN.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / (norms + _EPSILON)).astype(np.float32)


def score_matrix(
    query: np.ndarray, library: np.ndarray, metric: str = "cosine"
) -> np.ndarray:
    """Pairwise similarity between query windows and library windows.

    Every metric is oriented so that **higher means more similar**. Euclidean is
    therefore returned as a negative distance, which keeps the caller's ranking
    logic metric-agnostic.

    Args:
        query: ``(q, dim)`` float array of query window embeddings.
        library: ``(n, dim)`` float array of library window embeddings.
        metric: One of :data:`METRICS`.

    Returns:
        A ``(q, n)`` float32 score matrix.

    Raises:
        ValidationError: If the metric is unknown or the widths disagree.
    """
    q = np.asarray(query, dtype=np.float32)
    lib = np.asarray(library, dtype=np.float32)

    if q.ndim != 2 or lib.ndim != 2:
        raise ValidationError(
            f"Expected 2-D matrices, got shapes {q.shape} and {lib.shape}.",
            "embedding",
        )
    if q.shape[1] != lib.shape[1]:
        raise ValidationError(
            f"Embedding widths disagree: query is {q.shape[1]}-dimensional, "
            f"library is {lib.shape[1]}-dimensional. These are probably from "
            f"different models.",
            "embedding",
        )

    if metric == "cosine":
        return (normalize_rows(q) @ normalize_rows(lib).T).astype(np.float32)
    if metric == "dot":
        return (q @ lib.T).astype(np.float32)
    if metric == "euclidean":
        q_sq = np.sum(q**2, axis=1, keepdims=True)
        lib_sq = np.sum(lib**2, axis=1)
        dist_sq = np.maximum(q_sq + lib_sq - 2.0 * (q @ lib.T), 0.0)
        return (-np.sqrt(dist_sq)).astype(np.float32)

    raise ValidationError(
        f"Unknown metric {metric!r}. Choose one of {', '.join(METRICS)}.", "metric"
    )


@dataclass
class WindowMatch:
    """The best-scoring window pair between a query and one library item.

    Attributes:
        score: The similarity score of the winning pair.
        query_window: Index of the query window that won.
        item_window: Index of the item's window that won.
        start_time: Winning item window's start, in seconds within the sample.
        end_time: Winning item window's end. May exceed the sample duration when
            the window was zero-padded; clamp before seeking.
    """

    score: float
    query_window: int
    item_window: int
    start_time: float
    end_time: float


def reduce_to_items(
    scores: np.ndarray,
    keys: list[tuple[int, int, float, float]],
    into: dict[int, WindowMatch] | None = None,
) -> dict[int, WindowMatch]:
    """Collapse window-level scores to one best match per item.

    Takes the maximum over all ``(query window, item window)`` pairs, so a match
    on any part of either signal counts. Averaging instead would penalise a long
    recording for the silence around a short call.

    Accumulates into ``into`` so a caller can stream the library in batches
    without holding every score in memory. An item's windows may be split across
    batches; because each batch only ever raises an item's best score, the result
    is the same as scoring everything at once.

    Args:
        scores: ``(q, n)`` score matrix from :func:`score_matrix`.
        keys: ``(item_id, window_index, start_time, end_time)`` per column of
            ``scores``.
        into: Existing best-match map to update in place. ``None`` starts fresh.

    Returns:
        Mapping of item ID to its best :class:`WindowMatch`.

    Raises:
        ValidationError: If ``keys`` does not describe every column.
    """
    best = {} if into is None else into

    if scores.ndim != 2:
        raise ValidationError(
            f"Expected a 2-D score matrix, got shape {scores.shape}.", "scores"
        )
    if scores.shape[1] != len(keys):
        raise ValidationError(
            f"Score matrix has {scores.shape[1]} columns but {len(keys)} keys "
            f"were supplied.",
            "scores",
        )
    if scores.size == 0:
        return best

    # Best query window per library window, in one pass over the matrix.
    best_query_per_column = np.argmax(scores, axis=0)
    best_score_per_column = scores[best_query_per_column, np.arange(scores.shape[1])]

    for column, (item_id, window_index, start_time, end_time) in enumerate(keys):
        score = float(best_score_per_column[column])
        current = best.get(item_id)
        if current is None or score > current.score:
            best[item_id] = WindowMatch(
                score=score,
                query_window=int(best_query_per_column[column]),
                item_window=int(window_index),
                start_time=float(start_time),
                end_time=float(end_time),
            )
    return best


@dataclass
class FusedScore:
    """One item's combined score across models.

    Attributes:
        item_id: The item.
        score: Fused score used for ranking.
        per_model: Each contributing model's score.
        matches: Each contributing model's winning window pair.
    """

    item_id: int
    score: float
    per_model: dict[str, float] = field(default_factory=dict)
    matches: dict[str, WindowMatch] = field(default_factory=dict)

    @property
    def best_model(self) -> str | None:
        """Model that produced the highest individual score, if any."""
        if not self.per_model:
            return None
        return max(self.per_model, key=lambda model: self.per_model[model])


def fuse(
    per_model: dict[str, dict[int, WindowMatch]], method: str = "max"
) -> list[FusedScore]:
    """Combine per-model item scores into one descending ranking.

    ``max`` takes each item's best model, so a confident match from either model
    surfaces the item. ``mean`` averages over the models that scored the item,
    rewarding agreement.

    An item missing from a model's results is not treated as scoring zero: the
    model simply has no opinion, usually because that item has no embeddings for
    it yet. Substituting zero would penalise items awaiting a backfill.

    Args:
        per_model: Mapping of model ID to its item best-match map.
        method: One of :data:`FUSIONS`, or a model ID to rank by that model
            alone.

    Returns:
        :class:`FusedScore` entries, highest first. Ties break on ascending item
        ID so the ordering is deterministic.

    Raises:
        ValidationError: If ``method`` is neither a fusion nor a present model ID.
    """
    if method not in FUSIONS and method not in per_model:
        known = ", ".join(FUSIONS) + (
            f", or a model id ({', '.join(sorted(per_model))})" if per_model else ""
        )
        raise ValidationError(
            f"Unknown fusion {method!r}. Choose {known}.", "fusion"
        )

    contributing = (
        {method: per_model[method]} if method in per_model and method not in FUSIONS
        else per_model
    )

    item_ids: set[int] = set()
    for matches in contributing.values():
        item_ids.update(matches)

    fused: list[FusedScore] = []
    for item_id in item_ids:
        scores: dict[str, float] = {}
        matches: dict[str, WindowMatch] = {}
        for model_id, model_matches in contributing.items():
            match = model_matches.get(item_id)
            if match is not None:
                scores[model_id] = match.score
                matches[model_id] = match
        if not scores:
            continue
        values = list(scores.values())
        combined = max(values) if method != "mean" else float(np.mean(values))
        fused.append(
            FusedScore(
                item_id=item_id,
                score=float(combined),
                per_model=scores,
                matches=matches,
            )
        )

    fused.sort(key=lambda entry: (-entry.score, entry.item_id))
    return fused


def top_k(fused: list[FusedScore], k: int, min_score: float | None = None) -> list[FusedScore]:
    """Take the best ``k`` entries, optionally above a score floor.

    Args:
        fused: Descending list from :func:`fuse`.
        k: Maximum entries to return.
        min_score: Discard entries scoring below this.

    Returns:
        At most ``k`` entries.

    Raises:
        ValidationError: If ``k`` is not positive.
    """
    if k < 1:
        raise ValidationError(f"top_k must be at least 1, got {k}.", "top_k")
    selected = fused if min_score is None else [f for f in fused if f.score >= min_score]
    return selected[:k]
