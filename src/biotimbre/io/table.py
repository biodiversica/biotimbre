"""The library as a table: one row per item, one column per stored value.

For a spreadsheet or a dataframe rather than for another library — a bundle is
the format that round-trips. So every compound value is split into plain
columns (a location becomes ``latitude`` and ``longitude``, a band its ``low``
and ``high``), and an absent value is an empty cell rather than a word, so a
numeric column stays numeric.

Search results are the same rows, in rank order, with what the search found
about each — rank, scores, the matched window, distance — in front.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from biotimbre.domain.item import LibraryItem

if TYPE_CHECKING:
    from biotimbre.service import SearchHit

#: Columns, in order, each with how to read it off an item. Named as the
#: database names them, so a column means the same thing everywhere.
_ITEM_COLUMNS: tuple[tuple[str, Callable[[LibraryItem], object]], ...] = (
    ("item_id", lambda item: item.item_id),
    ("biotimbre_id", lambda item: item.biotimbre_id),
    # Separated as the sidecar separates them; the cell is quoted, so the
    # commas cannot be mistaken for column breaks.
    ("taxa", lambda item: ",".join(item.taxa)),
    ("date", lambda item: item.date),
    ("time", lambda item: item.time),
    ("latitude", lambda item: item.location.latitude),
    ("longitude", lambda item: item.location.longitude),
    (
        "sample_window_start",
        lambda item: item.sample_window.start if item.sample_window else None,
    ),
    (
        "sample_window_end",
        lambda item: item.sample_window.end if item.sample_window else None,
    ),
    ("duration_s", lambda item: item.duration_s),
    ("sample_rate", lambda item: item.sample_rate),
    ("channels", lambda item: item.channels),
    ("speed", lambda item: item.speed),
    ("device", lambda item: item.device),
    ("microphone", lambda item: item.microphone),
    ("recorded_by", lambda item: item.recorded_by),
    ("filtered_band_low", lambda item: item.filtered_band.low),
    ("filtered_band_high", lambda item: item.filtered_band.high),
    ("filter_type", lambda item: item.filter_type),
    ("filter_order", lambda item: item.filter_order),
    ("temperature", lambda item: item.temperature),
    ("humidity", lambda item: item.humidity),
    ("link_to_full_recording", lambda item: item.link_to_full_recording),
    ("notes", lambda item: item.notes),
    ("audio_path", lambda item: item.audio_path),
    ("audio_sha256", lambda item: item.audio_sha256),
    ("created_at", lambda item: item.created_at),
    ("updated_at", lambda item: item.updated_at),
)


def items_csv(
    items: Sequence[LibraryItem],
    *,
    labels: Mapping[str, str | None],
    embedding_counts: Mapping[int, Mapping[str, int]],
    model_ids: Sequence[str],
) -> str:
    """Write items as CSV text, header first.

    Args:
        items: The items, one row each, in the order given.
        labels: Biotimbre label by biotimbre ID. Missing or ``None`` is blank.
        embedding_counts: Window count per model, by item ID.
        model_ids: Models to give a window-count column each, in order. A
            model an item was never embedded with counts zero there.

    Returns:
        The CSV, with ``\\r\\n`` line endings as RFC 4180 has them.
    """
    return _table(
        items,
        leading=(),
        leading_rows=[() for _ in items],
        labels=labels,
        embedding_counts=embedding_counts,
        model_ids=model_ids,
    )


def results_csv(
    hits: Sequence[SearchHit],
    *,
    labels: Mapping[str, str | None],
    embedding_counts: Mapping[int, Mapping[str, int]],
    model_ids: Sequence[str],
) -> str:
    """Write search results as CSV text, in rank order.

    Each row is :func:`items_csv`'s row for the hit's item, after the columns
    that describe the match: ``rank``, the fused ``score``, the ``best_model``,
    the matched window in the item (``matched_start``, ``matched_end``), the
    ``distance_km`` from a location filter's centre, the ``query_speed``, and
    one ``score_<model>`` per model that scored any hit. A search with no query
    audio scores nothing, so those columns are left empty rather than zero.

    Args:
        hits: The results, best first.
        labels: As for :func:`items_csv`.
        embedding_counts: As for :func:`items_csv`.
        model_ids: As for :func:`items_csv`.

    Returns:
        The CSV text.
    """
    scored_by = sorted({model for hit in hits for model in hit.per_model})
    leading = (
        "rank",
        "score",
        "best_model",
        "matched_start",
        "matched_end",
        "distance_km",
        "query_speed",
        *(f"score_{model}" for model in scored_by),
    )
    rows = []
    for rank, hit in enumerate(hits, start=1):
        scored = bool(hit.per_model)
        span = hit.matched_span
        rows.append(
            (
                rank,
                hit.score if scored else None,
                hit.best_model,
                span[0] if span else None,
                span[1] if span else None,
                hit.distance_km,
                hit.query_speed if scored else None,
                *(hit.per_model.get(model) for model in scored_by),
            )
        )
    return _table(
        [hit.item for hit in hits],
        leading=leading,
        leading_rows=rows,
        labels=labels,
        embedding_counts=embedding_counts,
        model_ids=model_ids,
    )


def _table(
    items: Sequence[LibraryItem],
    *,
    leading: Sequence[str],
    leading_rows: Sequence[Sequence[object]],
    labels: Mapping[str, str | None],
    embedding_counts: Mapping[int, Mapping[str, int]],
    model_ids: Sequence[str],
) -> str:
    """Items as CSV, with ``leading`` columns in front of each row."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    names = [name for name, _ in _ITEM_COLUMNS]
    # The label sits beside the ID it names.
    names.insert(names.index("biotimbre_id") + 1, "biotimbre_label")
    writer.writerow(
        [*leading, *names, *(f"embeddings_{model}" for model in model_ids)]
    )

    for item, first in zip(items, leading_rows):
        values = [read(item) for _, read in _ITEM_COLUMNS]
        values.insert(names.index("biotimbre_label"), labels.get(item.biotimbre_id))
        counts = embedding_counts.get(item.item_id, {})
        values += [counts.get(model, 0) for model in model_ids]
        writer.writerow(
            "" if value is None else value for value in (*first, *values)
        )
    return buffer.getvalue()
