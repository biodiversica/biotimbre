"""The library as a table: one row per item, one column per stored value.

For a spreadsheet or a dataframe rather than for another library — a bundle is
the format that round-trips. So every compound value is split into plain
columns (a location becomes ``latitude`` and ``longitude``, a band its ``low``
and ``high``), and an absent value is an empty cell rather than a word, so a
numeric column stays numeric.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping, Sequence

from biotimbre.domain.item import LibraryItem

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
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    names = [name for name, _ in _ITEM_COLUMNS]
    # The label sits beside the ID it names.
    names.insert(names.index("biotimbre_id") + 1, "biotimbre_label")
    writer.writerow(names + [f"embeddings_{model}" for model in model_ids])

    for item in items:
        values = [read(item) for _, read in _ITEM_COLUMNS]
        values.insert(names.index("biotimbre_label"), labels.get(item.biotimbre_id))
        counts = embedding_counts.get(item.item_id, {})
        values += [counts.get(model, 0) for model in model_ids]
        writer.writerow("" if value is None else value for value in values)
    return buffer.getvalue()
