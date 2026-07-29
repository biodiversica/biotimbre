"""Data access for items, biotimbres, models and embedding vectors.

The repository owns SQL and nothing else: no audio decoding, no inference, no
filesystem writes beyond what SQLite does itself. It does enforce one rule that
is too important to leave to callers — changing an item's ``speed`` invalidates
that item's embeddings, so they are deleted in the same transaction as the update
(PLAN.md D2b.3). A stale vector cannot outlive the speed it was computed at.

Vectors are stored as raw little-endian float32 bytes. ``numpy.frombuffer`` on a
SQLite BLOB gives a read-only view over the driver's buffer, so every read is
copied before being handed out.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone

import numpy as np

from biotimbre.domain.errors import DuplicateError, NotFoundError, ValidationError
from biotimbre.domain.item import (
    DEFAULT_SPEED,
    FilteredBand,
    ItemDraft,
    LibraryItem,
    Location,
    SampleWindow,
    format_biotimbre_id,
    normalize_taxa,
    parse_biotimbre_id,
)
from biotimbre.models_registry import ModelSpec

#: float32: 4 bytes per dimension, matching numpy's default on all supported
#: platforms. Stored little-endian; SQLite BLOBs are byte-exact round trips.
VECTOR_DTYPE = np.float32

MAX_BIOTIMBRE_COUNT = 0x10000

_ITEM_COLUMNS = """
    item_id, biotimbre_id, date, time, latitude, longitude,
    sample_window_start, sample_window_end, device,
    filtered_band_low, filtered_band_high, noise_reduction, speed,
    link_to_full_recording, notes, audio_path, audio_sha256,
    duration_s, sample_rate, channels, created_at, updated_at
"""


def _utc_now() -> str:
    """Current time as an ISO-8601 UTC string, second resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _vector_to_blob(vector: np.ndarray) -> bytes:
    """Flatten a vector to contiguous little-endian float32 bytes."""
    arr = np.ascontiguousarray(vector, dtype=VECTOR_DTYPE).ravel()
    return arr.tobytes()


def _blob_to_vector(blob: bytes) -> np.ndarray:
    """Decode stored bytes back to a writable 1-D float32 array."""
    return np.frombuffer(blob, dtype=VECTOR_DTYPE).copy()


class Repository:
    """SQL access layer over an initialised library database.

    The repository does not own the connection's lifetime; the caller opens it
    (via :func:`biotimbre.db.connection.initialize`) and closes it.

    Args:
        conn: An initialised connection, with foreign keys enabled.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection, for callers that need a transaction."""
        return self._conn

    # ── biotimbres ────────────────────────────────────────────────────────────

    def allocate_biotimbre_id(self) -> str:
        """Reserve and return the lowest unused biotimbre ID.

        Fills gaps left by deletions rather than always taking the next highest,
        so a library that churns does not exhaust the 16-bit space prematurely.

        The row is inserted, so the ID is taken as soon as this returns. Call
        inside a transaction if the caller might roll back.

        Returns:
            Canonical 4-character uppercase hex ID.

        Raises:
            DuplicateError: If all 65 536 IDs are in use.
        """
        # Fixed-width uppercase hex sorts identically lexicographically and
        # numerically ('9' < 'A' in ASCII), so ORDER BY gives ascending values.
        used = [
            int(row["biotimbre_id"], 16)
            for row in self._conn.execute(
                "SELECT biotimbre_id FROM biotimbres ORDER BY biotimbre_id"
            )
        ]
        candidate = len(used)
        for expected, actual in enumerate(used):
            if actual != expected:
                candidate = expected
                break
        if candidate >= MAX_BIOTIMBRE_COUNT:
            raise DuplicateError(
                f"All {MAX_BIOTIMBRE_COUNT} biotimbre IDs are in use; "
                f"the 16-bit ID space is exhausted."
            )

        biotimbre_id = format_biotimbre_id(candidate)
        self._conn.execute(
            "INSERT INTO biotimbres (biotimbre_id, label, created_at) "
            "VALUES (?, NULL, ?)",
            (biotimbre_id, _utc_now()),
        )
        return biotimbre_id

    def ensure_biotimbre(self, biotimbre_id: str, label: str | None = None) -> str:
        """Create the biotimbre row if it does not exist.

        Args:
            biotimbre_id: ID in any accepted form; normalised on the way in.
            label: Optional name, applied only when creating the row.

        Returns:
            Canonical 4-character uppercase hex ID.
        """
        canonical = parse_biotimbre_id(biotimbre_id)
        self._conn.execute(
            "INSERT OR IGNORE INTO biotimbres (biotimbre_id, label, created_at) "
            "VALUES (?, ?, ?)",
            (canonical, label, _utc_now()),
        )
        return canonical

    def biotimbre_exists(self, biotimbre_id: str) -> bool:
        """Whether a biotimbre row exists.

        Args:
            biotimbre_id: ID in any accepted form.

        Returns:
            True if present.
        """
        canonical = parse_biotimbre_id(biotimbre_id)
        row = self._conn.execute(
            "SELECT 1 FROM biotimbres WHERE biotimbre_id = ?", (canonical,)
        ).fetchone()
        return row is not None

    def set_biotimbre_label(self, biotimbre_id: str, label: str | None) -> None:
        """Set or clear a biotimbre's human-readable label.

        Args:
            biotimbre_id: ID in any accepted form.
            label: New label, or ``None`` to clear it.

        Raises:
            NotFoundError: If the biotimbre does not exist.
        """
        canonical = parse_biotimbre_id(biotimbre_id)
        cursor = self._conn.execute(
            "UPDATE biotimbres SET label = ? WHERE biotimbre_id = ?",
            (label, canonical),
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"No biotimbre {canonical}.")

    def list_biotimbres(self) -> list[dict]:
        """All biotimbres with their item counts.

        Returns:
            Dicts of ``biotimbre_id``, ``label``, ``created_at``, ``item_count``,
            ordered by ID.
        """
        rows = self._conn.execute(
            "SELECT b.biotimbre_id, b.label, b.created_at, "
            "       COUNT(i.item_id) AS item_count "
            "FROM biotimbres b "
            "LEFT JOIN items i ON i.biotimbre_id = b.biotimbre_id "
            "GROUP BY b.biotimbre_id "
            "ORDER BY b.biotimbre_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def prune_empty_biotimbres(self) -> int:
        """Delete biotimbres that have no items and no label.

        A labelled biotimbre is kept even when empty, on the assumption that the
        label was deliberate and the items may come back.

        Returns:
            Number of rows deleted.
        """
        cursor = self._conn.execute(
            "DELETE FROM biotimbres WHERE label IS NULL AND biotimbre_id NOT IN "
            "(SELECT DISTINCT biotimbre_id FROM items)"
        )
        return cursor.rowcount

    # ── items ─────────────────────────────────────────────────────────────────

    def create_item(
        self,
        draft: ItemDraft,
        *,
        audio_path: str,
        audio_sha256: str,
        duration_s: float,
        sample_rate: int,
        channels: int,
    ) -> LibraryItem:
        """Insert a new item.

        The draft must already be validated; audio properties come from the
        caller, which is the only party that has read the file.

        If ``draft.biotimbre_id`` is ``None`` a fresh ID is allocated; otherwise
        the given biotimbre is created if needed, so an item can join a biotimbre
        that has no rows yet.

        Args:
            draft: Validated draft.
            audio_path: Stored path, relative to the data directory.
            audio_sha256: Hex digest of the stored file.
            duration_s: Sample length in seconds.
            sample_rate: Sample rate of the stored file.
            channels: Channel count of the stored file.

        Returns:
            The persisted :class:`LibraryItem`, with its assigned ``item_id``.

        Raises:
            ValidationError: If audio properties are not positive.
        """
        if duration_s <= 0:
            raise ValidationError(f"Duration {duration_s} s must be positive.", "audio")
        if sample_rate <= 0:
            raise ValidationError(
                f"Sample rate {sample_rate} Hz must be positive.", "audio"
            )
        if channels <= 0:
            raise ValidationError(f"Channel count {channels} must be positive.", "audio")

        biotimbre_id = (
            self.allocate_biotimbre_id()
            if draft.biotimbre_id is None
            else self.ensure_biotimbre(draft.biotimbre_id)
        )

        now = _utc_now()
        cursor = self._conn.execute(
            """
            INSERT INTO items (
                biotimbre_id, date, time, latitude, longitude,
                sample_window_start, sample_window_end, device,
                filtered_band_low, filtered_band_high, noise_reduction, speed,
                link_to_full_recording, notes, audio_path, audio_sha256,
                duration_s, sample_rate, channels, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                biotimbre_id,
                draft.date,
                draft.time,
                draft.location.latitude,
                draft.location.longitude,
                draft.sample_window.start,
                draft.sample_window.end,
                draft.device,
                draft.filtered_band.low,
                draft.filtered_band.high,
                int(draft.noise_reduction),
                draft.speed,
                draft.link_to_full_recording,
                draft.notes,
                audio_path,
                audio_sha256,
                float(duration_s),
                int(sample_rate),
                int(channels),
                now,
                now,
            ),
        )
        item_id = int(cursor.lastrowid)
        self._replace_taxa(item_id, draft.taxa)
        return self.get_item(item_id)

    def get_item(self, item_id: int) -> LibraryItem:
        """Fetch one item by ID.

        Args:
            item_id: The item's ID.

        Returns:
            The item.

        Raises:
            NotFoundError: If no such item exists.
        """
        row = self._conn.execute(
            f"SELECT {_ITEM_COLUMNS} FROM items WHERE item_id = ?", (item_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"No item with id {item_id}.")
        return self._row_to_item(row, self.get_taxa(item_id))

    def find_by_sha256(self, audio_sha256: str) -> list[LibraryItem]:
        """Find items whose stored audio has a given digest.

        Used to warn about re-adding a file already in the library. Byte-identical
        duplicates are not forbidden — the same audio can legitimately be filed
        under two biotimbres — so this reports rather than blocks.

        Args:
            audio_sha256: Hex digest to look for.

        Returns:
            Matching items, possibly empty.
        """
        rows = self._conn.execute(
            f"SELECT {_ITEM_COLUMNS} FROM items WHERE audio_sha256 = ? "
            f"ORDER BY item_id",
            (audio_sha256,),
        ).fetchall()
        return [self._row_to_item(row, self.get_taxa(row["item_id"])) for row in rows]

    def update_item(
        self,
        item_id: int,
        draft: ItemDraft,
        *,
        audio_path: str | None = None,
    ) -> tuple[LibraryItem, bool]:
        """Update an item's metadata.

        Args:
            item_id: Item to update.
            draft: Validated draft holding the new values. A ``biotimbre_id`` of
                ``None`` leaves the item's current biotimbre unchanged, since an
                update never allocates a new one.
            audio_path: New stored path, when the caller has moved the file
                (a biotimbre change relocates the audio).

        Returns:
            The updated item, and whether its embeddings were invalidated.

        Raises:
            NotFoundError: If no such item exists.
        """
        existing = self.get_item(item_id)
        biotimbre_id = (
            existing.biotimbre_id
            if draft.biotimbre_id is None
            else self.ensure_biotimbre(draft.biotimbre_id)
        )

        # Changing speed changes what the embeddings mean, so they cannot be kept.
        # Enforced here rather than in the service so no caller can skip it.
        speed_changed = draft.speed != existing.speed
        if speed_changed:
            self.delete_embeddings(item_id)

        self._conn.execute(
            """
            UPDATE items SET
                biotimbre_id = ?, date = ?, time = ?,
                latitude = ?, longitude = ?,
                sample_window_start = ?, sample_window_end = ?,
                device = ?, filtered_band_low = ?, filtered_band_high = ?,
                noise_reduction = ?, speed = ?,
                link_to_full_recording = ?, notes = ?,
                audio_path = ?, updated_at = ?
            WHERE item_id = ?
            """,
            (
                biotimbre_id,
                draft.date,
                draft.time,
                draft.location.latitude,
                draft.location.longitude,
                draft.sample_window.start,
                draft.sample_window.end,
                draft.device,
                draft.filtered_band.low,
                draft.filtered_band.high,
                int(draft.noise_reduction),
                draft.speed,
                draft.link_to_full_recording,
                draft.notes,
                audio_path if audio_path is not None else existing.audio_path,
                _utc_now(),
                item_id,
            ),
        )
        self._replace_taxa(item_id, draft.taxa)
        return self.get_item(item_id), speed_changed

    def delete_item(self, item_id: int) -> str:
        """Delete an item and, by cascade, its taxa and embeddings.

        Args:
            item_id: Item to delete.

        Returns:
            The stored audio path of the deleted item, so the caller can remove
            the file. The repository does not touch the filesystem.

        Raises:
            NotFoundError: If no such item exists.
        """
        item = self.get_item(item_id)
        self._conn.execute("DELETE FROM items WHERE item_id = ?", (item_id,))
        return item.audio_path

    def list_items(
        self,
        *,
        biotimbre_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[LibraryItem]:
        """List items, newest first.

        Args:
            biotimbre_id: Restrict to one biotimbre.
            limit: Maximum rows to return; ``None`` for all.
            offset: Rows to skip.

        Returns:
            Matching items.
        """
        sql = f"SELECT {_ITEM_COLUMNS} FROM items"
        params: list[object] = []
        if biotimbre_id is not None:
            sql += " WHERE biotimbre_id = ?"
            params.append(parse_biotimbre_id(biotimbre_id))
        sql += " ORDER BY item_id DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])

        rows = self._conn.execute(sql, params).fetchall()
        taxa_map = self.get_taxa_for([int(row["item_id"]) for row in rows])
        return [
            self._row_to_item(row, taxa_map.get(int(row["item_id"]), ()))
            for row in rows
        ]

    def count_items(self) -> int:
        """Total number of items in the library."""
        return int(self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])

    def distinct_speeds(self) -> list[float]:
        """Distinct ``speed`` values present among items, ascending.

        Drives the query-side speed sweep: the query must be embedded at each of
        these to be comparable with the items indexed at them (PLAN.md D2c).

        Returns:
            Ascending speeds. Empty for an empty library.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT speed FROM items ORDER BY speed"
        ).fetchall()
        return [float(row["speed"]) for row in rows]

    # ── taxa ──────────────────────────────────────────────────────────────────

    def get_taxa(self, item_id: int) -> tuple[str, ...]:
        """Taxon tags for one item, sorted.

        Args:
            item_id: The item's ID.

        Returns:
            Sorted tags; empty means unknown.
        """
        rows = self._conn.execute(
            "SELECT tag FROM item_taxa WHERE item_id = ? ORDER BY tag", (item_id,)
        ).fetchall()
        return tuple(row["tag"] for row in rows)

    def get_taxa_for(self, item_ids: Sequence[int]) -> dict[int, tuple[str, ...]]:
        """Taxon tags for many items in one query.

        Args:
            item_ids: Item IDs to look up.

        Returns:
            Mapping of item ID to sorted tags. Items with no tags are absent.
        """
        if not item_ids:
            return {}
        result: dict[int, list[str]] = {}
        for chunk in _chunked(list(item_ids), 500):
            placeholders = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT item_id, tag FROM item_taxa "
                f"WHERE item_id IN ({placeholders}) ORDER BY item_id, tag",
                chunk,
            ).fetchall()
            for row in rows:
                result.setdefault(int(row["item_id"]), []).append(row["tag"])
        return {item_id: tuple(tags) for item_id, tags in result.items()}

    def all_taxa(self) -> list[tuple[str, int]]:
        """Every tag in use with how many items carry it.

        Returns:
            ``(tag, count)`` pairs, most frequent first then alphabetical.
        """
        rows = self._conn.execute(
            "SELECT tag, COUNT(*) AS n FROM item_taxa "
            "GROUP BY tag ORDER BY n DESC, tag"
        ).fetchall()
        return [(row["tag"], int(row["n"])) for row in rows]

    def _replace_taxa(self, item_id: int, taxa: Iterable[str]) -> None:
        """Replace an item's tag set wholesale."""
        tags = normalize_taxa(list(taxa))
        self._conn.execute("DELETE FROM item_taxa WHERE item_id = ?", (item_id,))
        if tags:
            self._conn.executemany(
                "INSERT INTO item_taxa (item_id, tag) VALUES (?, ?)",
                [(item_id, tag) for tag in tags],
            )

    # ── models ────────────────────────────────────────────────────────────────

    def register_model(self, spec: ModelSpec, embedding_dim: int) -> None:
        """Record a model and its detected embedding dimension.

        Re-registering with a *different* dimension is refused: existing vectors
        for that ``model_id`` would silently become unreadable at the wrong width.

        Args:
            spec: The model's specification.
            embedding_dim: Dimension read from the ONNX graph.

        Raises:
            ValidationError: If ``embedding_dim`` is not positive, or conflicts
                with a previous registration.
        """
        if embedding_dim <= 0:
            raise ValidationError(
                f"Embedding dimension {embedding_dim} must be positive.",
                "embedding_dim",
            )

        existing = self.get_model(spec.model_id)
        if existing is not None and int(existing["embedding_dim"]) != embedding_dim:
            raise ValidationError(
                f"Model {spec.model_id!r} is already registered with dimension "
                f"{existing['embedding_dim']}, but the graph now reports "
                f"{embedding_dim}. Use a new model_id instead of changing an "
                f"existing one, or delete its embeddings first.",
                "embedding_dim",
            )

        self._conn.execute(
            """
            INSERT INTO models (
                model_id, source, filename, revision, sample_rate,
                window_duration_s, window_overlap, embedding_dim, registered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                source = excluded.source,
                filename = excluded.filename,
                revision = excluded.revision,
                sample_rate = excluded.sample_rate,
                window_duration_s = excluded.window_duration_s,
                window_overlap = excluded.window_overlap
            """,
            (
                spec.model_id,
                spec.source,
                spec.filename,
                spec.revision,
                spec.sample_rate,
                spec.window_duration_s,
                spec.window_overlap,
                embedding_dim,
                _utc_now(),
            ),
        )

    def get_model(self, model_id: str) -> dict | None:
        """Fetch a registered model's row.

        Args:
            model_id: The model's ID.

        Returns:
            The row as a dict, or ``None`` if not registered.
        """
        row = self._conn.execute(
            "SELECT * FROM models WHERE model_id = ?", (model_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def list_models(self) -> list[dict]:
        """All registered models with their embedding counts.

        Returns:
            Row dicts plus an ``embedding_count`` key, ordered by ID.
        """
        rows = self._conn.execute(
            "SELECT m.*, "
            "  (SELECT COUNT(*) FROM embeddings e WHERE e.model_id = m.model_id) "
            "  AS embedding_count "
            "FROM models m ORDER BY m.model_id"
        ).fetchall()
        return [dict(row) for row in rows]

    # ── embeddings ────────────────────────────────────────────────────────────

    def set_embeddings(
        self,
        item_id: int,
        model_id: str,
        windows: Sequence[tuple[float, float, np.ndarray]],
    ) -> int:
        """Replace all of an item's embeddings for one model.

        Replacing rather than appending keeps the window set consistent with the
        parameters that produced it: a re-run at a different overlap must not
        leave the old windows behind.

        Args:
            item_id: Item the windows belong to.
            model_id: Model that produced them.
            windows: ``(start_time, end_time, vector)`` per window, in order.
                Times are in the sample's own time base (PLAN.md D2b.2).

        Returns:
            Number of window rows written.

        Raises:
            NotFoundError: If the item or model does not exist.
            ValidationError: If a vector's width disagrees with the model's
                registered dimension.
        """
        model = self.get_model(model_id)
        if model is None:
            raise NotFoundError(
                f"Model {model_id!r} is not registered; register it before "
                f"storing embeddings."
            )
        if self._conn.execute(
            "SELECT 1 FROM items WHERE item_id = ?", (item_id,)
        ).fetchone() is None:
            raise NotFoundError(f"No item with id {item_id}.")

        expected_dim = int(model["embedding_dim"])
        rows = []
        for index, (start, end, vector) in enumerate(windows):
            flat = np.ascontiguousarray(vector, dtype=VECTOR_DTYPE).ravel()
            if flat.size != expected_dim:
                raise ValidationError(
                    f"Window {index} has dimension {flat.size}, but model "
                    f"{model_id!r} is registered with {expected_dim}.",
                    "embedding",
                )
            rows.append(
                (item_id, model_id, index, float(start), float(end), flat.tobytes())
            )

        self._conn.execute(
            "DELETE FROM embeddings WHERE item_id = ? AND model_id = ?",
            (item_id, model_id),
        )
        if rows:
            self._conn.executemany(
                "INSERT INTO embeddings "
                "(item_id, model_id, window_index, start_time, end_time, vector) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def get_embeddings(
        self, item_id: int, model_id: str
    ) -> list[tuple[int, float, float, np.ndarray]]:
        """Load one item's windows for one model, in window order.

        Args:
            item_id: The item's ID.
            model_id: The model's ID.

        Returns:
            ``(window_index, start_time, end_time, vector)`` tuples; empty if
            this item has no embeddings for this model.
        """
        rows = self._conn.execute(
            "SELECT window_index, start_time, end_time, vector FROM embeddings "
            "WHERE item_id = ? AND model_id = ? ORDER BY window_index",
            (item_id, model_id),
        ).fetchall()
        return [
            (
                int(row["window_index"]),
                float(row["start_time"]),
                float(row["end_time"]),
                _blob_to_vector(row["vector"]),
            )
            for row in rows
        ]

    def get_embedding_matrix(
        self, item_id: int, model_id: str
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """Load one item's windows as a matrix plus their time spans.

        Args:
            item_id: The item's ID.
            model_id: The model's ID.

        Returns:
            An ``(n_windows, dim)`` float32 array and a parallel list of
            ``(start_time, end_time)``. The array is ``(0, 0)``-shaped when there
            are no embeddings.
        """
        windows = self.get_embeddings(item_id, model_id)
        if not windows:
            return np.zeros((0, 0), dtype=VECTOR_DTYPE), []
        matrix = np.stack([vector for _, _, _, vector in windows])
        spans = [(start, end) for _, start, end, _ in windows]
        return matrix, spans

    def delete_embeddings(self, item_id: int, model_id: str | None = None) -> int:
        """Delete embeddings for an item.

        Args:
            item_id: The item's ID.
            model_id: Restrict to one model; ``None`` deletes every model's.

        Returns:
            Number of window rows deleted.
        """
        if model_id is None:
            cursor = self._conn.execute(
                "DELETE FROM embeddings WHERE item_id = ?", (item_id,)
            )
        else:
            cursor = self._conn.execute(
                "DELETE FROM embeddings WHERE item_id = ? AND model_id = ?",
                (item_id, model_id),
            )
        return cursor.rowcount

    def embedding_counts(self, item_id: int) -> dict[str, int]:
        """Window count per model for one item.

        Args:
            item_id: The item's ID.

        Returns:
            Mapping of model ID to window count; models with none are absent.
        """
        rows = self._conn.execute(
            "SELECT model_id, COUNT(*) AS n FROM embeddings "
            "WHERE item_id = ? GROUP BY model_id ORDER BY model_id",
            (item_id,),
        ).fetchall()
        return {row["model_id"]: int(row["n"]) for row in rows}

    def items_missing_embeddings(self, model_id: str) -> list[int]:
        """IDs of items with no embeddings for a model.

        Drives ``backfill`` when a model is added to an existing library.

        Args:
            model_id: The model's ID.

        Returns:
            Ascending item IDs.
        """
        rows = self._conn.execute(
            "SELECT item_id FROM items WHERE item_id NOT IN "
            "(SELECT DISTINCT item_id FROM embeddings WHERE model_id = ?) "
            "ORDER BY item_id",
            (model_id,),
        ).fetchall()
        return [int(row["item_id"]) for row in rows]

    def iter_embedding_batches(
        self,
        model_id: str,
        *,
        item_ids: Sequence[int] | None = None,
        batch_size: int = 8192,
    ) -> Iterator[tuple[list[tuple[int, int, float, float]], np.ndarray]]:
        """Stream a model's embeddings in batches for scoring.

        Rows are ordered by ``(item_id, window_index)`` so an item's windows stay
        together, which lets the caller reduce per item as it goes.

        Candidate restriction goes through a temporary table rather than an
        ``IN (...)`` list: SQLite caps host parameters (999 by default) and long
        ``IN`` lists degrade, which would bite exactly when a filter is selective
        over a large library.

        Args:
            model_id: Model whose embeddings to stream.
            item_ids: Restrict to these items. ``None`` streams all; an *empty*
                sequence yields nothing, since no candidate survived filtering.
            batch_size: Rows per batch.

        Yields:
            ``keys`` — ``(item_id, window_index, start_time, end_time)`` per row,
            and ``matrix`` — the corresponding ``(B, dim)`` float32 array.
        """
        if item_ids is not None and len(item_ids) == 0:
            return

        cursor = self._conn.cursor()
        temp_created = False
        try:
            if item_ids is None:
                cursor.execute(
                    "SELECT item_id, window_index, start_time, end_time, vector "
                    "FROM embeddings WHERE model_id = ? "
                    "ORDER BY item_id, window_index",
                    (model_id,),
                )
            else:
                self._conn.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS _search_candidates "
                    "(item_id INTEGER PRIMARY KEY)"
                )
                self._conn.execute("DELETE FROM _search_candidates")
                temp_created = True
                self._conn.executemany(
                    "INSERT OR IGNORE INTO _search_candidates (item_id) VALUES (?)",
                    [(int(i),) for i in item_ids],
                )
                cursor.execute(
                    "SELECT e.item_id, e.window_index, e.start_time, e.end_time, "
                    "       e.vector "
                    "FROM embeddings e "
                    "JOIN _search_candidates c ON c.item_id = e.item_id "
                    "WHERE e.model_id = ? "
                    "ORDER BY e.item_id, e.window_index",
                    (model_id,),
                )

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                keys = [
                    (
                        int(row["item_id"]),
                        int(row["window_index"]),
                        float(row["start_time"]),
                        float(row["end_time"]),
                    )
                    for row in rows
                ]
                matrix = np.stack([_blob_to_vector(row["vector"]) for row in rows])
                yield keys, matrix
        finally:
            cursor.close()
            if temp_created:
                self._conn.execute("DELETE FROM _search_candidates")

    # ── row mapping ───────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_item(row: sqlite3.Row, taxa: tuple[str, ...]) -> LibraryItem:
        """Build a :class:`LibraryItem` from a row and its tags."""
        return LibraryItem(
            item_id=int(row["item_id"]),
            biotimbre_id=row["biotimbre_id"],
            date=row["date"],
            time=row["time"],
            location=Location(float(row["latitude"]), float(row["longitude"])),
            sample_window=SampleWindow(
                float(row["sample_window_start"]), float(row["sample_window_end"])
            ),
            audio_path=row["audio_path"],
            audio_sha256=row["audio_sha256"],
            duration_s=float(row["duration_s"]),
            sample_rate=int(row["sample_rate"]),
            channels=int(row["channels"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            taxa=taxa,
            device=row["device"],
            filtered_band=FilteredBand(
                row["filtered_band_low"], row["filtered_band_high"]
            ),
            noise_reduction=bool(row["noise_reduction"]),
            speed=float(row["speed"]) if row["speed"] is not None else DEFAULT_SPEED,
            link_to_full_recording=row["link_to_full_recording"],
            notes=row["notes"],
        )


def _chunked(values: list[int], size: int) -> Iterator[list[int]]:
    """Split a list into chunks of at most ``size`` elements."""
    for start in range(0, len(values), size):
        yield values[start : start + size]
