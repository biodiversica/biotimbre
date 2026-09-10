"""LibraryService — the single API every frontend calls.

The CLI, the future GUI and any HTTP API are peers that call this class and hold
no logic of their own (PLAN.md D5). That is what makes the eventual public search
tool a frontend swap rather than a rewrite.

The service owns the pieces the layers below deliberately do not:

* **transactions** — an ingest writes the audio file, the item row and the
  embedding rows, and either all of it lands or none of it does;
* **the filesystem** — the repository never touches disk, so audio copying,
  moving and deletion happen here;
* **model lifecycle** — extractors are built once and reused, since loading a
  45 MB graph per item would dominate ingest time.
"""

from __future__ import annotations

import io
import math
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from biotimbre.config import Settings, load_settings
from biotimbre.db.connection import (
    LibraryIdentity,
    get_library_identity,
    initialize,
    set_library_name,
)
from biotimbre.db.repository import Repository
from biotimbre.domain.errors import (
    BiotimbreError,
    NotFoundError,
    ValidationError,
)
from biotimbre.domain.item import (
    FilteredBand,
    ItemDraft,
    LibraryItem,
    is_species_name,
)
from biotimbre.embeddings import audio as audio_io
from biotimbre.embeddings.extractor import ExtractorPool, WindowEmbeddings
from biotimbre.embeddings.query_cache import AudioKey, QueryEmbeddingCache
from biotimbre.models_registry import get_spec
from biotimbre.search.filters import SearchFilters, select_candidates
from biotimbre.search.vector_search import (
    FusedScore,
    WindowMatch,
    fuse,
    reduce_to_items,
    score_matrix,
    top_k,
)


def npy_filename(item_id: int, model_id: str) -> str:
    """Conventional name for an exported embedding file.

    One place, so a download from the interface and a ``biotimbre export-npy``
    on the command line produce the same filename for the same rows.

    Args:
        item_id: The item.
        model_id: The model.

    Returns:
        A bare filename, no directory.
    """
    return f"item_{item_id}_{model_id}.npy"


def npy_archive_filename(item_id: int, model_id: str) -> str:
    """Conventional name for a multi-window embedding export.

    Args:
        item_id: The item.
        model_id: The model.

    Returns:
        A bare filename, no directory.
    """
    return f"item_{item_id}_{model_id}.zip"


def window_npy_filename(
    item_id: int, model_id: str, index: int, start: float, end: float
) -> str:
    """Name of one window's file inside an export archive.

    The window's place in the sample is in the name: a loose ``.npy`` extracted
    from the archive still says which seconds it describes.

    Args:
        item_id: The item.
        model_id: The model.
        index: Window index, in window order.
        start: Window start in seconds, in the sample's own time base.
        end: Window end in seconds.

    Returns:
        A bare filename, no directory.
    """
    return (
        f"item_{item_id}_{model_id}_w{index:02d}_{start:.2f}-{end:.2f}s.npy"
    )


def audio_filename(item_id: int, suffix: str) -> str:
    """Conventional name for an exported audio file.

    Args:
        item_id: The item.
        suffix: Extension of the stored file, dot included.

    Returns:
        A bare filename, no directory.
    """
    return f"item_{item_id}{suffix}"


@dataclass(frozen=True)
class SearchHit:
    """One search result, with everything a UI needs to render and play it.

    Attributes:
        item: The matched library item.
        score: Fused similarity score.
        per_model: Each model's individual score.
        matches: Each model's winning window pair.
        distance_km: Distance from a location filter's centre, if one was set.
        query_speed: Speed the query was embedded at to produce this match,
            which equals the item's own speed (PLAN.md D2c).
    """

    item: LibraryItem
    score: float
    per_model: dict[str, float] = field(default_factory=dict)
    matches: dict[str, WindowMatch] = field(default_factory=dict)
    distance_km: float | None = None
    query_speed: float = 1.0

    @property
    def best_model(self) -> str | None:
        """Model that produced the highest individual score."""
        if not self.per_model:
            return None
        return max(self.per_model, key=lambda m: self.per_model[m])

    @property
    def matched_span(self) -> tuple[float, float] | None:
        """Winning window span from the best model, clamped to the sample.

        Clamping matters because a zero-padded final window legitimately reports
        an end past the file's duration, and a player cannot seek there.
        """
        model = self.best_model
        if model is None:
            return None
        match = self.matches[model]
        return (
            max(0.0, match.start_time),
            min(self.item.duration_s, match.end_time),
        )


@dataclass(frozen=True)
class IngestResult:
    """Outcome of adding one item.

    Attributes:
        item: The stored item.
        window_counts: Windows written per model.
        duplicate_of: Item IDs that already held byte-identical audio.
        warnings: Non-fatal advisories, e.g. a Nyquist mismatch.
        joined_species: The species tag whose existing biotimbre this item was
            filed under, or ``None`` when a biotimbre was allocated or given.
    """

    item: LibraryItem
    window_counts: dict[str, int] = field(default_factory=dict)
    duplicate_of: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    joined_species: str | None = None


class LibraryService:
    """High-level operations over a biotimbre library.

    Args:
        settings: Resolved settings. ``None`` loads them from the usual places.
        connection: An existing database connection, for tests. ``None`` opens
            (and initialises) the configured database.
        pool: A pre-built extractor pool to share. Creating one costs about two
            seconds of ONNX session setup per process, so a long-lived frontend
            should build it once at startup — where it can show progress — rather
            than paying for it on the first search. A pool passed in is *not*
            closed by :meth:`close`; the caller owns its lifetime.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        connection=None,
        pool: ExtractorPool | None = None,
    ):
        self.settings = settings if settings is not None else load_settings()
        self.settings.ensure_dirs()
        self._owns_connection = connection is None
        self._conn = connection if connection is not None else initialize(
            self.settings.db_path
        )
        self.repo = Repository(self._conn)
        self._owns_pool = pool is None
        self._pool: ExtractorPool | None = pool
        self._query_cache: QueryEmbeddingCache | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the model sessions and, if owned, the database connection."""
        if self._query_cache is not None:
            self._query_cache.clear()
            self._query_cache = None
        if self._pool is not None and self._owns_pool:
            self._pool.close()
            self._pool = None
        if self._owns_connection:
            self._conn.close()

    def __enter__(self) -> "LibraryService":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def pool(self) -> ExtractorPool:
        """The extractor pool, built on first use."""
        if self._pool is None:
            self._pool = ExtractorPool(
                self.settings.active_models,
                cache_dir=self.settings.model_cache_dir,
                max_batch=self.settings.inference_batch_size,
                threads=self.settings.inference_threads,
            )
        return self._pool

    @property
    def query_cache(self) -> QueryEmbeddingCache:
        """Cache of query embeddings, built on first use.

        Only the search path uses it. Ingest deliberately does not: every item is
        distinct, so caching there would grow memory through a bulk import for no
        hit rate at all.
        """
        if self._query_cache is None:
            self._query_cache = QueryEmbeddingCache(
                self.pool, max_entries=self.settings.query_cache_entries
            )
        return self._query_cache

    @contextmanager
    def _transaction(self):
        """Run a block as one database transaction."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── identity ──────────────────────────────────────────────────────────────

    def identity(self) -> LibraryIdentity:
        """The library's UUID and name (name is ``None`` until set)."""
        return get_library_identity(self._conn)

    def rename(self, name: str) -> LibraryIdentity:
        """Set or change the library's name.

        Args:
            name: The new name. Must be non-empty.

        Returns:
            The updated :class:`LibraryIdentity`.

        Raises:
            ValidationError: If ``name`` is empty or all whitespace.
        """
        set_library_name(self._conn, name)
        return self.identity()

    # ── biotimbres ────────────────────────────────────────────────────────────

    def get_biotimbre_label(self, biotimbre_id: str) -> str | None:
        """The human-readable label of a biotimbre, if it has one.

        Args:
            biotimbre_id: ID in any accepted form.

        Returns:
            The label, or ``None`` if unset or the biotimbre does not exist.
        """
        row = self.repo.get_biotimbre(biotimbre_id)
        return row["label"] if row else None

    def set_biotimbre_label(self, biotimbre_id: str, label: str | None) -> None:
        """Set or clear a biotimbre's label.

        Args:
            biotimbre_id: ID in any accepted form.
            label: New label, or ``None``/empty to clear it.

        Raises:
            NotFoundError: If the biotimbre does not exist.
        """
        with self._transaction():
            self.repo.set_biotimbre_label(biotimbre_id, label)

    # ── models ────────────────────────────────────────────────────────────────

    def ensure_models_registered(self) -> dict[str, int]:
        """Register every active model, reading dimensions from the graphs.

        Downloads model files on first call. Idempotent.

        Returns:
            Mapping of model ID to embedding dimension.
        """
        dims: dict[str, int] = {}
        for model_id in self.settings.active_models:
            extractor = self.pool.get(model_id)
            dim = extractor.embedding_dim()
            with self._transaction():
                self.repo.register_model(extractor.spec, dim)
            dims[model_id] = dim
        return dims

    def list_models(self) -> list[dict]:
        """Registered models with their embedding counts."""
        return self.repo.list_models()

    # ── ingest ────────────────────────────────────────────────────────────────

    def _nyquist_warnings(self, draft: ItemDraft) -> list[str]:
        """Advise when a sample's declared band exceeds a model's Nyquist.

        Resampling anti-alias-filters away content above the model's Nyquist, so
        a sample whose energy sits up there gets an embedding of whatever little
        remains. ``speed`` is the remedy; this turns a silent failure into a
        prompt (PLAN.md §7).
        """
        high = draft.filtered_band.high
        if high is None:
            return []

        shifted = high * draft.speed
        warnings = []
        for model_id in self.settings.active_models:
            nyquist = get_spec(model_id).nyquist_hz
            if shifted > nyquist:
                suggested = nyquist / high
                warnings.append(
                    f"{model_id}: content up to {high:g} Hz "
                    f"({shifted:g} Hz after speed {draft.speed:g}) exceeds the "
                    f"model's {nyquist:g} Hz limit, so the upper band will be "
                    f"filtered out before embedding. Consider speed "
                    f"<= {suggested:.3g}."
                )
        return warnings

    def add_item(
        self,
        audio_path: str | Path,
        draft: ItemDraft,
        *,
        compute_embeddings: bool = True,
    ) -> IngestResult:
        """Add a sample to the library.

        The audio is read for its properties, copied into the library tree, the
        row is inserted and embeddings are computed — all inside one transaction,
        so a failure part-way leaves neither a stray row nor a stray file.

        A submission that names a species already in the library is filed
        under that species' biotimbre rather than starting a second identity
        for it — one species, one biotimbre. Naming a ``biotimbre_id``
        explicitly overrides this; a rank above species does not trigger it.

        What is stored is always mono. A multi-channel file is averaged down
        first (:func:`~biotimbre.embeddings.audio.write_processed`), so every
        item's audio is the single channel the models and the spectrograms
        actually use.

        Args:
            audio_path: Path to the source audio file. It is copied, not moved;
                a multi-channel one is converted rather than copied.
            draft: The item's metadata. Validated if it has not been already.
            compute_embeddings: Set false to defer embedding to a later backfill.

        Returns:
            The :class:`IngestResult`.

        Raises:
            ValidationError: If the audio or metadata is unusable.
            EmbeddingError: If embedding fails.
        """
        source = Path(audio_path)
        if not source.is_file():
            raise ValidationError(f"No such audio file: {source}", "audio")

        validated = draft.validated()
        properties = audio_io.read_properties(source)

        # The library stores mono: it is what every model is fed and what a
        # spectrogram shows, so a multi-channel submission is folded down here
        # rather than kept in a form nothing downstream reads. The digest then
        # describes the stored file, which is what audio_sha256 means — and two
        # copies of the same stereo recording still recognise each other.
        staging: tempfile.TemporaryDirectory | None = None
        if properties.channels > 1:
            staging = tempfile.TemporaryDirectory(prefix="biotimbre-mono-")
            source = Path(staging.name) / f"{source.stem}.wav"
            properties = audio_io.write_processed(audio_path, source)

        joined = None
        if validated.biotimbre_id is None:
            validated, joined, species_warnings = self._join_species_biotimbre(
                validated
            )
        else:
            species_warnings = ()

        try:
            result = self._ingest(
                source, validated, properties, compute_embeddings=compute_embeddings
            )
        finally:
            if staging is not None:
                staging.cleanup()

        return replace(
            result,
            joined_species=joined,
            warnings=result.warnings + species_warnings,
        )

    def _join_species_biotimbre(
        self, draft: ItemDraft
    ) -> tuple[ItemDraft, str | None, tuple[str, ...]]:
        """Reuse the biotimbre a species already has, if it has one.

        A species is one sound-being: two recordings of *Guyalna bonaerensis*
        are the same biotimbre by definition, so a submission that names a
        species already in the library joins it rather than starting a second
        identity for it. Ranks above species ("cicadidae") say nothing about
        which being was recorded and are left alone, as is open nomenclature
        ("guyalna sp."), where the species is precisely what is unknown.

        Args:
            draft: A validated draft with no biotimbre of its own.

        Returns:
            The draft (with a biotimbre when one was found), the species tag
            that supplied it, and any advisories.
        """
        for tag in draft.taxa:
            if not is_species_name(tag):
                continue
            existing = self.repo.biotimbres_with_taxon(tag)
            if not existing:
                continue
            warnings: tuple[str, ...] = ()
            if len(existing) > 1:
                # Predates this rule, or someone filed the species twice by
                # hand. The oldest is the one to converge on, and saying so is
                # better than silently picking.
                warnings = (
                    f"{tag} is filed under {len(existing)} biotimbres; this "
                    f"sample joins the oldest, {existing[0]}.",
                )
            return replace(draft, biotimbre_id=existing[0]), tag, warnings
        return draft, None, ()

    def _ingest(
        self,
        source: Path,
        validated: ItemDraft,
        properties: audio_io.AudioProperties,
        *,
        compute_embeddings: bool,
    ) -> IngestResult:
        """Store one prepared file and its row. See :meth:`add_item`.

        Args:
            source: The exact audio to store, already mono.
            validated: Validated metadata.
            properties: The source's properties.
            compute_embeddings: Whether to embed now.

        Returns:
            The :class:`IngestResult`.
        """
        digest = audio_io.file_sha256(source)
        duplicates = tuple(item.item_id for item in self.repo.find_by_sha256(digest))
        warnings = self._nyquist_warnings(validated)

        if compute_embeddings:
            self.ensure_models_registered()
            # Computed before the transaction opens: inference takes seconds and
            # holding a write transaction open across it would block readers.
            embeddings = self.pool.embed_file(source, speed=validated.speed)
        else:
            embeddings = {}

        stored_path: Path | None = None
        try:
            with self._transaction():
                item = self.repo.create_item(
                    validated,
                    audio_path="",  # replaced below, once item_id is known
                    audio_sha256=digest,
                    duration_s=properties.duration_s,
                    sample_rate=properties.sample_rate,
                    channels=properties.channels,
                )

                # The stored path embeds both IDs, so it is only knowable after
                # the insert assigns item_id.
                destination = self.settings.audio_path_for(
                    item.biotimbre_id, item.item_id
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                stored_path = destination

                relative = self.settings.relative_audio_path(destination)
                self._conn.execute(
                    "UPDATE items SET audio_path = ? WHERE item_id = ?",
                    (relative, item.item_id),
                )

                counts = {
                    model_id: self.repo.set_embeddings(
                        item.item_id, model_id, result.as_rows()
                    )
                    for model_id, result in embeddings.items()
                }
                stored = self.repo.get_item(item.item_id)
        except Exception:
            # The row is gone by rollback, but the copied file is not.
            if stored_path is not None and stored_path.exists():
                stored_path.unlink()
            raise

        return IngestResult(
            item=stored,
            window_counts=counts,
            duplicate_of=duplicates,
            warnings=tuple(warnings),
        )

    def add_from_sidecar(
        self, audio_path: str | Path, sidecar_path: str | Path | None = None
    ) -> IngestResult:
        """Add a sample using metadata from its sidecar file.

        Args:
            audio_path: Path to the audio file.
            sidecar_path: Path to the sidecar. ``None`` looks for one beside the
                audio file.

        Returns:
            The :class:`IngestResult`.

        Raises:
            ValidationError: If no sidecar is found, or its contents are invalid.
        """
        from biotimbre.io.sidecar import draft_from_sidecar, find_sidecar

        resolved = (
            Path(sidecar_path) if sidecar_path is not None else find_sidecar(audio_path)
        )
        if resolved is None:
            raise ValidationError(
                f"No sidecar found for {Path(audio_path).name}. Expected a "
                f".txt file with the same stem.",
                "sidecar",
            )
        return self.add_item(audio_path, draft_from_sidecar(resolved))

    def import_directory(
        self, directory: str | Path, *, compute_embeddings: bool = True
    ) -> tuple[list[IngestResult], list[tuple[Path, str]]]:
        """Add every audio file in a directory that has a sidecar.

        One bad file does not abort the run: failures are collected and returned
        so a large import reports everything wrong with it at once.

        Args:
            directory: Directory to scan, non-recursively.
            compute_embeddings: Set false to defer embedding.

        Returns:
            Successful results, and ``(path, message)`` for each failure.
        """
        from biotimbre.io.sidecar import draft_from_sidecar, find_sidecar

        root = Path(directory)
        if not root.is_dir():
            raise ValidationError(f"Not a directory: {root}", "path")

        results: list[IngestResult] = []
        failures: list[tuple[Path, str]] = []

        for audio_file in sorted(root.iterdir()):
            if audio_file.suffix.lower() not in (".wav", ".flac", ".ogg", ".aiff"):
                continue
            try:
                sidecar = find_sidecar(audio_file)
                if sidecar is None:
                    failures.append((audio_file, "no sidecar file found"))
                    continue
                results.append(
                    self.add_item(
                        audio_file,
                        draft_from_sidecar(sidecar),
                        compute_embeddings=compute_embeddings,
                    )
                )
            except BiotimbreError as exc:
                failures.append((audio_file, str(exc)))

        return results, failures

    # ── edit and delete ───────────────────────────────────────────────────────

    def update_item(self, item_id: int, draft: ItemDraft) -> IngestResult:
        """Update an item's metadata, re-embedding if its speed changed.

        A biotimbre change relocates the audio file, since the stored path
        contains the biotimbre ID.

        Args:
            item_id: Item to update.
            draft: New metadata.

        Returns:
            An :class:`IngestResult` whose ``window_counts`` is populated only if
            re-embedding happened.

        Raises:
            NotFoundError: If the item does not exist.
        """
        existing = self.repo.get_item(item_id)
        validated = draft.validated()

        new_biotimbre = validated.biotimbre_id or existing.biotimbre_id
        moved_to: Path | None = None
        old_absolute = self.settings.absolute_audio_path(existing.audio_path)
        relative: str | None = None

        if new_biotimbre != existing.biotimbre_id:
            destination = self.settings.audio_path_for(new_biotimbre, item_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if old_absolute.exists():
                shutil.copy2(old_absolute, destination)
                moved_to = destination
            relative = self.settings.relative_audio_path(destination)

        try:
            with self._transaction():
                updated, invalidated = self.repo.update_item(
                    item_id, validated, audio_path=relative
                )
        except Exception:
            if moved_to is not None and moved_to.exists():
                moved_to.unlink()
            raise

        # Only remove the original once the row committed to the new path.
        if moved_to is not None and old_absolute.exists():
            old_absolute.unlink()

        counts: dict[str, int] = {}
        if invalidated:
            counts = self.compute_embeddings_for(item_id)

        return IngestResult(
            item=self.repo.get_item(item_id),
            window_counts=counts,
            warnings=tuple(self._nyquist_warnings(validated)),
        )

    def delete_item(self, item_id: int, *, remove_audio: bool = True) -> None:
        """Delete an item, its taxa, its embeddings and optionally its audio.

        Args:
            item_id: Item to delete.
            remove_audio: Whether to delete the stored audio file.

        Raises:
            NotFoundError: If the item does not exist.
        """
        with self._transaction():
            stored_path = self.repo.delete_item(item_id)
            self.repo.prune_empty_biotimbres()

        if remove_audio:
            absolute = self.settings.absolute_audio_path(stored_path)
            if absolute.exists():
                absolute.unlink()
            parent = absolute.parent
            # Remove the biotimbre directory once its last item is gone.
            if parent.is_dir() and parent != self.settings.audio_dir:
                try:
                    parent.rmdir()
                except OSError:
                    pass  # Other items still live here.

    # ── embeddings ────────────────────────────────────────────────────────────

    def compute_embeddings_for(
        self, item_id: int, *, model_ids: tuple[str, ...] | None = None
    ) -> dict[str, int]:
        """Compute and store embeddings for one existing item.

        Args:
            item_id: Item to embed.
            model_ids: Models to run. ``None`` runs every active model.

        Returns:
            Windows written per model.

        Raises:
            NotFoundError: If the item or its audio file is missing.
        """
        item = self.repo.get_item(item_id)
        source = self.settings.absolute_audio_path(item.audio_path)
        if not source.is_file():
            raise NotFoundError(
                f"Audio for item {item_id} is missing from {source}."
            )

        self.ensure_models_registered()
        targets = model_ids or self.settings.active_models

        samples, native_rate = audio_io.load_mono(source)
        counts: dict[str, int] = {}
        for model_id in targets:
            result = self.pool.get(model_id).embed_array(
                samples, native_rate, speed=item.speed
            )
            with self._transaction():
                counts[model_id] = self.repo.set_embeddings(
                    item_id, model_id, result.as_rows()
                )
        return counts

    def backfill(self, model_id: str) -> dict[int, int]:
        """Embed every item that lacks embeddings for a model.

        Run after adding a model to an existing library.

        Args:
            model_id: The model to backfill.

        Returns:
            Mapping of item ID to windows written.
        """
        self.ensure_models_registered()
        pending = self.repo.items_missing_embeddings(model_id)
        written: dict[int, int] = {}
        for item_id in pending:
            counts = self.compute_embeddings_for(item_id, model_ids=(model_id,))
            written[item_id] = counts.get(model_id, 0)
        return written

    def embeddings_export(self, item_id: int, model_id: str) -> tuple[bytes, str]:
        """One item's embeddings for one model, as a file per window.

        A window is the unit that means something — one vector, over one span
        of seconds — so no file here holds more than one of them. A model with
        a single window exports as a bare ``.npy``; several windows export as a
        zip of one ``.npy`` each, named with the window's index and its span.

        The same bytes :meth:`export_npy` writes, for a caller that has nowhere
        to write them — a browser download, say.

        Args:
            item_id: The item.
            model_id: The model.

        Returns:
            The file's bytes and the name to save them under. Each ``.npy``
            holds one 1-D vector of the model's width.

        Raises:
            NotFoundError: If the item has no embeddings for this model.
        """
        windows = self.repo.get_embeddings(item_id, model_id)
        if not windows:
            raise NotFoundError(
                f"Item {item_id} has no {model_id} embeddings to export."
            )

        def npy_bytes(vector: np.ndarray) -> bytes:
            buffer = io.BytesIO()
            np.save(buffer, vector)
            return buffer.getvalue()

        if len(windows) == 1:
            return npy_bytes(windows[0][3]), npy_filename(item_id, model_id)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for index, start, end, vector in windows:
                # A fixed timestamp, so exporting the same rows twice gives the
                # same bytes: the archive can be checksummed as provenance, and
                # a download matches a command-line export exactly. 1980-01-01
                # is the earliest a zip entry can record.
                entry = zipfile.ZipInfo(
                    window_npy_filename(item_id, model_id, index, start, end),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, npy_bytes(vector))
        return buffer.getvalue(), npy_archive_filename(item_id, model_id)

    def export_npy(self, item_id: int, model_id: str, destination: str | Path) -> Path:
        """Write one item's embeddings for one model to disk.

        Provided for interop and hand-off; the database remains the source of
        truth (PLAN.md D1). One window writes a ``.npy``; several write a zip
        of one ``.npy`` per window — see :meth:`embeddings_export`.

        Args:
            item_id: The item.
            model_id: The model.
            destination: Output path, or a directory to name the file inside.

        Returns:
            The written path.

        Raises:
            NotFoundError: If the item has no embeddings for this model.
        """
        data, name = self.embeddings_export(item_id, model_id)

        out = Path(destination)
        if out.is_dir():
            out = out / name
        # A name is not a promise about the contents: correct the extension
        # rather than leave a zip called .npy. Appending, not replacing, since
        # model ids carry dots of their own ("birdnet_v2.4").
        suffix = Path(name).suffix
        if out.suffix != suffix:
            out = out.with_suffix(out.suffix + suffix)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

    # ── read ──────────────────────────────────────────────────────────────────

    def get_item(self, item_id: int) -> LibraryItem:
        """Fetch one item.

        Args:
            item_id: The item's ID.

        Returns:
            The item.

        Raises:
            NotFoundError: If it does not exist.
        """
        return self.repo.get_item(item_id)

    def list_items(
        self, *, biotimbre_id: str | None = None, limit: int | None = None
    ) -> list[LibraryItem]:
        """List items, newest first.

        Args:
            biotimbre_id: Restrict to one biotimbre.
            limit: Maximum items to return.

        Returns:
            Matching items.
        """
        return self.repo.list_items(biotimbre_id=biotimbre_id, limit=limit)

    def get_audio(self, item_id: int) -> tuple[Path, str]:
        """Resolve an item's audio for playback.

        Frontends serve audio through this call rather than exposing the audio
        directory, which keeps one place to enforce access control if the search
        tool becomes public (PLAN.md D8).

        Args:
            item_id: The item's ID.

        Returns:
            The absolute path and its MIME type.

        Raises:
            NotFoundError: If the item or its file is missing.
        """
        item = self.repo.get_item(item_id)
        path = self.settings.absolute_audio_path(item.audio_path)
        if not path.is_file():
            raise NotFoundError(
                f"Audio for item {item_id} is missing from {path}."
            )
        mime = {
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".aiff": "audio/aiff",
        }.get(path.suffix.lower(), "application/octet-stream")
        return path, mime

    def stats(self) -> dict:
        """Summary counts for a dashboard or CLI status line."""
        return {
            "items": self.repo.count_items(),
            "biotimbres": len(self.repo.list_biotimbres()),
            "models": self.repo.list_models(),
            "taxa": self.repo.all_taxa(),
            "speeds": self.repo.distinct_speeds(),
        }

    # ── search ────────────────────────────────────────────────────────────────

    def filter_items(
        self, filters: SearchFilters | None = None, *, k: int | None = None
    ) -> list[SearchHit]:
        """Find items by metadata alone — no query audio, so nothing is scored.

        The same filters :meth:`search` applies before scoring can stand on
        their own: "every cicadidae recording within 5 km" is a question about
        metadata, and demanding a recording to ask it would be arbitrary.

        Args:
            filters: Metadata constraints. An empty instance matches the whole
                library.
            k: Maximum items. Defaults to the configured ``top_k``.

        Returns:
            Matching items, newest first, as hits with no score and no
            per-model detail — there is no query to compare them against. A
            location filter still reports each item's distance.
        """
        filters = filters or SearchFilters()
        k = k if k is not None else self.settings.default_top_k
        candidates, distances = select_candidates(self._conn, filters)
        if candidates is not None and not candidates:
            return []
        items = self.repo.list_items(item_ids=candidates, limit=k)
        return [
            SearchHit(
                item=item,
                score=0.0,
                distance_km=distances.get(item.item_id),
            )
            for item in items
        ]

    def search(
        self,
        audio_path: str | Path,
        *,
        filters: SearchFilters | None = None,
        k: int | None = None,
        metric: str | None = None,
        fusion: str | None = None,
        min_score: float | None = None,
        query_speed: float | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        filter_band: FilteredBand | None = None,
        filter_order: int = 4,
    ) -> list[SearchHit]:
        """Find the library items most similar to a short query recording.

        The query is limited to ``max_query_duration_s`` (15 s by default). This
        library matches short vocalizations; a longer query is slow to embed and
        analytically weaker, because max-over-window-pairs will latch onto
        whichever fragment happens to match. A longer file therefore needs an
        explicit ``start_time``/``end_time``, rather than a segment being guessed.

        The query is embedded at every speed present in the library, so an item
        indexed at a non-native speed is still findable from a natively-recorded
        query (PLAN.md D2c). Each item is scored against the query embedding
        computed at *that item's* speed.

        Args:
            audio_path: Query audio file.
            filters: Metadata constraints applied before scoring.
            k: Maximum hits. Defaults to the configured ``top_k``.
            metric: Similarity metric. Defaults to the configured metric.
            fusion: ``"max"``, ``"mean"``, or a model ID.
            min_score: Discard hits below this score.
            query_speed: Embed the query only at this speed, instead of sweeping
                the library's speeds. Use when the query itself needs shifting.
            start_time: Start of the segment to search with, in seconds.
            end_time: End of the segment to search with, in seconds.
            filter_band: Butterworth highpass/lowpass/bandpass cutoffs to apply
                to the query before embedding. ``None`` or an unset band applies
                no filter. Unlike ``items.filtered_band``, which only records
                filtering already done elsewhere, this actually runs the filter
                (see :func:`biotimbre.embeddings.audio.apply_filter`).
            filter_order: Order of the Butterworth filter. Ignored when
                ``filter_band`` is unset.

        Returns:
            Descending :class:`SearchHit` list.

        Raises:
            ValidationError: If the query audio is unusable, is longer than the
                limit without a segment being chosen, or the filter cutoffs are
                invalid.
        """
        filters = filters or SearchFilters()
        k = k if k is not None else self.settings.default_top_k
        metric = metric or self.settings.default_metric
        fusion = fusion or self.settings.default_fusion
        band = filter_band if filter_band is not None else FilteredBand()

        # Resolved from the header alone, so an over-long file is rejected before
        # anything is decoded or any model is loaded.
        properties = audio_io.read_properties(audio_path)
        segment = audio_io.resolve_query_segment(
            properties.duration_s,
            start_time,
            end_time,
            self.settings.max_query_duration_s,
        )

        self.ensure_models_registered()

        candidates, distances = select_candidates(self._conn, filters)
        if candidates is not None and not candidates:
            return []

        item_speeds = self._item_speeds(candidates)
        speeds = (
            [float(query_speed)]
            if query_speed is not None
            else sorted(set(item_speeds.values())) or [1.0]
        )

        def load_query() -> tuple[np.ndarray, int]:
            samples, native_rate = audio_io.load_mono(
                audio_path, start_time=segment[0], end_time=segment[1]
            )
            if band.is_unset:
                return samples, native_rate
            return (
                audio_io.apply_filter(
                    samples,
                    native_rate,
                    low_hz=band.low,
                    high_hz=band.high,
                    order=filter_order,
                ),
                native_rate,
            )

        # Cached per (audio, model, speed, filter), so re-running the same query
        # with a different metadata filter re-uses the embeddings instead of
        # re-decoding, re-filtering and re-running both models. The loader is
        # only called on a miss.
        filter_key = "" if band.is_unset else f"{band.low},{band.high},{filter_order}"
        queries = self.query_cache.embed(
            AudioKey.for_path(audio_path, segment=segment, filter_key=filter_key),
            speeds,
            loader=load_query,
        )
        resolved_speeds = _resolve_query_speeds(item_speeds, speeds)
        per_model = self._score_all_models(
            queries, resolved_speeds, candidates, metric
        )

        hits: list[SearchHit] = []
        for entry in top_k(fuse(per_model, fusion), k, min_score):
            item = self.repo.get_item(entry.item_id)
            hits.append(
                SearchHit(
                    item=item,
                    score=entry.score,
                    per_model=entry.per_model,
                    matches=entry.matches,
                    distance_km=distances.get(entry.item_id),
                    query_speed=resolved_speeds.get(entry.item_id, item.speed),
                )
            )
        return hits

    def _item_speeds(self, candidates: list[int] | None) -> dict[int, float]:
        """Speed each candidate item was indexed at."""
        if candidates is None:
            rows = self._conn.execute("SELECT item_id, speed FROM items").fetchall()
        else:
            rows = []
            for start in range(0, len(candidates), 500):
                chunk = candidates[start : start + 500]
                placeholders = ", ".join("?" * len(chunk))
                rows.extend(
                    self._conn.execute(
                        f"SELECT item_id, speed FROM items "
                        f"WHERE item_id IN ({placeholders})",
                        chunk,
                    ).fetchall()
                )
        return {int(row["item_id"]): float(row["speed"]) for row in rows}

    def _score_all_models(
        self,
        queries: dict[tuple[str, float], WindowEmbeddings],
        resolved_speeds: dict[int, float],
        candidates: list[int] | None,
        metric: str,
    ) -> dict[str, dict[int, WindowMatch]]:
        """Score every model, matching each item against its resolved query speed.

        Args:
            queries: Query embeddings keyed by ``(model_id, speed)``.
            resolved_speeds: The query speed to score each item against, from
                :func:`_resolve_query_speeds`. Every candidate item has an entry,
                so no item is silently left unscored.
            candidates: Item IDs to restrict to, or ``None`` for all.
            metric: Similarity metric.

        Returns:
            Per-model mapping of item ID to its best window match.
        """
        per_model: dict[str, dict[int, WindowMatch]] = {}

        for model_id in self.settings.active_models:
            best: dict[int, WindowMatch] = {}
            for keys, matrix in self.repo.iter_embedding_batches(
                model_id,
                item_ids=candidates,
                batch_size=self.settings.search_batch_size,
            ):
                # A batch may hold items indexed at different speeds, and each
                # must be scored against the query embedded at its own speed.
                # Grouping by speed keeps this to one matrix multiply per speed
                # rather than one per row.
                by_speed: dict[float, list[int]] = {}
                model_speeds = [s for m, s in queries if m == model_id]
                for position, key in enumerate(keys):
                    # resolved_speeds covers every candidate item; the fallback
                    # only guards against an item appearing in embeddings without
                    # a matching items row, which foreign keys should prevent.
                    speed = resolved_speeds.get(key[0])
                    if speed is None and model_speeds:
                        speed = _nearest_speed(1.0, model_speeds)
                    if speed is None:
                        continue
                    by_speed.setdefault(speed, []).append(position)

                for speed, positions in by_speed.items():
                    query = queries.get((model_id, speed))
                    if query is None:
                        continue
                    subset = matrix[positions]
                    subset_keys = [keys[position] for position in positions]
                    best = reduce_to_items(
                        score_matrix(query.vectors, subset, metric),
                        subset_keys,
                        into=best,
                    )
            if best:
                per_model[model_id] = best
        return per_model


def _nearest_speed(target: float, available: list[float]) -> float:
    """Pick the available speed closest to ``target``, in log space.

    Log space because speed is multiplicative: relative to 0.25, a speed of 0.5
    is as far away as 0.125, whereas a linear metric would call 0.5 much closer.

    Args:
        target: Desired speed.
        available: Speeds actually embedded. Must not be empty.

    Returns:
        The closest available speed.
    """
    return min(available, key=lambda s: abs(math.log(s) - math.log(target)))


def _resolve_query_speeds(
    item_speeds: dict[int, float], available: list[float]
) -> dict[int, float]:
    """Choose which query embedding each item is scored against.

    Normally an item's own speed was embedded and is used directly. When it was
    not — the only case being an explicit ``query_speed`` that pins the sweep to
    one value — the nearest available speed is substituted rather than the item
    being skipped.

    Skipping was the original behaviour and it was wrong: pinning a speed no item
    shared returned *zero* results, which reads as "nothing in the library
    matches" rather than "your speed did not line up". Substituting keeps the
    search total, and the speed actually used is reported on every hit.

    Args:
        item_speeds: Each candidate item's indexed speed.
        available: Speeds the query was embedded at. Must not be empty.

    Returns:
        Mapping of item ID to the query speed to score it against.
    """
    if not available:
        return {}
    options = sorted(set(available))
    exact = set(options)
    return {
        item_id: speed if speed in exact else _nearest_speed(speed, options)
        for item_id, speed in item_speeds.items()
    }
