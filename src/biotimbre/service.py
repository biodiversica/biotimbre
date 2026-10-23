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
from typing import Mapping, Sequence

import numpy as np

from biotimbre.config import DEFAULT_DB_NAME, Settings, load_settings
from biotimbre.db.connection import (
    LibraryIdentity,
    get_library_identity,
    initialize,
    open_read_only,
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
from biotimbre.io.bundle import (
    BUNDLE_SUFFIX,
    Bundle,
    ExportEntry,
    ModelFingerprint,
    SourceLibrary,
    bundle_from_entries,
    read_bundle,
    write_bundle,
)
from biotimbre.io.table import items_csv, results_csv
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
from biotimbre.sharing import (
    BundlePreview,
    EntryPlan,
    ImportOutcome,
    ImportPolicy,
    ImportReport,
    local_fingerprints,
    plan_import,
)


#: Name of the library-wide CSV export, from the interface and the CLI alike.
ITEMS_CSV_FILENAME = "biotimbre_items.csv"

#: Name of a search's CSV export.
RESULTS_CSV_FILENAME = "biotimbre_search_results.csv"


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


def npy_archive_filename(
    item_id: int, model_id: str, *, with_audio: bool = False
) -> str:
    """Conventional name for a multi-window embedding export.

    Args:
        item_id: The item.
        model_id: The model.
        with_audio: Whether the archive also holds each window's audio clip.

    Returns:
        A bare filename, no directory.
    """
    suffix = "_with_audio" if with_audio else ""
    return f"item_{item_id}_{model_id}{suffix}.zip"


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
    return f"{_window_stem(item_id, model_id, index, start, end)}.npy"


def window_wav_filename(
    item_id: int, model_id: str, index: int, start: float, end: float
) -> str:
    """Name of one window's audio clip inside an export archive.

    The same stem as the window's ``.npy``, so a vector and the audio it was
    computed from sort next to each other and pair up by name.
    """
    return f"{_window_stem(item_id, model_id, index, start, end)}.wav"


def _window_stem(
    item_id: int, model_id: str, index: int, start: float, end: float
) -> str:
    return f"item_{item_id}_{model_id}_w{index:02d}_{start:.2f}-{end:.2f}s"


def _wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    """Encode float32 mono samples as an IEEE-float WAV, unchanged."""
    from scipy.io import wavfile

    buffer = io.BytesIO()
    # scipy rather than soundfile, as for the browser's normalised preview:
    # soundfile writes to a buffer through cffi callbacks, which fail where
    # libffi does not match what cffi was built for.
    wavfile.write(buffer, rate, np.asarray(samples, dtype=np.float32))
    return buffer.getvalue()


def bundle_filename(item_ids: Sequence[int]) -> str:
    """Conventional name for a shareable bundle.

    Named for what is inside rather than for the library it left, since the
    recipient is who reads it and a sender's library name means nothing to
    them.

    Args:
        item_ids: The items going into the bundle.

    Returns:
        A bare filename, no directory.
    """
    if len(item_ids) == 1:
        return f"biotimbre_item_{item_ids[0]}{BUNDLE_SUFFIX}"
    return f"biotimbre_{len(item_ids)}_items{BUNDLE_SUFFIX}"


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


@dataclass(frozen=True)
class ExternalLibrary:
    """Another library's items, read for importing into this one.

    See :meth:`LibraryService.open_external_library`.

    Attributes:
        data_dir: Where that library lives.
        bundle: Every item that has its audio, as an unwritten bundle whose
            entries point at that library's own files.
        missing_audio: Item ids (in that library) left out because their
            stored audio is missing from disk.
    """

    data_dir: Path
    bundle: Bundle
    missing_audio: tuple[int, ...] = ()


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
        read_only: Promise that this service will not change the library. It
            makes the read path avoid the one write it would otherwise perform
            (see :meth:`ensure_models_registered`) and leaves the directories
            alone. It is a statement about the ``connection`` given, not an
            enforcement: what actually refuses a write is opening that
            connection with :func:`~biotimbre.db.connection.open_read_only`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        connection=None,
        pool: ExtractorPool | None = None,
        read_only: bool = False,
    ):
        self.settings = settings if settings is not None else load_settings()
        self.read_only = read_only
        if not read_only:
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

    def ensure_models_registered(
        self, model_ids: Sequence[str] | None = None
    ) -> dict[str, int]:
        """Register models, reading their dimensions from the graphs.

        Downloads model files on first call. Idempotent.

        A read-only service reads the dimensions and records nothing. The
        registration says which models a library was *built* with, and a
        service that cannot add an item has nothing to add to that; :meth:`search`
        calls this on every query, so without the exemption a search-only run
        would fail on its first search rather than on an ingest it never does.

        Args:
            model_ids: Restrict to these models. ``None`` registers every
                active model. Naming a subset avoids building a session for a
                model this operation has no use for — an import that only needs
                one model's dimension, say.

        Returns:
            Mapping of model ID to embedding dimension.
        """
        dims: dict[str, int] = {}
        for model_id in (
            self.settings.active_models if model_ids is None else model_ids
        ):
            extractor = self.pool.get(model_id)
            dim = extractor.embedding_dim()
            if not self.read_only:
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
        validated = draft.validated()

        joined = None
        if validated.biotimbre_id is None:
            validated, joined, species_warnings = self._join_species_biotimbre(
                validated
            )
        else:
            species_warnings = ()

        with self._mono_source(audio_path) as (source, properties, level_warnings):
            result = self._ingest(
                source, validated, properties, compute_embeddings=compute_embeddings
            )

        return replace(
            result,
            joined_species=joined,
            warnings=level_warnings + result.warnings + species_warnings,
        )

    @contextmanager
    def _mono_source(self, audio_path: str | Path):
        """Yield the exact audio to store, folding a multi-channel file down.

        The library stores mono: it is what every model is fed and what a
        spectrogram shows, so a multi-channel submission is converted here
        rather than kept in a form nothing downstream reads. The digest then
        describes the stored file, which is what ``audio_sha256`` means — and
        two copies of the same stereo recording still recognise each other.

        A floating-point file that goes beyond full scale is converted too,
        normalised to peak at -1 dBFS: stored as it came, it would clip on
        playback and skew Perch, whose embeddings follow the input level. That
        is reported, not done silently.

        Shared by every path that stores audio, so a sample that arrives in a
        bundle is normalised exactly as one that arrives from a form.

        Args:
            audio_path: The submitted file.

        Yields:
            The path to store, its :class:`~biotimbre.embeddings.audio.AudioProperties`,
            and any warning about its level. The path is the original when it is
            already mono and within full scale, and a temporary converted file
            otherwise — removed when the block ends.

        Raises:
            ValidationError: If the file is missing or unreadable as audio.
        """
        source = Path(audio_path)
        if not source.is_file():
            raise ValidationError(f"No such audio file: {source}", "audio")

        properties = audio_io.read_properties(source)
        peak = audio_io.over_range_peak(source)
        warnings = (
            () if peak is None else (audio_io.over_range_message(source.name, peak),)
        )
        if properties.channels == 1 and peak is None:
            yield source, properties, warnings
            return

        with tempfile.TemporaryDirectory(prefix="biotimbre-mono-") as staging:
            converted = Path(staging) / f"{source.stem}.wav"
            yield converted, audio_io.write_processed(source, converted), warnings

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
        embedding_rows: Mapping[str, list[tuple[float, float, np.ndarray]]] | None = None,
        biotimbre_label: str | None = None,
    ) -> IngestResult:
        """Store one prepared file and its row. See :meth:`add_item`.

        Args:
            source: The exact audio to store, already mono.
            validated: Validated metadata.
            properties: The source's properties.
            compute_embeddings: Whether to embed what is not supplied.
            embedding_rows: Vectors already computed elsewhere, per model —
                an import reusing a bundle's. Their models must be registered,
                and the caller is responsible for having established that they
                are comparable (see :mod:`biotimbre.sharing`). Models named
                here are not recomputed.
            biotimbre_label: Name to give the biotimbre if this item creates
                it. Applied inside the same transaction, so a failed ingest
                leaves no named-but-empty biotimbre behind, and ignored when
                the biotimbre already exists — a local name is not overwritten
                by a sender's.

        Returns:
            The :class:`IngestResult`.
        """
        digest = audio_io.file_sha256(source)
        duplicates = tuple(item.item_id for item in self.repo.find_by_sha256(digest))
        warnings = self._nyquist_warnings(validated)

        rows: dict[str, list[tuple[float, float, np.ndarray]]] = dict(
            embedding_rows or {}
        )
        missing = [
            model_id
            for model_id in self.settings.active_models
            if model_id not in rows
        ]
        if compute_embeddings and missing:
            self.ensure_models_registered(missing)
            # Computed before the transaction opens: inference takes seconds and
            # holding a write transaction open across it would block readers.
            embeddings = self.pool.embed_file(
                source, speed=validated.speed, models=missing
            )
            rows.update(
                {model_id: result.as_rows() for model_id, result in embeddings.items()}
            )

        stored_path: Path | None = None
        try:
            with self._transaction():
                if biotimbre_label is not None and validated.biotimbre_id is not None:
                    self.repo.ensure_biotimbre(
                        validated.biotimbre_id, biotimbre_label
                    )
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
                        item.item_id, model_id, window_rows
                    )
                    for model_id, window_rows in sorted(rows.items())
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
        self,
        audio_path: str | Path,
        sidecar_path: str | Path | None = None,
        *,
        compute_embeddings: bool = True,
    ) -> IngestResult:
        """Add a sample using metadata from its sidecar file.

        Args:
            audio_path: Path to the audio file.
            sidecar_path: Path to the sidecar. ``None`` looks for one beside the
                audio file.
            compute_embeddings: Set false to defer embedding to a later
                backfill, as :meth:`add_item` does.

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
        return self.add_item(
            audio_path,
            draft_from_sidecar(resolved),
            compute_embeddings=compute_embeddings,
        )

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

    def embeddings_export(
        self, item_id: int, model_id: str, *, with_audio: bool = False
    ) -> tuple[bytes, str]:
        """One item's embeddings for one model, as a file per window.

        A window is the unit that means something — one vector, over one span
        of seconds — so no file here holds more than one of them. A model with
        a single window exports as a bare ``.npy``; several windows export as a
        zip of one ``.npy`` each, named with the window's index and its span.

        With ``with_audio``, the export is always a zip, and each ``.npy`` has
        a ``.wav`` beside it of the same name: the samples the model was given
        for that window — speed-shifted, resampled to the model's rate and
        padded to its window length, exactly as embedding prepared them. See
        :meth:`window_clips`.

        The same bytes :meth:`export_npy` writes, for a caller that has nowhere
        to write them — a browser download, say.

        Args:
            item_id: The item.
            model_id: The model.
            with_audio: Whether to include each window's audio clip.

        Returns:
            The file's bytes and the name to save them under. Each ``.npy``
            holds one 1-D vector of the model's width.

        Raises:
            NotFoundError: If the item has no embeddings for this model, or,
                with audio, its audio file is missing.
            ValidationError: With audio, if the stored windows no longer match
                what the model would cut from the audio today.
        """
        windows = self.repo.get_embeddings(item_id, model_id)
        if not windows:
            raise NotFoundError(
                f"Item {item_id} has no {model_id} embeddings to export."
            )
        clips = self.window_clips(item_id, model_id, windows) if with_audio else None

        def npy_bytes(vector: np.ndarray) -> bytes:
            buffer = io.BytesIO()
            np.save(buffer, vector)
            return buffer.getvalue()

        if len(windows) == 1 and clips is None:
            return npy_bytes(windows[0][3]), npy_filename(item_id, model_id)

        def write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
            # A fixed timestamp, so exporting the same rows twice gives the
            # same bytes: the archive can be checksummed as provenance, and a
            # download matches a command-line export exactly. 1980-01-01 is
            # the earliest a zip entry can record.
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for position, (index, start, end, vector) in enumerate(windows):
                write(
                    archive,
                    window_npy_filename(item_id, model_id, index, start, end),
                    npy_bytes(vector),
                )
                if clips is not None:
                    write(
                        archive,
                        window_wav_filename(item_id, model_id, index, start, end),
                        clips[position],
                    )
        name = npy_archive_filename(item_id, model_id, with_audio=clips is not None)
        return buffer.getvalue(), name

    def window_clips(
        self,
        item_id: int,
        model_id: str,
        windows: Sequence[tuple[int, float, float, np.ndarray]] | None = None,
    ) -> list[bytes]:
        """Each stored window's model input, as WAV, in window order.

        Cut again rather than stored: the audio and the model's preprocessing
        are both kept, so the windows can be remade exactly — the same speed
        shift, the same single resample to the model's rate, the same framing
        and zero-padding as :meth:`compute_embeddings_for`. Each clip is mono
        float32 at the model's sample rate, one window long.

        Args:
            item_id: The item.
            model_id: The model.
            windows: The item's stored windows for this model, if the caller
                already has them.

        Returns:
            One WAV file's bytes per stored window.

        Raises:
            NotFoundError: If the audio file is missing, or there are no
                stored windows.
            ValidationError: If the model is unknown, or the windows cut today
                differ from the stored ones — the vectors were computed under
                other settings, and a clip would not be what they describe.
        """
        if windows is None:
            windows = self.repo.get_embeddings(item_id, model_id)
        if not windows:
            raise NotFoundError(
                f"Item {item_id} has no {model_id} embeddings to export."
            )
        try:
            spec = get_spec(model_id)
        except KeyError as exc:
            raise ValidationError(str(exc), "model_id") from None

        item = self.repo.get_item(item_id)
        source = self.settings.absolute_audio_path(item.audio_path)
        if not source.is_file():
            raise NotFoundError(
                f"Audio for item {item_id} is missing from {source}."
            )
        samples, native_rate = audio_io.load_mono(source)
        clips, spans = audio_io.prepare_windows(
            samples,
            native_rate,
            model_rate=spec.sample_rate,
            window_samples=spec.window_samples,
            stride_samples=spec.stride_samples,
            speed=item.speed,
        )

        stored = [(start, end) for _, start, end, _ in windows]
        if len(spans) != len(stored) or any(
            abs(a - c) > 1e-3 or abs(b - d) > 1e-3
            for (a, b), (c, d) in zip(spans, stored)
        ):
            raise ValidationError(
                f"Item {item_id}'s {model_id} embeddings were computed with "
                "other settings than the model's current ones, so its clips "
                "cannot be remade. Re-embed the item first.",
                "model_id",
            )
        return [_wav_bytes(clip, spec.sample_rate) for clip in clips]

    def export_npy(
        self,
        item_id: int,
        model_id: str,
        destination: str | Path,
        *,
        with_audio: bool = False,
    ) -> Path:
        """Write one item's embeddings for one model to disk.

        Provided for interop and hand-off; the database remains the source of
        truth (PLAN.md D1). One window writes a ``.npy``; several write a zip
        of one ``.npy`` per window — see :meth:`embeddings_export`.

        Args:
            item_id: The item.
            model_id: The model.
            destination: Output path, or a directory to name the file inside.
            with_audio: Whether to include each window's audio clip, as the
                model heard it. Always writes a zip.

        Returns:
            The written path.

        Raises:
            NotFoundError: If the item has no embeddings for this model.
        """
        data, name = self.embeddings_export(
            item_id, model_id, with_audio=with_audio
        )

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

    # ── sharing ───────────────────────────────────────────────────────────────

    def export_bundle(
        self,
        item_ids: Sequence[int],
        destination: str | Path,
        *,
        include_embeddings: bool = True,
    ) -> Path:
        """Write items to a shareable bundle file.

        Args:
            item_ids: Items to include, in the order they should appear.
            destination: Path to write. Its parent is created if needed; a
                path without the conventional suffix gets one.
            include_embeddings: Carry each item's vectors. False halves the
                file size and costs the recipient an inference pass.

        Returns:
            The path written.

        Raises:
            NotFoundError: If an item does not exist, or its stored audio is
                missing from disk.
        """
        path = Path(destination)
        if path.suffix != BUNDLE_SUFFIX:
            path = path.with_suffix(BUNDLE_SUFFIX)
        entries, models = self._export_entries(item_ids, include_embeddings)
        write_bundle(
            path, entries, models=models, source_library=self._source_library()
        )
        return path

    def export_bundle_bytes(
        self, item_ids: Sequence[int], *, include_embeddings: bool = True
    ) -> tuple[bytes, str]:
        """The same bundle, for a caller with nowhere to write it.

        The shape :meth:`embeddings_export` uses, so a browser download and a
        command-line export produce the same file under the same name.

        Args:
            item_ids: Items to include.
            include_embeddings: Carry each item's vectors.

        Returns:
            The file's bytes and the name to save them under.

        Raises:
            NotFoundError: If an item does not exist, or its audio is missing.
        """
        entries, models = self._export_entries(item_ids, include_embeddings)
        buffer = io.BytesIO()
        write_bundle(
            buffer, entries, models=models, source_library=self._source_library()
        )
        return buffer.getvalue(), bundle_filename(item_ids)

    def _source_library(self) -> SourceLibrary:
        """This library's identity, as a bundle records it."""
        identity = self.identity()
        return SourceLibrary(uuid=identity.uuid, name=identity.name)

    def _export_entries(
        self, item_ids: Sequence[int], include_embeddings: bool
    ) -> tuple[list[ExportEntry], dict[str, ModelFingerprint]]:
        """Gather what a bundle needs about each item.

        Args:
            item_ids: Items to gather.
            include_embeddings: Whether to read their vectors.

        Returns:
            The entries, and the fingerprints of every model registered here —
            the writer keeps the ones the entries actually use.

        Raises:
            NotFoundError: If an item does not exist, or its audio is missing.
        """
        if not item_ids:
            raise ValidationError("Nothing to export.", "item_ids")

        entries: list[ExportEntry] = []
        for item_id in item_ids:
            item = self.repo.get_item(item_id)
            audio = self.settings.absolute_audio_path(item.audio_path)
            if not audio.is_file():
                raise NotFoundError(
                    f"Item {item_id}'s audio is missing from disk ({audio}); "
                    f"there is nothing to share."
                )

            embeddings: dict[str, list[tuple[float, float, np.ndarray]]] = {}
            if include_embeddings:
                for model_id in self.repo.embedding_counts(item_id):
                    embeddings[model_id] = [
                        (start, end, vector)
                        for _, start, end, vector in self.repo.get_embeddings(
                            item_id, model_id
                        )
                    ]

            entries.append(
                ExportEntry(
                    draft=item.to_draft(),
                    audio=audio,
                    audio_sha256=item.audio_sha256,
                    duration_s=item.duration_s,
                    sample_rate=item.sample_rate,
                    channels=item.channels,
                    biotimbre_label=self.get_biotimbre_label(item.biotimbre_id),
                    embeddings=embeddings,
                    source_item_id=item.item_id,
                )
            )
        return entries, local_fingerprints(self.repo)

    def _register_for_import(self, bundle: Bundle) -> None:
        """Register the models an incoming bundle's vectors would need.

        A library that has never embedded anything has no ``models`` rows, and
        without one there is no local dimension to check a sender's vectors
        against — so a fresh library would throw away every vector it was sent
        and re-run inference it did not need to. Registering first is the same
        work that library's next ingest would do anyway.

        Only models this library actually uses are considered: vectors for a
        model absent from ``active_models`` are refused however good they are,
        because nothing here would ever search them.

        Failure is not an error. A machine without the model files — the case
        ``compute_embeddings=False`` exists for — simply does not reuse those
        vectors, and planning says so in the rejection reason.

        Args:
            bundle: The bundle about to be planned.
        """
        wanted = [
            model_id
            for model_id in bundle.models
            if model_id in self.settings.active_models
            and self.repo.get_model(model_id) is None
        ]
        if not wanted:
            return
        try:
            self.ensure_models_registered(wanted)
        except Exception:
            # Unobtainable model file, no network, an id this build does not
            # know: every one of them means the same thing here, and none of
            # them should stop an import that is otherwise fine.
            pass

    def inspect_bundle(
        self,
        path: str | Path,
        *,
        policy: ImportPolicy | None = None,
        into: str | Path | None = None,
    ) -> BundlePreview:
        """Read a bundle and say what importing it would do, without doing it.

        The same :func:`~biotimbre.sharing.plan_import` the import itself runs,
        so what is shown is what will happen — decided again at import time
        against the library as it is then, rather than replayed from here.

        Registers any model the bundle's vectors need and this library uses
        (see :meth:`_register_for_import`), which on a library that has never
        embedded anything may load a model file.

        Args:
            path: The bundle file.
            policy: The choices to plan under; defaults apply if omitted.
            into: Directory to extract the bundle's audio into, for a caller
                that wants to play or draw it. ``None`` uses a temporary
                directory and removes it before returning — the preview's
                metadata is then all that remains valid, and its entries'
                ``audio`` paths point at files that are gone.

        Returns:
            A :class:`~biotimbre.sharing.BundlePreview`.

        Raises:
            ValidationError: If the file is not a readable bundle.
        """
        if into is not None:
            bundle = read_bundle(path, into=into)
            self._register_for_import(bundle)
            return plan_import(self.repo, bundle, policy)

        with tempfile.TemporaryDirectory(prefix="biotimbre-preview-") as staging:
            bundle = read_bundle(path, into=staging)
            self._register_for_import(bundle)
            return plan_import(self.repo, bundle, policy)

    def import_bundle(
        self, path: str | Path, *, policy: ImportPolicy | None = None
    ) -> ImportReport:
        """Add a bundle's items to this library.

        Each entry is its own transaction, so an entry that cannot be stored
        does not undo the ones already added — the bargain
        :meth:`import_directory` makes.

        That resilience starts *after* the file is read. Reading is
        all-or-nothing: a member whose digest does not match, a path that
        escapes the archive, a field the library would refuse, and the whole
        bundle is rejected before anything is stored. A bundle is one transfer,
        and a damaged one is not worth salvaging piecemeal — ask the sender
        again rather than wonder which half arrived.

        What happens to an entry, and why, is decided by
        :func:`~biotimbre.sharing.plan_import`; see :mod:`biotimbre.sharing`
        for the positions the library takes on each kind of disagreement.

        Models the bundle's vectors need are registered first (see
        :meth:`_register_for_import`), so a library receiving its first items
        keeps the vectors it was sent instead of recomputing them.

        Args:
            path: The bundle file.
            policy: The choices to import under; defaults apply if omitted.

        Returns:
            An :class:`~biotimbre.sharing.ImportReport`, one outcome per entry.

        Raises:
            ValidationError: If the file is not a readable bundle. A failure
                in a single entry is reported rather than raised.
        """
        with tempfile.TemporaryDirectory(prefix="biotimbre-import-") as staging:
            return self._import_opened(read_bundle(path, into=staging), policy)

    def _import_opened(
        self, bundle: Bundle, policy: ImportPolicy | None
    ) -> ImportReport:
        """Plan an opened bundle against this library as it is now, and carry it out."""
        resolved = policy or ImportPolicy()
        self._register_for_import(bundle)
        preview = plan_import(self.repo, bundle, resolved)
        outcomes = tuple(self._import_entry(plan, resolved) for plan in preview.plans)
        return ImportReport(bundle=bundle, policy=resolved, outcomes=outcomes)

    def open_external_library(
        self, data_dir: str | Path, *, include_embeddings: bool = True
    ) -> ExternalLibrary:
        """Read every item of another library on this machine, for importing here.

        The other library is opened read-only and left exactly as it was. Its
        items come back as a bundle that was never written — the one
        :meth:`export_bundle` would produce for all of them — whose entries
        point at that library's own audio files, so nothing is copied until an
        import stores it. What happens to each entry is then decided exactly as
        for a bundle someone sent (see :mod:`biotimbre.sharing`): recordings
        already here are skipped, species join their local biotimbre, vectors
        from a model registered differently are recomputed.

        An item whose stored audio is missing from disk is left out and
        reported rather than failing the whole import: there is nothing to
        import for it, and the rest of the library is no less worth having.

        Args:
            data_dir: The other library's directory, the one holding its
                ``library.db``.
            include_embeddings: Carry its vectors across. False recomputes
                everything here.

        Returns:
            An :class:`ExternalLibrary`.

        Raises:
            ValidationError: If the directory is this library, holds no
                library, or has no item that can be imported.
            BiotimbreError: If its database cannot be opened, e.g. because it
                needs a schema upgrade first.
        """
        source_dir = Path(data_dir).expanduser()
        db_path = source_dir / DEFAULT_DB_NAME
        if not db_path.is_file():
            raise ValidationError(
                f"No biotimbre library in {source_dir}: there is no "
                f"{DEFAULT_DB_NAME} there.",
                "library",
            )
        if db_path.resolve() == self.settings.db_path.resolve():
            raise ValidationError(
                "That is the library already open; there is nothing to import from it.",
                "library",
            )

        settings = replace(
            self.settings,
            data_dir=source_dir,
            db_path=db_path,
            audio_dir=source_dir / "audio",
        )
        connection = open_read_only(db_path)
        try:
            source = LibraryService(settings, connection=connection, read_only=True)
            items = sorted(source.repo.list_items(), key=lambda item: item.item_id)
            present: list[int] = []
            missing: list[int] = []
            for item in items:
                audio = settings.absolute_audio_path(item.audio_path)
                (present if audio.is_file() else missing).append(item.item_id)
            if not present:
                raise ValidationError(
                    f"The library in {source_dir} has no items to import"
                    + (
                        f" ({len(missing)} item(s) have no audio on disk)."
                        if missing
                        else "."
                    ),
                    "library",
                )
            entries, models = source._export_entries(present, include_embeddings)
            bundle = bundle_from_entries(
                entries, models=models, source_library=source._source_library()
            )
        finally:
            connection.close()
        return ExternalLibrary(
            data_dir=source_dir, bundle=bundle, missing_audio=tuple(missing)
        )

    def inspect_library(
        self,
        source: str | Path | ExternalLibrary,
        *,
        policy: ImportPolicy | None = None,
    ) -> BundlePreview:
        """Say what importing another library's items would do, without doing it.

        The library-sized counterpart of :meth:`inspect_bundle`.

        Args:
            source: The other library's directory, or one already opened with
                :meth:`open_external_library` — which a caller re-planning
                under changing choices should pass, so the other library is
                read once.
            policy: The choices to plan under; defaults apply if omitted.

        Returns:
            A :class:`~biotimbre.sharing.BundlePreview`.

        Raises:
            ValidationError: As :meth:`open_external_library`.
        """
        external = self._external(source)
        self._register_for_import(external.bundle)
        return plan_import(self.repo, external.bundle, policy)

    def import_library(
        self,
        source: str | Path | ExternalLibrary,
        *,
        policy: ImportPolicy | None = None,
    ) -> ImportReport:
        """Add every item of another library to this one.

        The library-sized counterpart of :meth:`import_bundle`, with the same
        bargain: each item is its own transaction, so one that cannot be stored
        does not undo the others. The other library is not changed.

        Args:
            source: The other library's directory, or one already opened with
                :meth:`open_external_library`.
            policy: The choices to import under; defaults apply if omitted.

        Returns:
            An :class:`~biotimbre.sharing.ImportReport`, one outcome per item
            that had audio. Items without audio are in
            :attr:`ExternalLibrary.missing_audio`, not here.

        Raises:
            ValidationError: As :meth:`open_external_library`.
        """
        return self._import_opened(self._external(source).bundle, policy)

    def _external(self, source: str | Path | ExternalLibrary) -> ExternalLibrary:
        if isinstance(source, ExternalLibrary):
            return source
        return self.open_external_library(source)

    def _import_entry(self, plan: EntryPlan, policy: ImportPolicy) -> ImportOutcome:
        """Carry out one entry's plan. See :meth:`import_bundle`."""
        if plan.action == "skip":
            return ImportOutcome(
                plan=plan, item=None, action="skip", reason=plan.reason
            )

        try:
            with self._mono_source(plan.entry.audio) as (
                source,
                properties,
                level_warnings,
            ):
                result = self._ingest(
                    source,
                    plan.draft,
                    properties,
                    compute_embeddings=policy.compute_embeddings,
                    embedding_rows=plan.reusable_rows,
                    biotimbre_label=plan.entry.biotimbre_label,
                )
        except BiotimbreError as exc:
            return ImportOutcome(
                plan=plan, item=None, action="failed", reason=str(exc)
            )

        return ImportOutcome(
            plan=plan,
            item=result.item,
            window_counts=result.window_counts,
            action="add",
            warnings=plan.warnings + level_warnings + result.warnings,
        )

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

    def items_table(self) -> tuple[bytes, str]:
        """Every item as one CSV row, every stored value as a column.

        Rows run in item ID order — the order items were added — so two
        exports of a growing library line up row for row. Beside the item's
        own values come its biotimbre's label and, per registered model, how
        many windows it was embedded as. See :func:`~biotimbre.io.table.items_csv`.

        Returns:
            The CSV as UTF-8 bytes, and the name to save it under.
        """
        items = sorted(self.repo.list_items(), key=lambda item: item.item_id)
        text = items_csv(items, **self._table_context())
        return text.encode("utf-8"), ITEMS_CSV_FILENAME

    def results_table(self, hits: Sequence[SearchHit]) -> tuple[bytes, str]:
        """Search results as CSV: the items' rows, with the match in front.

        Exactly the hits given, in their order and as they were found — what
        the search showed, not what it might show if run again. See
        :func:`~biotimbre.io.table.results_csv`.

        Args:
            hits: The results, best first.

        Returns:
            The CSV as UTF-8 bytes, and the name to save it under.
        """
        text = results_csv(hits, **self._table_context())
        return text.encode("utf-8"), RESULTS_CSV_FILENAME

    def _table_context(self) -> dict:
        """What a CSV row needs beyond the item: labels and window counts."""
        return {
            "labels": self.repo.biotimbre_labels(),
            "embedding_counts": self.repo.all_embedding_counts(),
            "model_ids": [model["model_id"] for model in self.repo.list_models()],
        }

    def export_items_csv(self, destination: str | Path) -> Path:
        """Write :meth:`items_table` to disk.

        Args:
            destination: Output path, or a directory to name the file inside.

        Returns:
            The written path.
        """
        data, name = self.items_table()
        out = Path(destination)
        if out.is_dir():
            out = out / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return out

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

    def suggest(self, field: str, text: str = "") -> list[str]:
        """Earlier values of a metadata field that match what is being typed.

        So a tag, a device or a recordist entered once can be picked rather
        than typed again — and typed the same way each time, which is what
        lets a filter on it find them all.

        Args:
            field: ``taxa``, ``device``, ``microphone`` or ``recorded_by``.
            text: What has been typed so far.

        Returns:
            Matching values, best first. See
            :meth:`~biotimbre.db.repository.Repository.suggest`.
        """
        return self.repo.suggest(field, text)

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

        A segment that goes beyond full scale is normalised to peak at -1 dBFS
        before embedding. That is not reported here; callers ask
        :func:`~biotimbre.embeddings.audio.over_range_peak` to warn about it.

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
            # Within the segment, so a loud passage elsewhere in a long file
            # does not turn down the part being searched with.
            samples, _ = audio_io.normalise_over_range(samples)
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
