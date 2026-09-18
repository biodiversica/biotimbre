"""The shareable bundle: one item, or many, as a single file.

A library's items are already meant to be portable — ``biotimbre_id`` is a UUID
precisely so that one sound-being means the same thing in every library that
holds it (see :func:`~biotimbre.domain.item.generate_biotimbre_id`). What was
missing is a container to carry one from library to library, and that is this
module.

A bundle is a zip, ``.btz``, laid out as::

    manifest.json               format version, provenance, model fingerprints
    README.txt                  what this file is, for whoever opens the zip
    items/0001/item.json        every field of the item, authoritative
    items/0001/item.txt         the same metadata as a sidecar, for reading
    items/0001/audio.wav        the STORED mono audio, byte-identical
    items/0001/embeddings/perch_v2.npz

Three decisions shape the format:

**The stored audio travels, not the original.** ``audio_sha256`` describes the
file the library keeps, so shipping that exact file lets the receiving library
recognise the sample it already has, and spares it a second down-mix whose
output might not be bit-identical.

**``item_id`` does not travel.** It is a local ``AUTOINCREMENT`` counter and
means nothing anywhere else; it is carried only as ``source_item_id``, for
provenance. ``biotimbre_id`` travels, because that is the whole point.

**Embeddings travel with their model's fingerprint.** Vectors from two runs are
only comparable when the preprocessing recipe behind them is identical, so the
manifest records the full model spec and the importer is expected to compare it
before storing anything. Getting them for free saves the recipient an inference
pass and a 45 MB download; getting them *wrong* would silently corrupt search.

This module knows the format and nothing else: no database, no settings, no
library layout. It reads and writes value objects, so it can be tested without
a library and reused by the CLI, the interface, and anything later.

A bundle arrives from another person, so :func:`read_bundle` treats it as
hostile: it reads only members the manifest names, refuses paths that escape
the archive, caps what it will decompress, loads arrays with pickling off, and
puts every field through :meth:`~biotimbre.domain.item.ItemDraft.validated`
rather than trusting the JSON.
"""

from __future__ import annotations

import hashlib
import io
import json
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

import numpy as np

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import (
    FilteredBand,
    ItemDraft,
    Location,
    SampleWindow,
)
from biotimbre.io.sidecar import format_sidecar

#: Bumped when the layout changes in a way an older reader cannot handle. A
#: reader refuses a bundle from the future rather than guessing at it.
BUNDLE_VERSION = 1

#: Marks the archive as ours before anything else is believed about it.
BUNDLE_FORMAT = "biotimbre-bundle"

#: Conventional extension. The file is a plain zip, and deliberately so: a
#: recipient without the application can still open it and see what they were
#: sent.
BUNDLE_SUFFIX = ".btz"

MANIFEST_NAME = "manifest.json"
README_NAME = "README.txt"
ITEM_JSON_NAME = "item.json"
ITEM_SIDECAR_NAME = "item.txt"
AUDIO_NAME = "audio.wav"
EMBEDDINGS_DIR = "embeddings"

#: Vectors are stored as float32, the width they are computed and searched at.
VECTOR_DTYPE = np.float32

#: Ceilings applied while reading. None of them constrain a real bundle — a
#: thousand items of a minute each sits far below the byte cap — and each turns
#: a hand-crafted archive that would otherwise exhaust memory or disk into a
#: plain error.
MAX_ENTRIES = 10_000
MAX_TOTAL_BYTES = 8 * 1024**3
MAX_JSON_BYTES = 1 * 1024**2
MAX_EMBEDDINGS_BYTES = 256 * 1024**2

#: A member whose declared size dwarfs its compressed size by more than this is
#: refused unread. Audio and float vectors compress by a small factor; a ratio
#: in the thousands is a zip bomb, not a recording.
MAX_COMPRESSION_RATIO = 500

#: Model ids become path segments inside the archive, so a bundle cannot be
#: allowed to invent one like ``../../etc``. Matches the ids the registry uses
#: (``perch_v2``, ``birdnet_v2.4``).
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

#: Entry directories the writer produces: ``items/0001``. Read back with the
#: same pattern, so nothing else is ever opened as an entry.
_ENTRY_PATH = re.compile(r"^items/\d{4,}$")

_README = """\
This is a biotimbre bundle: one or more library items packaged for sharing.

It is a plain zip file. Inside, each item has its metadata in item.json (the
authoritative copy) and again in item.txt (the same fields in the library's
sidecar format, for reading), its audio as a mono WAV, and, when they were
included, its embedding vectors as .npz arrays.

Load it with:  biotimbre import-bundle <this file>
"""


def _utc_now() -> str:
    """Current time as an ISO-8601 UTC string, second resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _fail(message: str) -> ValidationError:
    """A bundle-level validation error, worded for whoever was sent the file."""
    return ValidationError(message, "bundle")


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceLibrary:
    """Where a bundle came from.

    Attributes:
        uuid: The exporting library's stable UUID, or ``None`` if it had none.
        name: Its human-readable name, or ``None`` if it was never named.
    """

    uuid: str | None = None
    name: str | None = None

    def to_json(self) -> dict:
        return {"uuid": self.uuid, "name": self.name}

    @classmethod
    def from_json(cls, data: object) -> "SourceLibrary":
        if not isinstance(data, dict):
            return cls()
        return cls(
            uuid=_optional_text(data.get("uuid"), "source_library.uuid"),
            name=_optional_text(data.get("name"), "source_library.name"),
        )


@dataclass(frozen=True)
class ModelFingerprint:
    """A model's identity, complete enough to prove two runs are comparable.

    ``model_id`` alone is not enough. The id names a preprocessing recipe, and
    a library that re-registered ``perch_v2`` at a different window overlap
    produces vectors that are not interchangeable with the ones this bundle
    carries. Every field that would change the numbers is here, so an importer
    can compare rather than assume.

    Attributes:
        model_id: The id the vectors are filed under.
        source: Provenance string, e.g. ``hf://google/perch``.
        filename: ONNX file within that source.
        revision: Pinned revision, or ``None`` for the default branch.
        sample_rate: Rate the graph is fed at.
        window_duration_s: Length of one inference window.
        window_overlap: Fractional overlap between consecutive windows.
        embedding_dim: Width of one vector.
    """

    model_id: str
    source: str
    filename: str
    revision: str | None
    sample_rate: int
    window_duration_s: float
    window_overlap: float
    embedding_dim: int

    def matches(self, other: "ModelFingerprint") -> bool:
        """Whether vectors from ``other`` may be compared with these.

        Args:
            other: The fingerprint to compare against, typically built from the
                receiving library's ``models`` row.

        Returns:
            True when every field that affects the numbers agrees.
        """
        return self == other

    def differences(self, other: "ModelFingerprint") -> tuple[str, ...]:
        """Field names on which two fingerprints disagree.

        Args:
            other: The fingerprint to compare against.

        Returns:
            The differing field names, so a caller can say *why* embeddings
            were refused instead of only that they were.
        """
        return tuple(
            name
            for name in (
                "model_id",
                "source",
                "filename",
                "revision",
                "sample_rate",
                "window_duration_s",
                "window_overlap",
                "embedding_dim",
            )
            if getattr(self, name) != getattr(other, name)
        )

    def to_json(self) -> dict:
        return {
            "model_id": self.model_id,
            "source": self.source,
            "filename": self.filename,
            "revision": self.revision,
            "sample_rate": self.sample_rate,
            "window_duration_s": self.window_duration_s,
            "window_overlap": self.window_overlap,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_json(cls, data: object) -> "ModelFingerprint":
        if not isinstance(data, dict):
            raise _fail("A model fingerprint must be an object.")
        model_id = _required_text(data.get("model_id"), "model.model_id")
        _check_model_id(model_id)
        return cls(
            model_id=model_id,
            source=_required_text(data.get("source"), "model.source"),
            filename=_required_text(data.get("filename"), "model.filename"),
            revision=_optional_text(data.get("revision"), "model.revision"),
            sample_rate=_required_int(data.get("sample_rate"), "model.sample_rate"),
            window_duration_s=_required_float(
                data.get("window_duration_s"), "model.window_duration_s"
            ),
            window_overlap=_required_float(
                data.get("window_overlap"), "model.window_overlap"
            ),
            embedding_dim=_required_int(
                data.get("embedding_dim"), "model.embedding_dim"
            ),
        )


@dataclass(frozen=True)
class AudioInfo:
    """The stored audio's properties, as the exporting library recorded them.

    Carried so a bundle can be listed — durations, rates — without decoding
    anything, and so the importer can check what it reads against what it was
    promised.

    Attributes:
        sha256: Hex digest of the WAV in this bundle.
        duration_s: Length in seconds.
        sample_rate: Native sample rate in Hz.
        channels: Channel count; one, for anything the library stored.
    """

    sha256: str
    duration_s: float
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class Provenance:
    """Which library an entry came from, and what it was called there.

    Attributes:
        library_uuid: The exporting library's UUID, or ``None``.
        library_name: Its name at export time, or ``None``.
        item_id: The item's id *in that library*. Meaningless as an id here;
            kept so a recipient can say where a sample came from, and so the
            sender can be told which of their items a question is about.
        exported_at: When the bundle was written, ISO-8601 UTC.
    """

    library_uuid: str | None
    library_name: str | None
    item_id: int | None
    exported_at: str


#: One model's vectors: ``(start_time, end_time, vector)`` per window, in window
#: order — the shape
#: :meth:`~biotimbre.db.repository.Repository.set_embeddings` takes, so an
#: importer hands them straight over.
WindowRows = list[tuple[float, float, np.ndarray]]


@dataclass(frozen=True)
class ExportEntry:
    """One item on its way into a bundle.

    Attributes:
        draft: The item's metadata. Its ``biotimbre_id`` must be set: a stored
            item always has one, and a bundle without it could not carry the
            identity that makes sharing worth doing.
        audio: Path to the stored WAV to copy in.
        audio_sha256: Digest of that file, as the library holds it.
        duration_s: Length in seconds.
        sample_rate: Native sample rate in Hz.
        channels: Channel count.
        biotimbre_label: The biotimbre's name, if it has one.
        embeddings: Vectors per model id; empty to ship metadata and audio only.
        source_item_id: The item's id in the exporting library.
    """

    draft: ItemDraft
    audio: Path
    audio_sha256: str
    duration_s: float
    sample_rate: int
    channels: int
    biotimbre_label: str | None = None
    embeddings: Mapping[str, WindowRows] = field(default_factory=dict)
    source_item_id: int | None = None


@dataclass(frozen=True)
class BundleEntry:
    """One item read back out of a bundle.

    Attributes:
        draft: Validated metadata, with ``biotimbre_id`` set.
        audio: Path to the extracted WAV, under the directory
            :func:`read_bundle` was given.
        audio_info: Properties as the sender recorded them. ``sha256`` has been
            verified against the extracted bytes.
        biotimbre_label: The biotimbre's name in the sending library, if any.
        embeddings: Vectors per model id, keyed by a model id present in
            :attr:`Bundle.models`.
        provenance: Where this entry came from.
    """

    draft: ItemDraft
    audio: Path
    audio_info: AudioInfo
    biotimbre_label: str | None
    embeddings: dict[str, WindowRows]
    provenance: Provenance

    @property
    def window_counts(self) -> dict[str, int]:
        """Windows carried per model, for a preview that has not stored yet."""
        return {model: len(rows) for model, rows in self.embeddings.items()}


@dataclass(frozen=True)
class Bundle:
    """An opened bundle.

    Attributes:
        format_version: The version the file declares.
        created_at: When it was written, ISO-8601 UTC.
        source_library: The library it came from.
        models: Fingerprint per model id whose vectors appear in it.
        entries: The items, in the order the manifest lists them.
    """

    format_version: int
    created_at: str
    source_library: SourceLibrary
    models: dict[str, ModelFingerprint]
    entries: tuple[BundleEntry, ...]

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def biotimbre_ids(self) -> tuple[str, ...]:
        """The distinct biotimbres represented, in first-seen order."""
        seen: dict[str, None] = {}
        for entry in self.entries:
            assert entry.draft.biotimbre_id is not None
            seen.setdefault(entry.draft.biotimbre_id, None)
        return tuple(seen)


# --------------------------------------------------------------------------
# JSON for one item
# --------------------------------------------------------------------------


def draft_to_json(draft: ItemDraft) -> dict:
    """Represent a draft as JSON-ready data.

    Field names are exactly the sidecar's
    (:data:`~biotimbre.io.sidecar.SIDECAR_FIELDS`), so the two formats stay one
    vocabulary and a person reading either recognises the other. Compound
    values become objects rather than the sidecar's ``"lat,lon"`` strings:
    JSON can name its parts, and a named part cannot be transposed.

    Args:
        draft: A validated draft.

    Returns:
        A dict of plain JSON types.
    """
    band = draft.filtered_band
    window = draft.sample_window
    return {
        "biotimbre_id": draft.biotimbre_id,
        "taxa": list(draft.taxa),
        "date": draft.date,
        "time": draft.time,
        "location": {
            "latitude": draft.location.latitude,
            "longitude": draft.location.longitude,
        },
        "sample_window": (
            None if window is None else {"start": window.start, "end": window.end}
        ),
        "device": draft.device,
        "microphone": draft.microphone,
        "recorded_by": draft.recorded_by,
        "filtered_band": (
            None if band.is_unset else {"low": band.low, "high": band.high}
        ),
        "filter_type": draft.filter_type,
        "filter_order": draft.filter_order,
        "temperature": draft.temperature,
        "humidity": draft.humidity,
        "speed": draft.speed,
        "link_to_full_recording": draft.link_to_full_recording,
        "notes": draft.notes,
    }


def draft_from_json(data: object) -> ItemDraft:
    """Build a validated draft from bundle JSON.

    Every value goes through :meth:`~biotimbre.domain.item.ItemDraft.validated`,
    so a bundle cannot introduce a record the library would have refused from a
    form. The value objects parse their own shapes, which is why an object, a
    pair and a string are all accepted for a location: the sender's writer may
    not have been this one.

    Args:
        data: The decoded ``item.json`` body.

    Returns:
        A validated :class:`~biotimbre.domain.item.ItemDraft`.

    Raises:
        ValidationError: If a mandatory field is missing or a value is invalid.
    """
    if not isinstance(data, dict):
        raise _fail("An item must be a JSON object.")

    missing = [name for name in ("date", "time", "location") if not data.get(name)]
    if missing:
        raise _fail(f"Item is missing mandatory field(s): {', '.join(missing)}.")

    return ItemDraft(
        date=_required_text(data.get("date"), "date"),
        time=_required_text(data.get("time"), "time"),
        location=_location_from_json(data.get("location")),
        sample_window=_window_from_json(data.get("sample_window")),
        biotimbre_id=data.get("biotimbre_id"),
        taxa=data.get("taxa") or (),
        device=data.get("device"),
        microphone=data.get("microphone"),
        recorded_by=data.get("recorded_by"),
        filtered_band=_band_from_json(data.get("filtered_band")),
        filter_type=data.get("filter_type"),
        filter_order=data.get("filter_order"),
        temperature=data.get("temperature"),
        humidity=data.get("humidity"),
        speed=1.0 if data.get("speed") is None else data["speed"],
        link_to_full_recording=data.get("link_to_full_recording"),
        notes=data.get("notes"),
    ).validated()


def _location_from_json(value: object) -> object:
    if isinstance(value, dict):
        if "latitude" not in value or "longitude" not in value:
            raise _fail("A location needs both 'latitude' and 'longitude'.")
        return Location(value["latitude"], value["longitude"])
    return value


def _window_from_json(value: object) -> object:
    if isinstance(value, dict):
        if "start" not in value or "end" not in value:
            raise _fail("A sample window needs both 'start' and 'end'.")
        return SampleWindow(value["start"], value["end"])
    return value


def _band_from_json(value: object) -> object:
    if value is None:
        return FilteredBand()
    if isinstance(value, dict):
        return FilteredBand(value.get("low"), value.get("high"))
    return value


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_bundle(
    destination: str | Path | BinaryIO,
    entries: Sequence[ExportEntry],
    *,
    models: Mapping[str, ModelFingerprint] | None = None,
    source_library: SourceLibrary | None = None,
    created_at: str | None = None,
) -> None:
    """Write entries to a bundle.

    Args:
        destination: Path to write, or an open binary stream — a stream so the
            interface can build a download in memory without a temporary file.
        entries: The items to include. Each must have a ``biotimbre_id`` and an
            audio file that exists.
        models: Fingerprint for every model whose vectors any entry carries.
            Only the caller can supply these: ``embedding_dim`` and the
            registered revision live in the library's ``models`` table, not in
            the vectors.
        source_library: The exporting library's identity, recorded so a
            recipient knows who sent this.
        created_at: Timestamp to record; defaults to now.

    Raises:
        ValidationError: If an entry is unusable, or an entry carries vectors
            for a model no fingerprint was given for — which would ship numbers
            nobody could prove comparable.
    """
    fingerprints = dict(models or {})
    prepared = [_prepare(entry, index, fingerprints) for index, entry in enumerate(entries, start=1)]

    manifest = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "created_at": created_at or _utc_now(),
        "source_library": (source_library or SourceLibrary()).to_json(),
        "models": {
            model_id: fingerprints[model_id].to_json()
            for model_id in sorted(_models_used(prepared))
        },
        "items": [{"path": path} for path, _, _ in prepared],
    }

    handle = _open_for_write(destination)
    try:
        with zipfile.ZipFile(handle, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST_NAME, _dump_json(manifest))
            archive.writestr(README_NAME, _README)
            for path, entry, body in prepared:
                _write_entry(archive, path, entry, body)
    finally:
        if handle is not destination:
            handle.close()


def _prepare(
    entry: ExportEntry, index: int, fingerprints: Mapping[str, ModelFingerprint]
) -> tuple[str, ExportEntry, dict]:
    """Validate one entry and build its ``item.json`` body."""
    draft = entry.draft.validated()
    if draft.biotimbre_id is None:
        raise _fail(
            f"Item {entry.source_item_id or index} has no biotimbre_id; there "
            f"would be no identity for the recipient to file it under."
        )
    if not Path(entry.audio).is_file():
        raise _fail(f"No such audio file: {entry.audio}")

    for model_id in entry.embeddings:
        _check_model_id(model_id)
        if model_id not in fingerprints:
            raise _fail(
                f"Item {entry.source_item_id or index} carries {model_id!r} "
                f"embeddings, but no fingerprint for that model was given. "
                f"Vectors without their model's parameters cannot be shown to "
                f"be comparable."
            )

    path = f"items/{index:04d}"
    body = draft_to_json(draft)
    body["biotimbre_label"] = entry.biotimbre_label
    body["source_item_id"] = entry.source_item_id
    body["audio"] = {
        "filename": AUDIO_NAME,
        "sha256": entry.audio_sha256,
        "duration_s": entry.duration_s,
        "sample_rate": entry.sample_rate,
        "channels": entry.channels,
    }
    body["embeddings"] = {
        model_id: f"{EMBEDDINGS_DIR}/{model_id}.npz"
        for model_id in sorted(entry.embeddings)
    }
    return path, entry, body


def _models_used(prepared: Iterable[tuple[str, ExportEntry, dict]]) -> set[str]:
    used: set[str] = set()
    for _, entry, _ in prepared:
        used.update(entry.embeddings)
    return used


def _write_entry(
    archive: zipfile.ZipFile, path: str, entry: ExportEntry, body: dict
) -> None:
    """Write one entry's four kinds of member."""
    archive.writestr(f"{path}/{ITEM_JSON_NAME}", _dump_json(body))
    archive.writestr(f"{path}/{ITEM_SIDECAR_NAME}", format_sidecar(entry.draft))
    archive.write(entry.audio, f"{path}/{AUDIO_NAME}")
    for model_id, rows in sorted(entry.embeddings.items()):
        archive.writestr(
            f"{path}/{EMBEDDINGS_DIR}/{model_id}.npz", _pack_embeddings(rows)
        )


def _pack_embeddings(rows: WindowRows) -> bytes:
    """Pack one model's windows into ``.npz`` bytes.

    Three parallel arrays rather than a structured one: a plain ``(n, dim)``
    float32 block is what search multiplies, and keeping it that shape means a
    reader gets the matrix without rebuilding it. Times are separate because
    they are the sample's own time base, not the vectors'.

    Args:
        rows: ``(start, end, vector)`` per window, in window order.

    Returns:
        The ``.npz`` file's bytes.
    """
    if not rows:
        vectors = np.zeros((0, 0), dtype=VECTOR_DTYPE)
        starts = np.zeros(0, dtype=np.float64)
        ends = np.zeros(0, dtype=np.float64)
    else:
        vectors = np.stack(
            [np.ascontiguousarray(vector, dtype=VECTOR_DTYPE).ravel() for _, _, vector in rows]
        )
        starts = np.asarray([float(start) for start, _, _ in rows], dtype=np.float64)
        ends = np.asarray([float(end) for _, end, _ in rows], dtype=np.float64)

    buffer = io.BytesIO()
    # Uncompressed: the zip deflates the member on the way in, and compressing
    # twice costs time for nothing.
    np.savez(buffer, vectors=vectors, starts=starts, ends=ends)
    return buffer.getvalue()


def _dump_json(data: dict) -> str:
    """JSON as a human would want to find it inside a zip."""
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def _open_for_write(destination: str | Path | BinaryIO) -> BinaryIO:
    if isinstance(destination, (str, Path)):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("wb")
    return destination


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_bundle(source: str | Path | BinaryIO, *, into: str | Path) -> Bundle:
    """Open a bundle, extracting its audio into a directory.

    The audio is written out because that is what an ingest needs — a file to
    read properties from and copy into the library — while metadata and vectors
    are small enough to come back in memory.

    Everything here treats the archive as untrusted: only members the manifest
    names are opened, each is size-capped, arrays load with pickling disabled,
    and each WAV's digest is checked against what the sender claimed before the
    file is handed back.

    Args:
        source: Path to the bundle, or an open binary stream.
        into: Directory to extract audio into. Created if absent. The caller
            owns it and should remove it when done — a temporary directory in
            the usual case.

    Returns:
        The opened :class:`Bundle`.

    Raises:
        ValidationError: If the file is not a bundle, is from a newer format
            version, breaks one of the caps, or holds an item the library
            would refuse.
    """
    destination = Path(into)
    destination.mkdir(parents=True, exist_ok=True)

    handle = _open_for_read(source)
    try:
        try:
            archive = zipfile.ZipFile(handle)
        except zipfile.BadZipFile as exc:
            raise _fail(f"Not a readable bundle: {exc}") from exc

        with archive:
            _check_archive_size(archive)
            manifest = _read_manifest(archive)
            models = {
                model_id: ModelFingerprint.from_json(data)
                for model_id, data in manifest["models"].items()
            }
            provenance_library = SourceLibrary.from_json(manifest["source_library"])

            entries = tuple(
                _read_entry(
                    archive,
                    path,
                    destination / f"{position:04d}",
                    models,
                    provenance_library,
                    manifest["created_at"],
                )
                for position, path in enumerate(manifest["items"], start=1)
            )

        return Bundle(
            format_version=manifest["format_version"],
            created_at=manifest["created_at"],
            source_library=provenance_library,
            models=models,
            entries=entries,
        )
    finally:
        if handle is not source:
            handle.close()


def _open_for_read(source: str | Path | BinaryIO) -> BinaryIO:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise _fail(f"No such bundle: {path}")
        return path.open("rb")
    return source


def _check_archive_size(archive: zipfile.ZipFile) -> None:
    """Refuse an archive that would cost too much to read, before reading it.

    Both checks come from the central directory, so nothing is decompressed to
    learn that it should not be.
    """
    total = 0
    for info in archive.infolist():
        total += info.file_size
        if (
            info.compress_size > 0
            and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
        ):
            raise _fail(
                f"{info.filename!r} expands {info.file_size / info.compress_size:.0f}x, "
                f"past the {MAX_COMPRESSION_RATIO}x limit. This is not a bundle "
                f"of recordings."
            )
    if total > MAX_TOTAL_BYTES:
        raise _fail(
            f"Bundle expands to {total / 1024**3:.1f} GiB, past the "
            f"{MAX_TOTAL_BYTES / 1024**3:.0f} GiB limit."
        )


def _read_manifest(archive: zipfile.ZipFile) -> dict:
    """Read and check the manifest, the only member opened by a fixed name."""
    try:
        raw = _read_member(archive, MANIFEST_NAME, MAX_JSON_BYTES)
    except KeyError:
        raise _fail(
            f"No {MANIFEST_NAME} in the archive. This is a zip file, but not a "
            f"biotimbre bundle."
        ) from None

    manifest = _decode_json(raw, MANIFEST_NAME)
    if manifest.get("format") != BUNDLE_FORMAT:
        raise _fail(
            f"Archive declares format {manifest.get('format')!r}, expected "
            f"{BUNDLE_FORMAT!r}."
        )

    version = manifest.get("format_version")
    if not isinstance(version, int) or version < 1:
        raise _fail(f"Bundle declares an unusable format_version: {version!r}.")
    if version > BUNDLE_VERSION:
        raise _fail(
            f"Bundle is format version {version}; this version of biotimbre "
            f"reads up to {BUNDLE_VERSION}. Update to open it."
        )

    raw_models = manifest.get("models") or {}
    if not isinstance(raw_models, dict):
        raise _fail("manifest 'models' must be an object.")

    items = manifest.get("items")
    if not isinstance(items, list):
        raise _fail("manifest 'items' must be a list.")
    if len(items) > MAX_ENTRIES:
        raise _fail(f"Bundle declares {len(items)} items, past the {MAX_ENTRIES} limit.")

    paths: list[str] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise _fail(f"manifest item {position} must be an object.")
        path = item.get("path")
        if not isinstance(path, str) or not _ENTRY_PATH.match(path):
            raise _fail(
                f"manifest item {position} has an unusable path {path!r}; "
                f"expected the form 'items/0001'."
            )
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise _fail("manifest lists the same item path twice.")

    return {
        "format_version": version,
        "created_at": _optional_text(manifest.get("created_at"), "created_at") or "",
        "source_library": manifest.get("source_library"),
        "models": raw_models,
        "items": paths,
    }


def _read_entry(
    archive: zipfile.ZipFile,
    path: str,
    audio_dir: Path,
    models: Mapping[str, ModelFingerprint],
    library: SourceLibrary,
    exported_at: str,
) -> BundleEntry:
    """Read one entry: its metadata, its audio, and any vectors it carries."""
    try:
        raw = _read_member(archive, f"{path}/{ITEM_JSON_NAME}", MAX_JSON_BYTES)
    except KeyError:
        raise _fail(f"{path} has no {ITEM_JSON_NAME}.") from None

    body = _decode_json(raw, f"{path}/{ITEM_JSON_NAME}")
    draft = draft_from_json(body)
    if draft.biotimbre_id is None:
        raise _fail(
            f"{path} has no biotimbre_id. There is no identity to file it "
            f"under, and allocating one here would invent a second identity "
            f"for a sample that already has one."
        )

    # Every member this entry names is resolved and checked before anything is
    # opened or written: a path that escapes the archive must fail on its own
    # terms, not be masked by whatever the I/O happens to hit first.
    audio_info = _read_audio_info(body.get("audio"), path)
    member = _entry_member(path, body["audio"].get("filename") or AUDIO_NAME)
    declared = _resolve_embeddings(path, body.get("embeddings"), models)

    audio_dir.mkdir(parents=True, exist_ok=True)
    audio = audio_dir / AUDIO_NAME
    digest = _extract_member(archive, member, audio)
    if digest != audio_info.sha256:
        audio.unlink(missing_ok=True)
        raise _fail(
            f"{member} does not match the digest the bundle claims for it "
            f"({digest[:12]}… vs {audio_info.sha256[:12]}…). The file is "
            f"damaged or was altered in transit."
        )

    embeddings = _load_embeddings(archive, declared, models)

    return BundleEntry(
        draft=draft,
        audio=audio,
        audio_info=audio_info,
        biotimbre_label=_optional_text(
            body.get("biotimbre_label"), "biotimbre_label"
        ),
        embeddings=embeddings,
        provenance=Provenance(
            library_uuid=library.uuid,
            library_name=library.name,
            item_id=_optional_int(body.get("source_item_id"), "source_item_id"),
            exported_at=exported_at,
        ),
    )


def _read_audio_info(data: object, path: str) -> AudioInfo:
    if not isinstance(data, dict):
        raise _fail(f"{path} has no audio block.")
    digest = _required_text(data.get("sha256"), "audio.sha256").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise _fail(f"{path} has an unusable audio digest {digest!r}.")
    return AudioInfo(
        sha256=digest,
        duration_s=_required_float(data.get("duration_s"), "audio.duration_s"),
        sample_rate=_required_int(data.get("sample_rate"), "audio.sample_rate"),
        channels=_required_int(data.get("channels"), "audio.channels"),
    )


def _resolve_embeddings(
    path: str, declared: object, models: Mapping[str, ModelFingerprint]
) -> dict[str, str]:
    """Check what vectors an entry claims, and where, without reading any.

    Separated from loading them so that a bad model id or an escaping path is
    refused before the entry's audio is written to disk.

    Args:
        path: The entry's directory.
        declared: The entry's ``embeddings`` block.
        models: Fingerprints the manifest supplied.

    Returns:
        Member name per model id.

    Raises:
        ValidationError: On an unusable model id or path, or vectors for a
            model the manifest did not fingerprint — which would leave nobody
            able to say whether they are comparable.
    """
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        raise _fail(f"{path} has an unusable embeddings block.")

    resolved: dict[str, str] = {}
    for model_id, relative in sorted(declared.items()):
        _check_model_id(model_id)
        if model_id not in models:
            raise _fail(
                f"{path} carries {model_id!r} embeddings, but the manifest "
                f"gives no fingerprint for that model, so there is no way to "
                f"tell whether they are comparable."
            )
        if not isinstance(relative, str):
            raise _fail(f"{path} has an unusable path for {model_id!r} embeddings.")
        resolved[model_id] = _entry_member(path, relative)
    return resolved


def _load_embeddings(
    archive: zipfile.ZipFile,
    declared: Mapping[str, str],
    models: Mapping[str, ModelFingerprint],
) -> dict[str, WindowRows]:
    """Read the vectors :func:`_resolve_embeddings` located."""
    result: dict[str, WindowRows] = {}
    for model_id, member in declared.items():
        try:
            raw = _read_member(archive, member, MAX_EMBEDDINGS_BYTES)
        except KeyError:
            raise _fail(f"{member} is named in the bundle but missing.") from None
        result[model_id] = _unpack_embeddings(
            raw, member, models[model_id].embedding_dim
        )
    return result


def _unpack_embeddings(raw: bytes, member: str, embedding_dim: int) -> WindowRows:
    """Decode one ``.npz`` member into window rows.

    Args:
        raw: The member's bytes.
        member: Its name, for error messages.
        embedding_dim: Width the model's fingerprint promises. A file that
            disagrees is refused rather than stored, since the repository would
            refuse it anyway and the reason is clearer here.

    Returns:
        ``(start, end, vector)`` per window, in window order.

    Raises:
        ValidationError: If the file is unreadable or internally inconsistent.
    """
    try:
        # allow_pickle stays off: this array came from someone else.
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            vectors = np.asarray(data["vectors"], dtype=VECTOR_DTYPE)
            starts = np.asarray(data["starts"], dtype=np.float64)
            ends = np.asarray(data["ends"], dtype=np.float64)
    except (KeyError, ValueError, OSError) as exc:
        raise _fail(f"Could not read embeddings from {member}: {exc}") from exc

    if vectors.ndim != 2:
        raise _fail(
            f"{member} holds a {vectors.ndim}-dimensional array; embeddings "
            f"are one vector per window, so two dimensions."
        )
    if not (vectors.shape[0] == starts.shape[0] == ends.shape[0]):
        raise _fail(
            f"{member} has {vectors.shape[0]} vectors but {starts.shape[0]} "
            f"start times and {ends.shape[0]} end times."
        )
    if vectors.shape[0] and vectors.shape[1] != embedding_dim:
        raise _fail(
            f"{member} holds {vectors.shape[1]}-wide vectors, but its model is "
            f"fingerprinted at {embedding_dim}."
        )

    return [
        (float(starts[index]), float(ends[index]), vectors[index])
        for index in range(vectors.shape[0])
    ]


def _entry_member(path: str, relative: str) -> str:
    """Resolve a path an entry names, refusing anything that leaves its folder.

    A member path is data the sender chose, so ``../../.ssh/authorized_keys``
    has to be impossible rather than unlikely. Resolution is done on the string
    — never against the filesystem — and the result must still sit under the
    entry's own directory.

    Args:
        path: The entry's directory, e.g. ``items/0001``.
        relative: The path named inside ``item.json``.

    Returns:
        The member name to open.

    Raises:
        ValidationError: If the path is absolute, escapes the entry, or has a
            component that could be read as one.
    """
    if not relative or relative.startswith("/") or "\\" in relative or ":" in relative:
        raise _fail(f"{path} names an unusable member path {relative!r}.")
    resolved = posixpath.normpath(f"{path}/{relative}")
    if resolved != f"{path}/{relative}" or not resolved.startswith(f"{path}/"):
        raise _fail(
            f"{path} names {relative!r}, which points outside its own folder."
        )
    return resolved


def _read_member(archive: zipfile.ZipFile, name: str, limit: int) -> bytes:
    """Read a member into memory, refusing one that exceeds ``limit``.

    The limit is applied to the bytes actually read, not to the size the
    archive declares, because the declared size is also the sender's data.
    """
    with archive.open(name) as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise _fail(f"{name} is larger than the {limit // 1024} KiB limit for it.")
    return data


def _extract_member(archive: zipfile.ZipFile, name: str, destination: Path) -> str:
    """Copy a member to disk in chunks, returning its SHA-256.

    Hashing while copying means the digest describes the bytes that landed, and
    the size cap is enforced against those same bytes.
    """

    digest = hashlib.sha256()
    written = 0
    try:
        with archive.open(name) as source, destination.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_TOTAL_BYTES:
                    raise _fail(f"{name} is larger than the bundle size limit.")
                digest.update(chunk)
                target.write(chunk)
    except KeyError:
        destination.unlink(missing_ok=True)
        raise _fail(f"{name} is named in the bundle but missing.") from None
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def _decode_json(raw: bytes, name: str) -> dict:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail(f"{name} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _fail(f"{name} must hold a JSON object.")
    return data


def _check_model_id(model_id: object) -> None:
    if not isinstance(model_id, str) or not _MODEL_ID.match(model_id):
        raise _fail(
            f"{model_id!r} is not a usable model id. Model ids become folder "
            f"names inside the bundle."
        )


# --------------------------------------------------------------------------
# Small typed readers, so a bad field names itself
# --------------------------------------------------------------------------


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{field_name} must be a non-empty string, got {value!r}.")
    return value.strip()


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _fail(f"{field_name} must be a string, got {value!r}.")
    return value.strip() or None


def _required_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{field_name} must be a number, got {value!r}.")
    if float(value) != int(value):
        raise _fail(f"{field_name} must be a whole number, got {value!r}.")
    return int(value)


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _required_int(value, field_name)


def _required_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{field_name} must be a number, got {value!r}.")
    return float(value)
