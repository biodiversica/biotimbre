"""Reader for the whitespace-separated metadata sidecar format.

One field per line, name first, value being the rest of the line::

    taxa	cicadidae
    date	2020-01-14
    time	09:40:21
    sample_window 5,10
    location -20,-46
    device	Audiomoth
    filtered_band	2000,10000

Both tabs and spaces separate the name from the value, and a value may itself
contain spaces (a device name, a note), so the split is on the *first* run of
whitespace only. Absent lines mean the field was not supplied — which for ``taxa``
means unknown, per the spec.
"""

from __future__ import annotations

import re
from pathlib import Path

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import ItemDraft

#: Field names accepted in a sidecar, mapped to the draft field they populate.
SIDECAR_FIELDS: dict[str, str] = {
    "biotimbre_id": "biotimbre_id",
    "taxa": "taxa",
    "date": "date",
    "time": "time",
    "location": "location",
    "device": "device",
    "microphone": "microphone",
    "recorded_by": "recorded_by",
    "filtered_band": "filtered_band",
    "filter_type": "filter_type",
    "filter_order": "filter_order",
    "temperature": "temperature",
    "humidity": "humidity",
    "sample_window": "sample_window",
    "speed": "speed",
    "link_to_full_recording": "link_to_full_recording",
    "notes": "notes",
}

#: Sidecar extensions tried when locating a file's metadata, in order.
SIDECAR_SUFFIXES = (".txt", ".TXT")

_SPLIT = re.compile(r"\s+")

#: Lines starting with these are ignored, so a sidecar can carry commentary.
_COMMENT_PREFIXES = ("#", ";")


def parse_sidecar(text: str, *, source: str = "<string>") -> dict[str, str]:
    """Parse sidecar text into a field mapping.

    Args:
        text: The file's contents.
        source: Name used in error messages.

    Returns:
        Mapping of field name to raw string value. Only recognised fields are
        included.

    Raises:
        ValidationError: On an unrecognised field name or a duplicate field.
            Silently ignoring a typo like ``locaton`` would drop a mandatory
            value and produce a confusing "missing field" error later.
    """
    fields: dict[str, str] = {}

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(_COMMENT_PREFIXES):
            continue

        parts = _SPLIT.split(line, maxsplit=1)
        name = parts[0].strip().lower()
        value = parts[1].strip() if len(parts) > 1 else ""

        if name not in SIDECAR_FIELDS:
            known = ", ".join(sorted(SIDECAR_FIELDS))
            raise ValidationError(
                f"{source} line {line_number}: unknown field {name!r}. "
                f"Known fields: {known}.",
                "sidecar",
            )
        if name in fields:
            raise ValidationError(
                f"{source} line {line_number}: field {name!r} appears twice.",
                "sidecar",
            )
        fields[name] = value

    return fields


def read_sidecar(path: str | Path) -> dict[str, str]:
    """Read and parse a sidecar file.

    Args:
        path: Path to the sidecar.

    Returns:
        Mapping of field name to raw string value.

    Raises:
        ValidationError: If the file cannot be read or contains a bad field.
    """
    file_path = Path(path)
    try:
        text = file_path.read_text("utf-8")
    except OSError as exc:
        raise ValidationError(
            f"Could not read sidecar {file_path}: {exc}", "sidecar"
        ) from exc
    return parse_sidecar(text, source=file_path.name)


def find_sidecar(audio_path: str | Path) -> Path | None:
    """Locate the sidecar belonging to an audio file.

    Looks for the audio file's stem with a sidecar extension, so
    ``GUYVIR.wav`` pairs with ``GUYVIR.txt``.

    Args:
        audio_path: Path to the audio file.

    Returns:
        The sidecar path, or ``None`` if there is none.
    """
    audio = Path(audio_path)
    for suffix in SIDECAR_SUFFIXES:
        candidate = audio.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def draft_from_fields(fields: dict[str, str]) -> ItemDraft:
    """Build a validated draft from parsed sidecar fields.

    Args:
        fields: Mapping from :func:`parse_sidecar`.

    Returns:
        A validated :class:`~biotimbre.domain.item.ItemDraft`.

    Raises:
        ValidationError: If a mandatory field is missing or a value is invalid.
            Mandatory fields are reported together, so a user fixing a sidecar
            sees everything wrong with it at once rather than one error per run.
    """
    mandatory = ("date", "time", "location")
    missing = [name for name in mandatory if not fields.get(name)]
    if missing:
        raise ValidationError(
            f"Missing mandatory field(s): {', '.join(missing)}. "
            f"Required: {', '.join(mandatory)}.",
            "sidecar",
        )

    optional: dict[str, object] = {}
    for name in (
        "device",
        "microphone",
        "recorded_by",
        "filter_type",
        "link_to_full_recording",
        "notes",
        "taxa",
    ):
        if fields.get(name):
            optional[name] = fields[name]
    if fields.get("biotimbre_id"):
        optional["biotimbre_id"] = fields["biotimbre_id"]
    if fields.get("sample_window"):
        optional["sample_window"] = fields["sample_window"]
    if fields.get("filtered_band"):
        optional["filtered_band"] = fields["filtered_band"]
    if fields.get("filter_order"):
        optional["filter_order"] = fields["filter_order"]
    if fields.get("temperature"):
        optional["temperature"] = fields["temperature"]
    if fields.get("humidity"):
        optional["humidity"] = fields["humidity"]
    if fields.get("speed"):
        optional["speed"] = fields["speed"]

    return ItemDraft(
        date=fields["date"],
        time=fields["time"],
        location=fields["location"],
        **optional,
    ).validated()


def draft_from_sidecar(path: str | Path) -> ItemDraft:
    """Read a sidecar file and build a validated draft from it.

    Args:
        path: Path to the sidecar.

    Returns:
        A validated :class:`~biotimbre.domain.item.ItemDraft`.

    Raises:
        ValidationError: If the file is unreadable or its contents invalid.
    """
    return draft_from_fields(read_sidecar(path))


#: Order fields are written in: identity, then when and where, then how it was
#: recorded, then the free text. Reading a generated sidecar top to bottom
#: should answer the questions in the order a person asks them.
_WRITE_ORDER: tuple[str, ...] = (
    "biotimbre_id",
    "taxa",
    "date",
    "time",
    "location",
    "sample_window",
    "device",
    "microphone",
    "recorded_by",
    "filtered_band",
    "filter_type",
    "filter_order",
    "temperature",
    "humidity",
    "speed",
    "link_to_full_recording",
    "notes",
)

#: Column the values line up in. Wide enough for the longest field name plus a
#: space, so a generated file reads as two columns rather than a ragged list.
_NAME_WIDTH = max(len(name) for name in _WRITE_ORDER) + 2


def format_sidecar(draft: ItemDraft) -> str:
    """Render a draft as sidecar text.

    The inverse of :func:`parse_sidecar`, and the reason this lives here: a
    format with a reader and a writer in different modules grows two opinions
    about itself. Round-trips through :func:`draft_from_fields`, save for what
    the format cannot hold — see below.

    Absent optional fields are omitted rather than written empty: the format
    reads absence as "not supplied", so an empty value would be a third state
    it has no meaning for. ``speed`` is always written even at its default,
    because it decides what an item's embeddings mean and leaving it implicit
    in a file meant for another person invites the wrong assumption.

    One thing does not survive: the format is one field per line, so a newline
    inside ``notes`` becomes a space. Where this matters — inside a bundle —
    the sidecar is the readable copy and ``item.json`` is the authoritative
    one.

    Args:
        draft: The draft to render. Validated first, so the output is
            canonical regardless of how loosely the caller built it.

    Returns:
        Sidecar text, one field per line, ending in a newline.

    Raises:
        ValidationError: If the draft does not validate.
    """
    validated = draft.validated()
    values: dict[str, str | None] = {
        "biotimbre_id": validated.biotimbre_id,
        "taxa": ", ".join(validated.taxa) if validated.taxa else None,
        "date": validated.date,
        "time": validated.time,
        "location": str(validated.location),
        "sample_window": (
            None if validated.sample_window is None else str(validated.sample_window)
        ),
        "device": validated.device,
        "microphone": validated.microphone,
        "recorded_by": validated.recorded_by,
        "filtered_band": (
            None if validated.filtered_band.is_unset else str(validated.filtered_band)
        ),
        "filter_type": validated.filter_type,
        "filter_order": (
            None if validated.filter_order is None else str(validated.filter_order)
        ),
        "temperature": _number(validated.temperature),
        "humidity": _number(validated.humidity),
        "speed": f"{validated.speed:g}",
        "link_to_full_recording": validated.link_to_full_recording,
        "notes": validated.notes,
    }

    lines = [
        f"{name.ljust(_NAME_WIDTH)}{_one_line(values[name])}"
        for name in _WRITE_ORDER
        if values[name] is not None
    ]
    return "\n".join(lines) + "\n"


def _number(value: float | None) -> str | None:
    """A float without a trailing ``.0``, or ``None`` when absent."""
    return None if value is None else f"{value:g}"


def _one_line(value: str | None) -> str:
    """Collapse whitespace so a value cannot break the one-field-per-line rule."""
    return _SPLIT.sub(" ", (value or "").strip())
