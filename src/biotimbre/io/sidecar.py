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
    "filtered_band": "filtered_band",
    "noise_reduction": "noise_reduction",
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
    mandatory = ("date", "time", "location", "sample_window")
    missing = [name for name in mandatory if not fields.get(name)]
    if missing:
        raise ValidationError(
            f"Missing mandatory field(s): {', '.join(missing)}. "
            f"Required: {', '.join(mandatory)}.",
            "sidecar",
        )

    optional: dict[str, object] = {}
    for name in ("device", "link_to_full_recording", "notes", "taxa"):
        if fields.get(name):
            optional[name] = fields[name]
    if fields.get("biotimbre_id"):
        optional["biotimbre_id"] = fields["biotimbre_id"]
    if fields.get("filtered_band"):
        optional["filtered_band"] = fields["filtered_band"]
    if fields.get("noise_reduction"):
        optional["noise_reduction"] = fields["noise_reduction"]
    if fields.get("speed"):
        optional["speed"] = fields["speed"]

    return ItemDraft(
        date=fields["date"],
        time=fields["time"],
        location=fields["location"],
        sample_window=fields["sample_window"],
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
