"""Library item types and their validation.

Two record types, deliberately distinct:

:class:`ItemDraft`
    What a user submits. No ``item_id``; ``biotimbre_id`` optional (absent means
    "allocate a new one"). Audio properties are unknown until the file is read.

:class:`LibraryItem`
    What the database holds. Every field resolved.

Validation happens in :meth:`ItemDraft.validated`, so an invalid draft cannot
reach the repository. Dates and times are kept as strings in their input format
because that is what the schema stores and what users typed; they are parsed
only to prove they are well-formed, and exposed as real objects via properties.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date as _date
from datetime import datetime, time as _time

from biotimbre.domain.errors import ValidationError

DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M:%S"

#: ``items.speed`` bounds. Below 0.01 or above 100 is far likelier to be a typo
#: than an intent, and either extreme resamples into uselessness.
MIN_SPEED = 0.01
MAX_SPEED = 100.0
DEFAULT_SPEED = 1.0

#: A biotimbre ID is a 16-bit value written as 4 uppercase hex digits.
BIOTIMBRE_ID_PATTERN = re.compile(r"^[0-9A-F]{4}$")
MAX_BIOTIMBRE_ID = 0xFFFF

#: Taxa tags are normalised to lowercase; separators inside a tag collapse to a
#: single space so "Orthoptera " and "orthoptera" are the same tag.
_TAG_WHITESPACE = re.compile(r"\s+")


def format_biotimbre_id(value: int) -> str:
    """Render a 16-bit integer as a biotimbre ID.

    Args:
        value: Integer in ``[0, 0xFFFF]``.

    Returns:
        4-character uppercase hex string, e.g. ``"00A3"``.

    Raises:
        ValidationError: If out of 16-bit range.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"Expected an integer, got {type(value).__name__}.", "biotimbre_id"
        )
    if not 0 <= value <= MAX_BIOTIMBRE_ID:
        raise ValidationError(
            f"{value} is outside the 16-bit range 0..{MAX_BIOTIMBRE_ID}.",
            "biotimbre_id",
        )
    return f"{value:04X}"


def parse_biotimbre_id(value: str | int) -> str:
    """Normalise a user-supplied biotimbre ID to canonical form.

    Accepts an integer, or a hex string with or without a ``0x`` prefix, in any
    case, with or without zero padding. ``"a3"``, ``"00A3"``, ``"0xa3"`` and
    ``163`` all normalise to ``"00A3"``.

    Args:
        value: The ID as supplied.

    Returns:
        Canonical 4-character uppercase hex string.

    Raises:
        ValidationError: If not a valid 16-bit hex value.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return format_biotimbre_id(value)

    if not isinstance(value, str):
        raise ValidationError(
            f"Expected a hex string or integer, got {type(value).__name__}.",
            "biotimbre_id",
        )

    text = value.strip()
    if not text:
        raise ValidationError("Must not be empty.", "biotimbre_id")
    if text.lower().startswith("0x"):
        text = text[2:]
    if not text:
        raise ValidationError(f"{value!r} has no digits.", "biotimbre_id")

    try:
        number = int(text, 16)
    except ValueError:
        raise ValidationError(
            f"{value!r} is not a hexadecimal number.", "biotimbre_id"
        ) from None
    return format_biotimbre_id(number)


def normalize_taxa(tags: object) -> tuple[str, ...]:
    """Normalise taxa tags to a sorted, de-duplicated, lowercase tuple.

    An absent or empty tag set is valid and means *unknown*, per the spec.
    A single comma-separated string is accepted, since that is how the sidecar
    files and a text input both supply tags.

    Args:
        tags: ``None``, a comma-separated string, or an iterable of strings.

    Returns:
        Sorted unique lowercase tags; empty if unknown.

    Raises:
        ValidationError: If a tag is not a string.
    """
    if tags is None:
        return ()
    if isinstance(tags, str):
        candidates: list[object] = tags.split(",")
    else:
        try:
            candidates = list(tags)  # type: ignore[arg-type]
        except TypeError:
            raise ValidationError(
                f"Expected a string or a list of strings, "
                f"got {type(tags).__name__}.",
                "taxa",
            ) from None

    cleaned: set[str] = set()
    for raw in candidates:
        if not isinstance(raw, str):
            raise ValidationError(
                f"Tags must be strings, got {type(raw).__name__}.", "taxa"
            )
        tag = _TAG_WHITESPACE.sub(" ", raw.strip().lower())
        if tag:
            cleaned.add(tag)
    return tuple(sorted(cleaned))


def _validate_date(value: str) -> str:
    """Validate a date and return it zero-padded.

    Re-formatting is not cosmetic. ``strptime`` accepts ``"2020-1-14"``, and
    storing it unpadded would break date range filtering, which compares these
    values as strings in SQL. Canonicalising keeps lenient input working while
    guaranteeing that lexicographic order equals chronological order.
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"Expected a YYYY-MM-DD string, got {type(value).__name__}.", "date"
        )
    text = value.strip()
    try:
        parsed = datetime.strptime(text, DATE_FORMAT)
    except ValueError:
        raise ValidationError(
            f"{value!r} is not a valid date. Expected YYYY-MM-DD.", "date"
        ) from None
    return parsed.strftime(DATE_FORMAT)


def _validate_time(value: str) -> str:
    """Validate a time and return it zero-padded.

    Canonicalised for the same reason as :func:`_validate_date`: ``"9:40:21"``
    parses, but would not sort correctly against ``"09:40:21"``.
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"Expected an HH:MM:SS string, got {type(value).__name__}.", "time"
        )
    text = value.strip()
    try:
        parsed = datetime.strptime(text, TIME_FORMAT)
    except ValueError:
        raise ValidationError(
            f"{value!r} is not a valid time. Expected HH:MM:SS (24-hour).", "time"
        ) from None
    return parsed.strftime(TIME_FORMAT)


def _as_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError("Expected a number, got a boolean.", field_name)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            raise ValidationError(
                f"{value!r} is not a number.", field_name
            ) from None
    raise ValidationError(
        f"Expected a number, got {type(value).__name__}.", field_name
    )


@dataclass(frozen=True)
class Location:
    """A recording location in decimal degrees.

    Attributes:
        latitude: Degrees north, in ``[-90, 90]``.
        longitude: Degrees east, in ``[-180, 180]``.
    """

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        lat = _as_float(self.latitude, "location.latitude")
        lon = _as_float(self.longitude, "location.longitude")
        if not -90.0 <= lat <= 90.0:
            raise ValidationError(
                f"Latitude {lat} is outside [-90, 90].", "location.latitude"
            )
        if not -180.0 <= lon <= 180.0:
            raise ValidationError(
                f"Longitude {lon} is outside [-180, 180].", "location.longitude"
            )
        object.__setattr__(self, "latitude", lat)
        object.__setattr__(self, "longitude", lon)

    @classmethod
    def parse(cls, value: object) -> "Location":
        """Parse a location from a string, pair, or existing instance.

        Accepts ``"-26,-48"`` (the sidecar format), ``(-26.0, -48.0)``, or a
        :class:`Location`.

        Args:
            value: The location as supplied.

        Returns:
            A validated :class:`Location`.

        Raises:
            ValidationError: If the shape or the values are wrong.
        """
        if isinstance(value, Location):
            return value
        if isinstance(value, str):
            parts = [p for p in value.split(",")]
            if len(parts) != 2:
                raise ValidationError(
                    f"{value!r} must be 'latitude,longitude'.", "location"
                )
            return cls(
                _as_float(parts[0], "location.latitude"),
                _as_float(parts[1], "location.longitude"),
            )
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValidationError(
                    f"Expected 2 values (latitude, longitude), got {len(value)}.",
                    "location",
                )
            return cls(
                _as_float(value[0], "location.latitude"),
                _as_float(value[1], "location.longitude"),
            )
        raise ValidationError(
            f"Cannot read a location from {type(value).__name__}.", "location"
        )

    def __str__(self) -> str:
        return f"{self.latitude},{self.longitude}"


@dataclass(frozen=True)
class FilteredBand:
    """Bandpass applied to the sample before it entered the library.

    Either bound may be ``None``, meaning that side is open — which is how the
    sidecar format's ``2000,-`` is represented.

    Attributes:
        low: Highpass cutoff in Hz, or ``None``.
        high: Lowpass cutoff in Hz, or ``None``.
    """

    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        low = None if self.low is None else _as_float(self.low, "filtered_band.low")
        high = (
            None if self.high is None else _as_float(self.high, "filtered_band.high")
        )
        for name, bound in (("filtered_band.low", low), ("filtered_band.high", high)):
            if bound is not None and bound < 0:
                raise ValidationError(
                    f"Frequency {bound} Hz cannot be negative.", name
                )
        if low is not None and high is not None and low >= high:
            raise ValidationError(
                f"Highpass cutoff ({low} Hz) must be below the lowpass cutoff "
                f"({high} Hz).",
                "filtered_band",
            )
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def is_unset(self) -> bool:
        """True when neither bound is given, i.e. no filtering was recorded."""
        return self.low is None and self.high is None

    @classmethod
    def parse(cls, value: object) -> "FilteredBand":
        """Parse a filtered band from a string, pair, or existing instance.

        ``"2000,-"`` yields ``FilteredBand(2000.0, None)``. An empty string, a
        lone ``"-"``, or ``None`` yields an unset band. The placeholders ``-``,
        ``""`` and ``none`` are all accepted for an open bound.

        Args:
            value: The band as supplied.

        Returns:
            A validated :class:`FilteredBand`.

        Raises:
            ValidationError: If the shape or the values are wrong.
        """
        if value is None:
            return cls()
        if isinstance(value, FilteredBand):
            return value

        if isinstance(value, str):
            text = value.strip()
            if text in ("", "-"):
                return cls()
            parts = text.split(",")
            if len(parts) != 2:
                raise ValidationError(
                    f"{value!r} must be 'highpass,lowpass'; use '-' for an "
                    f"open bound.",
                    "filtered_band",
                )
        elif isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValidationError(
                    f"Expected 2 values (highpass, lowpass), got {len(value)}.",
                    "filtered_band",
                )
            parts = list(value)  # type: ignore[assignment]
        else:
            raise ValidationError(
                f"Cannot read a filtered band from {type(value).__name__}.",
                "filtered_band",
            )

        def _bound(raw: object, name: str) -> float | None:
            if raw is None:
                return None
            if isinstance(raw, str):
                stripped = raw.strip()
                if stripped in ("", "-") or stripped.lower() == "none":
                    return None
                return _as_float(stripped, name)
            return _as_float(raw, name)

        return cls(
            _bound(parts[0], "filtered_band.low"),
            _bound(parts[1], "filtered_band.high"),
        )

    def __str__(self) -> str:
        low = "-" if self.low is None else f"{self.low:g}"
        high = "-" if self.high is None else f"{self.high:g}"
        return f"{low},{high}"


@dataclass(frozen=True)
class SampleWindow:
    """Where the sample sits inside the full source recording, in seconds.

    This is *not* the same axis as an embedding window's ``start_time``, which is
    relative to the start of the sample itself. See PLAN.md §3.

    Attributes:
        start: Offset of the sample's first frame within the full recording.
        end: Offset of the sample's last frame within the full recording.
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        start = _as_float(self.start, "sample_window.start")
        end = _as_float(self.end, "sample_window.end")
        if start < 0:
            raise ValidationError(
                f"Start {start} s cannot be negative.", "sample_window.start"
            )
        if end <= start:
            raise ValidationError(
                f"End ({end} s) must be greater than start ({start} s).",
                "sample_window",
            )
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        """Length of the window in seconds."""
        return self.end - self.start

    @classmethod
    def parse(cls, value: object) -> "SampleWindow":
        """Parse a sample window from a string, pair, or existing instance.

        Accepts ``"34,39"`` (the sidecar format), ``(34.0, 39.0)``, or a
        :class:`SampleWindow`.

        Args:
            value: The window as supplied.

        Returns:
            A validated :class:`SampleWindow`.

        Raises:
            ValidationError: If the shape or the values are wrong.
        """
        if isinstance(value, SampleWindow):
            return value
        if isinstance(value, str):
            parts = value.split(",")
            if len(parts) != 2:
                raise ValidationError(
                    f"{value!r} must be 'start,end' in seconds.", "sample_window"
                )
            return cls(
                _as_float(parts[0], "sample_window.start"),
                _as_float(parts[1], "sample_window.end"),
            )
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValidationError(
                    f"Expected 2 values (start, end), got {len(value)}.",
                    "sample_window",
                )
            return cls(
                _as_float(value[0], "sample_window.start"),
                _as_float(value[1], "sample_window.end"),
            )
        raise ValidationError(
            f"Cannot read a sample window from {type(value).__name__}.",
            "sample_window",
        )

    def __str__(self) -> str:
        return f"{self.start:g},{self.end:g}"


#: Accepted spellings for a boolean supplied as text. The spec writes
#: ``noise_reduction: [yes/no]``, and plain ``bool("no")`` is True, so text must
#: be parsed rather than coerced.
_TRUE_WORDS = frozenset({"yes", "y", "true", "t", "1", "on"})
_FALSE_WORDS = frozenset({"no", "n", "false", "f", "0", "off", ""})


def _parse_bool(value: object, field_name: str) -> bool:
    """Interpret a boolean supplied as a bool, number, or word.

    Args:
        value: The value as supplied.
        field_name: Field name for error reporting.

    Returns:
        The boolean meaning.

    Raises:
        ValidationError: If a string is not a recognised yes/no spelling.
            Guessing would silently invert the user's intent.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_WORDS:
            return True
        if text in _FALSE_WORDS:
            return False
        raise ValidationError(
            f"{value!r} is not a yes/no value. Use yes or no.", field_name
        )
    raise ValidationError(
        f"Expected yes/no, got {type(value).__name__}.", field_name
    )


def _validate_speed(value: object) -> float:
    speed = _as_float(value, "speed")
    if not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValidationError(
            f"Speed {speed} is outside [{MIN_SPEED}, {MAX_SPEED}]. "
            f"Values below 1.0 slow the audio down (shifting ultrasound into "
            f"a model's range); above 1.0 speed it up.",
            "speed",
        )
    return speed


def _clean_optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(
            f"Expected text, got {type(value).__name__}.", field_name
        )
    text = value.strip()
    return text or None


@dataclass(frozen=True)
class ItemDraft:
    """An item as submitted, before it is stored.

    Mandatory: ``date``, ``time``, ``location``, ``sample_window``. Everything
    else is optional, per the spec. ``biotimbre_id`` is optional in the sense
    that omitting it requests a freshly allocated ID; supplying one attaches this
    sample to an existing biotimbre.

    Call :meth:`validated` to obtain a normalised copy; the repository refuses
    drafts that have not been validated.

    Attributes:
        date: Recording date, ``YYYY-MM-DD``.
        time: Recording time, ``HH:MM:SS``.
        location: Recording location.
        sample_window: Position of the sample within the full recording.
        biotimbre_id: Existing biotimbre to join, or ``None`` to allocate.
        taxa: Taxon tags; empty means unknown.
        device: Recording device brand/model.
        filtered_band: Bandpass already applied to the sample.
        noise_reduction: Whether noise reduction was applied.
        speed: Playback-speed factor applied before embedding (PLAN.md D2b).
        link_to_full_recording: URL of the full recording.
        notes: Free text.
    """

    date: str
    time: str
    location: Location
    sample_window: SampleWindow
    biotimbre_id: str | None = None
    taxa: tuple[str, ...] = ()
    device: str | None = None
    filtered_band: FilteredBand = field(default_factory=FilteredBand)
    noise_reduction: bool = False
    speed: float = DEFAULT_SPEED
    link_to_full_recording: str | None = None
    notes: str | None = None

    def validated(self) -> "ItemDraft":
        """Return a normalised, fully validated copy of this draft.

        Coerces loosely-typed input (strings from a form or a sidecar file) into
        the proper value objects and canonical forms.

        Returns:
            A new :class:`ItemDraft` safe to persist.

        Raises:
            ValidationError: On the first field that cannot be accepted.
        """
        return replace(
            self,
            date=_validate_date(self.date),
            time=_validate_time(self.time),
            location=Location.parse(self.location),
            sample_window=SampleWindow.parse(self.sample_window),
            biotimbre_id=(
                None
                if self.biotimbre_id is None or str(self.biotimbre_id).strip() == ""
                else parse_biotimbre_id(self.biotimbre_id)
            ),
            taxa=normalize_taxa(self.taxa),
            device=_clean_optional_text(self.device, "device"),
            filtered_band=FilteredBand.parse(self.filtered_band),
            noise_reduction=_parse_bool(self.noise_reduction, "noise_reduction"),
            speed=_validate_speed(self.speed),
            link_to_full_recording=_clean_optional_text(
                self.link_to_full_recording, "link_to_full_recording"
            ),
            notes=_clean_optional_text(self.notes, "notes"),
        )

    @property
    def taxa_display(self) -> str:
        """Taxa as a comma-separated string, or ``"unknown"`` if empty."""
        return ", ".join(self.taxa) if self.taxa else "unknown"


@dataclass(frozen=True)
class LibraryItem:
    """A stored library item, with every field resolved.

    Attributes:
        item_id: Unique, never-reused counter assigned by the database.
        biotimbre_id: The biotimbre this sample belongs to. Several items may
            share one.
        date: Recording date, ``YYYY-MM-DD``.
        time: Recording time, ``HH:MM:SS``.
        location: Recording location.
        sample_window: Position of the sample within the full recording.
        audio_path: Path to the stored WAV, relative to the data directory.
        audio_sha256: Hex digest of the stored file, for integrity and
            duplicate detection.
        duration_s: Length of the stored sample in seconds.
        sample_rate: Sample rate of the stored file, as submitted.
        channels: Channel count of the stored file, as submitted.
        created_at: ISO-8601 UTC creation timestamp.
        updated_at: ISO-8601 UTC last-modification timestamp.
        taxa: Taxon tags; empty means unknown.
        device: Recording device brand/model.
        filtered_band: Bandpass already applied to the sample.
        noise_reduction: Whether noise reduction was applied.
        speed: Playback-speed factor used to compute this item's embeddings.
        link_to_full_recording: URL of the full recording.
        notes: Free text.
    """

    item_id: int
    biotimbre_id: str
    date: str
    time: str
    location: Location
    sample_window: SampleWindow
    audio_path: str
    audio_sha256: str
    duration_s: float
    sample_rate: int
    channels: int
    created_at: str
    updated_at: str
    taxa: tuple[str, ...] = ()
    device: str | None = None
    filtered_band: FilteredBand = field(default_factory=FilteredBand)
    noise_reduction: bool = False
    speed: float = DEFAULT_SPEED
    link_to_full_recording: str | None = None
    notes: str | None = None

    @property
    def recorded_date(self) -> _date:
        """:attr:`date` as a :class:`datetime.date`."""
        return datetime.strptime(self.date, DATE_FORMAT).date()

    @property
    def recorded_time(self) -> _time:
        """:attr:`time` as a :class:`datetime.time`."""
        return datetime.strptime(self.time, TIME_FORMAT).time()

    @property
    def recorded_at(self) -> datetime:
        """:attr:`date` and :attr:`time` combined. Naive: no timezone is stored."""
        return datetime.combine(self.recorded_date, self.recorded_time)

    @property
    def taxa_display(self) -> str:
        """Taxa as a comma-separated string, or ``"unknown"`` if empty."""
        return ", ".join(self.taxa) if self.taxa else "unknown"

    @property
    def is_speed_shifted(self) -> bool:
        """True when embeddings were computed at a non-native speed."""
        return self.speed != DEFAULT_SPEED

    def to_draft(self) -> ItemDraft:
        """Project back to an :class:`ItemDraft`, for editing flows."""
        return ItemDraft(
            date=self.date,
            time=self.time,
            location=self.location,
            sample_window=self.sample_window,
            biotimbre_id=self.biotimbre_id,
            taxa=self.taxa,
            device=self.device,
            filtered_band=self.filtered_band,
            noise_reduction=self.noise_reduction,
            speed=self.speed,
            link_to_full_recording=self.link_to_full_recording,
            notes=self.notes,
        )
