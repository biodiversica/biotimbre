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
import uuid
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

#: Taxa tags are normalised to lowercase; separators inside a tag collapse to a
#: single space so "Orthoptera " and "orthoptera" are the same tag.
_TAG_WHITESPACE = re.compile(r"\s+")


def generate_biotimbre_id() -> str:
    """Generate a new biotimbre ID.

    A biotimbre identifies a sound-being — a sonotype whose species identity
    may be unknown — and is meant to mean the same thing no matter which
    library it turns up in, so libraries can eventually be compared or merged.
    A UUID needs no library-wide bookkeeping to stay unique, unlike the old
    per-library-allocated 4-hex-char scheme this replaces.

    Returns:
        A random UUID as a canonical lowercase string.
    """
    return str(uuid.uuid4())


def parse_biotimbre_id(value: str) -> str:
    """Normalise a user-supplied biotimbre ID to canonical form.

    Args:
        value: The ID as supplied, in any form :class:`uuid.UUID` accepts.

    Returns:
        Canonical lowercase, hyphenated UUID string.

    Raises:
        ValidationError: If not a valid UUID.
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"Expected a string, got {type(value).__name__}.", "biotimbre_id"
        )

    text = value.strip()
    if not text:
        raise ValidationError("Must not be empty.", "biotimbre_id")

    try:
        return str(uuid.UUID(text))
    except ValueError:
        raise ValidationError(
            f"{value!r} is not a valid UUID.", "biotimbre_id"
        ) from None


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


def validate_date(value: str) -> str:
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


def validate_time(value: str) -> str:
    """Validate a time and return it zero-padded.

    Canonicalised for the same reason as :func:`validate_date`: ``"9:40:21"``
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


def _is_blank(value: object) -> bool:
    """True for the several ways a form or sidecar spells "nothing here"."""
    return value is None or (isinstance(value, str) and not value.strip())


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

    @property
    def display(self) -> str:
        """The band for reading, as ``"2000 - 10000"``.

        A comma between two frequencies reads like a decimal separator, so a
        closed band joins its bounds with a dash instead. With one side open
        the cutoff alone would not say which side it is, so it keeps the
        comparison that names it: ``"≥ 2000"`` for a highpass, ``"≤ 10000"``
        for a lowpass. An unset band is the empty string, for the caller to
        replace with its own wording.
        """
        if self.low is not None and self.high is not None:
            return f"{self.low:g} - {self.high:g}"
        if self.low is not None:
            return f"≥ {self.low:g}"
        if self.high is not None:
            return f"≤ {self.high:g}"
        return ""

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

    @classmethod
    def parse_optional(cls, value: object) -> "SampleWindow | None":
        """Parse a window that may be absent, returning ``None`` when it is.

        The window says where the sample sits inside a full recording, which
        only means anything when there is a full recording to point at — so it
        is optional. ``None``, an empty string, ``"-"`` and a pair of blanks
        all mean "not recorded". Giving one end without the other is an error
        rather than a half-window.

        Args:
            value: The window as supplied.

        Returns:
            A validated :class:`SampleWindow`, or ``None``.

        Raises:
            ValidationError: If only one end is given, or a value is invalid.
        """
        if value is None:
            return None
        if isinstance(value, str) and value.strip() in ("", "-"):
            return None
        if isinstance(value, (tuple, list)) and len(value) == 2:
            blank = [_is_blank(bound) for bound in value]
            if all(blank):
                return None
            if any(blank):
                raise ValidationError(
                    "Give both ends of the sample window, or neither.",
                    "sample_window.start" if blank[0] else "sample_window.end",
                )
        return cls.parse(value)

    @property
    def display(self) -> str:
        """The window for reading, as ``"5 - 10"``.

        Same reasoning as :attr:`FilteredBand.display`: a comma between two
        numbers reads as a decimal separator. Both bounds are always present.
        """
        return f"{self.start:g} - {self.end:g}"

    def __str__(self) -> str:
        return f"{self.start:g},{self.end:g}"


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


def _clean_optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _as_float(value, field_name)


def _clean_optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    as_float = _as_float(value, field_name)
    if as_float != int(as_float):
        raise ValidationError(f"{value!r} is not a whole number.", field_name)
    return int(as_float)


@dataclass(frozen=True)
class ItemDraft:
    """An item as submitted, before it is stored.

    Mandatory: ``date``, ``time``, ``location``. Everything else is optional,
    per the spec. ``biotimbre_id`` is optional in the sense
    that omitting it requests a freshly allocated ID; supplying one attaches this
    sample to an existing biotimbre.

    Call :meth:`validated` to obtain a normalised copy; the repository refuses
    drafts that have not been validated.

    Attributes:
        date: Recording date, ``YYYY-MM-DD``.
        time: Recording time, ``HH:MM:SS``.
        location: Recording location.
        sample_window: Position of the sample within the full recording, or
            ``None`` when it was not recorded — it only says anything when
            there is a full recording to point at.
        biotimbre_id: Existing biotimbre to join, or ``None`` to allocate.
        taxa: Taxon tags; empty means unknown.
        device: Recording device brand/model.
        microphone: Microphone brand/model.
        recorded_by: Person or organisation who recorded the sample.
        filtered_band: Bandpass already applied to the sample.
        filter_type: Kind of filter used to apply the band, e.g. "Butterworth".
        filter_order: Order of the filter used to apply the band.
        temperature: Ambient temperature at recording time, in degrees Celsius.
        humidity: Ambient relative humidity at recording time, in percent.
        speed: Playback-speed factor applied before embedding (PLAN.md D2b).
        link_to_full_recording: URL of the full recording.
        notes: Free text.
    """

    date: str
    time: str
    location: Location
    sample_window: SampleWindow | None = None
    biotimbre_id: str | None = None
    taxa: tuple[str, ...] = ()
    device: str | None = None
    microphone: str | None = None
    recorded_by: str | None = None
    filtered_band: FilteredBand = field(default_factory=FilteredBand)
    filter_type: str | None = None
    filter_order: int | None = None
    temperature: float | None = None
    humidity: float | None = None
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
            date=validate_date(self.date),
            time=validate_time(self.time),
            location=Location.parse(self.location),
            sample_window=SampleWindow.parse_optional(self.sample_window),
            biotimbre_id=(
                None
                if self.biotimbre_id is None or str(self.biotimbre_id).strip() == ""
                else parse_biotimbre_id(self.biotimbre_id)
            ),
            taxa=normalize_taxa(self.taxa),
            device=_clean_optional_text(self.device, "device"),
            microphone=_clean_optional_text(self.microphone, "microphone"),
            recorded_by=_clean_optional_text(self.recorded_by, "recorded_by"),
            filtered_band=FilteredBand.parse(self.filtered_band),
            filter_type=_clean_optional_text(self.filter_type, "filter_type"),
            filter_order=_clean_optional_int(self.filter_order, "filter_order"),
            temperature=_clean_optional_float(self.temperature, "temperature"),
            humidity=_clean_optional_float(self.humidity, "humidity"),
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
        sample_window: Position of the sample within the full recording, or
            ``None`` when it was not recorded.
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
        microphone: Microphone brand/model.
        recorded_by: Person or organisation who recorded the sample.
        filtered_band: Bandpass already applied to the sample.
        filter_type: Kind of filter used to apply the band, e.g. "Butterworth".
        filter_order: Order of the filter used to apply the band.
        temperature: Ambient temperature at recording time, in degrees Celsius.
        humidity: Ambient relative humidity at recording time, in percent.
        speed: Playback-speed factor used to compute this item's embeddings.
        link_to_full_recording: URL of the full recording.
        notes: Free text.
    """

    item_id: int
    biotimbre_id: str
    date: str
    time: str
    location: Location
    sample_window: SampleWindow | None
    audio_path: str
    audio_sha256: str
    duration_s: float
    sample_rate: int
    channels: int
    created_at: str
    updated_at: str
    taxa: tuple[str, ...] = ()
    device: str | None = None
    microphone: str | None = None
    recorded_by: str | None = None
    filtered_band: FilteredBand = field(default_factory=FilteredBand)
    filter_type: str | None = None
    filter_order: int | None = None
    temperature: float | None = None
    humidity: float | None = None
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
            microphone=self.microphone,
            recorded_by=self.recorded_by,
            filtered_band=self.filtered_band,
            filter_type=self.filter_type,
            filter_order=self.filter_order,
            temperature=self.temperature,
            humidity=self.humidity,
            speed=self.speed,
            link_to_full_recording=self.link_to_full_recording,
            notes=self.notes,
        )
