"""Adding many items from one table: a CSV with one row per recording.

The command-line counterpart of filling the add form many times, and the
table counterpart of ``import-dir``: where a directory pairs each recording
with a sidecar file, a table holds every recording's metadata as one row.

Columns are the sidecar's field names (``date``, ``time``, ``location``,
``taxa``, ...), its processing lines (``trim``, ``apply_highpass``, ...),
``latitude`` and ``longitude`` as an alternative to ``location`` (which a
sidecar takes too), and three of the table's own:

``file``
    The recording. A bare name, or a path relative to the table's folder, is
    looked for beside the table; an absolute path is used as given. A path
    written on another operating system is recognised and refused with a
    message saying so, rather than reported as a file that is not there.
``inat``
    An iNaturalist observation, by ID or URL, in place of ``file``: its
    recording is downloaded and its metadata fills what the row leaves empty,
    as ``add --inat`` does.
``inat_sound``
    Which of the observation's recordings, counting from 1.

Each row needs exactly one of ``file`` and ``inat``. An empty cell means the
field is not given. Spreadsheets that write ``;``-separated files with decimal
commas (common outside the English-speaking world) are read as such: with
``;`` as the separator, ``-26,9`` in a single-number column means -26.9.

One bad row does not stop the import: every row is tried, and each failure is
reported with its line number. A recording already in the library is skipped,
so an interrupted import can be run again. iNaturalist is asked for the
observations 200 at a time, and its recordings are fetched one at a time,
paced by :class:`~biotimbre.sources.inaturalist.Throttle`.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from biotimbre.domain.errors import BiotimbreError, DuplicateAudioError, ValidationError
from biotimbre.io.sidecar import (
    COORDINATE_FIELDS,
    PROCESSING_FIELDS,
    SIDECAR_FIELDS,
    draft_from_fields,
    fold_coordinates,
    processing_from_fields,
)

#: The table's own columns, besides the sidecar's fields.
TABLE_COLUMNS = ("file", "inat", "inat_sound")

#: Every column a table may have: its own, and every name a sidecar takes.
KNOWN_COLUMNS = (
    *TABLE_COLUMNS, *SIDECAR_FIELDS, *COORDINATE_FIELDS, *PROCESSING_FIELDS
)

#: Columns holding one number, which a ``;``-separated table may write with a
#: decimal comma. Pairs (``sample_window``, ``filtered_band``, ``trim``) keep
#: the comma between their two values, so they need decimal points.
_NUMBER_COLUMNS = frozenset({
    "latitude", "longitude", "temperature", "humidity", "speed",
    "filter_order", "apply_highpass", "apply_lowpass", "apply_filter_order",
    "inat_sound",
})

_DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")

#: Separators a table may use, tried in this order when the file does not
#: make it clear.
_SEPARATORS = ",;\t"


@dataclass(frozen=True)
class TableRow:
    """One row of a table, read but not yet checked.

    Attributes:
        line: Its line in the file, for messages.
        fields: Its non-empty cells, by column, with ``latitude`` and
            ``longitude`` folded into ``location``.
    """

    line: int
    fields: dict[str, str]

    @property
    def label(self) -> str:
        """What the row is about, for a report line."""
        if self.fields.get("file"):
            return self.fields["file"]
        if self.fields.get("inat"):
            return f"iNaturalist {self.fields['inat']}"
        return "(no file or inat)"


@dataclass
class TableReport:
    """What a table import did, row by row.

    Attributes:
        added: ``(row, result)`` for each item added.
        skipped: ``(row, item_ids)`` for each recording already in the library.
        failed: ``(row, message)`` for each row that could not be added.
    """

    added: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    failed: list = field(default_factory=list)


def read_table(path: str | Path) -> list[TableRow]:
    """Read a table's rows, checking its columns.

    Args:
        path: The CSV file. UTF-8, with or without the byte-order mark Excel
            writes; separated by commas, semicolons or tabs.

    Returns:
        Its rows, wholly empty ones left out.

    Raises:
        ValidationError: If the file cannot be read, has neither a ``file``
            nor an ``inat`` column, or has a column that is not known — a
            misspelt one would otherwise drop its values unseen.
    """
    table = Path(path)
    try:
        text = table.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"Could not read the table {table}: {exc}", "table") from exc
    if not text.strip():
        raise ValidationError(f"The table {table.name} is empty.", "table")

    separator = _separator(text)
    # From a stream, not a list of lines: a quoted cell may hold a line break,
    # notes in particular, and the reader then counts lines itself.
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=separator)
    header = [name.strip().lower() for name in next(reader)]

    unknown = [name for name in header if name and name not in KNOWN_COLUMNS]
    if unknown:
        raise ValidationError(
            f"{table.name}: unknown column(s) {', '.join(map(repr, unknown))}. "
            f"Known: {', '.join(KNOWN_COLUMNS)}.",
            "table",
        )
    repeated = sorted({name for name in header if name and header.count(name) > 1})
    if repeated:
        raise ValidationError(
            f"{table.name}: column(s) {', '.join(repeated)} appear more than once.",
            "table",
        )
    if "file" not in header and "inat" not in header:
        raise ValidationError(
            f"{table.name} needs a 'file' column (the recordings) or an 'inat' "
            f"column (iNaturalist observations), or both.",
            "table",
        )

    rows: list[TableRow] = []
    line = reader.line_num + 1
    for cells in reader:
        values = {
            name: cell.strip()
            for name, cell in zip(header, cells)
            if name and cell.strip()
        }
        if not values:
            line = reader.line_num + 1
            continue
        if separator == ";":
            for name in _NUMBER_COLUMNS & values.keys():
                if _DECIMAL_COMMA.match(values[name]):
                    values[name] = values[name].replace(",", ".")
        rows.append(TableRow(line=line, fields=values))
        line = reader.line_num + 1
    return rows


def _separator(text: str) -> str:
    """The table's separator: whichever its header line uses most."""
    first = text.splitlines()[0]
    counts = {sep: first.count(sep) for sep in _SEPARATORS}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def resolve_file(value: str, table_dir: Path) -> Path:
    """Where a row's ``file`` is.

    Args:
        value: The cell: a name, a path relative to the table, or an absolute
            path.
        table_dir: The folder the table is in.

    Returns:
        The recording's path, which exists.

    Raises:
        ValidationError: If it is not there — saying so when the path was
            written on another operating system.
    """
    on_windows = os.name == "nt"
    foreign_windows = not on_windows and (
        bool(PureWindowsPath(value).drive) or value.startswith("\\\\")
    )
    foreign_posix = on_windows and value.startswith("/")
    if foreign_windows or foreign_posix:
        kind = "a Windows" if foreign_windows else "a macOS or Linux"
        raise ValidationError(
            f"{value!r} is {kind} path, which does not exist on this computer. "
            f"Write the path as it is here, or just the file's name with the "
            f"table in the same folder as the recordings.",
            "file",
        )

    candidate = Path(value)
    if not candidate.is_absolute():
        # A relative path written with the other system's separator still
        # names the same place beside the table.
        parts = (
            PureWindowsPath(value).parts if "\\" in value and not on_windows
            else PurePosixPath(value).parts if "/" in value and on_windows
            else (value,)
        )
        candidate = table_dir.joinpath(*parts)
    if not candidate.is_file():
        where = (
            "" if Path(value).is_absolute()
            else f" (looked for it in {table_dir.resolve()}, the table's folder)"
        )
        raise ValidationError(f"No such recording: {value}{where}.", "file")
    return candidate


def _observation_fields(observation, sound) -> dict[str, str]:
    """What an observation says, in a table's columns."""
    fields: dict[str, str] = {"link_to_full_recording": observation.url}
    if observation.date:
        fields["date"] = observation.date
    if observation.time:
        fields["time"] = observation.time
    if observation.latitude is not None and observation.longitude is not None:
        fields["location"] = f"{observation.latitude},{observation.longitude}"
    if observation.taxon:
        fields["taxa"] = observation.taxon
    if observation.observer:
        fields["recorded_by"] = observation.observer
    return fields


def _merged_fields(row: TableRow, observation, sound) -> dict[str, str]:
    """The row's cells, over what its observation says.

    Raises:
        ValidationError: If date, time or location is still missing, saying
            why the observation could not supply it.
    """
    merged = {**_observation_fields(observation, sound), **row.fields}
    credit = observation.credit_for(sound)
    merged["notes"] = "\n".join(
        text for text in (row.fields.get("notes"), credit) if text
    )
    missing = [name for name in ("date", "time", "location") if not merged.get(name)]
    if missing:
        hints = []
        if "location" in missing:
            hints.append(
                "iNaturalist hides this observation's location"
                if observation.obscured else "the observation has no location"
            )
        if "date" in missing or "time" in missing:
            hints.append("the observation has no date or time")
        raise ValidationError(
            f"needs {', '.join(missing)} in the table: {'; '.join(hints)}.",
            "table",
        )
    return merged


def _sound_number(row: TableRow) -> int:
    text = row.fields.get("inat_sound")
    if text is None:
        return 1
    try:
        number = float(text)
    except ValueError:
        number = 0
    if number < 1 or number != int(number):
        raise ValidationError(
            f"inat_sound must be a whole number from 1, got {text!r}.", "inat_sound"
        )
    return int(number)


#: Called as each row is done: ``(row, outcome, detail)``, where ``outcome``
#: is ``"added"``, ``"skipped"`` or ``"failed"`` and ``detail`` the
#: :class:`~biotimbre.service.IngestResult`, the item IDs, or the message.
RowCallback = Callable[[TableRow, str, object], None]


def import_table(
    service,
    path: str | Path,
    *,
    compute_embeddings: bool = True,
    skip_duplicates: bool = True,
    on_row: RowCallback | None = None,
    on_wait: Callable[[float, str], None] | None = None,
    on_download: Callable[[TableRow], Callable[[int, int], None]] | None = None,
    client=None,
    throttle=None,
) -> TableReport:
    """Add every row of a table to the library.

    Args:
        service: The :class:`~biotimbre.service.LibraryService` to add to.
        path: The table.
        compute_embeddings: Set false to defer embedding to a backfill.
        skip_duplicates: Skip a recording already in the library rather than
            store it again, so a table can be imported again after an
            interruption.
        on_row: Told as each row is done, for reporting as it goes.
        on_wait: Told before a long wait for iNaturalist, with its length and
            reason.
        on_download: Given a row about to be downloaded, returns the
            ``(received, total)`` progress callback for its download.
        client: An open HTTP client for iNaturalist, for tests.
        throttle: Paces iNaturalist, for tests. ``None`` makes one that calls
            ``on_wait``.

    Returns:
        The :class:`TableReport`.

    Raises:
        ValidationError: If the table itself cannot be read (see
            :func:`read_table`). A bad row is reported, not raised.
    """
    from biotimbre.sources import inaturalist

    table = Path(path)
    rows = read_table(table)
    report = TableReport()

    def done(row: TableRow, outcome: str, detail: object) -> None:
        getattr(report, outcome).append((row, detail))
        if on_row is not None:
            on_row(row, outcome, detail)

    # Every observation first, 200 to a request, rather than one request per
    # row: far fewer requests to pace.
    wanted: dict[int, int] = {}
    for row in rows:
        if "inat" in row.fields and "file" not in row.fields:
            try:
                wanted[row.line] = inaturalist.parse_observation_id(row.fields["inat"])
            except BiotimbreError:
                pass  # reported with the row
    observations: dict[int, object] = {}
    throttle = throttle or inaturalist.Throttle(on_wait=on_wait)
    if wanted and service.settings.inaturalist_enabled:
        observations = inaturalist.fetch_observations(
            list(dict.fromkeys(wanted.values())), client=client, throttle=throttle
        )

    for row in rows:
        try:
            result = _import_row(
                service, row, table.parent,
                observations=observations,
                compute_embeddings=compute_embeddings,
                skip_duplicates=skip_duplicates,
                client=client,
                throttle=throttle,
                on_download=on_download,
            )
        except DuplicateAudioError as exc:
            done(row, "skipped", exc.item_ids)
        except BiotimbreError as exc:
            done(row, "failed", str(exc))
        else:
            done(row, "added", result)
    return report


def _import_row(
    service,
    row: TableRow,
    table_dir: Path,
    *,
    observations,
    compute_embeddings: bool,
    skip_duplicates: bool,
    client,
    throttle,
    on_download,
):
    """Add one row, raising what is wrong with it."""
    from biotimbre.sources import inaturalist

    fields = fold_coordinates(row.fields)
    has_file, has_inat = "file" in fields, "inat" in fields
    if has_file == has_inat:
        raise ValidationError(
            "give file or inat, one of the two." if not has_file
            else "give file or inat, not both: inat fetches the recording.",
            "table",
        )
    if "inat_sound" in fields and not has_inat:
        raise ValidationError("inat_sound needs inat.", "table")

    def add(audio: Path, metadata: dict[str, str]):
        return service.add_item(
            audio,
            draft_from_fields(metadata),
            processing=processing_from_fields(metadata, source=f"line {row.line}"),
            compute_embeddings=compute_embeddings,
            skip_duplicates=skip_duplicates,
        )

    if has_file:
        return add(resolve_file(fields.pop("file"), table_dir), fields)

    if not service.settings.inaturalist_enabled:
        raise ValidationError(
            "fetching from iNaturalist is turned off for this library "
            "([sources.inaturalist] enabled = false).",
            "inat",
        )
    observation_id = inaturalist.parse_observation_id(fields.pop("inat"))
    observation = observations.get(observation_id)
    if observation is None:
        raise ValidationError(
            f"iNaturalist has no observation {observation_id}, or it is private.",
            "inat",
        )
    number = _sound_number(row)
    fields.pop("inat_sound", None)
    if not 1 <= number <= len(observation.sounds):
        raise ValidationError(
            f"observation {observation_id} has {len(observation.sounds)} "
            f"recording(s), so inat_sound {number} does not exist.",
            "inat",
        )
    sound = observation.sounds[number - 1]
    metadata = _merged_fields(TableRow(row.line, fields), observation, sound)

    with tempfile.TemporaryDirectory(prefix="biotimbre-inat-") as directory:
        target = Path(directory) / f"inat-{observation_id}.wav"
        received = {"bytes": 0}
        progress = on_download(row) if on_download is not None else None

        def note(got: int, total: int) -> None:
            received["bytes"] = got
            if progress is not None:
                progress(got, total)

        audio = throttle.call(
            lambda: inaturalist.download_sound(
                sound, target, client=client, on_progress=note
            ),
            media=True,
        )
        throttle.record_media(received["bytes"])
        result = add(audio, metadata)

    caveats = []
    if observation.obscured:
        caveats.append("iNaturalist hides this observation's location")
    if not sound.license_code:
        caveats.append(
            "all rights are reserved by the recordist: check before sharing it"
        )
    if observation.quality_grade and observation.quality_grade != "research":
        caveats.append("not research grade: the identification may still change")
    return dataclasses.replace(result, warnings=(*result.warnings, *caveats))


def check_table(path: str | Path) -> list[tuple[TableRow, str | None]]:
    """What :func:`import_table` would make of each row, changing nothing.

    Everything that can be known without the network or the models is
    checked: the columns, each recording's path, each row's metadata and
    processing, and each observation ID's form. Whether an observation exists
    and what it says is left to the import.

    Returns:
        Each row, with what is wrong with it or ``None``.

    Raises:
        ValidationError: If the table itself cannot be read.
    """
    from biotimbre.sources import inaturalist

    table = Path(path)
    checked: list[tuple[TableRow, str | None]] = []
    for row in read_table(table):
        try:
            fields = fold_coordinates(row.fields)
            has_file, has_inat = "file" in fields, "inat" in fields
            if has_file == has_inat:
                raise ValidationError(
                    "give file or inat, one of the two." if not has_file
                    else "give file or inat, not both.",
                    "table",
                )
            if has_file:
                resolve_file(fields.pop("file"), table.parent)
                draft_from_fields(fields)
            else:
                inaturalist.parse_observation_id(fields.pop("inat"))
                _sound_number(row)
            processing_from_fields(fields, source=f"line {row.line}")
        except BiotimbreError as exc:
            checked.append((row, str(exc)))
        else:
            checked.append((row, None))
    return checked
