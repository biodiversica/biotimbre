"""Command-line interface.

A thin frontend over :class:`~biotimbre.service.LibraryService`: it parses
arguments, formats output, and holds no library logic of its own (PLAN.md D5).
Output is English only for now; the translated interface arrives with the GUI in
phase 8. ``--help`` and errors are drawn with Rich (see :mod:`biotimbre.richhelp`),
so every argument and option carries a full explanation: the CLI is the only
interface of an install without the ``gui`` extra.
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from biotimbre import __version__
from biotimbre.config import load_settings
from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import FilteredBand, ItemDraft, Location, SampleWindow
from biotimbre.embeddings import audio as audio_io
from biotimbre.embeddings.runtime import onnxruntime_problem
from biotimbre.io.bundle import BUNDLE_SUFFIX
from biotimbre.io.sidecar import PROCESSING_FIELDS, SIDECAR_FIELDS
from biotimbre.models_registry import MODEL_SPECS
from biotimbre.richhelp import (
    RichHelpParser,
    download_progress,
    print_error,
    print_warning,
)
from biotimbre.search.filters import SearchFilters
from biotimbre.service import IngestResult, LibraryService, SearchHit
from biotimbre.processing import DEFAULT_FILTER_ORDER, Processing
from biotimbre.sharing import ImportPolicy


@contextmanager
def _service(args: argparse.Namespace) -> Iterator[LibraryService]:
    """Open the library from global CLI options, for one command.

    On the way out, prints what the service noted that is not an error, such
    as a newly registered model with more than one output.
    """
    settings = _load_settings(args)
    if args.models:
        settings = dataclasses.replace(
            settings, active_models=tuple(args.models), active_models_chosen=True
        )
    with LibraryService(settings) as service:
        try:
            yield service
        finally:
            for notice in service.pop_notices():
                print_warning(notice)


def _load_settings(args: argparse.Namespace):
    """Load the configuration, reporting a bad one as a user error."""
    try:
        return load_settings(args.config, data_dir=args.data_dir)
    except ValueError as exc:
        # load_settings raises ValueError for a config file it cannot use: an
        # unknown key, a custom model missing its sample rate. That is the
        # user's to fix, not a crash.
        raise ValidationError(str(exc), "config file") from None


def _print_ingest(result: IngestResult) -> None:
    """Report one added or updated item."""
    item = result.item
    print(f"item {item.item_id}  biotimbre {item.biotimbre_id}")
    print(f"  audio    {item.audio_path}")
    print(
        f"  sample   {item.duration_s:.2f} s, {item.sample_rate} Hz, "
        f"{item.channels} ch"
    )
    print(f"  taxa     {item.taxa_display}")
    if item.is_speed_shifted:
        print(f"  speed    {item.speed:g} (embeddings computed speed-shifted)")
    for model_id, count in sorted(result.window_counts.items()):
        print(f"  {model_id:<14} {count} window(s)")
    if result.duplicate_of:
        ids = ", ".join(str(i) for i in result.duplicate_of)
        print(f"  note: identical audio already stored as item(s) {ids}")
    for warning in result.warnings:
        print(f"  warning: {warning}")


def _print_bundle_header(bundle) -> None:
    """Say where a bundle came from, before saying what is in it."""
    library = bundle.source_library
    origin = library.name or "an unnamed library"
    if library.uuid:
        origin += f" ({library.uuid})"
    print(f"Bundle from {origin}")
    print(f"  written  {bundle.created_at or 'unknown'}")
    print(f"  items    {len(bundle)}")
    if bundle.models:
        for model_id, fingerprint in sorted(bundle.models.items()):
            print(
                f"  {model_id:<14} {fingerprint.embedding_dim}d, "
                f"{fingerprint.window_duration_s:g} s windows at "
                f"{fingerprint.window_overlap:g} overlap"
            )
    else:
        print("  models   none (metadata and audio only)")


def _print_plan(plan, *, outcome=None) -> None:
    """Report one entry, either as a prediction or as what happened.

    Args:
        plan: The entry's plan.
        outcome: The result of carrying it out, or ``None`` for a dry run —
            which is the only difference between the two reports.
    """
    if outcome is None:
        verb = "add " if plan.action == "add" else "skip"
    elif outcome.action == "add":
        verb = f"added item {outcome.item.item_id}"
    elif outcome.action == "failed":
        verb = "FAILED"
    else:
        verb = "skipped"

    draft = plan.draft
    print(
        f"{plan.index:>3}. {verb}  biotimbre {plan.biotimbre_id[:8]}  "
        f"{draft.date} {draft.time}  {draft.taxa_display}"
    )

    source = plan.provenance
    if source.item_id is not None:
        print(f"       was item {source.item_id} in {source.library_name or 'it'}")
    # A rebase is not reported here: every one of them produces a warning that
    # names the sender's id and says why it was not kept, and printing the id
    # twice reads as two different facts.

    reason = plan.reason if outcome is None else outcome.reason
    if reason:
        print(f"       {reason}")

    if outcome is not None and outcome.window_counts:
        for model_id, count in sorted(outcome.window_counts.items()):
            how = "reused" if model_id in plan.reusable else "computed"
            print(f"       {model_id:<14} {count} window(s), {how}")
    elif plan.reusable:
        for model_id in plan.reusable:
            windows = len(plan.entry.embeddings[model_id])
            print(f"       {model_id:<14} {windows} window(s), reusable")

    for model_id, why in sorted(plan.rejected.items()):
        print(f"       {model_id:<14} not reused: {why}")

    warnings = plan.warnings if outcome is None else outcome.warnings
    for warning in warnings:
        print(f"       warning: {warning}")


def _stdout() -> "Console":
    """A Rich console on stdout, for output that is more than lines of text.

    Plain text whenever stdout is not a terminal — a file, a pipe, a test —
    so what scripts read is unchanged by the styling and the links.
    """
    from rich.console import Console

    return Console(highlight=False, soft_wrap=True)


def _audio_link(settings, item) -> "Text":
    """An item's stored audio, as its absolute path, linked to the file.

    The link is a terminal hyperlink to a ``file://`` URL: clicked in a
    terminal that supports them (GNOME Terminal, Konsole, Windows Terminal,
    iTerm2, kitty, WezTerm, VS Code's), it opens the recording in the
    system's player. Elsewhere it is the path, to copy into one.
    """
    from rich.style import Style
    from rich.text import Text

    path = settings.absolute_audio_path(item.audio_path).resolve()
    return Text(str(path), style=Style(link=path.as_uri()))


def _url_link(url: str) -> "Text":
    """A URL as a terminal hyperlink, for terminals that do not spot it."""
    from rich.style import Style
    from rich.text import Text

    return Text(url, style=Style(link=url))


def _print_hits(
    hits: list[SearchHit], settings, *, show_all_models: bool, scored: bool = True
) -> None:
    """Report search results, one block each, ruled apart.

    The first line of each — rank, score, item, biotimbre — is what scripts
    match on and stays as it was; the rest are labelled rows beneath it.

    Args:
        hits: The results, best first.
        settings: The library's settings, to find each item's audio.
        show_all_models: Also print each model's score.
        scored: False for a search by filters alone, which has no query to
            score against: the results are listed newest first.
    """
    from rich.rule import Rule
    from rich.text import Text

    if not hits:
        print("No matches." if scored else "No items match these filters.")
        return

    console = _stdout()
    for rank, hit in enumerate(hits, start=1):
        item = hit.item
        span = hit.matched_span
        console.print(Rule(style="dim"))
        header = Text(f"{rank:>3}. ")
        if scored:
            header.append(f"score {hit.score:+.4f}", style="bold")
            header.append("  ")
        header.append(f"item {item.item_id}", style="bold")
        header.append(f"  biotimbre {item.biotimbre_id}", style="dim")
        console.print(header)

        rows: list[tuple[str, Text | str]] = [
            ("taxa", item.taxa_display),
            ("recorded", f"{item.date} {item.time}"),
        ]
        where = str(item.location)
        if hit.distance_km is not None:
            where += f"   ({hit.distance_km:.1f} km from the search centre)"
        rows.append(("location", where))
        if span is not None:
            rows.append((
                "matched",
                f"{span[0]:.2f}-{span[1]:.2f} s of {item.duration_s:.2f} s, "
                f"best {hit.best_model}",
            ))
        if show_all_models and len(hit.per_model) > 1:
            rows.append((
                "models",
                "   ".join(
                    f"{model} {value:+.4f}"
                    for model, value in sorted(hit.per_model.items())
                ),
            ))
        if item.is_speed_shifted:
            rows.append(("speed", f"indexed at {item.speed:g}"))
        if item.device:
            rows.append(("device", item.device))
        rows.append(("audio", _audio_link(settings, item)))
        if item.link_to_full_recording:
            rows.append(("full", _url_link(item.link_to_full_recording)))

        for label, value in rows:
            # One line each, never wrapped: a path cut in two cannot be
            # pasted, and a log should hold each value whole.
            line = Text(f"      {label:<9} ", style="dim")
            line.append(value if isinstance(value, Text) else Text(value))
            console.print(line)
    console.print(Rule(style="dim"))


# ── commands ──────────────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> int:
    """Create the library and register the active models."""
    with _service(args) as service:
        identity = service.identity()
        if args.name:
            identity = service.rename(args.name)
        elif identity.name is None:
            print(
                "This library has no name yet. A name is required so it can be "
                "identified when shared with others:\n"
                "  biotimbre init --name \"My Library\""
            )
            return 1

        print(f'Library "{identity.name}" ({identity.uuid})')
        print(f"  at       {service.settings.data_dir}")
        print(f"  database {service.settings.db_path}")
        print(f"  audio    {service.settings.audio_dir}")
        if args.skip_models:
            print("  models   not registered (--skip-models)")
            return 0
        print("  resolving models (downloads on first run)...")
        for model_id, dim in sorted(service.ensure_models_registered().items()):
            print(f"  {model_id:<14} {dim} dimensions")
    return 0


def _processing_flags(args: argparse.Namespace) -> dict[str, object]:
    """The processing options given on the command line, and only those.

    Only those, so that over a sidecar they replace the lines for the same
    choice and leave the rest: an explicit --apply-filter-order 4 still wins over a
    sidecar's 6, though 4 is also the default.
    """
    given: dict[str, object] = {}
    if args.trim is not None:
        given["trim"] = tuple(args.trim)
    if args.apply_highpass is not None:
        given["highpass"] = args.apply_highpass
    if args.apply_lowpass is not None:
        given["lowpass"] = args.apply_lowpass
    if args.apply_order is not None:
        given["filter_order"] = args.apply_order
    if args.normalise:
        given["normalise"] = True
    return given


def _flags_draft(
    args: argparse.Namespace, fallback: dict[str, object] | None = None
) -> ItemDraft:
    """The metadata the options give, with ``fallback`` for those not given.

    Args:
        args: Parsed add options.
        fallback: Values to use where an option is absent — what an
            iNaturalist observation says. ``None`` when the options are all
            there is.

    Raises:
        ValidationError: If date, time or location is still missing, naming
            the options that would supply it.
    """
    known = dict(fallback or {})

    def pick(name: str) -> object:
        value = getattr(args, name)
        return value if value is not None else known.get(name)

    reasons = known.pop("_missing", {})
    missing = [
        f"--{name}" + (f" ({reasons[name]})" if name in reasons else "")
        for name in ("date", "time", "latitude", "longitude")
        if pick(name) is None
    ]
    if missing:
        alternative = (
            "" if fallback is not None
            else " Alternatively supply a sidecar file with --sidecar."
        )
        raise ValidationError(
            f"Missing required option(s): {', '.join(missing)}.{alternative}",
            "arguments",
        )
    return ItemDraft(
        date=pick("date"),
        time=pick("time"),
        location=Location(pick("latitude"), pick("longitude")),
        sample_window=SampleWindow.parse_optional(
            (args.window_start, args.window_end)
        ),
        biotimbre_id=args.biotimbre_id,
        taxa=tuple(args.taxa) if args.taxa else tuple(known.get("taxa", ())),
        device=args.device,
        microphone=args.microphone,
        recorded_by=pick("recorded_by"),
        filtered_band=FilteredBand(args.highpass, args.lowpass),
        filter_type=args.filter_type,
        filter_order=args.filter_order,
        temperature=args.temperature,
        humidity=args.humidity,
        speed=args.speed,
        link_to_full_recording=pick("link"),
        notes="\n".join(
            text for text in (args.notes, known.get("credit")) if text
        ) or None,
    )


def _observation_metadata(observation, sound) -> dict[str, object]:
    """What an observation says, as fallbacks for :func:`_flags_draft`.

    The same fields the add form fills from it: when and where, the taxon,
    the observer, the observation page as the full recording, and a credit
    line for the recording used. An obscured location is left out — it names
    a random point — and the reason is kept for the error that asks for it.
    """
    values: dict[str, object] = {
        "date": observation.date,
        "time": observation.time,
        "latitude": observation.latitude,
        "longitude": observation.longitude,
        "taxa": (observation.taxon,) if observation.taxon else (),
        "recorded_by": observation.observer,
        "link": observation.url,
        "credit": observation.credit_for(sound),
    }
    why_location = (
        "iNaturalist hides this observation's location"
        if observation.obscured else "the observation has no location"
    )
    values["_missing"] = {
        "date": "the observation has no date",
        "time": "the observation has no time",
        "latitude": why_location,
        "longitude": why_location,
    }
    return values


def _one_audio_source(args: argparse.Namespace) -> None:
    """Refuse AUDIO and --inat together, or neither where one is needed."""
    if args.inat_sound is not None and args.inat is None:
        raise ValidationError("--inat-sound needs --inat.", "arguments")
    if args.audio is not None and args.inat is not None:
        raise ValidationError(
            "Give AUDIO or --inat, not both: --inat fetches the recording.",
            "arguments",
        )


def cmd_add(args: argparse.Namespace) -> int:
    """Add one sample, from a sidecar, from explicit flags, or from iNaturalist."""
    _one_audio_source(args)
    if args.audio is None and args.inat is None:
        raise ValidationError(
            "Give AUDIO, the recording to add, or --inat OBSERVATION to fetch it "
            "from iNaturalist.",
            "arguments",
        )
    if args.inat is not None and args.sidecar:
        raise ValidationError(
            "--sidecar and --inat are two sources of metadata; give one. With "
            "--inat, options such as --date or --taxa replace what the "
            "observation says.",
            "arguments",
        )
    processing = _processing_flags(args)
    with _service(args) as service:
        if args.inat is not None:
            with _inat_recording(args, service.settings) as (path, observation, sound):
                result = service.add_item(
                    path,
                    _flags_draft(args, _observation_metadata(observation, sound)),
                    compute_embeddings=not args.no_embeddings,
                    processing=Processing(**processing),
                )
        elif args.sidecar or not args.date:
            result = service.add_from_sidecar(
                args.audio,
                args.sidecar,
                compute_embeddings=not args.no_embeddings,
                processing_overrides=processing,
            )
        else:
            result = service.add_item(
                args.audio,
                _flags_draft(args),
                compute_embeddings=not args.no_embeddings,
                processing=Processing(**processing),
            )
        _print_ingest(result)
    return 0


@contextmanager
def _inat_recording(args: argparse.Namespace, settings) -> Iterator[tuple]:
    """Fetch the recording --inat names, and say what to know about it.

    Yields:
        The downloaded file, the observation, and which of its recordings it
        is. The file is removed when the block ends.

    Raises:
        ValidationError: If the source is turned off for this library, the
            observation has no such recording, or ffmpeg is needed and absent.
        BiotimbreError: If iNaturalist cannot be reached.
    """
    if not settings.inaturalist_enabled:
        raise ValidationError(
            "Fetching from iNaturalist is turned off for this library "
            "([sources.inaturalist] enabled = false in the config file).",
            "inat",
        )
    from biotimbre.sources import inaturalist

    observation = inaturalist.fetch_observation(
        inaturalist.parse_observation_id(args.inat)
    )
    count = len(observation.sounds)
    if not count:
        raise ValidationError(
            f"iNaturalist observation {observation.id} has no recordings.", "inat"
        )
    number = args.inat_sound if args.inat_sound is not None else 1
    if not 1 <= number <= count:
        raise ValidationError(
            f"Observation {observation.id} has {count} recording(s); "
            f"--inat-sound must be 1 to {count}.",
            "inat",
        )
    sound = observation.sounds[number - 1]

    print(
        f"iNaturalist observation {observation.id}: "
        f"{observation.taxon or 'unidentified'}"
        + (f", {observation.date}" if observation.date else "")
        + (f" {observation.time}" if observation.time else "")
        + (f", by {observation.observer}" if observation.observer else "")
    )
    if count > 1:
        for index, other in enumerate(observation.sounds, start=1):
            marker = "*" if index == number else " "
            print(
                f"  {marker} recording {index}: "
                f"{other.license_code or 'all rights reserved'}"
            )
        print("  (--inat-sound N picks another)")
    if not sound.license_code:
        print_warning(
            "all rights are reserved by the recordist: check before storing or "
            "exporting this recording."
        )
    if observation.quality_grade and observation.quality_grade != "research":
        print_warning(
            "the observation is not research grade: its identification may "
            "still change."
        )

    with tempfile.TemporaryDirectory(prefix="biotimbre-inat-") as directory:
        # download_sound gives it the suffix for what it holds.
        target = Path(directory) / f"inat-{observation.id}.wav"
        with download_progress(f"recording {number} of {observation.id}") as on_progress:
            path = inaturalist.download_sound(sound, target, on_progress=on_progress)
        yield path, observation, sound


#: Fields ``edit --clear`` can empty, by the name the option takes, with the
#: draft field each one is and the value that means "not recorded".
_CLEARABLE: dict[str, tuple[str, object]] = {
    "taxa": ("taxa", ()),
    "sample_window": ("sample_window", None),
    "link": ("link_to_full_recording", None),
    "device": ("device", None),
    "microphone": ("microphone", None),
    "recorded_by": ("recorded_by", None),
    "temperature": ("temperature", None),
    "humidity": ("humidity", None),
    "notes": ("notes", None),
    "filtered_band": ("filtered_band", FilteredBand()),
    "filter_type": ("filter_type", None),
    "filter_order": ("filter_order", None),
}

#: Plain options of edit, by argparse dest, and the draft field each sets.
_EDITABLE = {
    "date": "date",
    "time": "time",
    "biotimbre_id": "biotimbre_id",
    "device": "device",
    "microphone": "microphone",
    "recorded_by": "recorded_by",
    "filter_type": "filter_type",
    "filter_order": "filter_order",
    "temperature": "temperature",
    "humidity": "humidity",
    "speed": "speed",
    "link": "link_to_full_recording",
    "notes": "notes",
}


def _edited_draft(args: argparse.Namespace, before: ItemDraft) -> ItemDraft:
    """The item's metadata with the options given applied to it."""
    changes: dict[str, object] = {
        field: getattr(args, dest)
        for dest, field in _EDITABLE.items()
        if getattr(args, dest) is not None
    }
    if args.taxa is not None:
        changes["taxa"] = tuple(args.taxa)
    if args.latitude is not None or args.longitude is not None:
        changes["location"] = Location(
            args.latitude if args.latitude is not None else before.location.latitude,
            args.longitude if args.longitude is not None else before.location.longitude,
        )
    if args.window_start is not None or args.window_end is not None:
        window = before.sample_window
        changes["sample_window"] = SampleWindow.parse_optional(
            (
                args.window_start if args.window_start is not None
                else (window.start if window else None),
                args.window_end if args.window_end is not None
                else (window.end if window else None),
            )
        )
    if args.highpass is not None or args.lowpass is not None:
        band = before.filtered_band or FilteredBand()
        changes["filtered_band"] = FilteredBand(
            args.highpass if args.highpass is not None else band.low,
            args.lowpass if args.lowpass is not None else band.high,
        )

    for name in args.clear or ():
        field, empty = _CLEARABLE[name]
        if field in changes:
            raise ValidationError(
                f"--clear {name} and an option setting {name} were both given.",
                "arguments",
            )
        changes[field] = empty

    if not changes:
        raise ValidationError(
            "Nothing to change: give at least one option, or --clear. See "
            "biotimbre edit --help.",
            "arguments",
        )
    return dataclasses.replace(before, **changes)


def _show(value: object) -> str:
    """A stored value as edit reports it."""
    if value is None or value == () or value == FilteredBand():
        return "-"
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value)
    return str(value)


def cmd_edit(args: argparse.Namespace) -> int:
    """Change an item's stored metadata."""
    with _service(args) as service:
        before = service.get_item(args.item_id).to_draft().validated()
        result = service.update_item(args.item_id, _edited_draft(args, before))
        after = result.item.to_draft().validated()
        item = result.item
        print(f"item {item.item_id}  biotimbre {item.biotimbre_id}")
        changed = [
            (field.name, getattr(before, field.name), getattr(after, field.name))
            for field in dataclasses.fields(ItemDraft)
            if getattr(before, field.name) != getattr(after, field.name)
        ]
        if not changed:
            print("  nothing changed: the values given are the ones stored")
        width = max((len(name) for name, _, _ in changed), default=0)
        for name, old, new in changed:
            print(f"  {name:<{width}}  {_show(old)} -> {_show(new)}")
        for model_id, count in sorted(result.window_counts.items()):
            print(f"  {model_id:<14} {count} window(s), embedded again")
        for warning in result.warnings:
            print(f"  warning: {warning}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Add every sidecar-paired audio file in a directory."""
    if Path(args.directory).suffix == BUNDLE_SUFFIX:
        # A bundle is one file, not a directory of them, and it carries far
        # more than a sidecar can. Saying so beats "Not a directory".
        print_error(
            f"{args.directory} is a shared bundle. Load it with:\n"
            f"  biotimbre import-bundle {args.directory}"
        )
        return 2

    with _service(args) as service:
        results, failures = service.import_directory(
            args.directory, compute_embeddings=not args.no_embeddings
        )
        for result in results:
            _print_ingest(result)
            print()
        print(f"Added {len(results)} item(s).")
        if failures:
            print(f"{len(failures)} failure(s):", file=sys.stderr)
            for path, message in failures:
                print(f"  {path.name}: {message}", file=sys.stderr)
            return 1
    return 0


def cmd_import_table(args: argparse.Namespace) -> int:
    """Add every row of a CSV table: one recording, or observation, per row."""
    from biotimbre import table_import

    if args.dry_run:
        checked = table_import.check_table(args.table)
        problems = 0
        for row, problem in checked:
            if problem is None:
                print(f"line {row.line:>4}  ok      {row.label}")
            else:
                problems += 1
                print(f"line {row.line:>4}  FAILED  {row.label}: {problem}")
        print(
            f"\n{len(checked) - problems} row(s) ready, {problems} with problems. "
            f"Nothing was changed; iNaturalist was not asked whether the "
            f"observations exist."
        )
        return 1 if problems else 0

    def on_row(row, outcome, detail) -> None:
        if outcome == "added":
            print(f"line {row.line:>4}  added   item {detail.item.item_id}  {row.label}")
            for warning in detail.warnings:
                print(f"            warning: {warning}")
        elif outcome == "skipped":
            ids = ", ".join(str(i) for i in detail)
            print(f"line {row.line:>4}  skipped {row.label}: already item(s) {ids}")
        else:
            print(f"line {row.line:>4}  FAILED  {row.label}: {detail}")

    def on_wait(seconds: float, why: str) -> None:
        print_warning(f"waiting {seconds:.0f} s: {why}.")

    with _service(args) as service:
        report = table_import.import_table(
            service,
            args.table,
            compute_embeddings=not args.no_embeddings,
            skip_duplicates=not args.keep_duplicates,
            on_row=on_row,
            on_wait=on_wait,
        )
    print(
        f"\nAdded {len(report.added)}, skipped {len(report.skipped)}, "
        f"failed {len(report.failed)}."
    )
    return 1 if report.failed else 0


def cmd_list(args: argparse.Namespace) -> int:
    """List stored items."""
    with _service(args) as service:
        items = service.list_items(biotimbre_id=args.biotimbre_id, limit=args.limit)
        if not items:
            print("Library is empty.")
            return 0
        print(f"{'id':>5}  {'btid':<8} {'date':<10} {'time':<8} "
              f"{'dur':>6}  {'speed':>5}  taxa")
        for item in items:
            print(
                f"{item.item_id:>5}  {item.biotimbre_id[:8]:<8} {item.date:<10} "
                f"{item.time:<8} {item.duration_s:>5.1f}s  {item.speed:>5g}  "
                f"{item.taxa_display}"
            )
        print(f"\n{len(items)} item(s) shown. biotimbre IDs shown truncated; "
              f"see `show ITEM_ID` for the full ID.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show one item in full."""
    from rich.text import Text

    with _service(args) as service:
        item = service.get_item(args.item_id)
        counts = service.repo.embedding_counts(args.item_id)
        rows = [
            ("item ID", item.item_id),
            ("biotimbre ID", item.biotimbre_id),
            ("taxa", item.taxa_display),
            ("date", item.date),
            ("time", item.time),
            ("location", str(item.location)),
            ("device", item.device or "-"),
            ("microphone", item.microphone or "-"),
            ("recorded_by", item.recorded_by or "-"),
            ("filtered_band", str(item.filtered_band)),
            ("filter_type", item.filter_type or "-"),
            ("filter_order", item.filter_order if item.filter_order is not None else "-"),
            (
                "temperature",
                f"{item.temperature:g} °C" if item.temperature is not None else "-",
            ),
            ("humidity", f"{item.humidity:g} %" if item.humidity is not None else "-"),
            ("speed", f"{item.speed:g}"),
            ("sample_window", str(item.sample_window or "-")),
            (
                "link",
                _url_link(item.link_to_full_recording)
                if item.link_to_full_recording else "-",
            ),
            ("notes", item.notes or "-"),
            ("audio", _audio_link(service.settings, item)),
            ("sha256", item.audio_sha256),
            ("duration", f"{item.duration_s:.3f} s"),
            ("sample_rate", f"{item.sample_rate} Hz"),
            ("channels", item.channels),
            ("created", item.created_at),
            ("updated", item.updated_at),
        ]
        width = max(len(str(name)) for name, _ in rows)
        console = _stdout()
        for name, value in rows:
            if isinstance(value, Text):  # a link: the audio's file, or a URL
                console.print(Text(f"{name:<{width}}  ") + value)
            else:
                print(f"{name:<{width}}  {value}")
        print(f"{'embeddings':<{width}}  " + (
            ", ".join(f"{m}={n}" for m, n in sorted(counts.items())) or "none"
        ))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete an item."""
    with _service(args) as service:
        item = service.get_item(args.item_id)
        if not args.yes:
            prompt = (
                f"Delete item {item.item_id} (biotimbre {item.biotimbre_id}, "
                f"{item.taxa_display})? [y/N] "
            )
            if input(prompt).strip().lower() not in ("y", "yes"):
                print("Cancelled.")
                return 1
        service.delete_item(args.item_id, remove_audio=not args.keep_audio)
        print(f"Deleted item {args.item_id}.")
    return 0


#: search options that only mean something with a query recording.
#: Options whose name is not their destination's, for messages.
_QUERY_ONLY_FLAGS = {"apply_order": "--apply-filter-order"}

_QUERY_ONLY = (
    "start_time",
    "end_time",
    "query_speed",
    "metric",
    "fusion",
    "min_score",
    "apply_highpass",
    "apply_lowpass",
    "apply_order",
)


def cmd_search(args: argparse.Namespace) -> int:
    """Search the library with a query recording, or by its filters alone."""
    _one_audio_source(args)
    if args.inat_filters and args.inat is None:
        raise ValidationError("--inat-filters needs --inat.", "arguments")
    if args.include_self and args.item is None:
        raise ValidationError("--include-self needs --item.", "arguments")
    sources = [
        name for name, value in (
            ("AUDIO", args.audio), ("--inat", args.inat), ("--item", args.item)
        )
        if value is not None
    ]
    if len(sources) > 1:
        raise ValidationError(
            f"Give one query: {' or '.join(sources)}, not more than one.",
            "arguments",
        )
    if sources and args.speed:
        raise ValidationError(
            "--speed is for a search by the filters alone. With a query "
            "recording, --query-speed picks the speed: it embeds the query "
            "at it and searches only the items stored at it.",
            "arguments",
        )
    with _service(args) as service, ExitStack() as stack:
        query = Path(args.audio) if args.audio is not None else None
        observation = None
        if args.inat is not None:
            query, observation, _sound = stack.enter_context(
                _inat_recording(args, service.settings)
            )
        exclude: tuple[int, ...] = ()
        if args.item is not None:
            item = service.get_item(args.item)
            query, _mime = service.get_audio(args.item)
            query = Path(query)
            print(
                f"Query: item {item.item_id} ({item.taxa_display}, "
                f"{item.date} {item.time})"
            )
            if not args.include_self:
                # It would come first, scoring 1.0 against itself, and say
                # nothing about the rest of the library.
                exclude = (item.item_id,)
        filters = _search_filters(
            args, observation if args.inat_filters else None, exclude=exclude
        )
        if query is None:
            given = [
                _QUERY_ONLY_FLAGS.get(name, "--" + name.replace("_", "-"))
                for name in _QUERY_ONLY
                if getattr(args, name) is not None
            ]
            if given:
                verb, them = ("applies", "it") if len(given) == 1 else ("apply", "them")
                raise ValidationError(
                    f"{', '.join(given)} only {verb} to a query recording; give "
                    f"one, or leave {them} out to search by the filters alone.",
                    "arguments",
                )
            hits = service.filter_items(filters, k=args.top_k)
            _print_hits(hits, service.settings, show_all_models=False, scored=False)
            _write_results(service, hits, args.csv)
            return 0
        hits = service.search(
            query,
            filters=filters,
            k=args.top_k,
            metric=args.metric,
            fusion=args.fusion,
            min_score=args.min_score,
            query_speed=args.query_speed,
            start_time=args.start_time,
            end_time=args.end_time,
            filter_band=FilteredBand(args.apply_highpass, args.apply_lowpass),
            filter_order=(
                args.apply_order if args.apply_order is not None
                else DEFAULT_FILTER_ORDER
            ),
        )
        # After the search, which has already refused an unusable segment, and
        # over the segment it resolved: a start alone means the cap from there.
        start, end = audio_io.resolve_query_segment(
            audio_io.read_properties(query).duration_s,
            args.start_time,
            args.end_time,
            service.settings.max_query_duration_s,
        )
        peak = audio_io.over_range_peak(query, start_time=start, end_time=end)
        if peak is not None:
            print_warning(audio_io.over_range_message(query.name, peak))
        if not hits and args.query_speed is not None:
            speeds = service.stats()["speeds"]
            if float(args.query_speed) not in speeds:
                # Otherwise "No matches." reads as the library having nothing
                # like the query, when it has nothing at that speed at all.
                in_use = ", ".join(f"{s:g}" for s in speeds) or "none"
                print(
                    f"No item is stored at speed {args.query_speed:g}, so there "
                    f"is nothing to compare the query with. Speeds in use: "
                    f"{in_use}."
                )
                _write_results(service, hits, args.csv)
                return 0
        _print_hits(hits, service.settings, show_all_models=True)
        _write_results(service, hits, args.csv)
    return 0


def _write_results(service, hits, destination: str | None) -> None:
    """Write the hits as the search page's results table, if asked to."""
    if destination is None:
        return
    data, name = service.results_table(hits)
    out = Path(destination)
    if out.is_dir():
        out = out / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"\nWrote {len(hits)} result(s) to {out}")


#: Radius around an observation's coordinates that --inat-filters searches,
#: as the search page's own observation fetch does.
OBSERVATION_RADIUS_KM = 25.0


def _search_filters(
    args: argparse.Namespace, observation=None, *, exclude: tuple[int, ...] = ()
) -> SearchFilters:
    """The search's filters, completed from an observation for --inat-filters.

    What the observation supplies only fills a filter its option left unset:
    its taxon for --taxa, and its coordinates, within --radius-km or 25 km, for
    the location. Not its date, which says when one recording was made rather
    than when the animal calls. What was taken is printed, since nothing else
    would show it.
    """
    taxa = tuple(args.taxa or ())
    latitude, longitude, radius = args.latitude, args.longitude, args.radius_km
    if observation is not None:
        taken = []
        if not taxa and observation.taxon:
            taxa = (observation.taxon,)
            taken.append(f"taxa {observation.taxon!r}")
        if latitude is None and longitude is None:
            if observation.latitude is not None and observation.longitude is not None:
                latitude, longitude = observation.latitude, observation.longitude
                radius = radius if radius is not None else OBSERVATION_RADIUS_KM
                taken.append(
                    f"within {radius:g} km of {latitude:g},{longitude:g}"
                )
            elif observation.obscured:
                print_warning(
                    "iNaturalist hides this observation's location, so it is "
                    "not used as a filter."
                )
        if taken:
            print(f"Filters from the observation: {'; '.join(taken)}.")
    return SearchFilters(
        taxa=taxa,
        taxa_match_all=args.taxa_all,
        biotimbre_ids=tuple(args.biotimbre_id or ()),
        date_from=args.date_from,
        date_to=args.date_to,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius,
        exclude_item_ids=tuple(dict.fromkeys((*(args.exclude or ()), *exclude))),
        speeds=tuple(args.speed or ()),
    )


def cmd_models(args: argparse.Namespace) -> int:
    """List registered models."""
    with _service(args) as service:
        models = service.list_models()
        if not models:
            print("No models registered yet. Run 'biotimbre init'.")
            return 0
        for model in models:
            print(f"{model['model_id']}")
            print(f"  source        {model['source']}")
            print(f"  file          {model['filename']}")
            if model["revision"]:
                print(f"  revision      {model['revision']}")
            print(f"  sample_rate   {model['sample_rate']} Hz")
            print(f"  window        {model['window_duration_s']} s "
                  f"(overlap {model['window_overlap']})")
            print(f"  dimensions    {model['embedding_dim']}")
            print(f"  vectors       {model['embedding_count']}")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    """Embed items that lack a model's embeddings."""
    with _service(args) as service:
        written = service.backfill(args.model_id)
        if not written:
            print(f"Every item already has {args.model_id} embeddings.")
            return 0
        total = sum(written.values())
        print(f"Embedded {len(written)} item(s), {total} window(s) written.")
    return 0


def cmd_export_npy(args: argparse.Namespace) -> int:
    """Export one item's embeddings: a .npy per window, zipped if several."""
    with _service(args) as service:
        out = service.export_npy(
            args.item_id, args.model_id, args.output, with_audio=args.with_audio
        )
        print(f"Wrote {out}")
    return 0


def cmd_export_csv(args: argparse.Namespace) -> int:
    """Export every item's metadata as one CSV row."""
    with _service(args) as service:
        out = service.export_items_csv(args.output)
        print(f"Wrote {out}")
    return 0


def cmd_export_bundle(args: argparse.Namespace) -> int:
    """Package items into a file that another library can load."""
    from biotimbre.io.table import read_item_ids

    item_ids = list(args.item_id or ())
    if args.items_from:
        item_ids.extend(read_item_ids(args.items_from))
    item_ids = list(dict.fromkeys(item_ids))
    if not item_ids:
        raise ValidationError(
            "No items to export: give their IDs, or --items-from a CSV with an "
            "item_id column.",
            "arguments",
        )
    with _service(args) as service:
        path = service.export_bundle(
            item_ids,
            args.output,
            include_embeddings=not args.no_embeddings,
        )
        size = path.stat().st_size / 1024
        print(f"Wrote {path} ({size:,.0f} KiB)")
        print(f"  {len(item_ids)} item(s)")
        if args.no_embeddings:
            print("  no embeddings: the receiving library recomputes them.")
    return 0


def _import_policy(args: argparse.Namespace) -> ImportPolicy:
    """The choices import-bundle and import-library share."""
    return ImportPolicy(
        trust_embeddings=not args.distrust_embeddings,
        compute_embeddings=not args.no_embeddings,
        duplicate_audio=args.duplicates,
        species_conflict=args.species.replace("-", "_"),
    )


def _run_import(args: argparse.Namespace, inspect, perform) -> int:
    """Preview or carry out an import, and report it entry by entry."""
    policy = _import_policy(args)
    if args.dry_run:
        preview = inspect(policy)
        _print_bundle_header(preview.bundle)
        print()
        for plan in preview.plans:
            _print_plan(plan)
        print(
            f"\nWould add {len(preview.to_add)}, skip {len(preview.to_skip)}. "
            f"Nothing was changed."
        )
        return 0

    report = perform(policy)
    _print_bundle_header(report.bundle)
    print()
    for outcome in report.outcomes:
        _print_plan(outcome.plan, outcome=outcome)
    print(
        f"\nAdded {len(report.added)}, skipped {len(report.skipped)}"
        + (f", failed {len(report.failed)}" if report.failed else "")
        + "."
    )
    if report.failed:
        for outcome in report.failed:
            print(
                f"  entry {outcome.plan.index}: {outcome.reason}",
                file=sys.stderr,
            )
        return 1
    return 0


def cmd_import_bundle(args: argparse.Namespace) -> int:
    """Load a shared bundle into this library."""
    with _service(args) as service:
        return _run_import(
            args,
            lambda policy: service.inspect_bundle(args.bundle, policy=policy),
            lambda policy: service.import_bundle(args.bundle, policy=policy),
        )


def cmd_import_library(args: argparse.Namespace) -> int:
    """Add every item of another library to this one."""
    with _service(args) as service:
        external = service.open_external_library(args.library)
        if external.missing_audio:
            ids = ", ".join(str(item_id) for item_id in external.missing_audio)
            print_warning(
                f"{len(external.missing_audio)} item(s) have no audio "
                f"on disk and are left out: {ids}"
            )
        return _run_import(
            args,
            lambda policy: service.inspect_library(external, policy=policy),
            lambda policy: service.import_library(external, policy=policy),
        )


def cmd_gui(args: argparse.Namespace) -> int:
    """Start the web interface."""
    try:
        from biotimbre.ui.app import run as run_ui
    except ImportError as exc:
        print_error(
            "the GUI needs the 'gui' extra: uv tool install \"biotimbre[gui]\", "
            f"or uv sync --extra gui in a checkout ({exc})"
        )
        return 2

    settings = _load_settings(args)
    # Kept as overrides rather than only folded into ``settings``: opening a
    # library from the browser reloads the configuration, and a flag typed for
    # this run has to survive that.
    overrides: dict[str, object] = {}
    if args.models:
        overrides["active_models"] = tuple(args.models)
        overrides["active_models_chosen"] = True
    if args.no_map:
        # The one part of the interface that reaches the network just by being
        # looked at, so it can be refused for a single run without editing the
        # config file.
        overrides["map_enabled"] = False
    if args.search_only:
        overrides["search_only"] = True
    if overrides:
        settings = dataclasses.replace(settings, **overrides)

    # Only --data-dir opens a library straight away. A config file's data_dir
    # is where a library might be, not a decision to open it, so the chooser
    # asks and offers it first. A search-only run has no chooser to ask with,
    # and keeps opening what its configuration names.
    suggested = None
    if args.data_dir is None and not settings.search_only:
        if settings.data_dir_chosen:
            suggested = settings.data_dir
        settings = dataclasses.replace(settings, data_dir_chosen=False)

    if settings.search_only and not settings.data_dir_chosen:
        # The chooser is what a run without a library falls back on, and a
        # search-only run does not have one: it may not open a library, let
        # alone create one. So the library has to be named here.
        print_error(
            "--search-only needs a library to search. Pass --data-dir, "
            "or set library.data_dir in the config file."
        )
        return 2

    if settings.data_dir_chosen:
        print(f"Library at {settings.data_dir}")
    else:
        # Nothing was chosen, so nothing is created: the interface asks.
        print("No library given — choose one in the browser.")
    if settings.search_only:
        print("Search only: the library page is not served and the database "
              "is opened read-only.")
    # Asked here as well as where the models load, so the console says it too:
    # the interface still starts, and shows the same message on the page.
    problem = onnxruntime_problem()
    if problem:
        print_warning(problem)
    print(f"Serving on http://{args.host}:{args.port}")
    run_ui(
        settings,
        host=args.host,
        port=args.port,
        show=not args.no_browser,
        reload=args.reload,
        overrides=overrides,
        suggested_library=suggested,
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show library totals."""
    with _service(args) as service:
        identity = service.identity()
        print(f'library     {identity.name or "(unnamed)"} ({identity.uuid})')
        stats = service.stats()
        print(f"items       {stats['items']}")
        print(f"biotimbres  {stats['biotimbres']}")
        speeds = stats["speeds"]
        print(f"speeds      " + (", ".join(f"{s:g}" for s in speeds) or "-"))
        print("models")
        for model in stats["models"]:
            print(f"  {model['model_id']:<14} {model['embedding_count']} vectors "
                  f"({model['embedding_dim']}d)")
        if stats["taxa"]:
            print("taxa")
            for tag, count in stats["taxa"]:
                print(f"  {tag:<24} {count}")
    return 0


# ── argument parsing ──────────────────────────────────────────────────────────

#: Said under every command's help, since the options it names are not listed
#: there: argparse only accepts them before the command.
_GLOBAL_NOTE = (
    "Global options (--config, --data-dir, --models) go before the command, "
    "e.g. biotimbre --data-dir ~/my-library {command} ...; see biotimbre --help."
)

#: The sidecar format, for the commands that read one. A CLI-only install has
#: no README at hand, so the help says it in full.
_SIDECAR_HELP = (
    "A sidecar is a text file with one field per line: the name, then spaces "
    "or a tab, then the value, e.g. 'date 2020-01-14', 'time 09:40:21', "
    "'location -20,-46' (latitude,longitude; or 'latitude -20' and "
    "'longitude -46' as two lines), 'taxa cicadidae, hemiptera', "
    "'filtered_band 2000,10000' (- for an open end). Lines starting with # or "
    "; are comments. Required: date, time, and location or latitude and "
    "longitude. Optional: "
    + ", ".join(f for f in SIDECAR_FIELDS if f not in ("date", "time", "location"))
    + ". A misspelt field name is an error, not ignored.\n\n"
    "A sidecar can also say how to process the audio before it is stored, "
    "as add's Processing options do: "
    + "; ".join(f"'{name}' ({what})" for name, what in PROCESSING_FIELDS.items())
    + ". The stored metadata is corrected to match: sample_window for a trim, "
    "filtered_band, filter_type and filter_order for a filter, a line in the "
    "notes for normalisation."
)

#: How to define a model of one's own. In the help of the command that lists
#: models, since that is where someone looking for one goes.
_CUSTOM_MODELS_HELP = (
    "Other models are defined in the config file, one table each: "
    "[models.custom.my_model] with repo_id = \"owner/repo\" and filename = "
    "\"model.onnx\" (optionally revision = \"<commit>\") for a Hugging Face "
    "model, or url = \"https://...\" (optionally sha256 = \"<hex>\") for a "
    "file elsewhere; plus sample_rate, window_duration_s and optionally "
    "window_overlap (default 0.5). Local files are not accepted. The graph "
    "must take raw mono audio, shaped (windows, samples), and return one "
    "embedding per window. Once registered, the "
    "library remembers the model, so it keeps working without that config "
    "file. Its definition may not change once vectors exist under its id."
)

_DATE_HELP = "YYYY-MM-DD"
_TIME_HELP = "HH:MM:SS"


def _command(
    subparsers,
    name: str,
    *,
    panel: str,
    summary: str,
    description: str,
    examples: str = "",
) -> argparse.ArgumentParser:
    """Add a subcommand whose help has a summary, a description and examples.

    Args:
        subparsers: The top-level parser's subparsers action.
        name: The command's name.
        panel: The panel the top-level help lists it under.
        summary: One sentence, for that listing.
        description: Paragraphs for the command's own help. The first is shown
            plain, the rest dimmed.
        examples: Example invocations, one per line, shown verbatim.
    """
    epilog = ""
    if examples:
        epilog += "Examples:\n" + examples.strip("\n") + "\n\n"
    epilog += _GLOBAL_NOTE.format(command=name)
    return subparsers.add_parser(
        name,
        panel=panel,
        help=summary,
        description=f"{summary}\n\n{description}" if description else summary,
        epilog=epilog,
    )


def _add_import_options(parser: argparse.ArgumentParser, source: str) -> None:
    """The options import-bundle and import-library share.

    Args:
        parser: The subcommand's parser.
        source: What is being imported, for the help text.
    """
    preview = parser.add_argument_group("Preview")
    preview.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Read the {source} and report, entry by entry, what would be added "
             "or skipped and why — including which embeddings would be reused — "
             "without changing this library. Run this first with an unfamiliar "
             f"{source}.",
    )

    embeddings = parser.add_argument_group("Embeddings")
    embeddings.add_argument(
        "--distrust-embeddings",
        action="store_true",
        help=f"Ignore every vector the {source} carries and recompute them from "
             "its audio, the only part that can be verified. Without this flag, "
             "vectors are reused when their model fingerprint (source, file, "
             "revision, sample rate, window length, overlap and dimension) "
             "matches this library's exactly, and recomputed otherwise.",
    )
    embeddings.add_argument(
        "--no-embeddings",
        action="store_true",
        help=f"Do not compute the embeddings the {source} did not supply. The "
             "items are stored but cannot be found by search until "
             "`biotimbre backfill MODEL_ID` fills the gaps. Useful for a bulk "
             "import on a machine without the model files.",
    )

    conflicts = parser.add_argument_group("Conflicts")
    conflicts.add_argument(
        "--duplicates",
        choices=("skip", "add"),
        default="skip",
        help="What to do with a recording this library already holds, matched "
             "by the digest of its audio. Under the same biotimbre it is the "
             "same item and is always skipped. Under a different biotimbre the "
             "two libraries disagree about which being it is: 'skip' reports "
             "it and leaves it out; 'add' stores it anyway.",
    )
    conflicts.add_argument(
        "--species",
        choices=("join-local", "keep-incoming"),
        default="join-local",
        help="What to do when an incoming species (a binomial such as "
             "'guyalna bonaerensis') already has a biotimbre here under another "
             "ID. 'join-local' files the item under the local biotimbre — one "
             "species, one biotimbre — and reports the sender's ID. "
             "'keep-incoming' keeps the sender's ID, leaving two identities for "
             "one species.",
    )


def _inat_options(parser: argparse.ArgumentParser, *, searching: bool) -> None:
    """The options add and search share for fetching from iNaturalist."""
    group = parser.add_argument_group("iNaturalist")
    group.add_argument(
        "--inat",
        metavar="OBSERVATION",
        help="Fetch the recording from this iNaturalist observation, by ID or "
             "URL, instead of giving AUDIO. "
             + (
                 "It becomes the query; its taxon and location are used as "
                 "filters only with --inat-filters."
                 if searching else
                 "Its date, time, location, taxon and observer fill the "
                 "metadata, its page becomes --link-to-full-recording, and a "
                 "credit line with the "
                 "recording's licence goes in the notes; any of these options "
                 "given replaces what the observation says. An obscured "
                 "location is not used: give --latitude and --longitude."
             )
             + " Most recordings there are AAC (.m4a), which needs ffmpeg on "
             "the PATH to decode.",
    )
    group.add_argument(
        "--inat-sound",
        type=int,
        metavar="N",
        help="Which of the observation's recordings to use, counting from 1. "
             "Default: the first. The recordings and their licences are listed "
             "when there are several.",
    )
    if searching:
        group.add_argument(
            "--inat-filters",
            action="store_true",
            help="Also narrow the search by the observation: to its taxon, "
                 "unless --taxa is given, and to within --radius-km (25 km if "
                 "not given) of its location, unless --latitude/--longitude "
                 "are. Not by its date, which says when one recording was made, "
                 "not when the animal calls.",
        )


def _metadata_options(
    parser: argparse.ArgumentParser, *, editing: bool
) -> argparse._ArgumentGroup:
    """The metadata options add and edit share.

    Args:
        parser: The command's parser.
        editing: For edit, where each option changes one stored field and
            none is required; for add, where they describe a new item.

    Returns:
        The "Embedding" group, for the command to add its own options to.
    """
    required = parser.add_argument_group(
        "When and where" if editing else
        "Required metadata (when no sidecar is used)"
    )
    required.add_argument(
        "--date",
        metavar=_DATE_HELP,
        help="Recording date, e.g. 2021-03-05.",
    )
    required.add_argument(
        "--time",
        metavar=_TIME_HELP,
        help="Recording time of day, 24-hour, e.g. 22:10:00.",
    )
    # Scalar rather than a "lat,lon" string: argparse treats a value starting
    # with '-' as an option, so "--location -26,-48" cannot work, and every
    # southern-hemisphere latitude is negative. The comma forms remain the
    # sidecar syntax, where there is no such ambiguity.
    required.add_argument(
        "--latitude",
        type=float,
        help="Latitude of the recording site in decimal degrees, -90 to 90; "
             "negative is south, e.g. -26.9.",
    )
    required.add_argument(
        "--longitude",
        type=float,
        help="Longitude of the recording site in decimal degrees, -180 to 180; "
             "negative is west, e.g. -48.7.",
    )

    identity = parser.add_argument_group("Identity")
    identity.add_argument(
        "--taxa",
        nargs="+",
        metavar="TAG",
        help=("Replace the item's taxon tags with these; --clear taxa removes "
              "them all. " if editing else "")
             + "Taxon tags, at any rank: species, genus, family, order... e.g. "
             "--taxa cicadidae hemiptera. Stored lowercase. Quote a binomial so "
             "it stays one tag: --taxa 'Guyalna bonaerensis'."
             + ("" if editing else " Omit when the taxon is unknown."),
    )
    identity.add_argument(
        "--biotimbre-id",
        metavar="UUID",
        help=("Move the item to this biotimbre (a sound-being: one sonotype, "
              "whose species may be unknown); its stored audio moves with it. "
              if editing else
              "File the item under this existing biotimbre (a sound-being: one "
              "sonotype, whose species may be unknown) instead of the one the "
              "taxa would choose or a new one. ")
             + "Copy the full ID from `biotimbre show ITEM_ID`.",
    )

    recording = parser.add_argument_group("Recording details")
    recording.add_argument(
        "--window-start",
        type=float,
        metavar="SECONDS",
        help="Where this sample starts within the full recording given by "
             "--link-to-full-recording, in seconds. "
             + ("Changes only the start; the end is kept." if editing else
                "Give --window-end with it, or neither."),
    )
    recording.add_argument(
        "--window-end",
        type=float,
        metavar="SECONDS",
        help="Where this sample ends within the full recording, in seconds.",
    )
    recording.add_argument(
        "--link-to-full-recording",
        "--link",
        dest="link",
        metavar="URL",
        help="URL of the full recording this sample was taken from, e.g. an "
             "iNaturalist observation or an archive entry. --link is short for "
             "it.",
    )
    recording.add_argument("--device", help="Recorder brand and model, e.g. Audiomoth.")
    recording.add_argument("--microphone", help="Microphone brand and model.")
    recording.add_argument(
        "--recorded-by", help="Person or organisation who made the recording."
    )
    recording.add_argument(
        "--temperature",
        type=float,
        metavar="CELSIUS",
        help="Air temperature at recording time, in °C.",
    )
    recording.add_argument(
        "--humidity",
        type=float,
        metavar="PERCENT",
        help="Relative humidity at recording time, 0 to 100.",
    )
    recording.add_argument("--notes", help="Free-text notes. Quote text with spaces.")

    filtering = parser.add_argument_group("Filtering already applied to the audio")
    filtering.add_argument(
        "--highpass",
        type=float,
        metavar="HZ",
        help="Highpass cutoff the audio was already filtered with, in Hz. This "
             "records what was done; it does not filter.",
    )
    filtering.add_argument(
        "--lowpass",
        type=float,
        metavar="HZ",
        help="Lowpass cutoff the audio was already filtered with, in Hz. A band "
             "above a model's limit (16 kHz for Perch, 24 kHz for BirdNET) makes "
             "add suggest a --speed.",
    )
    filtering.add_argument(
        "--filter-type",
        help="Kind of filter used, e.g. Butterworth.",
    )
    filtering.add_argument(
        "--filter-order", type=int, help="Order of that filter, e.g. 4."
    )

    embedding = parser.add_argument_group("Embedding")
    embedding.add_argument(
        "--speed",
        type=float,
        default=None if editing else 1.0,
        help=("Change the speed factor; the item is embedded again with every "
              "active model at the new speed. " if editing else "")
             + "Speed factor applied before embedding, between 0.01 and 100. Each "
             "model hears only up to half its sample rate (16 kHz for Perch, "
             "24 kHz for BirdNET), so ultrasound — a katydid at 35 kHz, a bat — "
             "is near-silence to it. Below 1 slows the audio down and lowers its "
             "pitch: at 0.25, 40 kHz becomes 10 kHz. The stored audio and its "
             "playback are not changed, and a search finds the item from a "
             "query at natural speed. Keep one value for similar recordings.",
    )
    return embedding


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    known_models = ", ".join(sorted(MODEL_SPECS))
    parser = RichHelpParser(
        prog="biotimbre",
        description=(
            "Build a local library of animal vocalizations and search it by "
            "sound.\n\n"
            "Each recording you add (an item) is stored with its metadata and "
            "with embeddings — vectors computed by bioacoustic models — so a "
            "short query recording can be ranked against the whole library. A "
            "library is one directory holding library.db (SQLite: metadata and "
            "vectors) and audio/ (the stored recordings); move or copy the "
            "directory and the library goes with it.\n\n"
            "The options below are global and go before the command. Run "
            "biotimbre COMMAND --help for a command's own arguments and options."
        ),
        epilog=(
            "Getting started:\n"
            '  biotimbre --data-dir ~/my-library init --name "My Library"\n'
            "  biotimbre --data-dir ~/my-library add GUYVIR.wav --sidecar GUYVIR.txt\n"
            "  biotimbre --data-dir ~/my-library search query.wav -k 5\n"
            "  biotimbre --data-dir ~/my-library gui"
        ),
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Read settings from this TOML file. Without it, biotimbre uses the "
             "file $BIOTIMBRE_CONFIG names, then biotimbre.toml in the current "
             "directory, then built-in defaults. Relative paths in the file "
             "resolve against the file's own directory. It sets the library "
             "location, the active models, search defaults (top_k, metric, "
             "fusion, the 15 s query limit) and the interface.",
    )
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        help="The library directory to work on: the one holding library.db and "
             "audio/. Overrides library.data_dir from the config file; without "
             "either, ./data is used. Created if it does not exist.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_ID",
        help="Compute embeddings with these models for this run only, instead of "
             "models.active from the config file. By default every known model "
             f"is used: the built-in ones ({known_models}), those defined under "
             "[models.custom.<id>] in the config file, and those the library "
             "remembers from earlier runs. Since this option takes several "
             "values, put -- after them, or write --models=ID for one: "
             "biotimbre --models perch_v2 -- add x.wav.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"biotimbre {__version__}",
        help="Show the version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Library ──

    p_init = _command(
        subparsers,
        "init",
        panel="Library",
        summary="Create the library, or rename it, and fetch the models.",
        description=(
            "Creates the data directory, its database and its audio folder, "
            "names the library, and registers the active models — downloading "
            "their ONNX files from Hugging Face the first time (cached under "
            "$HF_HOME, by default ~/.cache/huggingface). Safe to run again: an "
            "existing library is kept, and --name renames it.\n\n"
            "The name travels with every bundle exported from this library, so "
            "a recipient can tell where items came from. Each library also has "
            "a UUID of its own, made here and never changed."
        ),
        examples=(
            '  biotimbre --data-dir ~/my-library init --name "Field Recordings 2024"\n'
            '  biotimbre --data-dir ~/my-library init --name "New Name"   # rename\n'
            "  biotimbre --data-dir ~/my-library init --skip-models       # offline"
        ),
    )
    p_init.add_argument(
        "--name",
        help="The library's human-readable name. Required the first time; give "
             "it again later to rename the library. Quote a name with spaces.",
    )
    p_init.add_argument(
        "--skip-models",
        action="store_true",
        help="Create the library without downloading or registering models, for "
             "a machine that is offline right now. They are fetched the first "
             "time something needs them (add, search, backfill).",
    )
    p_init.set_defaults(func=cmd_init)

    # ── Items ──

    p_add = _command(
        subparsers,
        "add",
        panel="Items",
        summary="Add one recording to the library.",
        description=(
            "Copies the audio into the library (averaged to mono if it has "
            "several channels), records its metadata, and computes its "
            "embeddings with every active model. Any format libsndfile reads "
            "is accepted: WAV, FLAC, OGG, AIFF and more.\n\n"
            "Metadata comes from one of three places. With --inat, from the "
            "iNaturalist observation the recording is fetched from, with any "
            "option given taking precedence. With --sidecar, or when "
            "--date is not given, it is read from a sidecar file (format below) — "
            "without --sidecar, the file next to AUDIO with the same name and a "
            ".txt suffix. Otherwise it comes from the options below, "
            "where --date, --time, --latitude and --longitude are required. "
            "The two are not combined: when a sidecar is read, every metadata "
            "option is ignored, --speed included.\n\n"
            "The Processing options trim, filter or normalise the stored copy, "
            "and apply either way; over a sidecar's processing lines, each one "
            "given replaces the line for the same choice. The metadata, which "
            "describes AUDIO as given, is corrected to describe what is "
            "stored.\n\n"
            "An item tagged with a species (a binomial, e.g. 'Guyalna "
            "bonaerensis') already in the library joins that species' "
            "biotimbre; anything else starts a new biotimbre unless "
            "--biotimbre-id names one.\n\n"
            + _SIDECAR_HELP
        ),
        examples=(
            "  biotimbre add GUYVIR.wav                       # reads GUYVIR.txt\n"
            "  biotimbre add GUYVIR.wav --sidecar meta/guyvir.txt\n"
            "  biotimbre add katydid.wav --date 2021-03-05 --time 22:10:00 \\\n"
            "      --latitude -26.9 --longitude -48.7 --taxa orthoptera insecta \\\n"
            "      --device Audiomoth --highpass 8000 --lowpass 40000 --speed 0.25\n"
            "  biotimbre add dawn.wav --trim 92 99 --apply-highpass 1500 --normalise\n"
            "  biotimbre add --inat 400139850                # fetch and fill from it\n"
            "  biotimbre add --inat 400139850 --trim 2 7 --taxa cicadidae"
        ),
    )
    p_add.add_argument(
        "audio",
        nargs="?",
        help="The recording to add. It is copied into the library; the original "
             "is not moved or changed. Leave it out with --inat, which fetches "
             "the recording instead.",
    )
    _inat_options(p_add, searching=False)

    source = p_add.add_argument_group("Metadata source")
    source.add_argument(
        "--sidecar",
        metavar="FILE",
        help="Read the metadata from this sidecar file rather than from the "
             "options below. Default: AUDIO's name with a .txt suffix, used "
             "whenever --date is not given.",
    )

    embedding = _metadata_options(p_add, editing=False)

    processing = p_add.add_argument_group("Processing (applied before storing)")
    processing.add_argument(
        "--trim",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="Store only this part of AUDIO, from START to END seconds. "
             "sample_window is set to match: to START..END, or shifted by "
             "them when sample_window already says where AUDIO sits in a "
             "longer recording.",
    )
    processing.add_argument(
        "--apply-highpass",
        type=float,
        metavar="HZ",
        help="Filter out what is below this frequency, with a zero-phase "
             "Butterworth highpass, before storing. Unlike --highpass, this "
             "filters; filtered_band, filter_type and filter_order are then set "
             "to describe it.",
    )
    processing.add_argument(
        "--apply-lowpass",
        type=float,
        metavar="HZ",
        help="Filter out what is above this frequency, with a zero-phase "
             "Butterworth lowpass, before storing.",
    )
    processing.add_argument(
        "--apply-filter-order",
        "--apply-order",
        dest="apply_order",
        type=int,
        metavar="N",
        help="Order of the filter --apply-highpass/--apply-lowpass applies, 1 "
             "to 16. Default: 4. --apply-order is short for it.",
    )
    processing.add_argument(
        "--normalise",
        action="store_true",
        help="Scale the stored copy so its loudest sample is at -1 dBFS, and "
             "add a line saying so, with the gain used, to the notes. Applied "
             "after trimming and filtering.",
    )

    embedding.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Store the audio and metadata only, skipping the models. The item "
             "cannot be found by search until `biotimbre backfill MODEL_ID` "
             "embeds it.",
    )
    p_add.set_defaults(func=cmd_add)

    p_import = _command(
        subparsers,
        "import-dir",
        panel="Items",
        summary="Add every audio file in a directory that has a sidecar file.",
        description=(
            "Scans DIRECTORY (not its subdirectories) for .wav, .flac, .ogg and "
            ".aiff files and adds each one with the metadata from its sidecar: "
            "the file of the same name with a .txt suffix, so GUYVIR.wav pairs "
            "with GUYVIR.txt. A file without a sidecar, or with invalid "
            "metadata, is reported and skipped; the rest are still added.\n\n"
            "Exits with status 1 if any file failed. To load a shared .btz "
            "bundle, use import-bundle instead.\n\n"
            + _SIDECAR_HELP
        ),
        examples=(
            "  biotimbre import-dir recordings/\n"
            "  biotimbre import-dir recordings/ --no-embeddings   # embed later"
        ),
    )
    p_import.add_argument(
        "directory", help="The directory holding the audio and sidecar files."
    )
    p_import.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Store audio and metadata only; embed later with "
             "`biotimbre backfill MODEL_ID`. Much faster for a large directory.",
    )
    p_import.set_defaults(func=cmd_import)

    p_table = _command(
        subparsers,
        "import-table",
        panel="Items",
        summary="Add many recordings from a CSV table, one row each.",
        description=(
            "Like import-dir, but with every recording's metadata in one "
            "table instead of a sidecar beside each. Each row is one item; "
            "the columns are the sidecar's field names (date, time, location, "
            "latitude and longitude or location, taxa, device, ...), its "
            "processing lines (trim, apply_highpass, apply_lowpass, "
            "apply_filter_order, normalise), and three of the table's own: "
            "file, inat and inat_sound. An empty cell gives nothing. A column "
            "that is not one of these is an error, so a misspelt one cannot "
            "drop its values unseen.\n\n"
            "file: the recording. A bare name is looked for in the table's "
            "folder, so keep the table beside the recordings; a path relative "
            "to that folder works too, and an absolute path is used as given. "
            "A path written on another operating system (C:\\... read on "
            "Linux, /home/... on Windows) is reported as such.\n\n"
            "inat: an iNaturalist observation ID or URL instead of file. Its "
            "recording is downloaded, and its date, time, location, taxon, "
            "observer and page fill whatever the row leaves empty, with a "
            "credit line in the notes, as add --inat does. inat_sound picks "
            "one of its recordings (default 1). iNaturalist is asked for the "
            "observations 200 at a time and the recordings one at a time, at "
            "most one request a second and under 5 GB of audio an hour, and "
            "a request it refuses as too many is retried after the wait it "
            "names. Waits longer than a second are announced.\n\n"
            "The table may be separated by commas, semicolons or tabs, as "
            "spreadsheets save CSV. With semicolons, a single number may use "
            "a decimal comma (-26,9); pairs such as trim or sample_window "
            "always need decimal points, since their comma separates the two "
            "values. Use latitude and longitude rather than location there.\n\n"
            "Every row is tried, and each is reported as it is done with its "
            "line number. A recording already in the library is skipped, so "
            "an import that was interrupted can simply be run again. Exits "
            "with status 1 if any row failed."
        ),
        examples=(
            "  biotimbre import-table recordings/table.csv --dry-run\n"
            "  biotimbre import-table recordings/table.csv\n"
            "\n"
            "  # table.csv, beside the recordings. Row 2 is a file; row 3 is\n"
            "  # fetched from iNaturalist, which fills all but its taxon.\n"
            "  file,inat,date,time,latitude,longitude,taxa,trim,normalise\n"
            "  GUYVIR.wav,,2020-01-12,09:40:21,-26.9,-48.7,cicadidae,\"1,4\",yes\n"
            "  ,400139850,,,,,cicadidae,,"
        ),
    )
    p_table.add_argument(
        "table",
        help="The CSV table. Bare file names in it are looked for in its folder.",
    )
    p_table.add_argument(
        "--dry-run",
        action="store_true",
        help="Check every row — columns, paths, metadata, processing, the form "
             "of each observation ID — and change nothing. iNaturalist is not "
             "asked, so whether an observation exists is only found out by "
             "the import.",
    )
    p_table.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Store audio and metadata only; embed later with "
             "`biotimbre backfill MODEL_ID`. Much faster for a large table.",
    )
    p_table.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Store a recording the library already holds as another item, "
             "instead of skipping it. Without it, running the same table again "
             "adds nothing twice.",
    )
    p_table.set_defaults(func=cmd_import_table)

    p_list = _command(
        subparsers,
        "list",
        panel="Items",
        summary="List the items in the library, one line each.",
        description=(
            "Shows each item's ID, the first 8 characters of its biotimbre ID, "
            "date, time, duration, embedding speed and taxa, newest first. Use "
            "show ITEM_ID for everything about one item."
        ),
        examples=(
            "  biotimbre list\n"
            "  biotimbre list --limit 20\n"
            "  biotimbre list --biotimbre-id 4f6c1e2a-...   # one sound-being"
        ),
    )
    p_list.add_argument(
        "--biotimbre-id",
        metavar="UUID",
        help="Show only the items of this biotimbre. Takes the full ID, as "
             "`biotimbre show ITEM_ID` prints it.",
    )
    p_list.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Show only the N most recently added items. Default: all.",
    )
    p_list.set_defaults(func=cmd_list)

    p_show = _command(
        subparsers,
        "show",
        panel="Items",
        summary="Show every stored field of one item.",
        description=(
            "Prints the item's metadata, its full biotimbre ID, where its audio "
            "is stored with the file's SHA-256 digest, the audio's duration, "
            "sample rate and channels, and how many embedding windows each "
            "model holds for it."
        ),
        examples="  biotimbre show 12",
    )
    p_show.add_argument(
        "item_id", type=int, help="The item's ID, as list and search print it."
    )
    p_show.set_defaults(func=cmd_show)

    p_delete = _command(
        subparsers,
        "delete",
        panel="Items",
        summary="Delete one item, its embeddings and its stored audio.",
        description=(
            "Asks for confirmation first. The item ID is never reused. When the "
            "last item of a biotimbre is deleted, the biotimbre's audio folder "
            "goes with it. This cannot be undone: export a bundle first if you "
            "may want the item back."
        ),
        examples=(
            "  biotimbre delete 12\n"
            "  biotimbre delete 12 --yes --keep-audio"
        ),
    )
    p_delete.add_argument("item_id", type=int, help="The item to delete.")
    p_delete.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Delete without asking. For scripts.",
    )
    p_delete.add_argument(
        "--keep-audio",
        action="store_true",
        help="Leave the stored audio file on disk; only the database records "
             "are removed.",
    )
    p_delete.set_defaults(func=cmd_delete)

    p_edit = _command(
        subparsers,
        "edit",
        panel="Items",
        summary="Change an item's stored metadata.",
        description=(
            "Each option replaces one stored field; every field not mentioned "
            "is kept. --clear empties fields. The audio is not changed — to "
            "trim or filter it, delete the item and add it again with "
            "--trim, --apply-highpass, --apply-lowpass.\n\n"
            "Two changes do more than rewrite a field: --biotimbre-id moves "
            "the item, and its stored audio, to another biotimbre; --speed "
            "embeds it again with every active model. The changes are "
            "printed, old value then new."
        ),
        examples=(
            "  biotimbre edit 12 --taxa 'guyalna bonaerensis' cicadidae\n"
            "  biotimbre edit 12 --latitude -26.91 --notes 'calling from a fence post'\n"
            "  biotimbre edit 12 --clear device microphone\n"
            "  biotimbre edit 12 --speed 0.25          # embeds the item again"
        ),
    )
    p_edit.add_argument("item_id", type=int, help="The item to change.")
    _metadata_options(p_edit, editing=True)
    p_edit.add_argument(
        "--clear",
        nargs="+",
        choices=sorted(_CLEARABLE),
        metavar="FIELD",
        help="Empty these fields, so they read as not recorded: "
             + ", ".join(sorted(_CLEARABLE))
             + ". date, time, location and speed cannot be cleared.",
    )
    p_edit.set_defaults(func=cmd_edit)

    # ── Search ──

    p_search = _command(
        subparsers,
        "search",
        panel="Search",
        summary="Find the library items that sound most like a query recording, "
                "or that match filters.",
        description=(
            "Embeds the query with every active model and ranks the library's "
            "items by their best-matching pair of windows. For each hit it "
            "prints the score, the item, the span of the item that matched and "
            "the model that scored best.\n\n"
            "The query is AUDIO, a recording fetched with --inat, or a library "
            "item's own with --item.\n\n"
            "Without any of them, it searches by the filters alone: every item that "
            "matches them, newest first and unscored, up to --top-k. With no "
            "filters either, that is the most recent items.\n\n"
            "A query is at most 15 s (search.max_query_duration_s in the config "
            "file). A shorter file is used whole; a longer one needs "
            "--start-time and --end-time and is refused rather than cut.\n\n"
            "Filters narrow the library before scoring, so results are the best "
            "within the filtered set. The query is embedded at every speed "
            "present in the library, so items stored with --speed are found "
            "from a recording at natural speed."
        ),
        examples=(
            "  biotimbre search query.wav\n"
            "  biotimbre search query.wav -k 5 --taxa cicadidae \\\n"
            "      --latitude -26 --longitude -48 --radius-km 50 --date-from 2020-01-01\n"
            "  biotimbre search dawn_chorus.wav --start-time 92 --end-time 99\n"
            "  biotimbre search query.wav --fusion perch_v2 --min-score 0.6\n"
            "  biotimbre search noisy.wav --apply-highpass 1500     # cut the hum\n"
            "  biotimbre search --taxa cicadidae --date-from 2024-01-01 -k 50\n"
            "  biotimbre search --inat 400139850 --inat-filters\n"
            "  biotimbre search --item 12 --taxa cicadidae      # more like item 12"
        ),
    )
    p_search.add_argument(
        "audio",
        nargs="?",
        help="The query recording: a short clip of the sound to look for, in any "
             "format libsndfile reads. Leave it out to search by the filters "
             "alone. Give it before options that take several values, such as "
             "--taxa, which would otherwise take it as one of theirs.",
    )

    segment = p_search.add_argument_group("Query segment")
    segment.add_argument(
        "--start-time",
        type=float,
        metavar="SECONDS",
        help="Start of the part of the query file to search with, in seconds "
             "from its beginning. Required for a file longer than 15 s. Alone, "
             "it searches up to 15 s from there.",
    )
    segment.add_argument(
        "--end-time",
        type=float,
        metavar="SECONDS",
        help="End of that part, in seconds from the file's beginning. At most "
             "15 s after --start-time.",
    )
    segment.add_argument(
        "--query-speed",
        type=float,
        metavar="FACTOR",
        help="Embed the query at this one speed instead of at every speed in "
             "the library, and search only the items stored at this speed. For "
             "a query that itself needs shifting, e.g. 0.25 for an ultrasonic "
             "query, to compare with items added with --speed 0.25. biotimbre "
             "stats lists the speeds in use.",
    )

    _inat_options(p_search, searching=True)

    from_library = p_search.add_argument_group("Query from the library")
    from_library.add_argument(
        "--item",
        type=int,
        metavar="ITEM_ID",
        help="Search with this item's stored recording, instead of AUDIO or "
             "--inat: find what in the library sounds like something already "
             "in it. The segment, query filter and speed options apply to it "
             "as to a file; an item longer than 15 s needs --start-time and "
             "--end-time. The item itself is left out of the results.",
    )
    from_library.add_argument(
        "--include-self",
        action="store_true",
        help="Keep the --item among the results. It comes first, scoring about "
             "1 against itself: a check that the search works, not a finding.",
    )

    p_search.add_argument(
        "--csv",
        metavar="FILE",
        help="Also write the results to this CSV, as the search page's export "
             "does: rank, score, best model, matched span, distance, query "
             "speed and each model's score, then every stored field of the "
             "item. An existing folder gets biotimbre_search_results.csv. "
             "export-bundle --items-from takes it back, to share the results "
             "kept after checking them.",
    )

    query_filter = p_search.add_argument_group(
        "Query filter (applied to the query before embedding)"
    )
    query_filter.add_argument(
        "--apply-highpass",
        type=float,
        metavar="HZ",
        help="Filter out what is below this frequency in the query, with a "
             "zero-phase Butterworth highpass: wind, traffic, mains hum. The "
             "library is not changed.",
    )
    query_filter.add_argument(
        "--apply-lowpass",
        type=float,
        metavar="HZ",
        help="Filter out what is above this frequency in the query.",
    )
    query_filter.add_argument(
        "--apply-filter-order",
        "--apply-order",
        dest="apply_order",
        type=int,
        metavar="N",
        help="Order of that filter, 1 to 16. Default: 4. --apply-order is short "
             "for it.",
    )

    ranking = p_search.add_argument_group("Ranking")
    ranking.add_argument(
        "-k",
        "--top-k",
        type=int,
        metavar="N",
        help="Return at most N hits. Default: search.top_k from the config "
             "file, 10 if unset.",
    )
    ranking.add_argument(
        "--metric",
        choices=("cosine", "dot", "euclidean"),
        help="How two embedding windows are compared. 'cosine' (the angle "
             "between vectors, -1 to 1) ignores the vectors' lengths; 'dot' "
             "does not; 'euclidean' ranks by negative distance, so scores are "
             "0 or below. Default: search.metric from the config file, cosine "
             "if unset.",
    )
    ranking.add_argument(
        "--fusion",
        metavar="max|mean|MODEL_ID",
        help="How the models' scores combine into one. 'max' takes each item's "
             "best model, so a confident match from either surfaces it; 'mean' "
             "averages the models, rewarding agreement; a model ID, e.g. "
             "perch_v2, ranks by that model alone. Default: search.fusion from "
             "the config file, max if unset.",
    )
    ranking.add_argument(
        "--min-score",
        type=float,
        metavar="SCORE",
        help="Leave out hits scoring below this, after fusion, on the "
             "--metric's scale: e.g. 0.5 for cosine.",
    )

    filters = p_search.add_argument_group("Filters (applied before scoring)")
    filters.add_argument(
        "--taxa",
        nargs="+",
        metavar="TAG",
        help="Only items carrying any of these taxon tags (all of them with "
             "--taxa-all). Case does not matter; quote a binomial: "
             "--taxa 'Guyalna bonaerensis'.",
    )
    filters.add_argument(
        "--taxa-all",
        action="store_true",
        help="Require every tag given to --taxa, not just one of them.",
    )
    filters.add_argument(
        "--biotimbre-id",
        nargs="+",
        metavar="UUID",
        help="Only items of these biotimbres. Full IDs, as show prints them.",
    )
    filters.add_argument(
        "--date-from",
        metavar=_DATE_HELP,
        help="Only items recorded on or after this date.",
    )
    filters.add_argument(
        "--date-to",
        metavar=_DATE_HELP,
        help="Only items recorded on or before this date.",
    )
    filters.add_argument(
        "--latitude",
        type=float,
        help="Centre of a location filter, in decimal degrees (negative is "
             "south). Give --longitude and --radius-km with it.",
    )
    filters.add_argument(
        "--longitude",
        type=float,
        help="Centre of a location filter, in decimal degrees (negative is "
             "west).",
    )
    filters.add_argument(
        "--radius-km",
        type=float,
        metavar="KM",
        help="Only items recorded within this great-circle distance of "
             "--latitude/--longitude. Each hit then shows its distance.",
    )
    filters.add_argument(
        "--speed",
        nargs="+",
        type=float,
        metavar="FACTOR",
        help="Search by the filters alone only: list the items stored with any "
             "of these speed factors, as add --speed gave them — 1 for those at "
             "natural speed, 0.25 for those slowed down 4 times. biotimbre stats "
             "lists the speeds in use. With a query recording, use --query-speed "
             "instead.",
    )
    filters.add_argument(
        "--exclude",
        nargs="+",
        type=int,
        metavar="ITEM_ID",
        help="Leave these items out, e.g. the item the query was cut from.",
    )
    p_search.set_defaults(func=cmd_search)

    # ── Library ──

    p_models = _command(
        subparsers,
        "models",
        panel="Library",
        summary="List the models registered in this library.",
        description=(
            "For each model: where its ONNX file comes from (a Hugging Face "
            "repository or a URL) and which revision or file digest, the "
            "sample rate it takes audio at, its window length and overlap, the "
            "size of its vectors, and how many vectors the library holds for "
            "it. A model is registered by init, or by the first command that "
            "uses it.\n\n"
            + _CUSTOM_MODELS_HELP
        ),
        examples="  biotimbre models",
    )
    p_models.set_defaults(func=cmd_models)

    # ── Embeddings & export ──

    p_backfill = _command(
        subparsers,
        "backfill",
        panel="Embeddings & export",
        summary="Compute one model's embeddings for every item that lacks them.",
        description=(
            "Fills the gaps left by --no-embeddings, by an import, or by adding "
            "a model to models.active in an existing library. Items that "
            "already have the model's embeddings are left alone, so it is safe "
            "to run again."
        ),
        examples=(
            "  biotimbre backfill perch_v2\n"
            "  biotimbre backfill birdnet_v2.4"
        ),
    )
    p_backfill.add_argument(
        "model_id",
        help=f"The model to embed with: a built-in one ({known_models}), or one "
             "defined in the config file or remembered by the library.",
    )
    p_backfill.set_defaults(func=cmd_backfill)

    p_export = _command(
        subparsers,
        "export-npy",
        panel="Embeddings & export",
        summary="Export one item's embeddings for one model as NumPy files.",
        description=(
            "One window writes a single .npy file holding its vector. Several "
            "windows write a .zip with one .npy per window, each named with the "
            "window's index and the span of seconds it covers. The right "
            "suffix is added to OUTPUT when it is missing."
        ),
        examples=(
            "  biotimbre export-npy 12 perch_v2 exports/\n"
            "  biotimbre export-npy 12 birdnet_v2.4 item12.zip --with-audio"
        ),
    )
    p_export.add_argument("item_id", type=int, help="The item to export.")
    p_export.add_argument(
        "model_id",
        help="Whose embeddings to export, as `biotimbre models` lists them.",
    )
    p_export.add_argument(
        "output",
        help="The file to write, or an existing directory to write it into "
             "under a name made from the item ID and model.",
    )
    p_export.add_argument(
        "--with-audio",
        action="store_true",
        help="Put each window's audio beside its .npy as a .wav: the samples "
             "exactly as the model was given them — resampled to the model's "
             "rate, speed-shifted and padded to its window length. Always "
             "writes a zip.",
    )
    p_export.set_defaults(func=cmd_export_npy)

    p_export_csv = _command(
        subparsers,
        "export-csv",
        panel="Embeddings & export",
        summary="Export every item's metadata as a CSV table.",
        description=(
            "One row per item, one column per stored field, including the "
            "item and biotimbre IDs and the audio path. Opens in a spreadsheet "
            "or pandas. Embeddings are not included; see export-npy."
        ),
        examples=(
            "  biotimbre export-csv items.csv\n"
            "  biotimbre export-csv exports/   # writes exports/biotimbre_items.csv"
        ),
    )
    p_export_csv.add_argument(
        "output",
        help="The CSV file to write, or an existing directory to write "
             "biotimbre_items.csv into.",
    )
    p_export_csv.set_defaults(func=cmd_export_csv)

    # ── Sharing ──

    p_export_bundle = _command(
        subparsers,
        "export-bundle",
        panel="Sharing",
        summary="Package items into a .btz bundle that another library can import.",
        description=(
            f"A bundle ({BUNDLE_SUFFIX}) is a zip holding each item's metadata, "
            "its stored audio and its embedding vectors with the fingerprints "
            "of the models that made them. The recipient loads it with "
            "import-bundle, which reuses the vectors when their models match. "
            "Biotimbre IDs are UUIDs, so the same sound-being keeps its "
            "identity in every library."
        ),
        examples=(
            "  biotimbre export-bundle 12 -o guyalna.btz\n"
            "  biotimbre export-bundle 12 13 14 -o batch.btz\n"
            "  biotimbre export-bundle 12 -o audio-only.btz --no-embeddings\n"
            "  biotimbre search q.wav --taxa cicadidae --csv hits.csv\n"
            "  biotimbre export-bundle --items-from hits.csv -o checked.btz"
        ),
    )
    p_export_bundle.add_argument(
        "item_id",
        type=int,
        nargs="*",
        help="The items to include, by ID; several may be given. May be left "
             "out with --items-from.",
    )
    p_export_bundle.add_argument(
        "--items-from",
        metavar="CSV",
        help="Also include every item in this CSV's item_id column: the "
             "results search --csv wrote, the library's export-csv, or any "
             "table with that column. To share only some search results, "
             "delete the other rows in a spreadsheet first.",
    )
    p_export_bundle.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE",
        help=f"The bundle file to write. The {BUNDLE_SUFFIX} suffix is added if "
             "missing, and missing parent directories are created.",
    )
    p_export_bundle.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Leave the vectors out, so the recipient computes them on import. "
             "It saves little space — audio is most of a bundle, vectors about "
             "5-10% — so it is mainly for a recipient whose models differ "
             "anyway, who could not reuse them.",
    )
    p_export_bundle.set_defaults(func=cmd_export_bundle)

    p_import_bundle = _command(
        subparsers,
        "import-bundle",
        panel="Sharing",
        summary="Load a .btz bundle shared from another library.",
        description=(
            "Adds the bundle's items with their audio and metadata. A recording "
            "this library already holds is skipped, so importing the same "
            "bundle twice is harmless. Embeddings are reused when their model "
            "fingerprint matches, and computed otherwise, so every imported "
            "item is searchable at once.\n\n"
            "A bundle from someone else is treated as untrusted: its paths, "
            "sizes and audio digests are checked and every field is validated "
            "as if typed in. Exits with status 1 if any entry failed."
        ),
        examples=(
            "  biotimbre import-bundle guyalna.btz --dry-run\n"
            "  biotimbre import-bundle guyalna.btz\n"
            "  biotimbre import-bundle batch.btz --species keep-incoming"
        ),
    )
    p_import_bundle.add_argument("bundle", help=f"The {BUNDLE_SUFFIX} file to load.")
    _add_import_options(p_import_bundle, "bundle")
    p_import_bundle.set_defaults(func=cmd_import_bundle)

    p_import_library = _command(
        subparsers,
        "import-library",
        panel="Sharing",
        summary="Add every item of another library on this computer to this one.",
        description=(
            "Works as if the other library had exported all its items to a "
            "bundle, with the same rules and options as import-bundle, but "
            "without writing the bundle. The other library is opened "
            "read-only and not changed. Its items with no audio on disk are "
            "left out and listed. Repeating the import is safe: what is "
            "already here is skipped.\n\n"
            "The other library must be at the current schema version; open an "
            "older one with this biotimbre once to upgrade it."
        ),
        examples=(
            "  biotimbre import-library ~/field-2024 --dry-run\n"
            "  biotimbre import-library ~/field-2024"
        ),
    )
    p_import_library.add_argument(
        "library",
        help="The other library's directory: the one holding its library.db.",
    )
    _add_import_options(p_import_library, "library")
    p_import_library.set_defaults(func=cmd_import_library)

    # ── Library ──

    p_stats = _command(
        subparsers,
        "stats",
        panel="Library",
        summary="Show the library's name and totals.",
        description=(
            "The library's name and UUID; how many items and biotimbres it "
            "holds; the embedding speeds in use; each model's vector count and "
            "dimension; and how many items carry each taxon tag."
        ),
        examples="  biotimbre stats",
    )
    p_stats.set_defaults(func=cmd_stats)

    # ── Interface ──

    p_gui = _command(
        subparsers,
        "gui",
        panel="Interface",
        summary="Start the web interface in a browser.",
        description=(
            "Serves the library and search pages on a local web server and "
            "opens them in the default browser. Stop it with Ctrl+C. Needs the "
            "gui extra: uv tool install \"biotimbre[gui]\".\n\n"
            "Without --data-dir the interface asks which library to open, "
            "offering the one the config file names, rather than creating one "
            "where it was started."
        ),
        examples=(
            "  biotimbre gui\n"
            "  biotimbre --data-dir ~/my-library gui --no-browser --port 8090\n"
            "  biotimbre --data-dir ~/my-library gui --search-only --host 0.0.0.0"
        ),
    )
    server = p_gui.add_argument_group("Server")
    server.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to listen on. The default answers this computer only. "
             "0.0.0.0 lets other machines on the network reach it — with no "
             "login, so pair it with --search-only.",
    )
    server.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on. Change it if 8080 is taken or to run two "
             "libraries at once.",
    )
    server.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window; open the printed address yourself. "
             "For a server or a remote machine.",
    )
    server.add_argument(
        "--reload",
        action="store_true",
        help="Restart the server when biotimbre's source files change. For "
             "development only.",
    )
    page = p_gui.add_argument_group("Pages")
    page.add_argument(
        "--no-map",
        action="store_true",
        help="Hide the location map for this run. Its basemap tiles are fetched "
             "from OpenStreetMap as a page is drawn, the only thing the "
             "interface fetches from the network on its own; everything else "
             "works offline. Location filters still work. To turn it off for "
             "good, set map.enabled = false in the config file.",
    )
    page.add_argument(
        "--search-only",
        action="store_true",
        help="Serve the search page alone, with the library opened read-only: "
             "no library page, no library chooser, nothing that can change what "
             "is stored. For letting other people search a library without "
             "being able to change it. Needs --data-dir or library.data_dir in "
             "the config file.",
    )
    p_gui.set_defaults(func=cmd_gui)

    return parser


def _enable_faulthandler() -> None:
    """Print a traceback if a native library crashes the process.

    Without it, a crash inside an extension module (onnxruntime with an old
    Visual C++ runtime, say) ends the process with no word on the console.
    Left alone when something, such as pytest, has already turned it on, or
    when stderr is not a real file and there is nowhere to write it.
    """
    if faulthandler.is_enabled():
        return
    try:
        faulthandler.enable()
    except (AttributeError, OSError, ValueError, RuntimeError):
        pass


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list. ``None`` uses ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    _enable_faulthandler()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (BiotimbreError, FileNotFoundError) as exc:
        print_error(str(exc))
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
