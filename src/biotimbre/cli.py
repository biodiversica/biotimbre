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
from pathlib import Path

from biotimbre import __version__
from biotimbre.config import load_settings
from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import FilteredBand, ItemDraft, Location, SampleWindow
from biotimbre.embeddings import audio as audio_io
from biotimbre.embeddings.runtime import onnxruntime_problem
from biotimbre.io.bundle import BUNDLE_SUFFIX
from biotimbre.io.sidecar import SIDECAR_FIELDS
from biotimbre.models_registry import MODEL_SPECS
from biotimbre.richhelp import RichHelpParser, print_error, print_warning
from biotimbre.search.filters import SearchFilters
from biotimbre.service import IngestResult, LibraryService, SearchHit
from biotimbre.sharing import ImportPolicy


def _service(args: argparse.Namespace) -> LibraryService:
    """Build a service from global CLI options."""
    settings = load_settings(args.config, data_dir=args.data_dir)
    if args.models:
        settings = type(settings)(
            **{**settings.__dict__, "active_models": tuple(args.models)}
        )
    return LibraryService(settings)


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


def _print_hits(hits: list[SearchHit], *, show_all_models: bool) -> None:
    """Report search results."""
    if not hits:
        print("No matches.")
        return

    for rank, hit in enumerate(hits, start=1):
        item = hit.item
        span = hit.matched_span
        print(
            f"{rank:>3}. score {hit.score:+.4f}  item {item.item_id}  "
            f"biotimbre {item.biotimbre_id}"
        )
        print(
            f"      {item.taxa_display}  |  {item.date} {item.time}  |  "
            f"{item.location}"
        )
        if span is not None:
            print(
                f"      matched {span[0]:.2f}-{span[1]:.2f} s of "
                f"{item.duration_s:.2f} s  (best: {hit.best_model})"
            )
        if hit.distance_km is not None:
            print(f"      {hit.distance_km:.1f} km from the search centre")
        if item.is_speed_shifted:
            print(f"      indexed at speed {item.speed:g}")
        if show_all_models and len(hit.per_model) > 1:
            breakdown = "  ".join(
                f"{model}={score:+.4f}"
                for model, score in sorted(hit.per_model.items())
            )
            print(f"      {breakdown}")
        if item.device:
            print(f"      device {item.device}")
        print(f"      audio  {item.audio_path}")


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


def cmd_add(args: argparse.Namespace) -> int:
    """Add one sample, from a sidecar or from explicit flags."""
    with _service(args) as service:
        if args.sidecar or not args.date:
            result = service.add_from_sidecar(
                args.audio,
                args.sidecar,
                compute_embeddings=not args.no_embeddings,
            )
        else:
            missing = [
                name
                for name, value in (
                    ("--time", args.time),
                    ("--latitude", args.latitude),
                    ("--longitude", args.longitude),
                )
                if value is None
            ]
            if missing:
                raise ValidationError(
                    f"Missing required option(s): {', '.join(missing)}. "
                    f"Alternatively supply a sidecar file with --sidecar.",
                    "arguments",
                )
            draft = ItemDraft(
                date=args.date,
                time=args.time,
                location=Location(args.latitude, args.longitude),
                sample_window=SampleWindow.parse_optional(
                    (args.window_start, args.window_end)
                ),
                biotimbre_id=args.biotimbre_id,
                taxa=args.taxa or (),
                device=args.device,
                microphone=args.microphone,
                recorded_by=args.recorded_by,
                filtered_band=FilteredBand(args.highpass, args.lowpass),
                filter_type=args.filter_type,
                filter_order=args.filter_order,
                temperature=args.temperature,
                humidity=args.humidity,
                speed=args.speed,
                link_to_full_recording=args.link,
                notes=args.notes,
            )
            result = service.add_item(
                args.audio, draft, compute_embeddings=not args.no_embeddings
            )
        _print_ingest(result)
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
            ("link", item.link_to_full_recording or "-"),
            ("notes", item.notes or "-"),
            ("audio", item.audio_path),
            ("sha256", item.audio_sha256),
            ("duration", f"{item.duration_s:.3f} s"),
            ("sample_rate", f"{item.sample_rate} Hz"),
            ("channels", item.channels),
            ("created", item.created_at),
            ("updated", item.updated_at),
        ]
        width = max(len(str(name)) for name, _ in rows)
        for name, value in rows:
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


def cmd_search(args: argparse.Namespace) -> int:
    """Search the library with a query recording."""
    with _service(args) as service:
        filters = SearchFilters(
            taxa=tuple(args.taxa or ()),
            taxa_match_all=args.taxa_all,
            biotimbre_ids=tuple(args.biotimbre_id or ()),
            date_from=args.date_from,
            date_to=args.date_to,
            latitude=args.latitude,
            longitude=args.longitude,
            radius_km=args.radius_km,
            exclude_item_ids=tuple(args.exclude or ()),
        )
        hits = service.search(
            args.audio,
            filters=filters,
            k=args.top_k,
            metric=args.metric,
            fusion=args.fusion,
            min_score=args.min_score,
            query_speed=args.query_speed,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        # After the search, which has already refused an unusable segment, and
        # over the segment it resolved: a start alone means the cap from there.
        start, end = audio_io.resolve_query_segment(
            audio_io.read_properties(args.audio).duration_s,
            args.start_time,
            args.end_time,
            service.settings.max_query_duration_s,
        )
        peak = audio_io.over_range_peak(args.audio, start_time=start, end_time=end)
        if peak is not None:
            name = Path(args.audio).name
            print_warning(audio_io.over_range_message(name, peak))
        _print_hits(hits, show_all_models=True)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List registered models."""
    with _service(args) as service:
        models = service.list_models()
        if not models:
            print("No models registered yet. Run 'biotimbre init'.")
            return 0
        for model in models:
            print(f"{model['model_id']}")
            print(f"  source        {model['source']}/{model['filename']}")
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
    with _service(args) as service:
        path = service.export_bundle(
            args.item_id,
            args.output,
            include_embeddings=not args.no_embeddings,
        )
        size = path.stat().st_size / 1024
        print(f"Wrote {path} ({size:,.0f} KiB)")
        print(f"  {len(args.item_id)} item(s)")
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

    settings = load_settings(args.config, data_dir=args.data_dir)
    # Kept as overrides rather than only folded into ``settings``: opening a
    # library from the browser reloads the configuration, and a flag typed for
    # this run has to survive that.
    overrides: dict[str, object] = {}
    if args.models:
        overrides["active_models"] = tuple(args.models)
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
    "'location -20,-46' (latitude,longitude), 'taxa cicadidae, hemiptera', "
    "'filtered_band 2000,10000' (- for an open end). Lines starting with # or "
    "; are comments. Required: date, time, location. Optional: "
    + ", ".join(f for f in SIDECAR_FIELDS if f not in ("date", "time", "location"))
    + ". A misspelt field name is an error, not ignored."
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
             "models.active from the config file (by default all of them). "
             f"Known models: {known_models}. Since this option takes several "
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
            "Metadata comes from one of two places. With --sidecar, or when "
            "--date is not given, it is read from a sidecar file (format below) — "
            "without --sidecar, the file next to AUDIO with the same name and a "
            ".txt suffix. Otherwise it comes from the options below, "
            "where --date, --time, --latitude and --longitude are required. "
            "The two are not combined: when a sidecar is read, every metadata "
            "option is ignored, --speed included.\n\n"
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
            "      --device Audiomoth --highpass 8000 --lowpass 40000 --speed 0.25"
        ),
    )
    p_add.add_argument(
        "audio",
        help="The recording to add. It is copied into the library; the original "
             "is not moved or changed.",
    )

    source = p_add.add_argument_group("Metadata source")
    source.add_argument(
        "--sidecar",
        metavar="FILE",
        help="Read the metadata from this sidecar file rather than from the "
             "options below. Default: AUDIO's name with a .txt suffix, used "
             "whenever --date is not given.",
    )

    required = p_add.add_argument_group(
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

    identity = p_add.add_argument_group("Identity")
    identity.add_argument(
        "--taxa",
        nargs="+",
        metavar="TAG",
        help="Taxon tags, at any rank: species, genus, family, order... e.g. "
             "--taxa cicadidae hemiptera. Stored lowercase. Quote a binomial so "
             "it stays one tag: --taxa 'Guyalna bonaerensis'. Omit when the "
             "taxon is unknown.",
    )
    identity.add_argument(
        "--biotimbre-id",
        metavar="UUID",
        help="File the item under this existing biotimbre (a sound-being: one "
             "sonotype, whose species may be unknown) instead of the one the "
             "taxa would choose or a new one. Copy the full ID from "
             "`biotimbre show ITEM_ID`.",
    )

    recording = p_add.add_argument_group("Recording details")
    recording.add_argument(
        "--window-start",
        type=float,
        metavar="SECONDS",
        help="Where this sample starts within the full recording given by "
             "--link, in seconds. Give --window-end with it, or neither.",
    )
    recording.add_argument(
        "--window-end",
        type=float,
        metavar="SECONDS",
        help="Where this sample ends within the full recording, in seconds.",
    )
    recording.add_argument(
        "--link",
        metavar="URL",
        help="URL of the full recording this sample was taken from, e.g. an "
             "iNaturalist observation or an archive entry.",
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

    filtering = p_add.add_argument_group("Filtering already applied to the audio")
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

    embedding = p_add.add_argument_group("Embedding")
    embedding.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed factor applied before embedding, between 0.01 and 100. Each "
             "model hears only up to half its sample rate (16 kHz for Perch, "
             "24 kHz for BirdNET), so ultrasound — a katydid at 35 kHz, a bat — "
             "is near-silence to it. Below 1 slows the audio down and lowers its "
             "pitch: at 0.25, 40 kHz becomes 10 kHz. The stored audio and its "
             "playback are not changed, and a search finds the item from a "
             "query at natural speed. Keep one value for similar recordings.",
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

    # ── Search ──

    p_search = _command(
        subparsers,
        "search",
        panel="Search",
        summary="Find the library items that sound most like a query recording.",
        description=(
            "Embeds the query with every active model and ranks the library's "
            "items by their best-matching pair of windows. For each hit it "
            "prints the score, the item, the span of the item that matched and "
            "the model that scored best.\n\n"
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
            "  biotimbre search query.wav --fusion perch_v2 --min-score 0.6"
        ),
    )
    p_search.add_argument(
        "audio",
        help="The query recording: a short clip of the sound to look for, in any "
             "format libsndfile reads.",
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
             "the library. For a query that itself needs shifting, e.g. 0.25 "
             "for an ultrasonic query. An item stored at another speed is "
             "compared at the nearest one embedded, and each hit says which.",
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
            "For each model: where its ONNX file comes from, the sample rate it "
            "takes audio at, its window length and overlap, the size of its "
            "vectors, and how many vectors the library holds for it. A model "
            "is registered by init, or by the first command that uses it."
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
        help=f"The model to embed with. Known models: {known_models}.",
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
        help=f"Whose embeddings to export. Known models: {known_models}.",
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
            "  biotimbre export-bundle 12 -o small.btz --no-embeddings"
        ),
    )
    p_export_bundle.add_argument(
        "item_id",
        type=int,
        nargs="+",
        help="The items to include, by ID; several may be given.",
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
        help="Leave the vectors out: about half the size, at the cost of the "
             "recipient computing them on import.",
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
