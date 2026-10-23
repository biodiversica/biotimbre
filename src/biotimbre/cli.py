"""Command-line interface.

A thin frontend over :class:`~biotimbre.service.LibraryService`: it parses
arguments, formats output, and holds no library logic of its own (PLAN.md D5).
Output is English only for now; the translated interface arrives with the GUI in
phase 8.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from biotimbre.config import load_settings
from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import FilteredBand, ItemDraft, Location, SampleWindow
from biotimbre.embeddings import audio as audio_io
from biotimbre.io.bundle import BUNDLE_SUFFIX
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
        print(
            f"error: {args.directory} is a shared bundle. Load it with:\n"
            f"  biotimbre import-bundle {args.directory}",
            file=sys.stderr,
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
            print(f"warning: {audio_io.over_range_message(name, peak)}", file=sys.stderr)
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
            print(
                f"warning: {len(external.missing_audio)} item(s) have no audio "
                f"on disk and are left out: {ids}",
                file=sys.stderr,
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
        print(
            f"error: the GUI needs the 'gui' extra: uv sync --extra gui ({exc})",
            file=sys.stderr,
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

    if settings.search_only and not settings.data_dir_chosen:
        # The chooser is what a run without a library falls back on, and a
        # search-only run does not have one: it may not open a library, let
        # alone create one. So the library has to be named here.
        print(
            "error: --search-only needs a library to search. Pass --data-dir, "
            "or set library.data_dir in the config file.",
            file=sys.stderr,
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
    print(f"Serving on http://{args.host}:{args.port}")
    run_ui(
        settings,
        host=args.host,
        port=args.port,
        show=not args.no_browser,
        reload=args.reload,
        overrides=overrides,
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


def _add_import_options(parser: argparse.ArgumentParser, source: str) -> None:
    """The options import-bundle and import-library share.

    Args:
        parser: The subcommand's parser.
        source: What is being imported, for the help text.
    """
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Say what would happen and change nothing.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help=f"Do not compute what the {source} did not supply; leave it for "
             "`backfill`.",
    )
    parser.add_argument(
        "--distrust-embeddings",
        action="store_true",
        help=f"Ignore the {source}'s vectors and recompute from its audio.",
    )
    parser.add_argument(
        "--duplicates",
        choices=("skip", "add"),
        default="skip",
        help="What to do with a recording this library already holds under "
             "another biotimbre. Default: skip.",
    )
    parser.add_argument(
        "--species",
        choices=("join-local", "keep-incoming"),
        default="join-local",
        help="What to do when an incoming species already has a biotimbre "
             "here. Default: join-local, one species one biotimbre.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="biotimbre",
        description="Local library of animal vocalizations with similarity search.",
    )
    parser.add_argument("--config", help="Path to a biotimbre.toml config file.")
    parser.add_argument("--data-dir", help="Override the library data directory.")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_ID",
        help="Override the active models for this invocation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init", help="Create the library.")
    p_init.add_argument(
        "--name",
        help="Set (or change) the library's name. Required the first time.",
    )
    p_init.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not download or register models.",
    )
    p_init.set_defaults(func=cmd_init)

    p_add = subparsers.add_parser("add", help="Add one sample.")
    p_add.add_argument("audio", help="Audio file to add.")
    p_add.add_argument("--sidecar", help="Metadata sidecar file.")
    p_add.add_argument("--biotimbre-id", help="Join an existing biotimbre.")
    p_add.add_argument("--date", help="Recording date, YYYY-MM-DD.")
    p_add.add_argument("--time", help="Recording time, HH:MM:SS.")
    # Scalar rather than a "lat,lon" string: argparse treats a value starting
    # with '-' as an option, so "--location -26,-48" cannot work, and every
    # southern-hemisphere latitude is negative. The comma forms remain the
    # sidecar syntax, where there is no such ambiguity.
    p_add.add_argument("--latitude", type=float, help="Recording latitude.")
    p_add.add_argument("--longitude", type=float, help="Recording longitude.")
    p_add.add_argument(
        "--window-start",
        type=float,
        help="Optional sample start, in seconds within the full recording.",
    )
    p_add.add_argument(
        "--window-end",
        type=float,
        help="Optional sample end, in seconds within the full recording.",
    )
    p_add.add_argument("--taxa", nargs="+", help="Taxon tags.")
    p_add.add_argument("--device", help="Recording device brand/model.")
    p_add.add_argument("--microphone", help="Microphone brand/model.")
    p_add.add_argument("--recorded-by", help="Person or organisation who recorded it.")
    p_add.add_argument(
        "--highpass", type=float, help="Highpass cutoff already applied, in Hz."
    )
    p_add.add_argument(
        "--lowpass", type=float, help="Lowpass cutoff already applied, in Hz."
    )
    p_add.add_argument(
        "--filter-type",
        help="Kind of filter used to apply the band, e.g. Butterworth.",
    )
    p_add.add_argument(
        "--filter-order", type=int, help="Order of the filter used to apply the band."
    )
    p_add.add_argument(
        "--temperature",
        type=float,
        help="Ambient temperature at recording time, in degrees Celsius.",
    )
    p_add.add_argument(
        "--humidity",
        type=float,
        help="Ambient relative humidity at recording time, in percent.",
    )
    p_add.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed factor for embedding: <1 slows down (bringing ultrasound "
             "into a model's range), >1 speeds up. Default 1.0.",
    )
    p_add.add_argument("--link", help="URL of the full recording.")
    p_add.add_argument("--notes", help="Free-text notes.")
    p_add.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Store metadata only; embed later with 'backfill'.",
    )
    p_add.set_defaults(func=cmd_add)

    p_import = subparsers.add_parser(
        "import-dir", help="Add every sidecar-paired audio file in a directory."
    )
    p_import.add_argument("directory", help="Directory to scan.")
    p_import.add_argument(
        "--no-embeddings", action="store_true", help="Store metadata only."
    )
    p_import.set_defaults(func=cmd_import)

    p_list = subparsers.add_parser("list", help="List items.")
    p_list.add_argument("--biotimbre-id", help="Restrict to one biotimbre.")
    p_list.add_argument("--limit", type=int, help="Maximum items to show.")
    p_list.set_defaults(func=cmd_list)

    p_show = subparsers.add_parser("show", help="Show one item in full.")
    p_show.add_argument("item_id", type=int)
    p_show.set_defaults(func=cmd_show)

    p_delete = subparsers.add_parser("delete", help="Delete an item.")
    p_delete.add_argument("item_id", type=int)
    p_delete.add_argument("-y", "--yes", action="store_true", help="Skip confirmation.")
    p_delete.add_argument(
        "--keep-audio", action="store_true", help="Leave the audio file on disk."
    )
    p_delete.set_defaults(func=cmd_delete)

    p_search = subparsers.add_parser("search", help="Find similar samples.")
    p_search.add_argument("audio", help="Query audio file.")
    p_search.add_argument("-k", "--top-k", type=int, help="Maximum hits.")
    p_search.add_argument(
        "--start-time",
        type=float,
        help="Start of the segment to search with, in seconds. Required when the "
             "query file is longer than the search limit (15 s by default).",
    )
    p_search.add_argument(
        "--end-time",
        type=float,
        help="End of the segment to search with, in seconds.",
    )
    p_search.add_argument("--metric", choices=("cosine", "dot", "euclidean"))
    p_search.add_argument(
        "--fusion", help="'max', 'mean', or a model id to rank by that model alone."
    )
    p_search.add_argument("--min-score", type=float, help="Discard hits below this.")
    p_search.add_argument(
        "--query-speed",
        type=float,
        help="Embed the query only at this speed instead of sweeping the "
             "library's speeds.",
    )
    p_search.add_argument("--taxa", nargs="+", help="Require these taxon tags.")
    p_search.add_argument(
        "--taxa-all", action="store_true", help="Require every listed tag, not any."
    )
    p_search.add_argument("--biotimbre-id", nargs="+", help="Restrict to biotimbres.")
    p_search.add_argument("--date-from", help="Earliest date, YYYY-MM-DD.")
    p_search.add_argument("--date-to", help="Latest date, YYYY-MM-DD.")
    p_search.add_argument("--latitude", type=float, help="Search centre latitude.")
    p_search.add_argument("--longitude", type=float, help="Search centre longitude.")
    p_search.add_argument("--radius-km", type=float, help="Search radius in km.")
    p_search.add_argument(
        "--exclude", nargs="+", type=int, help="Item IDs to omit from results."
    )
    p_search.set_defaults(func=cmd_search)

    p_models = subparsers.add_parser("models", help="List registered models.")
    p_models.set_defaults(func=cmd_models)

    p_backfill = subparsers.add_parser(
        "backfill", help="Embed items missing a model's embeddings."
    )
    p_backfill.add_argument("model_id")
    p_backfill.set_defaults(func=cmd_backfill)

    p_export = subparsers.add_parser(
        "export-npy",
        help="Export one item's embeddings: one .npy per window, zipped when "
             "there is more than one.",
    )
    p_export.add_argument("item_id", type=int)
    p_export.add_argument("model_id")
    p_export.add_argument("output", help="Output file or directory.")
    p_export.add_argument(
        "--with-audio",
        action="store_true",
        help="Also include each window's audio as WAV, as the model was given "
             "it: at the model's sample rate, speed-shifted and padded. Always "
             "writes a zip.",
    )
    p_export.set_defaults(func=cmd_export_npy)

    p_export_csv = subparsers.add_parser(
        "export-csv",
        help="Export every item's metadata as CSV: one row per item, one "
             "column per stored value.",
    )
    p_export_csv.add_argument("output", help="Output file or directory.")
    p_export_csv.set_defaults(func=cmd_export_csv)

    p_export_bundle = subparsers.add_parser(
        "export-bundle",
        help="Package items into a shareable file another library can load.",
    )
    p_export_bundle.add_argument(
        "item_id", type=int, nargs="+", help="Items to include."
    )
    p_export_bundle.add_argument(
        "-o",
        "--output",
        required=True,
        help=f"Output file. The {BUNDLE_SUFFIX} suffix is added if absent.",
    )
    p_export_bundle.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Leave the vectors out. Halves the file; costs the recipient an "
             "inference pass.",
    )
    p_export_bundle.set_defaults(func=cmd_export_bundle)

    p_import_bundle = subparsers.add_parser(
        "import-bundle",
        help="Load a bundle shared by another library.",
    )
    p_import_bundle.add_argument("bundle", help=f"The {BUNDLE_SUFFIX} file.")
    _add_import_options(p_import_bundle, "bundle")
    p_import_bundle.set_defaults(func=cmd_import_bundle)

    p_import_library = subparsers.add_parser(
        "import-library",
        help="Add every item of another library on this machine to this "
             "one. The other library is not changed.",
    )
    p_import_library.add_argument(
        "library", help="The other library's directory (holding library.db)."
    )
    _add_import_options(p_import_library, "library")
    p_import_library.set_defaults(func=cmd_import_library)

    p_stats = subparsers.add_parser("stats", help="Show library totals.")
    p_stats.set_defaults(func=cmd_stats)

    p_gui = subparsers.add_parser("gui", help="Start the web interface.")
    p_gui.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Defaults to loopback; this is a local library.",
    )
    p_gui.add_argument("--port", type=int, default=8080, help="Bind port.")
    p_gui.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser window."
    )
    p_gui.add_argument(
        "--reload", action="store_true", help="Auto-reload on edits (development)."
    )
    p_gui.add_argument(
        "--no-map",
        action="store_true",
        help=(
            "Do not show the location map. Its basemap tiles are fetched from "
            "OpenStreetMap as a page is drawn; everything else works offline."
        ),
    )
    p_gui.add_argument(
        "--search-only",
        action="store_true",
        help=(
            "Serve the search page alone, with the library opened read-only. "
            "For giving other people a way to search a library without being "
            "able to change it. Requires --data-dir or a configured library."
        ),
    )
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list. ``None`` uses ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BiotimbreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
