"""Command-line interface.

A thin frontend over :class:`~biotimbre.service.LibraryService`: it parses
arguments, formats output, and holds no library logic of its own (PLAN.md D5).
Output is English only for now; the translated interface arrives with the GUI in
phase 8.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from biotimbre.config import load_settings
from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import FilteredBand, ItemDraft, Location, SampleWindow
from biotimbre.search.filters import SearchFilters
from biotimbre.service import IngestResult, LibraryService, SearchHit


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
        print(f"Library at {service.settings.data_dir}")
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
            result = service.add_from_sidecar(args.audio, args.sidecar)
        else:
            missing = [
                name
                for name, value in (
                    ("--time", args.time),
                    ("--latitude", args.latitude),
                    ("--longitude", args.longitude),
                    ("--window-start", args.window_start),
                    ("--window-end", args.window_end),
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
                sample_window=SampleWindow(args.window_start, args.window_end),
                biotimbre_id=args.biotimbre_id,
                taxa=args.taxa or (),
                device=args.device,
                filtered_band=FilteredBand(args.highpass, args.lowpass),
                noise_reduction=args.noise_reduction or False,
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
        print(f"{'id':>5}  {'btid':<6} {'date':<10} {'time':<8} "
              f"{'dur':>6}  {'speed':>5}  taxa")
        for item in items:
            print(
                f"{item.item_id:>5}  {item.biotimbre_id:<6} {item.date:<10} "
                f"{item.time:<8} {item.duration_s:>5.1f}s  {item.speed:>5g}  "
                f"{item.taxa_display}"
            )
        print(f"\n{len(items)} item(s) shown.")
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
            ("filtered_band", str(item.filtered_band)),
            ("noise_reduction", "yes" if item.noise_reduction else "no"),
            ("speed", f"{item.speed:g}"),
            ("sample_window", str(item.sample_window)),
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
            device=args.device,
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
    """Export one item's embeddings to a .npy file."""
    with _service(args) as service:
        out = service.export_npy(args.item_id, args.model_id, args.output)
        print(f"Wrote {out}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show library totals."""
    with _service(args) as service:
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
        help="Sample start, in seconds within the full recording.",
    )
    p_add.add_argument(
        "--window-end",
        type=float,
        help="Sample end, in seconds within the full recording.",
    )
    p_add.add_argument("--taxa", nargs="+", help="Taxon tags.")
    p_add.add_argument("--device", help="Recording device brand/model.")
    p_add.add_argument(
        "--highpass", type=float, help="Highpass cutoff already applied, in Hz."
    )
    p_add.add_argument(
        "--lowpass", type=float, help="Lowpass cutoff already applied, in Hz."
    )
    p_add.add_argument(
        "--noise-reduction", action="store_true", help="Noise reduction was applied."
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
    p_search.add_argument("--device", help="Device substring to require.")
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
        "export-npy", help="Export one item's embeddings to .npy."
    )
    p_export.add_argument("item_id", type=int)
    p_export.add_argument("model_id")
    p_export.add_argument("output", help="Output file or directory.")
    p_export.set_defaults(func=cmd_export_npy)

    p_stats = subparsers.add_parser("stats", help="Show library totals.")
    p_stats.set_defaults(func=cmd_stats)

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
