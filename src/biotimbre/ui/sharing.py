"""Receiving a bundle someone sent, or another whole library: look, decide, import.

Importing is not one act but a handful of decisions — whose biotimbre wins when
two libraries have catalogued the same species, whether to keep a recording that
is already here under another identity, whether to trust a stranger's vectors —
and the library has a default position on each (see :mod:`biotimbre.sharing`).
Defaults chosen for someone are still choices made on their behalf, so this
dialog puts all four in front of the reader and re-plans the moment one moves.
The list underneath is not an explanation of what the buttons mean; it is the
actual outcome, recomputed.

That is affordable because
:meth:`~biotimbre.service.LibraryService.inspect_bundle` writes nothing, and
neither does :meth:`~biotimbre.service.LibraryService.inspect_library`, which
plans importing every item of another library on this machine the same way. The
same :func:`~biotimbre.sharing.plan_import` runs again on import, against the
library as it is then, so what was shown is what happens.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Callable

from nicegui import ui

from biotimbre.domain.errors import BiotimbreError
from biotimbre.i18n import Translator
from biotimbre.io.bundle import BUNDLE_SUFFIX
from biotimbre.sharing import BundlePreview, EntryPlan, ImportPolicy, ImportReport
from biotimbre.ui.layout import notify_error
from biotimbre.ui.browser import DirectoryBrowser
from biotimbre.ui.sections import foldable_card
from biotimbre.ui.setup import recent_libraries
from biotimbre.ui.state import get_state

#: What the upload control accepts. One extension, because a bundle is one
#: thing; anything else is rejected before it reaches the reader as an error.
ACCEPTED = BUNDLE_SUFFIX

#: Icons per outcome, so the list can be read down the left edge without
#: parsing a word of it.
_ACTION_ICONS = {
    "add": "add_circle_outline",
    "skip": "block",
    "failed": "error_outline",
}


def import_card(t: Translator, on_imported) -> None:
    """The card that takes a bundle file or another library and opens the preview.

    Args:
        t: Active translator.
        on_imported: Called after items are added, to refresh the page.
    """
    state = get_state()

    with foldable_card("import", t("share.import_heading")):
        ui.label(t("share.import_hint")).classes("text-sm text-muted")

        async def handle_upload(event) -> None:
            # The file has to outlive the upload: the dialog reads it again on
            # every change of policy, and once more to import. So the staging
            # directory lives until the dialog is closed, which is what makes
            # _open_preview await it rather than just open it.
            staging = tempfile.TemporaryDirectory(prefix="biotimbre-bundle-")
            target = Path(staging.name) / Path(event.file.name).name
            try:
                await event.file.save(target)
                await _open_preview(
                    t,
                    lambda policy: state.service.inspect_bundle(
                        target, policy=policy
                    ),
                    lambda policy: state.service.import_bundle(
                        target, policy=policy
                    ),
                    on_imported,
                )
            except BiotimbreError as exc:
                notify_error(t, exc)
            except Exception as exc:  # noqa: BLE001
                # Anything unexpected still has to reach the reader; a silent
                # handler leaves the control looking inert.
                notify_error(t, exc)
                raise
            finally:
                uploader.reset()
                staging.cleanup()

        uploader = ui.upload(
            label=t("share.upload"),
            on_upload=handle_upload,
            auto_upload=True,
            max_files=1,
        ).props(f'accept="{ACCEPTED}" flat bordered').classes("w-full").mark(
            "upload-bundle"
        )

        ui.separator().classes("my-1")
        _library_controls(t, on_imported)


def _library_controls(t: Translator, on_imported) -> None:
    """Pick another library on this machine and preview importing all of it.

    The folder listing is the chooser's own (:mod:`biotimbre.ui.browser`),
    read on the server rather than through the chooser's HTTP route, which is
    withdrawn once a library is open.
    """
    state = get_state()

    ui.label(t("share.library_heading")).classes("text-base font-medium")
    ui.label(t("share.library_hint")).classes("text-sm text-muted")

    current = state.settings.data_dir.resolve()
    others = [
        path
        for path in recent_libraries()
        if Path(path).expanduser().resolve() != current
    ]

    with ui.row().classes("items-center gap-2 w-full no-wrap"):
        path_input = ui.input(t("share.library_path")).props(
            "dense outlined clearable"
        ).classes("grow").mark("library-path")
        browse_button = ui.button(
            t("share.library_browse"), icon="folder_open"
        ).props("flat dense no-caps")
        preview_button = ui.button(
            t("share.library_preview"), icon="library_add"
        ).props("dense no-caps").mark("library-preview")

    if others:
        with ui.row().classes("items-center gap-1 flex-wrap"):
            ui.label(t("setup.recent")).classes("text-xs text-muted")
            for path in others:
                ui.button(
                    path, on_click=lambda p=path: path_input.set_value(p)
                ).props("flat dense no-caps size=sm").classes("font-mono")

    def follow(data: dict) -> None:
        if data["is_library"]:
            path_input.set_value(data["path"])

    browser = DirectoryBrowser(t, follow, closable=True)
    browser.hide()

    async def browse() -> None:
        typed = (path_input.value or "").strip()
        await browser.show(typed or str(state.settings.data_dir.resolve().parent))

    async def preview() -> None:
        path = (path_input.value or "").strip()
        if not path:
            ui.notify(t("share.library_no_path"), type="warning", position="top")
            return
        preview_button.props("loading")
        try:
            external = await state.run(state.service.open_external_library, path)
        except BiotimbreError as exc:
            notify_error(t, exc)
            return
        finally:
            preview_button.props(remove="loading")
        browser.hide()
        await _open_preview(
            t,
            lambda policy: state.service.inspect_library(external, policy=policy),
            lambda policy: state.service.import_library(external, policy=policy),
            on_imported,
            origin=str(external.data_dir),
            missing_audio=external.missing_audio,
        )

    browse_button.on_click(browse)
    preview_button.on_click(preview)
    path_input.on("keydown.enter", lambda _: preview())


async def _open_preview(
    t: Translator,
    inspect: Callable[[ImportPolicy], BundlePreview],
    perform: Callable[[ImportPolicy], ImportReport],
    on_imported,
    *,
    origin: str | None = None,
    missing_audio: tuple[int, ...] = (),
) -> None:
    """Show what an import would do, and let the reader change it.

    Waits for the dialog to close before returning, so the caller can keep
    whatever ``inspect`` and ``perform`` read — a staged bundle file — for
    exactly as long as it is needed: every change of policy plans again, and
    so does the import.

    Args:
        t: Active translator.
        inspect: Plans the import under a policy, writing nothing.
        perform: Carries the import out under a policy.
        on_imported: Called after a successful import.
        origin: Where the items are coming from, for a library on this
            machine. ``None`` for a bundle, which says when it was written
            instead.
        missing_audio: Items of that library left out for want of audio.
    """
    state = get_state()
    policy = ImportPolicy()

    try:
        preview = await state.run(inspect, policy)
    except BiotimbreError as exc:
        notify_error(t, exc)
        return

    with ui.dialog() as dialog, ui.card().classes(
        "w-full max-w-4xl p-4 gap-3 bg-card"
    ):
        _bundle_header(t, preview, origin=origin)
        if missing_audio:
            ui.label(
                t(
                    "share.library_missing_audio",
                    count=len(missing_audio),
                    ids=", ".join(str(item_id) for item_id in missing_audio),
                )
            ).classes("text-sm text-notice")

        plans_container = ui.column().classes("w-full gap-2")
        footer_label = ui.label().classes("text-sm text-muted")

        async def replan() -> None:
            """Re-decide every entry under the policy as it now stands."""
            nonlocal preview
            try:
                preview = await state.run(inspect, policy)
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            _draw_plans(t, plans_container, preview)
            footer_label.set_text(
                t(
                    "share.summary_preview",
                    add=len(preview.to_add),
                    skip=len(preview.to_skip),
                )
            )
            import_button.set_enabled(bool(preview.to_add))

        async def on_policy_change(**changes) -> None:
            """Adopt a changed decision and show what it now does.

            Awaited rather than scheduled: a timer created inside an event
            handler belongs to whatever slot happens to be current, and the
            redraw's own ``clear()`` can then delete the element running it.
            """
            nonlocal policy
            policy = replace(policy, **changes)
            await replan()

        _policy_controls(t, policy, on_policy_change)
        ui.separator()
        _draw_plans(t, plans_container, preview)

        async def do_import() -> None:
            import_button.set_enabled(False)
            import_button.props("loading")
            try:
                report = await state.run(perform, policy)
            except BiotimbreError as exc:
                notify_error(t, exc)
                import_button.props(remove="loading")
                import_button.set_enabled(True)
                return

            dialog.close()
            _report_notice(t, report)
            on_imported()

        with ui.row().classes("w-full items-center justify-between pt-1"):
            footer_label.set_text(
                t(
                    "share.summary_preview",
                    add=len(preview.to_add),
                    skip=len(preview.to_skip),
                )
            )
            with ui.row().classes("gap-2"):
                ui.button(t("common.cancel"), on_click=dialog.close).props("flat")
                # Marked so a test can name it: "Import items" is the card's
                # heading, so the word alone is not unique.
                import_button = ui.button(
                    t("share.import"), on_click=do_import
                ).mark("bundle-import")
                import_button.set_enabled(bool(preview.to_add))

    await dialog


def _bundle_header(
    t: Translator, preview: BundlePreview, *, origin: str | None = None
) -> None:
    """Who sent this, when or from where, and what models their vectors came from."""
    bundle = preview.bundle
    library = bundle.source_library

    with ui.row().classes("items-baseline gap-2"):
        ui.label(
            t("share.from_library", name=library.name or t("share.unnamed_library"))
        ).classes("text-lg font-medium")
        if library.uuid:
            ui.label(f"#{library.uuid[:8]}").classes(
                "text-xs font-mono text-muted"
            ).tooltip(library.uuid)

    details = [t("share.item_count", count=len(bundle))]
    if origin is not None:
        details.append(origin)
    elif bundle.created_at:
        details.append(t("share.written", when=bundle.created_at))
    ui.label(" · ".join(details)).classes("text-sm text-muted")

    if bundle.models:
        for model_id, fingerprint in sorted(bundle.models.items()):
            ui.label(
                t(
                    "share.model_line",
                    model=model_id,
                    dim=fingerprint.embedding_dim,
                    duration=f"{fingerprint.window_duration_s:g}",
                    overlap=f"{fingerprint.window_overlap:g}",
                )
            ).classes("text-xs text-muted font-mono")
    else:
        ui.label(
            t("share.library_models_none" if origin else "share.models_none")
        ).classes("text-xs text-muted")


def _policy_controls(t: Translator, policy: ImportPolicy, on_change) -> None:
    """The four decisions, laid out so each one's effect is visible below.

    Args:
        t: Active translator.
        policy: The policy the dialog opened with, for the controls' initial
            values.
        on_change: Called with the changed field as a keyword argument.
    """
    # Each control is marked so a test can name the decision rather than its
    # wording, which is translated and free to change.
    ui.label(t("share.policy_heading")).classes("text-sm font-medium pt-1")

    with ui.row().classes("w-full gap-6 items-start flex-wrap"):
        with ui.column().classes("gap-1 min-w-64"):
            ui.switch(
                t("share.trust"),
                value=policy.trust_embeddings,
                on_change=lambda e: on_change(trust_embeddings=e.value),
            ).props("dense").mark("policy-trust")
            ui.label(t("share.trust_hint")).classes("text-xs text-muted")

        with ui.column().classes("gap-1 min-w-64"):
            ui.switch(
                t("share.compute"),
                value=policy.compute_embeddings,
                on_change=lambda e: on_change(compute_embeddings=e.value),
            ).props("dense").mark("policy-compute")
            ui.label(t("share.compute_hint")).classes("text-xs text-muted")

    with ui.row().classes("w-full gap-6 items-start flex-wrap"):
        with ui.column().classes("gap-1 min-w-64"):
            ui.select(
                {
                    "skip": t("share.duplicates_skip"),
                    "add": t("share.duplicates_add"),
                },
                value=policy.duplicate_audio,
                label=t("share.duplicates"),
                on_change=lambda e: on_change(duplicate_audio=e.value),
            ).props("dense outlined").classes("w-64").mark("policy-duplicates")
            ui.label(t("share.duplicates_hint")).classes("text-xs text-muted")

        with ui.column().classes("gap-1 min-w-64"):
            ui.select(
                {
                    "join_local": t("share.species_join"),
                    "keep_incoming": t("share.species_keep"),
                },
                value=policy.species_conflict,
                label=t("share.species"),
                on_change=lambda e: on_change(species_conflict=e.value),
            ).props("dense outlined").classes("w-64").mark("policy-species")
            ui.label(t("share.species_hint")).classes("text-xs text-muted")


def _draw_plans(t: Translator, container, preview: BundlePreview) -> None:
    """Replace the entry list with the current plan.

    Args:
        t: Active translator.
        container: The column to fill.
        preview: The plan to draw.
    """
    container.clear()
    with container:
        if not preview.plans:
            ui.label(t("share.empty_bundle")).classes("text-sm text-muted")
            return
        for plan in preview.plans:
            _plan_row(t, plan)


def _plan_row(t: Translator, plan: EntryPlan, outcome=None) -> None:
    """One entry: what happens to it, and everything that bears on that.

    Args:
        t: Active translator.
        plan: The entry's plan.
        outcome: The result of carrying it out, or ``None`` while previewing.
    """
    action = plan.action if outcome is None else outcome.action
    draft = plan.draft

    with ui.row().classes("w-full items-start gap-2"):
        ui.icon(_ACTION_ICONS.get(action, "help_outline"), size="1.2rem").classes(
            "pt-0.5 text-muted" if action == "skip" else "pt-0.5"
        )
        with ui.column().classes("gap-0.5 flex-1 min-w-0"):
            with ui.row().classes("items-baseline gap-2 flex-wrap"):
                ui.label(_action_label(t, plan, outcome)).classes(
                    "text-sm font-medium"
                )
                ui.label(draft.taxa_display).classes("text-sm")
                ui.label(f"{draft.date} {draft.time}").classes("text-xs text-muted")
                ui.label(plan.biotimbre_id[:8]).classes(
                    "text-xs font-mono text-muted"
                ).tooltip(plan.biotimbre_id)

            source = plan.provenance
            if source.item_id is not None:
                ui.label(
                    t(
                        "share.was_item",
                        item_id=source.item_id,
                        library=source.library_name or t("share.unnamed_library"),
                    )
                ).classes("text-xs text-muted")

            reason = plan.reason if outcome is None else outcome.reason
            if reason:
                ui.label(reason).classes("text-xs text-muted")

            _embedding_lines(t, plan, outcome)

            warnings = plan.warnings if outcome is None else outcome.warnings
            for warning in warnings:
                with ui.row().classes("items-start gap-1 text-notice"):
                    ui.icon("info_outline", size="1rem").classes("pt-0.5")
                    ui.label(warning).classes("text-xs")


def _action_label(t: Translator, plan: EntryPlan, outcome) -> str:
    """The verb for one entry, before or after the fact."""
    if outcome is None:
        return t("share.will_add") if plan.action == "add" else t("share.will_skip")
    if outcome.action == "add" and outcome.item is not None:
        return t("share.added", item_id=outcome.item.item_id)
    if outcome.action == "failed":
        return t("share.failed")
    return t("share.skipped")


def _embedding_lines(t: Translator, plan: EntryPlan, outcome) -> None:
    """Say what happens to each model's vectors, and why when they are refused."""
    if outcome is not None:
        for model_id, count in sorted(outcome.window_counts.items()):
            key = "share.reused" if model_id in plan.reusable else "share.computed"
            ui.label(t(key, model=model_id, count=count)).classes(
                "text-xs text-muted font-mono"
            )
    else:
        for model_id in plan.reusable:
            ui.label(
                t(
                    "share.reusable",
                    model=model_id,
                    count=len(plan.entry.embeddings[model_id]),
                )
            ).classes("text-xs text-muted font-mono")

    for model_id, why in sorted(plan.rejected.items()):
        ui.label(t("share.not_reused", model=model_id, reason=why)).classes(
            "text-xs text-muted"
        )


def _report_notice(t: Translator, report: ImportReport) -> None:
    """Say what the import did, and open the detail if anything needs reading.

    A clean import is a notification and nothing more. One that skipped or
    failed something has a reason the reader has not seen yet, so it gets a
    dialog rather than a message that disappears.

    Args:
        t: Active translator.
        report: What happened.
    """
    summary = t(
        "share.summary_done",
        added=len(report.added),
        skipped=len(report.skipped),
    )
    if not report.skipped and not report.failed:
        ui.notify(summary, type="positive", position="top")
        return

    with ui.dialog() as dialog, ui.card().classes(
        "w-full max-w-4xl p-4 gap-3 bg-card"
    ):
        ui.label(summary).classes("text-lg font-medium")
        if report.failed:
            ui.label(t("share.failed_count", count=len(report.failed))).classes(
                "text-sm text-notice"
            )
        with ui.column().classes("w-full gap-2"):
            for outcome in report.outcomes:
                _plan_row(t, outcome.plan, outcome=outcome)
        with ui.row().classes("w-full justify-end"):
            ui.button(t("common.close"), on_click=dialog.close).props("flat")
    dialog.open()
