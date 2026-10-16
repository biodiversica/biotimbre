"""Admin page: add, inspect, edit and delete library items.

The list is a table so a large library stays scannable; detail, editing and
deletion happen in dialogs opened from a row. Showing a spectrogram and player per
row would not survive a few hundred items.

Adding runs inference, which takes seconds, so it is offloaded to a worker thread
and the button reports progress rather than the page appearing to hang.
"""

from __future__ import annotations

from nicegui import app, ui

from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import LibraryItem
from biotimbre.embeddings import audio as audio_io
from biotimbre.i18n import Translator
from biotimbre.sources import inaturalist
from biotimbre.ui import inat, maps
from biotimbre.ui.sections import foldable_card
from biotimbre.ui.sharing import import_card
from biotimbre.ui.suggestions import library_suggestions
from biotimbre.ui.details import open_detail
from biotimbre.ui.clip_editor import (
    ACCEPTED,
    HANDOFF_KEY,
    LIBRARY_WORDING,
    ClipControls,
    ClipState,
    StagedMedia,
    Upload,
    clip_editor,
    take_over,
)
from biotimbre.ui.forms import metadata_form
from biotimbre.ui.layout import current_views, notify_error
from biotimbre.ui.state import get_state

def _fill_metadata_from(t: Translator, form, observation) -> None:
    """Write what an observation knows into the metadata form.

    Filled, not imposed: every field stays editable, because only the person
    adding the sample knows whether iNaturalist has it right. What is left
    alone is as deliberate as what is written — an obscured location would be
    a randomised point, and a note already typed is not overwritten.

    Args:
        t: Active translator.
        form: The metadata form.
        observation: The fetched observation.
    """
    if observation.date:
        form.inputs["date"].value = observation.date
    if observation.time:
        form.inputs["time"].value = observation.time
    if observation.latitude is not None and observation.longitude is not None:
        form.inputs["location"].coordinates = (
            observation.latitude,
            observation.longitude,
        )
    if observation.taxon:
        # The select starts with no options and grows as tags are typed, so a
        # value it has never heard of renders as nothing.
        taxa = form.inputs["taxa"]
        taxa.set_options(
            sorted({*(taxa.options or []), observation.taxon}),
            value=[observation.taxon],
        )
    if observation.observer:
        form.inputs["recorded_by"].value = observation.observer

    # The observation page is the full recording this sample came out of, which
    # is what the field means — and what a trimmed segment's sample_window then
    # refers to.
    form.inputs["link"].value = observation.url

    credit = observation.credit
    notes = (form.inputs["notes"].value or "").strip()
    if credit not in notes:
        form.inputs["notes"].value = f"{notes}\n{credit}".strip()


def _fill_processing_fields(form, add_state: ClipState) -> None:
    """Describe the processing in the metadata fields it belongs in.

    A cut is what ``sample_window`` means — where this sample sits in the
    recording it came from — and a filter applied here is filtering "already
    applied to the sample" by the time it is stored. Writing them into the form
    keeps the record true to the audio without the submitter retyping what they
    just chose; the fields stay editable, since only they know whether the
    upload was itself already an excerpt.
    """
    if add_state.is_cut:
        form.inputs["window_start"].value = round(add_state.start, 2)
        form.inputs["window_end"].value = round(add_state.end, 2)
    if add_state.is_filtered:
        form.inputs["highpass"].value = add_state.highpass
        form.inputs["lowpass"].value = add_state.lowpass
        form.inputs["filter_order"].value = add_state.filter_order
        if not (form.inputs["filter_type"].value or "").strip():
            form.inputs["filter_type"].value = "Butterworth"


def _note_normalisation(t: Translator, form, add_state: ClipState) -> None:
    """Keep one line in the notes saying the stored copy was normalised.

    Notes are free text the submitter also writes in, so only the line this
    wrote is touched: replaced when the gain changes, removed when the box is
    unticked, and left out of any line the submitter typed themselves.
    """
    notes = form.inputs["notes"].value or ""
    previous = add_state.normalise_note
    lines = [
        line for line in notes.splitlines() if previous is None or line != previous
    ]
    note = None
    if add_state.normalised_gain_db is not None:
        note = t(
            "admin.normalised_note",
            target=f"{audio_io.NORMALISED_PEAK_DBFS:g}",
            gain=f"{add_state.normalised_gain_db:+.1f}",
        )
        lines.append(note)
    add_state.normalise_note = note
    updated = "\n".join(lines).strip()
    if updated != notes.strip():
        form.inputs["notes"].value = updated


def _stat_tile(label: str, value: str) -> None:
    """One labelled figure in the library summary."""
    with ui.column().classes(
        "items-start gap-0 px-4 py-2 rounded-lg bg-surface min-w-28"
    ):
        ui.label(value).classes("text-xl font-medium leading-tight")
        ui.label(label).classes("text-xs uppercase tracking-wide text-muted")


async def _summary_row(t: Translator) -> None:
    """Library totals: how many items, and how many biotimbres they form."""
    state = get_state()
    stats = await state.run(state.service.stats)

    with ui.row().classes("gap-3 flex-wrap"):
        _stat_tile(t("admin.column.id"), str(stats["items"]))
        _stat_tile(t("field.biotimbre_id"), str(stats["biotimbres"]))


# ── add ───────────────────────────────────────────────────────────────────────


def _add_card(t: Translator, on_added) -> None:
    """The add-a-sample form."""
    state = get_state()
    add_state = ClipState()

    #: Publishes the staged file, so it can be played before it is added.
    media = StagedMedia()
    #: The editor of the file now staged, for Save to settle first.
    editor: dict[str, ClipControls | None] = {"controls": None}

    with foldable_card("add", t("admin.add_heading")) as add_panel:

        preview = ui.column().classes("w-full gap-1")

        def show_no_file() -> None:
            """Put the preview back to its empty state."""
            preview.clear()
            with preview:
                ui.label(t("admin.no_file")).classes("text-sm text-muted")

        def clear_file() -> None:
            """Drop the staged file, so choosing one is not a one-way door.

            The upload element has to be reset too: it goes on showing the
            file it sent, and until this existed the only way to be rid of it
            was reloading the page.
            """
            add_state.discard()
            media.stop()
            uploader.reset()
            results.clear()
            show_no_file()

        show_no_file()

        async def handle_upload(event) -> None:
            # event.file is a NiceGUI FileUpload: .name plus async read()/save().
            # Streamed to disk rather than read into memory, since a sample can be
            # a long recording.
            try:
                target = add_state.begin(event.file.name)
                await event.file.save(target)
                upload = await state.run(add_state.finish, target, event.file.name)
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            except Exception as exc:  # noqa: BLE001
                # Anything unexpected must reach the user. A handler that only
                # logged left the button silently inert, which is how the
                # event.content/event.file API change went unnoticed.
                notify_error(t, exc)
                raise

            await show_staged(upload)

        async def show_staged(upload: Upload) -> None:
            """Draw the preview and the controls for a staged file.

            Split from the upload handler so a recording fetched from
            elsewhere arrives at exactly the same form: whatever staged it,
            what follows is one path.
            """
            preview.clear()
            with preview:

                def applied(clip: ClipState) -> None:
                    _fill_processing_fields(form, clip)
                    _note_normalisation(t, form, clip)

                editor["controls"] = None
                await clip_editor(
                    t,
                    add_state,
                    upload,
                    media=media,
                    wording=LIBRARY_WORDING,
                    on_clear=clear_file,
                    bind_speed=form.inputs["speed"],
                    on_applied=applied,
                    on_ready=lambda controls: editor.update(controls=controls),
                )

        def handle_rejected(_event) -> None:
            """Report a file the browser refused, e.g. wrong type or too large."""
            ui.notify(t("error.no_audio"), type="warning", position="top")

        # Marked because the page carries a second upload control — the bundle
        # importer below — and "the upload element" stopped being unambiguous
        # the moment it did.
        uploader = ui.upload(
            label=t("admin.upload"),
            on_upload=handle_upload,
            on_rejected=handle_rejected,
            auto_upload=True,
            max_files=1,
        ).props(f'accept="{ACCEPTED}" flat bordered').classes("w-full").mark(
            "upload-audio"
        )

        async def stage_observation_sound(observation, sound, on_progress) -> None:
            """Stage one of an observation's recordings for adding."""
            target = add_state.begin(f"inat-{observation.id}.wav")
            path = await state.run(
                inaturalist.download_sound, sound, target, on_progress=on_progress
            )
            upload = await state.run(
                add_state.finish, path, f"inat-{observation.id}{path.suffix}"
            )
            await show_staged(upload)

        if state.settings.inaturalist_enabled:
            inat.fetch_panel(
                t,
                stage=stage_observation_sound,
                after=lambda observation: _fill_metadata_from(t, form, observation),
            )

        form = metadata_form(t, map_picker=maps.enabled(), suggest=library_suggestions)
        results = ui.column().classes("w-full gap-1")

        async def save() -> None:
            form.clear_errors()
            results.clear()
            if add_state.upload is None:
                ui.notify(t("error.no_audio"), type="warning", position="top")
                return
            # A cut-off typed but not yet committed would otherwise be left
            # out of what is stored.
            controls = editor["controls"]
            if controls is not None and not await controls.settle():
                return

            try:
                draft = form.draft()
            except ValidationError as exc:
                if not form.show_error(exc):
                    notify_error(t, exc)
                return

            save_button.disable()
            spinner.set_visibility(True)
            try:
                result = await state.run(
                    state.service.add_item, add_state.source, draft
                )
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            finally:
                spinner.set_visibility(False)
                save_button.enable()

            if form.label:
                # Blank means "don't touch it": on a fresh biotimbre there is
                # nothing to clear, and on one the user joined by ID an empty
                # field should not wipe out a label set elsewhere.
                try:
                    await state.run(
                        state.service.set_biotimbre_label,
                        result.item.biotimbre_id,
                        form.label,
                    )
                except BiotimbreError as exc:
                    notify_error(t, exc)

            ui.notify(
                t(
                    "admin.saved",
                    item_id=result.item.item_id,
                    biotimbre_id=result.item.biotimbre_id,
                ),
                type="positive",
                position="top",
            )
            with results:
                _ingest_report(t, result)
            add_state.discard()
            media.stop()
            uploader.reset()
            show_no_file()
            on_added()

        with ui.row().classes("items-center gap-3 mt-2"):
            save_button = ui.button(
                t("admin.save_item"), icon="add", on_click=save
            ).props("text-color=button-text")
            spinner = ui.row().classes("items-center gap-2")
            with spinner:
                ui.spinner(size="1.2rem")
                ui.label(t("admin.adding")).classes("text-sm text-muted")
            spinner.set_visibility(False)

        async def adopt(token: str) -> None:
            """Stage a query brought over from the search page.

            Staged, not added: the form opens with the recording prepared as
            it was searched with, every choice still open to change, and the
            metadata still to give before Save.
            """
            add_panel.value = True
            results.clear()
            try:
                taken = await take_over(token, add_state)
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            if taken is None:
                ui.notify(t("admin.handoff_gone"), type="warning", position="top")
                return
            upload, handoff = taken
            form.inputs["speed"].value = handoff.speed
            if handoff.observation is not None:
                _fill_metadata_from(t, form, handoff.observation)
            ui.notify(t("admin.handoff_taken"), type="info", position="top")
            await show_staged(upload)

        def take_pending() -> None:
            token = _pending_handoff()
            if token is not None:
                # After the page is up, rather than holding it back while the
                # recording is decoded and pictured.
                with add_panel:
                    ui.timer(0, lambda: adopt(token), once=True)

        take_pending()
        # The page outlives a trip to the search page now, so a query sent
        # from there has to be picked up on the way back, not only on load.
        views = current_views()
        if views is not None:
            views.when_shown("admin", take_pending)


def _pending_handoff() -> str | None:
    """The query the search page left for the add form, taken once."""
    try:
        return app.storage.user.pop(HANDOFF_KEY, None)
    except RuntimeError:
        return None


def _ingest_report(t: Translator, result) -> None:
    """Show window counts and any advisories from an ingest."""
    counts = ", ".join(
        f"{model}: {t('admin.windows_written', count=count)}"
        for model, count in sorted(result.window_counts.items())
    )
    if counts:
        ui.label(counts).classes("text-sm text-muted")

    if result.joined_species:
        with ui.row().classes("items-start gap-2"):
            ui.icon("link", size="1.1rem")
            ui.label(
                t(
                    "admin.joined_species",
                    species=result.joined_species,
                    biotimbre_id=result.item.biotimbre_id[:8],
                )
            ).classes("text-sm text-muted").tooltip(result.item.biotimbre_id)

    if result.duplicate_of:
        with ui.row().classes("items-start gap-2 text-notice"):
            ui.icon("content_copy", size="1.1rem")
            ui.label(
                t(
                    "admin.duplicate_warning",
                    items=", ".join(str(i) for i in result.duplicate_of),
                )
            ).classes("text-sm")

    for warning in result.warnings:
        with ui.row().classes("items-start gap-2 text-notice"):
            ui.icon("graphic_eq", size="1.1rem")
            ui.label(warning).classes("text-sm")


# ── detail, edit, delete ──────────────────────────────────────────────────────


async def _open_detail(t: Translator, item_id: int, on_changed) -> None:
    """Show one item, with the library's edit and delete actions."""

    def actions(item: LibraryItem, dialog: ui.dialog) -> None:
        ui.button(
            t("common.delete"),
            icon="delete",
            on_click=lambda: _confirm_delete(t, item, dialog, on_changed),
        ).props("flat color=negative")
        ui.button(
            t("common.edit"),
            icon="edit",
            on_click=lambda: _open_edit(t, item, dialog, on_changed),
        )

    await open_detail(t, item_id, actions=actions, on_missing=on_changed)


def _confirm_delete(t: Translator, item: LibraryItem, parent, on_changed) -> None:
    """Ask before deleting, naming what will go."""
    state = get_state()

    with ui.dialog() as confirm, ui.card().classes("p-4 gap-3 max-w-md bg-card"):
        ui.label(
            t("admin.delete_confirm", item_id=item.item_id, taxa=item.taxa_display)
        ).classes("text-sm")

        async def do_delete() -> None:
            try:
                await state.run(state.service.delete_item, item.item_id)
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            confirm.close()
            parent.close()
            ui.notify(
                t("admin.deleted", item_id=item.item_id),
                type="positive",
                position="top",
            )
            on_changed()

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button(t("common.cancel"), on_click=confirm.close).props("flat")
            ui.button(t("common.delete"), on_click=do_delete).props("color=negative")
    confirm.open()


async def _open_edit(t: Translator, item: LibraryItem, parent, on_changed) -> None:
    """Edit an item's metadata, warning that a speed change re-embeds."""
    state = get_state()
    label = await state.run(state.service.get_biotimbre_label, item.biotimbre_id)

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl p-4 gap-3 bg-card"):
        ui.label(t("admin.edit_heading", item_id=item.item_id)).classes(
            "text-lg font-medium"
        )
        form = metadata_form(
            t,
            item,
            allow_new_biotimbre=False,
            biotimbre_label=label,
            suggest=library_suggestions,
        )

        warning = ui.row().classes("items-center gap-2 text-notice")
        with warning:
            ui.icon("autorenew", size="1.1rem")
            ui.label(t("admin.speed_change_warning")).classes("text-sm")
        warning.set_visibility(False)

        def check_speed() -> None:
            warning.set_visibility(form.speed != item.speed)

        form.inputs["speed"].on_value_change(lambda _: check_speed())

        async def save() -> None:
            form.clear_errors()
            try:
                draft = form.draft()
            except ValidationError as exc:
                if not form.show_error(exc):
                    notify_error(t, exc)
                return

            save_button.disable()
            try:
                result = await state.run(
                    state.service.update_item, item.item_id, draft
                )
                # A biotimbre change means the label field still shows the
                # *previous* biotimbre's label; applying it to the new one
                # would silently overwrite an unrelated group's name.
                if draft.biotimbre_id == item.biotimbre_id:
                    await state.run(
                        state.service.set_biotimbre_label,
                        result.item.biotimbre_id,
                        form.label,
                    )
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            finally:
                save_button.enable()

            dialog.close()
            parent.close()
            ui.notify(
                t("admin.updated", item_id=item.item_id),
                type="positive",
                position="top",
            )
            if result.window_counts:
                summary = ", ".join(
                    f"{model} {count}"
                    for model, count in sorted(result.window_counts.items())
                )
                ui.notify(
                    t("admin.reembedded", summary=summary),
                    type="info",
                    position="top",
                )
            on_changed()

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button(t("common.cancel"), on_click=dialog.close).props("flat")
            save_button = ui.button(
                t("common.save"), icon="save", on_click=save
            ).props("text-color=button-text")
    dialog.open()


# ── list ──────────────────────────────────────────────────────────────────────


def _item_points(items) -> list[maps.MapPoint]:
    """Turn library items into plottable points.

    Args:
        items: The items currently listed.

    Returns:
        One point per item, labelled the way its table row is.
    """
    return [
        maps.MapPoint(
            latitude=item.location.latitude,
            longitude=item.location.longitude,
            label=f"#{item.item_id} · {item.taxa_display} · {item.date}",
            item_id=item.item_id,
        )
        for item in items
    ]


async def _items_table(
    t: Translator, container, on_changed, library_map=None
) -> None:
    """Render the item list into ``container``, and plot it if there is a map.

    Args:
        t: Active translator.
        container: Where the table goes.
        on_changed: Called after an edit or a deletion.
        library_map: The page's map, fed from the same list the table is built
            from — the items are already loaded here, and a second query for
            the same rows would be pure waste.
    """
    state = get_state()
    # Everything is read before the container is emptied: an await after the
    # clear leaves the list blank for as long as it takes, and a redraw that is
    # cancelled at that point leaves it blank for good.
    items = await state.run(state.service.list_items)
    biotimbres = await state.run(state.service.repo.list_biotimbres)
    labels = {b["biotimbre_id"]: b["label"] for b in biotimbres}

    if library_map is not None:
        library_map.show(_item_points(items))

    container.clear()
    with container:
        if not items:
            ui.label(t("admin.no_items")).classes("text-muted")
            return

        ui.label(t("admin.item_count", count=len(items))).classes(
            "text-xs text-muted"
        )
        columns = [
            {"name": "item_id", "label": t("admin.column.id"), "field": "item_id",
             "sortable": True, "align": "left"},
            {"name": "biotimbre_id", "label": t("admin.column.biotimbre"),
             "field": "biotimbre_id", "sortable": True, "align": "left"},
            {"name": "biotimbre_label", "label": t("admin.column.biotimbre_label"),
             "field": "biotimbre_label", "sortable": True, "align": "left"},
            {"name": "taxa", "label": t("admin.column.taxa"), "field": "taxa",
             "align": "left"},
            {"name": "date", "label": t("admin.column.date"), "field": "date",
             "sortable": True, "align": "left"},
        ]
        rows = [
            {
                "item_id": item.item_id,
                "biotimbre_id": item.biotimbre_id[:8],
                "biotimbre_label": labels.get(item.biotimbre_id) or "",
                "taxa": item.taxa_display,
                "date": f"{item.date} {item.time}",
            }
            for item in items
        ]
        table = ui.table(
            columns=columns, rows=rows, row_key="item_id", pagination=15
        ).classes("w-full")
        table.on(
            "rowClick",
            lambda event: _open_detail(t, int(event.args[1]["item_id"]), on_changed),
        )


async def render(t: Translator) -> None:
    """Render the whole admin page.

    Args:
        t: Active translator.
    """
    # Containers are created in display order: the totals, the map and the
    # list, then the add form and the bundle importer. Each section folds.
    # The totals, the map and the list fold together: they all describe what
    # the library holds, and the totals are no use apart from the listing.
    with foldable_card("library", t("admin.library_section")):
        with ui.column().classes("w-full gap-1"):
            ui.label(t("admin.heading")).classes("text-sm font-medium")
            summary_container = ui.column().classes("w-full")
        # One map for the whole library, above the list it plots — never one
        # per row. Created here rather than further down because ``redraw``
        # closes over it and ``refresh`` can schedule one from any callback.
        # A map folded away when the page loads sizes itself when unfolded:
        # NiceGUI's Leaflet watches for becoming visible.
        library_map = None
        if maps.enabled():
            with ui.column().classes("w-full gap-1"):
                ui.label(t("map.library_hint")).classes("text-xs text-muted")
                # Opens the same dialog as clicking the item's table row.
                library_map = maps.LocationMap(
                    on_open=lambda item_id: _open_detail(t, item_id, refresh),
                    open_label=t("map.open_item"),
                )
        table_container = ui.column().classes("w-full gap-2")
    add_container = ui.column().classes("w-full")
    import_container = ui.column().classes("w-full")
    # Home for the timers ``refresh`` schedules. See below for why they cannot
    # live wherever the caller happens to be.
    schedule_anchor = ui.element("div")

    async def redraw() -> None:
        """Re-render the totals and the list after any change."""
        summary_container.clear()
        with summary_container:
            await _summary_row(t)
        await _items_table(t, table_container, refresh, library_map)

    def refresh() -> None:
        """Schedule a redraw from a synchronous callback.

        The timer is anchored on the page rather than in the caller's slot. A
        dialog's callback runs inside the list subtree, so the timer would be
        created there too, and the redraw's own ``clear()`` would then delete
        the element running it — cancelling the redraw mid-way and leaving the
        list empty until the page was reloaded.
        """
        with schedule_anchor:
            ui.timer(0.1, redraw, once=True)

    with add_container:
        _add_card(t, refresh)
    with import_container:
        import_card(t, refresh)
    await redraw()
