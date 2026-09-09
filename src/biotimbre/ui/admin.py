"""Admin page: add, inspect, edit and delete library items.

The list is a table so a large library stays scannable; detail, editing and
deletion happen in dialogs opened from a row. Showing a spectrogram and player per
row would not survive a few hundred items.

Adding runs inference, which takes seconds, so it is offloaded to a worker thread
and the button reports progress rather than the page appearing to hang.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from nicegui import ui

from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import LibraryItem
from biotimbre.embeddings import audio as audio_io
from biotimbre.i18n import Translator
from biotimbre.models_registry import get_spec
from biotimbre.ui.forms import metadata_form
from biotimbre.ui.layout import notify_error
from biotimbre.ui.state import get_state
from biotimbre.viz import render_spectrogram

#: Accepted upload extensions, matching what soundfile can decode.
ACCEPTED = ".wav,.flac,.ogg,.aiff,.aif,.w64"

#: Spectrogram render size for the detail dialog. Taller than a list thumbnail:
#: this is the one place a sample is meant to be read, not just recognised. The
#: image is scaled to the dialog's width, so it is the ratio that decides how
#: tall it ends up on screen.
DETAIL_WIDTH = 900
DETAIL_HEIGHT = 360


@dataclass
class Upload:
    """The file currently staged for adding.

    Attributes:
        path: Temporary copy on disk.
        name: Original filename, shown to the user.
        properties: Decoded audio properties.
    """

    path: Path
    name: str
    properties: audio_io.AudioProperties


@dataclass
class AddState:
    """Mutable state of the add form."""

    upload: Upload | None = None
    directory: Path = field(
        default_factory=lambda: Path(tempfile.mkdtemp(prefix="biotimbre-upload-"))
    )

    def begin(self, name: str) -> Path:
        """Discard any previous upload and return where the new one should go.

        Split from :meth:`finish` so the caller can stream the file to disk rather
        than holding it in memory — a query or sample can be a long recording, and
        an hour at 96 kHz is hundreds of megabytes.

        Args:
            name: Original filename. Only its basename is used, so an upload name
                cannot escape the staging directory.

        Returns:
            The path to write to.
        """
        self.discard()
        return self.directory / Path(name).name

    def finish(self, path: Path, name: str) -> Upload:
        """Record a staged file after it has been written.

        Args:
            path: Where the file was written, from :meth:`begin`.
            name: Original filename, for display.

        Returns:
            The staged :class:`Upload`.

        Raises:
            ValidationError: If the file is not decodable audio.
        """
        # Read properties now so an unusable file is rejected at upload time
        # rather than after the user has filled in the whole form.
        properties = audio_io.read_properties(path)
        self.upload = Upload(path=path, name=name, properties=properties)
        return self.upload

    def replace(self, name: str, data: bytes) -> Upload:
        """Stage a file from bytes. Convenience for callers holding the content.

        Args:
            name: Original filename.
            data: File bytes.

        Returns:
            The staged :class:`Upload`.

        Raises:
            ValidationError: If the bytes are not decodable audio.
        """
        target = self.begin(name)
        target.write_bytes(data)
        return self.finish(target, name)

    def discard(self) -> None:
        """Delete the staged file, if any."""
        if self.upload is not None and self.upload.path.exists():
            self.upload.path.unlink()
        self.upload = None


def _nyquist_lines(t: Translator) -> str:
    """Describe where each model's limit falls, for a spectrogram caption."""
    limits = ", ".join(
        f"{model_id} {get_spec(model_id).nyquist_hz / 1000:g} kHz"
        for model_id in get_state().settings.active_models
    )
    return f"{t('admin.model_limits')} ({limits})"


def _spectrogram_of(path: Path, *, width: int = 860, height: int = 170) -> str:
    """Render a staged upload as an inline data URI.

    Inline rather than a route: a staged file has no item ID, and exposing a
    path-addressable endpoint for arbitrary temp files would undo the property
    that requests can only name items.
    """
    colors = get_state().settings.theme.spectrogram_colors
    return render_spectrogram(
        path, width=width, height=height, colors=colors
    ).data_uri()


def _stat_tile(label: str, value: str) -> None:
    """One labelled figure in the library summary."""
    with ui.column().classes(
        "items-start gap-0 px-4 py-2 rounded-lg bg-surface min-w-28"
    ):
        ui.label(value).classes("text-xl font-medium leading-tight")
        ui.label(label).classes("text-xs uppercase tracking-wide text-muted")


async def _summary_row(t: Translator) -> None:
    """Library totals: item and biotimbre counts, vectors per model, speeds used."""
    state = get_state()
    stats = await state.run(state.service.stats)

    with ui.row().classes("gap-3 flex-wrap"):
        _stat_tile(t("admin.column.id"), str(stats["items"]))
        _stat_tile(t("field.biotimbre_id"), str(stats["biotimbres"]))
        for model in stats["models"]:
            _stat_tile(model["model_id"], str(model["embedding_count"]))
        if stats["speeds"]:
            _stat_tile(
                t("field.speed"),
                ", ".join(f"{speed:g}" for speed in stats["speeds"]),
            )


# ── add ───────────────────────────────────────────────────────────────────────


def _add_card(t: Translator, on_added) -> None:
    """The add-a-sample form."""
    state = get_state()
    add_state = AddState()

    with ui.card().classes("w-full p-4 gap-3 bg-card"):
        ui.label(t("admin.add_heading")).classes("text-lg font-medium")

        preview = ui.column().classes("w-full gap-1")
        with preview:
            ui.label(t("admin.no_file")).classes("text-sm text-muted")

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

            props = upload.properties
            preview.clear()
            with preview:
                ui.label(
                    t(
                        "admin.file_chosen",
                        name=upload.name,
                        duration=f"{props.duration_s:.2f}",
                        rate=props.sample_rate,
                        channels=props.channels,
                    )
                ).classes("text-sm")
                try:
                    uri = await state.run(_spectrogram_of, upload.path)
                    ui.image(uri).classes("w-full rounded").props("no-spinner")
                    ui.label(_nyquist_lines(t)).classes("text-xs text-muted")
                except BiotimbreError as exc:
                    notify_error(t, exc)

        def handle_rejected(_event) -> None:
            """Report a file the browser refused, e.g. wrong type or too large."""
            ui.notify(t("error.no_audio"), type="warning", position="top")

        ui.upload(
            label=t("admin.upload"),
            on_upload=handle_upload,
            on_rejected=handle_rejected,
            auto_upload=True,
            max_files=1,
        ).props(f'accept="{ACCEPTED}" flat bordered').classes("w-full")

        form = metadata_form(t)
        results = ui.column().classes("w-full gap-1")

        async def save() -> None:
            form.clear_errors()
            results.clear()
            if add_state.upload is None:
                ui.notify(t("error.no_audio"), type="warning", position="top")
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
                    state.service.add_item, add_state.upload.path, draft
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
            preview.clear()
            with preview:
                ui.label(t("admin.no_file")).classes("text-sm text-muted")
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


def _ingest_report(t: Translator, result) -> None:
    """Show window counts and any advisories from an ingest."""
    counts = ", ".join(
        f"{model}: {t('admin.windows_written', count=count)}"
        for model, count in sorted(result.window_counts.items())
    )
    if counts:
        ui.label(counts).classes("text-sm text-muted")

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


def _metadata_rows(
    t: Translator, item: LibraryItem, counts: dict[str, int], biotimbre_label: str | None
) -> None:
    """Render an item's stored values as a definition list."""
    rows = [
        (t("field.biotimbre_id"), item.biotimbre_id),
        (t("field.biotimbre_label"), biotimbre_label or t("common.none")),
        (t("field.taxa"), item.taxa_display),
        (t("field.date"), f"{item.date} {item.time}"),
        (t("field.location"), str(item.location)),
        (t("field.device"), item.device or t("common.none")),
        (t("field.microphone"), item.microphone or t("common.none")),
        (t("field.recorded_by"), item.recorded_by or t("common.none")),
        (t("field.filtered_band"), item.filtered_band.display or t("common.none")),
        (t("field.filter_type"), item.filter_type or t("common.none")),
        (
            t("field.filter_order"),
            str(item.filter_order) if item.filter_order is not None else t("common.none"),
        ),
        (
            t("field.temperature"),
            f"{item.temperature:g} °C" if item.temperature is not None else t("common.none"),
        ),
        (
            t("field.humidity"),
            f"{item.humidity:g} %" if item.humidity is not None else t("common.none"),
        ),
        (t("field.speed"), f"{item.speed:g}"),
        (
            t("field.sample_window"),
            f"{item.sample_window.display} {t('common.seconds')}",
        ),
        (t("field.duration"), f"{item.duration_s:.3f} {t('common.seconds')}"),
        (
            t("field.sample_rate"),
            f"{item.sample_rate} {t('common.hertz')} · {item.channels} ch",
        ),
        (
            t("field.embeddings"),
            ", ".join(f"{model} {count}" for model, count in sorted(counts.items()))
            or t("admin.embeddings_none"),
        ),
        (t("field.notes"), item.notes or t("common.none")),
        (t("field.created"), item.created_at),
        (t("field.updated"), item.updated_at),
    ]
    with ui.grid(columns=2).classes("gap-x-4 gap-y-1 text-sm"):
        for label, value in rows:
            ui.label(label).classes("text-muted")
            ui.label(str(value))

        # The link is a field like any other, so it keeps its place in the list
        # rather than hiding when unset; only its value is clickable.
        ui.label(t("field.link")).classes("text-muted")
        if item.link_to_full_recording:
            ui.link(
                item.link_to_full_recording,
                item.link_to_full_recording,
                new_tab=True,
            ).classes("text-link break-all").tooltip(
                t("admin.open_full_recording")
            )
        else:
            ui.label(t("common.none"))


async def _open_detail(t: Translator, item_id: int, on_changed) -> None:
    """Show one item with its spectrogram, player and actions."""
    state = get_state()
    try:
        item = await state.run(state.service.get_item, item_id)
        counts = await state.run(state.service.repo.embedding_counts, item_id)
        label = await state.run(state.service.get_biotimbre_label, item.biotimbre_id)
    except BiotimbreError:
        ui.notify(t("error.not_found"), type="warning", position="top")
        on_changed()
        return

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl p-4 gap-3 bg-card"):
        ui.label(t("admin.detail", item_id=item.item_id)).classes(
            "text-lg font-medium"
        )
        # The card is a flex column that scrolls, so anything without an
        # intrinsic height is free to shrink away: the player collapsed to
        # nothing and the spectrogram lost most of its height. Neither may
        # shrink.
        ui.image(
            state.spectrogram_url(
                item.item_id, width=DETAIL_WIDTH, height=DETAIL_HEIGHT
            )
        ).classes("w-full rounded shrink-0").props("no-spinner")
        ui.audio(state.audio_url(item.item_id)).classes("w-full shrink-0")
        _metadata_rows(t, item, counts, label)

        with ui.row().classes("justify-end gap-2 w-full"):
            ui.button(t("common.close"), on_click=dialog.close).props("flat")
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
    dialog.open()


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
            t, item, allow_new_biotimbre=False, biotimbre_label=label
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


async def _items_table(t: Translator, container, on_changed) -> None:
    """Render the item list into ``container``."""
    state = get_state()
    # Everything is read before the container is emptied: an await after the
    # clear leaves the list blank for as long as it takes, and a redraw that is
    # cancelled at that point leaves it blank for good.
    items = await state.run(state.service.list_items)
    biotimbres = await state.run(state.service.repo.list_biotimbres)
    labels = {b["biotimbre_id"]: b["label"] for b in biotimbres}

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
            {"name": "duration", "label": t("admin.column.duration"),
             "field": "duration", "sortable": True, "align": "right"},
            {"name": "speed", "label": t("admin.column.speed"), "field": "speed",
             "sortable": True, "align": "right"},
        ]
        rows = [
            {
                "item_id": item.item_id,
                "biotimbre_id": item.biotimbre_id[:8],
                "biotimbre_label": labels.get(item.biotimbre_id) or "",
                "taxa": item.taxa_display,
                "date": f"{item.date} {item.time}",
                "duration": f"{item.duration_s:.2f}",
                "speed": f"{item.speed:g}",
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
    ui.label(t("admin.heading")).classes("text-xl font-medium")

    # Containers are created in display order: totals, the add form, then the list.
    summary_container = ui.column().classes("w-full")
    add_container = ui.column().classes("w-full")
    table_container = ui.column().classes("w-full gap-2")
    # Home for the timers ``refresh`` schedules. See below for why they cannot
    # live wherever the caller happens to be.
    schedule_anchor = ui.element("div")

    async def redraw() -> None:
        """Re-render the totals and the list after any change."""
        summary_container.clear()
        with summary_container:
            await _summary_row(t)
        await _items_table(t, table_container, refresh)

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
    await redraw()
