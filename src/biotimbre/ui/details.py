"""The item detail dialog, shared by the library and the search results.

One dialog wherever an item is found, so a result shows exactly what the
library would — the full picture, the player and every stored value — rather
than a second, drifting summary of it. Only the actions differ: the library
adds editing and deleting, search does not.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from biotimbre.domain.errors import BiotimbreError
from biotimbre.domain.item import LibraryItem
from biotimbre.i18n import Translator
from biotimbre.ui.exports import export_button
from biotimbre.ui.playback import item_player
from biotimbre.ui.state import get_state
from biotimbre.ui.visuals import ViewChoice, audio_view, view_toggle

#: Spectrogram render size for the detail dialog. Taller than a list thumbnail:
#: this is the one place a sample is meant to be read, not just recognised. The
#: image is scaled to the dialog's width, so it is the ratio that decides how
#: tall it ends up on screen.
DETAIL_WIDTH = 900
DETAIL_HEIGHT = 360

#: On-screen height of the detail view, so a waveform and a spectrogram of the
#: same item occupy the same place in the dialog.
DETAIL_VIEW_HEIGHT = 290


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
            f"{item.sample_window.display} {t('common.seconds')}"
            if item.sample_window
            else t("common.none"),
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


async def open_detail(
    t: Translator,
    item_id: int,
    *,
    actions: Callable[[LibraryItem, ui.dialog], None] | None = None,
    on_missing: Callable[[], None] | None = None,
) -> None:
    """Show one item: its picture, a player, every stored value and exports.

    Read-only unless the caller adds to it. The library passes its edit and
    delete buttons; a search result opens the same dialog without them, since
    changing the library is the library's business.

    Args:
        t: Active translator.
        item_id: The item to show.
        actions: Draws further buttons into the dialog's action row, given
            the item and the dialog. ``None`` adds nothing.
        on_missing: Called if the item has gone, e.g. deleted from another tab,
            for the caller to refresh whatever still lists it.
    """
    state = get_state()
    try:
        item = await state.run(state.service.get_item, item_id)
        counts = await state.run(state.service.repo.embedding_counts, item_id)
        label = await state.run(state.service.get_biotimbre_label, item.biotimbre_id)
        # The waveform is drawn from the audio itself, not from a rendered image.
        audio_path, _ = await state.run(state.service.get_audio, item_id)
    except BiotimbreError:
        ui.notify(t("error.not_found"), type="warning", position="top")
        if on_missing is not None:
            on_missing()
        return

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl p-4 gap-3 bg-card"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(t("admin.detail", item_id=item.item_id)).classes(
                "text-lg font-medium"
            )
            choice = ViewChoice()
            view_toggle(t, choice, lambda: draw_view())
        # The card is a flex column that scrolls, so anything without an
        # intrinsic height is free to shrink away: the player collapsed to
        # nothing and the picture lost most of its height. Neither may shrink.
        visual = ui.column().classes("w-full gap-0 shrink-0")

        async def draw_view() -> None:
            visual.clear()
            with visual:
                await audio_view(
                    view=choice.view,
                    spectrogram_url=state.spectrogram_url(
                        item.item_id, width=DETAIL_WIDTH, height=DETAIL_HEIGHT
                    ),
                    audio_path=audio_path,
                    start=0.0,
                    end=item.duration_s,
                    span=None,
                    height=DETAIL_VIEW_HEIGHT,
                    sample_rate=item.sample_rate,
                )

        await draw_view()
        # Heard at the speed it was embedded at, as on the add form, or as
        # recorded: a slowed item can be listened to either way.
        item_player(t, state.audio_url(item.item_id), item.speed).classes(
            "w-full shrink-0"
        )
        _metadata_rows(t, item, counts, label)

        with ui.row().classes("items-center justify-end gap-2 w-full"):
            export_button(t, item.item_id, counts)
            ui.button(t("common.close"), on_click=dialog.close).props("flat")
            if actions is not None:
                actions(item, dialog)
    dialog.open()
