"""The shared iNaturalist fetch control.

Both pages ask the same question — give me observation N — and differ only in
what they do with the answer: the search page makes it the query, the library
page stages it for adding. So the field, the button, the progress bar, the
provenance line and the warnings live here, and each page passes in what to do
with what comes back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from nicegui import ui

from biotimbre.domain.errors import BiotimbreError
from biotimbre.i18n import Translator
from biotimbre.sources import inaturalist
from biotimbre.sources.inaturalist import Observation, Sound
from biotimbre.ui.layout import notify_error
from biotimbre.ui.state import get_state

#: How often the bar catches up with the download, in seconds.
_TICK_S = 0.1


def observation_report(
    t: Translator,
    observation: Observation,
    sound: Sound,
    *,
    on_sound: Callable[[Observation, Sound], Awaitable[None]],
) -> None:
    """Say where a recording came from, and what to know before trusting it.

    Args:
        t: Active translator.
        observation: The fetched observation.
        sound: The recording currently in use.
        on_sound: Called when another of its recordings is picked.
    """
    with ui.row().classes("items-center gap-2 flex-wrap"):
        ui.link(
            t(
                "inat.fetched",
                id=observation.id,
                taxon=observation.taxon or t("inat.unidentified"),
            ),
            observation.url,
            new_tab=True,
        ).classes("text-xs text-link")
        if sound.attribution:
            ui.label(sound.attribution).classes("text-xs text-muted")

    if len(observation.sounds) > 1:
        ui.select(
            {
                index: f"{t('inat.sound')} {index + 1}"
                f" — {other.license_code or 'all rights reserved'}"
                for index, other in enumerate(observation.sounds)
            },
            value=observation.sounds.index(sound),
            on_change=lambda event: on_sound(
                observation, observation.sounds[int(event.value)]
            ),
        ).props("dense outlined options-dense").classes("w-64")

    notices = []
    if observation.obscured:
        notices.append(t("inat.obscured"))
    if not sound.license_code:
        notices.append(t("inat.reserved"))
    if observation.quality_grade and observation.quality_grade != "research":
        notices.append(t("inat.needs_id"))
    for notice in notices:
        with ui.row().classes("items-start gap-2 text-notice"):
            ui.icon("info", size="1.1rem")
            ui.label(notice).classes("text-xs")


def fetch_panel(
    t: Translator,
    *,
    stage: Callable[..., Awaitable[None]],
    after: Callable[[Observation], None] | None = None,
) -> None:
    """Render the field, button, progress bar and provenance line.

    Args:
        t: Active translator.
        stage: ``async (observation, sound, on_progress)`` — takes the chosen
            recording and does whatever the page does with it. Raising a
            :class:`~biotimbre.domain.errors.BiotimbreError` reports it and
            leaves the page as it was.
        after: Called once a recording is staged, to fill in whatever the page
            fills in from an observation.
    """
    state = get_state()

    #: How far the download has got, written from the worker thread and read by
    #: a timer on the event loop. Passing bytes rather than touching the bar
    #: directly keeps element updates on one thread.
    arriving = {"received": 0, "total": 0}

    def note_progress(received: int, total: int) -> None:
        arriving["received"] = received
        arriving["total"] = total

    def show_progress() -> None:
        """Move the bar, or leave it sweeping when the size is unknown."""
        total = arriving["total"]
        if total > 0:
            progress.set_value(min(1.0, arriving["received"] / total))
            progress.props(remove="indeterminate")
        else:
            progress.props("indeterminate")

    async def use_sound(observation: Observation, sound: Sound) -> bool:
        """Stage one recording, with the bar running."""
        arriving.update(received=0, total=0)
        fetch_button.disable()
        progress_row.set_visibility(True)
        # The API call and any decode have no size to report, so the bar sweeps
        # until bytes start arriving and again while ffmpeg works.
        progress.props("indeterminate")
        ticker = ui.timer(_TICK_S, show_progress)
        try:
            await stage(observation, sound, note_progress)
        except BiotimbreError as exc:
            notify_error(t, exc)
            return False
        finally:
            ticker.delete()
            progress_row.set_visibility(False)
            fetch_button.enable()
        return True

    async def switch_sound(observation: Observation, sound: Sound) -> None:
        """Take a different recording from the same observation."""
        if await use_sound(observation, sound):
            provenance.clear()
            with provenance:
                observation_report(t, observation, sound, on_sound=switch_sound)

    async def fetch() -> None:
        """Fetch the observation named in the field."""
        provenance.clear()
        try:
            observation_id = inaturalist.parse_observation_id(field.value)
        except BiotimbreError as exc:
            notify_error(t, exc)
            return

        fetch_button.disable()
        progress_row.set_visibility(True)
        progress.props("indeterminate")
        ticker = ui.timer(_TICK_S, show_progress)
        try:
            observation = await state.run(
                inaturalist.fetch_observation, observation_id
            )
        except BiotimbreError as exc:
            notify_error(t, exc)
            return
        finally:
            ticker.delete()
            progress_row.set_visibility(False)
            fetch_button.enable()

        if not observation.sounds:
            ui.notify(
                t("inat.no_sounds", id=observation.id),
                type="warning",
                position="top",
            )
            return

        sound = observation.sounds[0]
        if not await use_sound(observation, sound):
            return
        if after is not None:
            after(observation)
        with provenance:
            observation_report(t, observation, sound, on_sound=switch_sound)

    with ui.row().classes("items-center gap-2 w-full"):
        field = ui.input(t("inat.observation")).props("dense outlined").classes("grow")
        fetch_button = ui.button(
            t("inat.fetch"), icon="cloud_download", on_click=fetch
        ).props("flat dense no-caps")
    field.on("keydown.enter", lambda _: fetch())

    progress_row = ui.row().classes("items-center gap-2 w-full")
    with progress_row:
        ui.label(t("inat.fetching")).classes("text-xs text-muted")
        progress = ui.linear_progress(
            value=0.0, show_value=False, size="6px"
        ).props("instant-feedback rounded").classes("grow")
    progress_row.set_visibility(False)

    provenance = ui.column().classes("w-full gap-1")
