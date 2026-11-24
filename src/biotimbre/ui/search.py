"""Search page: query, filters, and ranked results with players.

Listening is the verification step — a similarity score is a claim the curator has
to check by ear (PLAN.md D8). So the query stays playable at the top of the page
while every result carries its own player, and each result's spectrogram marks the
window that actually produced the score rather than leaving it to be hunted for.

The query is prepared with the same editor the library page uses for a sample
(:mod:`biotimbre.ui.clip_editor`) — cut, filtered, normalised, heard at its speed
— and the prepared copy is what is searched with. The one difference is the 15 s
query cap (D10), which the segment picker enforces: for a longer recording the
selection is explicit and visible rather than a silent truncation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from nicegui import app, ui

from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import DEFAULT_SPEED
from biotimbre.i18n import Translator
from biotimbre.search.filters import SearchFilters
from biotimbre.search.vector_search import FUSIONS, METRICS
from biotimbre.service import SearchHit
from biotimbre.sources import inaturalist
from biotimbre.ui.clip_editor import (
    ACCEPTED,
    HANDOFF_KEY,
    SEARCH_HANDOFF_KEY,
    SEARCH_WORDING,
    ClipControls,
    ClipState,
    StagedMedia,
    Upload,
    clip_editor,
    hand_over,
    search_with_item,
    take_over,
)
from biotimbre.ui.clipboard import copy_on_click
from biotimbre.ui.coordinates import CoordinatesInput
from biotimbre.ui.details import open_detail
from biotimbre.ui import inat, maps
from biotimbre.ui.exports import SelectionBar, export_button
from biotimbre.ui.forms import input_values, set_inputs
from biotimbre.ui.layout import current_views, notify_error
from biotimbre.ui.playback import item_player
from biotimbre.ui.visuals import ViewChoice, audio_view, view_toggle
from biotimbre.ui.state import get_state
from biotimbre.ui.suggestions import library_suggestions, suggest_tags

#: Spectrogram size in a result row.
RESULT_WIDTH = 560

#: Radius assumed when an observation's coordinates fill the location filter.
#: Wide enough that a nearby recording still matches, and visible in the panel
#: for anyone who wants it tighter.
OBSERVATION_RADIUS_KM = 25.0
RESULT_HEIGHT = 110


def _fill_filters_from(t: Translator, inputs: dict, observation) -> None:
    """Put what an observation knows into the search filters.

    Filled rather than applied silently: the panel is opened so the narrowing
    is visible and can be cleared.

    The date is deliberately not among them. An observation's day says when
    that one recording was made, not when the animal sings; narrowing the
    library to it would hide every other recording of the same thing.

    Args:
        t: Active translator.
        inputs: The filter panel's inputs.
        observation: The fetched observation.
    """
    if observation.taxon:
        # The select starts with no options and grows as tags are typed, so a
        # value it has never heard of renders as nothing: it has to be offered
        # before it can be chosen.
        taxa = inputs["taxa"]
        taxa.set_options(
            sorted({*(taxa.options or []), observation.taxon}),
            value=[observation.taxon],
        )
    if observation.latitude is not None and observation.longitude is not None:
        # A radius is required with a centre, and one has to be assumed; the
        # filter is visible and editable, which is the point of filling it in.
        # It is written first because the map follows these three inputs, and
        # the centre is what completes them: an order that sets it last leaves
        # the map framing a centre it is about to reframe.
        inputs["radius_km"].value = inputs["radius_km"].value or OBSERVATION_RADIUS_KM
        inputs["location"].coordinates = (observation.latitude, observation.longitude)
    inputs["panel"].value = True


def _hit_points(t: Translator, hits) -> list[maps.MapPoint]:
    """Turn search hits into plottable points.

    The popup says which recording a dot is, in the same terms as the card
    below it — a dot the reader cannot connect to a result is just decoration.

    Args:
        t: Active translator.
        hits: What the search returned.

    Returns:
        One point per hit, in rank order.
    """
    points = []
    for rank, hit in enumerate(hits, start=1):
        item = hit.item
        parts = [f"{rank}. {item.taxa_display}", f"#{item.item_id}", item.date]
        if hit.distance_km is not None:
            parts.append(t("search.distance", km=f"{hit.distance_km:.1f}"))
        points.append(
            maps.MapPoint(
                latitude=item.location.latitude,
                longitude=item.location.longitude,
                label=" · ".join(part for part in parts if part),
                item_id=item.item_id,
            )
        )
    return points


def _embedding_counts(item_ids: list[int]) -> dict[int, dict[str, int]]:
    """Embedding counts per item, for the export menus."""
    repo = get_state().service.repo
    return {item_id: repo.embedding_counts(item_id) for item_id in item_ids}


async def _result_card(
    t: Translator,
    hit: SearchHit,
    rank: int,
    counts: dict[str, int],
    *,
    on_tick: Callable[[int, bool], None] | None = None,
) -> ui.checkbox | None:
    """One search result: score, metadata, marked spectrogram and a player.

    A hit with no per-model scores came from a metadata-only search, where
    there is no query to compare against: the score, the per-model column and
    the matched window are all omitted rather than shown as zeroes.

    Args:
        on_tick: Told ``(item_id, ticked)`` when the card's selection box
            changes. ``None`` draws no box.

    Returns:
        The selection box, if one was drawn.
    """
    state = get_state()
    item = hit.item
    span = hit.matched_span
    scored = bool(hit.per_model)
    tick: ui.checkbox | None = None

    with ui.card().classes("w-full p-3 gap-2 bg-card"):
        with ui.row().classes("items-start justify-between w-full gap-4"):
            with ui.column().classes("gap-1 grow"):
                with ui.row().classes("items-baseline gap-3"):
                    if on_tick is not None:
                        tick = ui.checkbox(
                            on_change=lambda event: on_tick(item.item_id, event.value)
                        ).props("dense").tooltip(t("export.select")).mark(
                            "select-result"
                        )
                    ui.label(f"{rank}.").classes("text-muted")
                    if scored:
                        ui.label(f"{hit.score:+.4f}").classes(
                            "text-lg font-medium tabular-nums"
                        )
                    ui.label(item.taxa_display).classes("text-sm")
                    # Shortened to stay out of the way; a click copies the
                    # whole ID, for the filter or another library.
                    copy_on_click(
                        t,
                        ui.label(item.biotimbre_id[:8])
                        .classes("font-mono text-xs px-2 py-0.5 rounded bg-chip")
                        .tooltip(t("search.copy_id", biotimbre_id=item.biotimbre_id))
                        .mark("result-biotimbre-id"),
                        item.biotimbre_id,
                    )
                with ui.row().classes("gap-3 text-xs text-muted flex-wrap"):
                    ui.label(f"#{item.item_id}")
                    ui.label(f"{item.date} {item.time}")
                    ui.label(str(item.location))
                    if item.device:
                        ui.label(item.device)
                    if hit.distance_km is not None:
                        ui.label(t("search.distance", km=f"{hit.distance_km:.1f}"))
                    if item.is_speed_shifted:
                        ui.label(
                            t("search.indexed_at_speed", speed=f"{item.speed:g}")
                        ).classes("text-notice")

            with ui.column().classes("items-end gap-1 min-w-40"):
                for model, score in sorted(hit.per_model.items()):  # empty if unscored
                    weight = (
                        "font-medium" if model == hit.best_model else "text-muted"
                    )
                    ui.label(f"{model} {score:+.4f}").classes(
                        f"text-xs tabular-nums {weight}"
                    )

        if span is not None:
            ui.label(
                t(
                    "search.matched_window",
                    start=f"{span[0]:.2f}",
                    end=f"{span[1]:.2f}",
                )
                + f"  ({t('search.best_model', model=hit.best_model)})"
            ).classes("text-xs text-muted")

        # Each result has its own view: looking at one waveform is no reason
        # to redraw every other card on the page.
        choice = ViewChoice()
        with ui.row().classes("items-center gap-3 w-full"):
            visual = ui.column().classes("gap-0").style(f"width: {RESULT_WIDTH}px")

            async def draw_view() -> None:
                visual.clear()
                with visual:
                    await audio_view(
                        view=choice.view,
                        spectrogram_url=state.spectrogram_url(
                            item.item_id, width=RESULT_WIDTH, height=RESULT_HEIGHT
                        ),
                        audio_path=state.settings.absolute_audio_path(
                            item.audio_path
                        ),
                        start=0.0,
                        end=item.duration_s,
                        span=span,
                        height=RESULT_HEIGHT,
                        # One column per pixel of the row's own width: a finer
                        # envelope would be drawn onto the same pixels, and
                        # this markup is sent once per result.
                        columns=RESULT_WIDTH,
                        sample_rate=item.sample_rate,
                    )

            await draw_view()
            # Sized from nothing rather than from its content: the row wraps,
            # and the playback caption on one line is wide enough to push the
            # whole column below the picture. It still wraps on a narrow
            # screen, where there is no room for a player beside it.
            with ui.column().classes("grow basis-0 min-w-64 gap-1"):
                # As the models heard it, with the recording itself a click away.
                item_player(t, state.audio_url(item.item_id), item.speed).classes(
                    "w-full"
                )
                with ui.row().classes("items-center gap-3"):
                    view_toggle(t, choice, draw_view).mark("result-view")
                    # The library's own dialog, without its edit and delete:
                    # everything stored about the item, read where it was found.
                    ui.button(
                        t("search.details"),
                        icon="info",
                        on_click=lambda: open_detail(t, item.item_id),
                    ).props("flat dense no-caps").mark("result-details")
                    ui.button(
                        t("search.find_similar"),
                        icon="manage_search",
                        on_click=lambda: search_with_item(t, item.item_id),
                    ).props("flat dense no-caps").tooltip(
                        t("search.find_similar_help")
                    ).mark("result-find-similar")
                    export_button(t, item.item_id, counts)
                    if item.link_to_full_recording:
                        ui.link(
                            t("admin.open_full_recording"),
                            item.link_to_full_recording,
                            new_tab=True,
                        ).classes("text-xs text-link")
    return tick


async def _download_results_csv(t: Translator, hits: list[SearchHit]) -> None:
    """Send the displayed results to the browser as one CSV."""
    state = get_state()
    try:
        data, name = await state.run(state.service.results_table, hits)
    except BiotimbreError as exc:
        notify_error(t, exc)
        return
    ui.download.content(data, name, "text/csv")


def _filters_panel(t: Translator) -> dict[str, ui.element]:
    """Render the filter inputs and return them by name."""
    inputs: dict[str, ui.element] = {}

    def dense(element):
        return element.props("dense outlined").classes("w-full")

    # The panel itself travels with the inputs, so a caller that fills filters
    # in — a fetched observation does — can open it and show its work.
    panel = ui.expansion(t("search.filters"), icon="filter_alt").classes(
        "w-full"
    ).props("dense")
    inputs["panel"] = panel
    with panel:
        ui.label(t("search.filters_hint")).classes("text-xs text-muted mb-2")
        with ui.grid(columns=2).classes("w-full gap-3"):
            inputs["taxa"] = dense(
                ui.select(
                    options=[],
                    value=[],
                    label=t("search.filter.taxa"),
                    multiple=True,
                    new_value_mode="add-unique",
                ).props("use-chips").mark("filter-taxa")
            )
            # The tags the library holds, as on the add form: a filter on a
            # tag spelt differently from the items' finds nothing.
            suggest_tags(
                inputs["taxa"], lambda text: library_suggestions("taxa", text)
            )
            with ui.column().classes("justify-center"):
                inputs["taxa_all"] = ui.switch(t("search.filter.taxa_all"))
            inputs["date_from"] = dense(
                ui.input(t("search.filter.date_from")).props('mask="####-##-##"')
            )
            inputs["date_to"] = dense(
                ui.input(t("search.filter.date_to")).props('mask="####-##-##"')
            )
            inputs["location"] = dense(
                CoordinatesInput(t("search.filter.centre"))
            )
            inputs["radius_km"] = dense(
                ui.number(
                    t("search.filter.radius"), step=1, suffix=t("common.kilometres")
                )
            )
            inputs["biotimbre_id"] = dense(
                ui.select(
                    options=[],
                    value=[],
                    label=t("search.filter.biotimbre_id"),
                    multiple=True,
                    new_value_mode="add-unique",
                ).props("use-chips").tooltip(t("search.filter.biotimbre_id_help"))
            )

        with ui.row().classes("justify-end w-full"):
            ui.button(
                t("search.filter.clear"),
                icon="filter_alt_off",
                on_click=lambda: clear_filters(inputs),
            ).props("flat dense no-caps")
    return inputs


def clear_filters(inputs: dict[str, ui.element]) -> None:
    """Empty every filter, leaving the panel as it was first drawn.

    Filters accumulate — some typed, some filled in by a fetched observation —
    and a search that finds nothing is more often over-filtered than genuinely
    empty. Undoing that one field at a time is the tedious way to find out.

    Args:
        inputs: The filter panel's inputs.
    """
    for name in ("taxa", "biotimbre_id"):
        # Multi-selects keep what has been typed into them as options; only
        # the choice is cleared, so the tags stay available.
        inputs[name].value = []
    inputs["taxa_all"].value = False
    for name in ("date_from", "date_to"):
        inputs[name].value = ""
    inputs["location"].coordinates = None
    inputs["radius_km"].value = None


def _options_panel(t: Translator) -> dict[str, ui.element]:
    """Render the scoring options and return them by name."""
    settings = get_state().settings
    inputs: dict[str, ui.element] = {}

    def dense(element):
        return element.props("dense outlined").classes("w-full")

    with ui.expansion(t("search.options"), icon="tune").classes("w-full").props(
        "dense"
    ):
        with ui.grid(columns=2).classes("w-full gap-3"):
            inputs["top_k"] = dense(
                ui.number(
                    t("search.top_k"), value=settings.default_top_k, min=1, step=1
                )
            )
            inputs["min_score"] = dense(
                ui.number(t("search.min_score"), step=0.05)
            )
            inputs["metric"] = dense(
                ui.select(
                    options=list(METRICS),
                    value=settings.default_metric,
                    label=t("search.metric"),
                )
            )
            fusion_options = {
                "max": t("search.fusion_max"),
                "mean": t("search.fusion_mean"),
                **{model: model for model in settings.active_models},
            }
            default_fusion = (
                settings.default_fusion
                if settings.default_fusion in fusion_options
                else "max"
            )
            inputs["fusion"] = dense(
                ui.select(
                    options=fusion_options,
                    value=default_fusion,
                    label=t("search.fusion"),
                )
            )
    return inputs


def _build_filters(
    inputs: dict[str, ui.element], *, query_speed: float | None
) -> SearchFilters:
    """Assemble :class:`SearchFilters` from the panel's inputs.

    Args:
        inputs: The filter panel's inputs.
        query_speed: Speed the query is embedded at. Only items indexed at
            this same speed are candidates — an item embedded at a different
            speed is not comparable, so it is filtered out rather than scored.
            ``None`` leaves the speed unconstrained, for a search with no query
            audio: there, the speed is a property of the query that does not
            exist, and filtering by it would quietly hide most of the library.

    Raises:
        ValidationError: If the values are inconsistent, e.g. a centre with no
            radius, or the centre is not readable as coordinates.
    """

    def text(name: str) -> str | None:
        value = inputs[name].value
        return value.strip() if isinstance(value, str) and value.strip() else None

    location = inputs["location"].location()
    return SearchFilters(
        taxa=tuple(inputs["taxa"].value or ()),
        taxa_match_all=bool(inputs["taxa_all"].value),
        biotimbre_ids=tuple(inputs["biotimbre_id"].value or ()),
        date_from=text("date_from"),
        date_to=text("date_to"),
        latitude=location.latitude if location else None,
        longitude=location.longitude if location else None,
        radius_km=inputs["radius_km"].value,
        speeds=() if query_speed is None else (query_speed,),
    )


async def render(t: Translator) -> None:
    """Render the whole search page.

    Args:
        t: Active translator.
    """
    state = get_state()
    cap = state.settings.max_query_duration_s
    # Prepared the way the library page prepares a sample, so what is heard
    # — cut, filtered, normalised — is exactly what is searched with.
    query = ClipState()
    media = StagedMedia()
    #: The editor of the staged query, for a search to settle and read.
    editor: dict[str, ClipControls | None] = {"controls": None}
    #: The query speed, kept from one recording to the next: it says which
    #: speed of the library to search, not anything about one file.
    query_speed = {"value": DEFAULT_SPEED}
    #: Where the query came from: an iNaturalist observation and which of
    #: its recordings, so the add form can take its metadata and credit it;
    #: or a library item, which its results leave out unless asked not to.
    origin: dict = {
        "observation": None, "sound": None, "item": None, "include_self": False,
    }
    #: The results on screen, as :func:`draw_results` takes them, so a change
    #: of language can draw them again rather than search again.
    shown_results: dict = {"value": None}

    with ui.card().classes("w-full p-4 gap-3 bg-card"):
        ui.label(t("search.upload_hint", seconds=f"{cap:g}")).classes(
            "text-xs text-muted"
        )
        ui.label(t("search.query_optional")).classes("text-xs text-muted")

        query_area = ui.column().classes("w-full gap-1")

        def clear_query() -> None:
            """Drop the staged recording, the same way the library page does.

            Results are cleared with it: they answer a query that is gone.
            """
            query.discard()
            media.stop()
            editor["controls"] = None
            origin.update(observation=None, sound=None, item=None)
            uploader.reset()
            show_pickers(True)
            query_area.clear()
            results_area.clear()
            shown_results["value"] = None
            if result_map is not None:
                # The dots answered the query that just went away.
                result_map.show(())

        def show_pickers(visible: bool) -> None:
            """Offer the ways of choosing a recording, or put them away.

            Away while one is loaded: there is one query at a time, and
            removing it is how another is chosen.
            """
            uploader.set_visibility(visible)
            if inat_panel is not None:
                inat_panel.set_picking(visible)

        async def show_query(upload: Upload) -> None:
            """Draw the editor for a staged recording, however it was staged."""
            show_pickers(False)
            def ready(controls: ClipControls) -> None:
                editor["controls"] = controls
                controls.speed.on_value_change(
                    lambda event: query_speed.update(value=event.value)
                )
                # Here, with the controls, rather than after the editor has
                # returned: that waits on the first restage, and the button
                # should not arrive a moment after everything else.
                if not state.settings.search_only:
                    ui.button(
                        t("search.to_library"),
                        icon="library_add",
                        on_click=to_library,
                    ).props("flat dense no-caps").tooltip(
                        t("search.to_library_hint")
                    ).mark("to-library")

            editor["controls"] = None
            query_area.clear()
            with query_area:
                if origin["item"] is not None:
                    await item_origin_row(origin["item"])
                await clip_editor(
                    t,
                    query,
                    upload,
                    media=media,
                    wording=SEARCH_WORDING,
                    on_clear=clear_query,
                    max_span=cap,
                    speed=query_speed["value"],
                    on_ready=ready,
                )

        async def item_origin_row(item_id: int) -> None:
            """Say which library item the query is, and whether it is a result."""
            try:
                item = await state.run(state.service.get_item, item_id)
                described = item.taxa_display
            except BiotimbreError:
                described = "?"
            with ui.row().classes("items-center gap-3 w-full"):
                ui.label(
                    t("search.query_from_item", id=item_id, taxa=described)
                ).classes("text-sm").mark("query-from-item")
                ui.checkbox(
                    t("search.include_self"),
                    value=origin["include_self"],
                    on_change=lambda event: origin.update(
                        include_self=bool(event.value)
                    ),
                ).props("dense").tooltip(t("search.include_self_help"))

        async def take_item(token: str) -> None:
            """Make a library item, sent from another page, the query."""
            clear_query()
            try:
                taken = await take_over(token, query)
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            if taken is None:
                ui.notify(t("search.item_gone"), type="warning", position="top")
                return
            upload, handoff = taken
            query_speed["value"] = handoff.speed
            origin.update(
                observation=None, sound=None, item=handoff.item_id,
                include_self=False,
            )
            await show_query(upload)

        async def to_library() -> None:
            """Take the query, as prepared, to the library page's add form.

            Not added: the form opens with everything chosen here, still
            editable, and with the metadata still to give.
            """
            controls = editor["controls"]
            if query.upload is None or controls is None:
                return
            # So the settings carried over are those in the fields, including
            # a cut-off typed and not yet committed.
            await controls.settle()
            try:
                token = await hand_over(
                    query,
                    speed=float(controls.speed.value or DEFAULT_SPEED),
                    observation=origin["observation"],
                    sound=origin["sound"],
                )
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            app.storage.user[HANDOFF_KEY] = token
            views = current_views()
            if views is not None:
                await views.show("admin")
            else:
                ui.navigate.to("/")

        async def handle_upload(event) -> None:
            # event.file is a NiceGUI FileUpload: .name plus async read()/save().
            # Streamed to disk, since a query may be a long recording from which
            # only a 15 s segment is searched.
            try:
                target = query.begin(event.file.name)
                await event.file.save(target)
                upload = await state.run(query.finish, target, event.file.name)
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            except Exception as exc:  # noqa: BLE001
                # Surfaced, not just logged: a silently inert button is how the
                # event.content/event.file API change went unnoticed.
                notify_error(t, exc)
                raise
            origin.update(observation=None, sound=None, item=None)
            await show_query(upload)

        def handle_rejected(_event) -> None:
            """Report a file the browser refused, e.g. wrong type or too large."""
            ui.notify(t("search.no_query"), type="warning", position="top")

        async def stage_observation_sound(observation, sound, on_progress) -> None:
            """Make one of an observation's recordings the query."""
            target = query.begin(f"inat-{observation.id}.wav")
            path = await state.run(
                inaturalist.download_sound, sound, target, on_progress=on_progress
            )
            # Named by the observation, and by where download_sound actually
            # wrote it, which may be a .wav beside the name it was given.
            upload = await state.run(
                query.finish, path, f"inat-{observation.id}{path.suffix}"
            )
            origin.update(observation=observation, sound=sound, item=None)
            await show_query(upload)

        uploader = ui.upload(
            label=t("search.upload"),
            on_upload=handle_upload,
            on_rejected=handle_rejected,
            auto_upload=True,
            max_files=1,
        ).props(f'accept="{ACCEPTED}" flat bordered').classes("w-full")

        inat_panel = (
            inat.fetch_panel(
                t,
                stage=stage_observation_sound,
                after=lambda observation, _sound: _fill_filters_from(
                    t, filters, observation
                ),
            )
            if state.settings.inaturalist_enabled
            else None
        )

    filters = _filters_panel(t)
    options = _options_panel(t)

    with ui.row().classes("items-center gap-3"):
        run_button = ui.button(
            t("search.run"), icon="search", on_click=lambda: run_search()
        ).props("text-color=button-text")
        spinner = ui.row().classes("items-center gap-2")
        with spinner:
            ui.spinner(size="1.2rem")
            ui.label(t("search.running")).classes("text-sm text-muted")
        spinner.set_visibility(False)

    if not state.models.ready:
        loading = ui.label(t("app.loading_models_hint")).classes(
            "text-notice text-sm"
        )
        # The page is no longer reloaded when warm-up ends, so the notice
        # takes itself away.
        loading_timer = ui.timer(1.5, lambda: settle_loading())

        def settle_loading() -> None:
            if state.models.ready:
                loading.delete()
                loading_timer.cancel()

    result_map = None
    if maps.enabled():

        def pick_centre(latitude: float, longitude: float) -> None:
            """Make the clicked point the location filter's centre.

            Filled in rather than applied: the panel opens so the narrowing is
            visible, exactly as a fetched observation's coordinates are. The
            map then zooms to the area the radius selects, so the click reads
            as choosing a region rather than as dropping a pin on a continent.
            """
            # Radius first, centre second: see :func:`_fill_filters_from`.
            filters["radius_km"].value = (
                filters["radius_km"].value or OBSERVATION_RADIUS_KM
            )
            filters["location"].coordinates = (latitude, longitude)
            filters["panel"].value = True

        result_map = maps.map_card(
            t,
            heading=t("map.heading"),
            hint=t("map.pick_hint"),
            on_pick=pick_centre,
            # The same read-only dialog as the result's Details button.
            on_open=lambda item_id: open_detail(t, item_id),
            open_label=t("map.open_item"),
        )

        def show_centre() -> None:
            """Redraw the centre and radius from whatever the filter now holds.

            The view follows too. A 25 km circle on a world map is a speck, so
            picking a centre — by clicking, by typing, or by fetching an
            observation — is also a request to see the area it selects, and a
            change of radius is a request to see the new one.
            """
            latitude, longitude = filters["location"].coordinates or (None, None)
            result_map.set_focus(
                latitude,
                longitude,
                filters["radius_km"].value,
                refit=True,
            )

        # Bound to the inputs rather than called from each place that writes
        # them: the centre is set by clicking the map, by typing, by a fetched
        # observation and by Clear filters, and all four have to agree.
        for name in ("location", "radius_km"):
            filters[name].on_value_change(show_centre)

    results_area = ui.column().classes("w-full gap-2")

    async def run_search() -> None:
        path = None
        speed = DEFAULT_SPEED
        if query.upload is not None:
            controls = editor["controls"]
            # A cut-off typed but not yet committed would otherwise be left
            # out of what is searched with. A failure has already said why.
            if controls is None or not await controls.settle():
                return
            # The prepared copy, cut to the cap: it is searched whole.
            path = query.source
            speed = float(controls.speed.value or DEFAULT_SPEED)

        try:
            search_filters = _build_filters(filters, query_speed=speed)
        except ValidationError as exc:
            notify_error(t, exc)
            return
        top_k = int(options["top_k"].value or state.settings.default_top_k)

        if path is None:
            # The speed filter is implied by the query, not asked for: without
            # a query it would silently narrow a metadata search to one speed.
            search_filters = _build_filters(filters, query_speed=None)
            if search_filters.is_empty:
                ui.notify(t("search.no_query"), type="warning", position="top")
                return
        elif origin["item"] is not None and not origin["include_self"]:
            # It would come first, scoring 1.0 against itself.
            search_filters = dataclasses.replace(
                search_filters,
                exclude_item_ids=(*search_filters.exclude_item_ids, origin["item"]),
            )

        run_button.disable()
        spinner.set_visibility(True)
        results_area.clear()
        shown_results["value"] = None
        try:
            if path is None:
                hits = await state.run(
                    state.service.filter_items, search_filters, k=top_k
                )
            else:
                hits = await state.run(
                    state.service.search,
                    path,
                    filters=search_filters,
                    k=top_k,
                    metric=options["metric"].value,
                    fusion=options["fusion"].value,
                    min_score=options["min_score"].value,
                    query_speed=speed,
                )
        except BiotimbreError as exc:
            notify_error(t, exc)
            return
        finally:
            spinner.set_visibility(False)
            run_button.enable()

        # In one trip, so the cards can offer an export without each of them
        # reaching into the database from the render pass.
        counts = await state.run(_embedding_counts, [h.item.item_id for h in hits])
        shown_results["value"] = {
            "hits": hits, "counts": counts, "scored": path is not None
        }
        await draw_results(hits, counts, scored=path is not None)

    async def draw_results(hits, counts, *, scored: bool) -> None:
        """Render the result list.

        Args:
            hits: What the search returned.
            counts: Embedding counts per item, for the export menus.
            scored: Whether the hits came from a query rather than filters.
        """
        if result_map is not None:
            result_map.show(_hit_points(t, hits))
        results_area.clear()
        with results_area:
            if not hits:
                ui.label(t("search.no_matches")).classes("text-muted")
                return
            with ui.row().classes("items-baseline gap-3"):
                ui.label(t("search.results")).classes("text-lg font-medium")
                ui.label(t("search.result_count", count=len(hits))).classes(
                    "text-xs text-muted"
                )
                # The results on screen, not the library: what this search
                # showed, in the order it showed them.
                ui.button(
                    t("search.export_csv"),
                    icon="table_view",
                    on_click=lambda: _download_results_csv(t, hits),
                ).props("flat dense no-caps").tooltip(
                    t("search.export_csv_help")
                ).mark("export-results-csv")
                select_all = ui.checkbox(
                    t("export.select_all"),
                    on_change=lambda event: tick_all(bool(event.value)),
                ).props("dense").mark("select-all-results")

            # Ticked results, for sharing several as one bundle. Drawn anew,
            # and so empty, with every search: a selection from the last
            # results cannot be exported with the new ones by mistake.
            order = [hit.item.item_id for hit in hits]
            ticked: set[int] = set()
            boxes: dict[int, ui.checkbox] = {}

            def tick(item_id: int, value: bool) -> None:
                (ticked.add if value else ticked.discard)(item_id)
                bar.update([i for i in order if i in ticked])

            def tick_all(value: bool) -> None:
                for box in boxes.values():
                    box.value = value  # each box's handler updates the bar

            def untick() -> None:
                select_all.value = False
                tick_all(False)

            bar = SelectionBar(t, on_clear=untick)
            if not scored:
                ui.label(t("search.metadata_only")).classes("text-xs text-muted")
            for rank, hit in enumerate(hits, start=1):
                box = await _result_card(
                    t, hit, rank, counts.get(hit.item.item_id, {}), on_tick=tick
                )
                if box is not None:
                    boxes[hit.item.item_id] = box
            if any(h.query_speed != 1.0 for h in hits):
                speeds = sorted({h.query_speed for h in hits if h.query_speed != 1.0})
                ui.label(
                    t(
                        "search.query_speed_note",
                        speed=", ".join(f"{s:g}" for s in speeds),
                    )
                ).classes("text-xs text-notice")

    async def snapshot() -> dict:
        """The page's work, to carry across a change of language."""
        kept = {
            "filters": input_values(filters),
            "options": input_values(options),
            "speed": query_speed["value"],
            "results": shown_results["value"],
            "query": None,
            "shown": None,
        }
        if query.upload is not None:
            speed = query_speed["value"]
            controls = editor["controls"]
            if controls is not None:
                # A cut-off typed and not yet committed counts as chosen.
                await controls.settle()
                speed = float(controls.speed.value or DEFAULT_SPEED)
            kept["query"] = await hand_over(
                query,
                speed=speed,
                observation=origin["observation"],
                sound=origin["sound"],
                item_id=origin["item"],
            )
            if inat_panel is not None:
                kept["shown"] = inat_panel.shown
        return kept

    async def restore(kept: dict) -> None:
        """Put back what :func:`snapshot` kept, on the page that replaced it."""
        set_inputs(filters, kept["filters"])
        set_inputs(options, kept["options"])
        query_speed["value"] = kept["speed"]
        if kept["query"] is not None:
            try:
                taken = await take_over(kept["query"], query)
            except BiotimbreError as exc:
                notify_error(t, exc)
                taken = None
            if taken is not None:
                upload, handoff = taken
                query_speed["value"] = handoff.speed
                origin.update(
                    observation=handoff.observation,
                    sound=handoff.sound,
                    item=handoff.item_id,
                )
                await show_query(upload)
                if kept["shown"] is not None and inat_panel is not None:
                    inat_panel.show(*kept["shown"])
        results = kept["results"]
        if results is not None:
            shown_results["value"] = results
            await draw_results(
                results["hits"], results["counts"], scored=results["scored"]
            )

    views = current_views()
    if views is not None:
        await views.keep("search", snapshot, restore)

    def take_pending_item() -> None:
        """Take up a library item another page sent here as the query."""
        try:
            token = app.storage.user.pop(SEARCH_HANDOFF_KEY, None)
        except RuntimeError:
            token = None
        if token is not None:
            # After the page is up, rather than holding it back while the
            # recording is decoded and pictured.
            ui.timer(0, lambda: take_item(token), once=True)

    take_pending_item()
    # The page outlives a trip to the library page, so an item sent from
    # there has to be picked up on the way back, not only on load.
    if views is not None:
        views.when_shown("search", take_pending_item)
