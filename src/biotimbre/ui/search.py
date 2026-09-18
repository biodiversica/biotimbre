"""Search page: query, filters, and ranked results with players.

Listening is the verification step — a similarity score is a claim the curator has
to check by ear (PLAN.md D8). So the query stays playable at the top of the page
while every result carries its own player, and each result's spectrogram marks the
window that actually produced the score rather than leaving it to be hunted for.

The 15 s query cap (D10) is why the segment picker exists. For a longer recording
the selection is explicit and visible rather than a silent truncation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nicegui import ui

from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import DEFAULT_SPEED, MAX_SPEED, MIN_SPEED, FilteredBand
from biotimbre.embeddings import audio as audio_io
from biotimbre.i18n import Translator
from biotimbre.search.filters import SearchFilters
from biotimbre.search.vector_search import FUSIONS, METRICS
from biotimbre.service import SearchHit
from biotimbre.sources import inaturalist
from biotimbre.ui.admin import ACCEPTED
from biotimbre.ui import inat, maps
from biotimbre.ui.exports import export_button
from biotimbre.ui.layout import notify_error
from biotimbre.ui.spectrograms import place_span_marker
from biotimbre.ui.visuals import audio_view, view_toggle
from biotimbre.ui.state import get_state

#: Spectrogram size for the query, where the segment is chosen.
QUERY_WIDTH = 860
QUERY_HEIGHT = 180

#: Spectrogram size in a result row.
RESULT_WIDTH = 560

#: Radius assumed when an observation's coordinates fill the location filter.
#: Wide enough that a nearby recording still matches, and visible in the panel
#: for anyone who wants it tighter.
OBSERVATION_RADIUS_KM = 25.0
RESULT_HEIGHT = 110


@dataclass
class QueryState:
    """The recording currently staged for searching.

    Attributes:
        token: Registry token for the staged file.
        name: Original filename.
        properties: Decoded audio properties.
        start: Selected segment start in seconds.
        end: Selected segment end in seconds.
    """

    token: str | None = None
    name: str = ""
    properties: audio_io.AudioProperties | None = None
    start: float = 0.0
    end: float = 0.0

    @property
    def duration(self) -> float:
        """Full duration of the staged file."""
        return self.properties.duration_s if self.properties else 0.0

    @property
    def needs_segment(self) -> bool:
        """Whether the file exceeds the cap and so requires an explicit segment."""
        return self.duration > get_state().settings.max_query_duration_s

    def segment(self) -> tuple[float | None, float | None]:
        """Arguments to pass to the service.

        A file within the cap is searched whole, which keeps the cache key and the
        behaviour identical to the command line. Only a longer file sends explicit
        bounds.
        """
        if not self.needs_segment:
            return None, None
        return round(self.start, 3), round(self.end, 3)


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
        inputs["latitude"].value = observation.latitude
        inputs["longitude"].value = observation.longitude
        # A radius is required with a centre, and one has to be assumed; the
        # filter is visible and editable, which is the point of filling it in.
        inputs["radius_km"].value = inputs["radius_km"].value or OBSERVATION_RADIUS_KM
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
            )
        )
    return points


def _embedding_counts(item_ids: list[int]) -> dict[int, dict[str, int]]:
    """Embedding counts per item, for the export menus."""
    repo = get_state().service.repo
    return {item_id: repo.embedding_counts(item_id) for item_id in item_ids}


async def _result_card(
    t: Translator, hit: SearchHit, rank: int, counts: dict[str, int]
) -> None:
    """One search result: score, metadata, marked spectrogram and a player.

    A hit with no per-model scores came from a metadata-only search, where
    there is no query to compare against: the score, the per-model column and
    the matched window are all omitted rather than shown as zeroes.
    """
    state = get_state()
    item = hit.item
    span = hit.matched_span
    scored = bool(hit.per_model)

    with ui.card().classes("w-full p-3 gap-2 bg-card"):
        with ui.row().classes("items-start justify-between w-full gap-4"):
            with ui.column().classes("gap-1 grow"):
                with ui.row().classes("items-baseline gap-3"):
                    ui.label(f"{rank}.").classes("text-muted")
                    if scored:
                        ui.label(f"{hit.score:+.4f}").classes(
                            "text-lg font-medium tabular-nums"
                        )
                    ui.label(item.taxa_display).classes("text-sm")
                    ui.label(item.biotimbre_id[:8]).classes(
                        "font-mono text-xs px-2 py-0.5 rounded bg-chip"
                    ).tooltip(item.biotimbre_id)
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

        with ui.row().classes("items-center gap-3 w-full"):
            with ui.column().classes("gap-0").style(f"width: {RESULT_WIDTH}px"):
                await audio_view(
                    spectrogram_url=state.spectrogram_url(
                        item.item_id, width=RESULT_WIDTH, height=RESULT_HEIGHT
                    ),
                    audio_path=state.settings.absolute_audio_path(item.audio_path),
                    start=0.0,
                    end=item.duration_s,
                    span=span,
                    height=RESULT_HEIGHT,
                    # One column per pixel of the row's own width: a finer
                    # envelope would be drawn onto the same pixels, and this
                    # markup is sent once per result.
                    columns=RESULT_WIDTH,
                )
            with ui.column().classes("grow gap-1"):
                ui.audio(state.audio_url(item.item_id)).classes("w-full")
                with ui.row().classes("items-center gap-3"):
                    export_button(t, item.item_id, counts)
                    if item.link_to_full_recording:
                        ui.link(
                            t("admin.open_full_recording"),
                            item.link_to_full_recording,
                            new_tab=True,
                        ).classes("text-xs text-link")


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
                ).props("use-chips")
            )
            with ui.column().classes("justify-center"):
                inputs["taxa_all"] = ui.switch(t("search.filter.taxa_all"))
            inputs["date_from"] = dense(
                ui.input(t("search.filter.date_from")).props('mask="####-##-##"')
            )
            inputs["date_to"] = dense(
                ui.input(t("search.filter.date_to")).props('mask="####-##-##"')
            )
            inputs["latitude"] = dense(
                ui.number(t("search.filter.latitude"), format="%.6f", step=0.000001)
            )
            inputs["longitude"] = dense(
                ui.number(t("search.filter.longitude"), format="%.6f", step=0.000001)
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
    for name in ("latitude", "longitude", "radius_km"):
        inputs[name].value = None


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
            inputs["speed"] = dense(
                ui.number(
                    t("search.speed"),
                    value=DEFAULT_SPEED,
                    step=0.05,
                    min=MIN_SPEED,
                    max=MAX_SPEED,
                ).tooltip(t("search.speed_help"))
            )
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
            inputs["highpass"] = dense(
                ui.number(
                    t("search.highpass"), step=100, suffix=t("common.hertz")
                ).tooltip(t("search.filter_help"))
            )
            inputs["lowpass"] = dense(
                ui.number(
                    t("search.lowpass"), step=100, suffix=t("common.hertz")
                ).tooltip(t("search.filter_help"))
            )
            inputs["filter_order"] = dense(
                ui.number(
                    t("search.filter_order"), value=4, step=1, min=1
                ).tooltip(t("search.filter_order_help"))
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
        ValidationError: If the values are inconsistent, e.g. a partial location.
    """

    def text(name: str) -> str | None:
        value = inputs[name].value
        return value.strip() if isinstance(value, str) and value.strip() else None

    return SearchFilters(
        taxa=tuple(inputs["taxa"].value or ()),
        taxa_match_all=bool(inputs["taxa_all"].value),
        biotimbre_ids=tuple(inputs["biotimbre_id"].value or ()),
        date_from=text("date_from"),
        date_to=text("date_to"),
        latitude=inputs["latitude"].value,
        longitude=inputs["longitude"].value,
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
    query = QueryState()

    ui.label(t("search.heading")).classes("text-xl font-medium")

    with ui.card().classes("w-full p-4 gap-3 bg-card"):
        ui.label(t("search.query")).classes("text-sm font-medium")
        ui.label(t("search.upload_hint", seconds=f"{cap:g}")).classes(
            "text-xs text-muted"
        )
        ui.label(t("search.query_optional")).classes("text-xs text-muted")

        query_area = ui.column().classes("w-full gap-2")

        async def clear_query() -> None:
            """Drop the staged recording, the same way the library page does.

            Results are cleared with it: they answer a query that is gone.
            """
            if query.token is not None:
                state.queries.discard(query.token)
            query.token = None
            query.name = ""
            query.properties = None
            query.start = 0.0
            query.end = 0.0
            uploader.reset()
            results_area.clear()
            if result_map is not None:
                # The dots answered the query that just went away.
                result_map.show(())
            await draw_query()

        async def draw_query() -> None:
            """Redraw the query preview, player and segment picker."""
            query_area.clear()
            if query.token is None or query.properties is None:
                return

            with query_area:
                with ui.row().classes("items-center gap-2"):
                    ui.label(
                        t(
                            "search.chosen",
                            name=query.name,
                            duration=f"{query.duration:.2f}",
                            rate=query.properties.sample_rate,
                        )
                    ).classes("text-sm")
                    ui.button(
                        t("common.clear_file"), icon="close", on_click=clear_query
                    ).props("flat dense no-caps color=negative")

                with ui.row().classes("items-center justify-between w-full"):
                    span_label = ui.label().classes("text-xs text-muted")
                    view_toggle(t, lambda: redraw_views())

                def play_segment() -> None:
                    """Point the player at the chosen segment.

                    A media fragment, not a trimmed file: the browser seeks to
                    the start and stops at the end by itself, and one cached
                    file still serves every segment. A browser that ignores the
                    end simply plays on, which is what it did before.
                    """
                    source = state.query_audio_url(query.token)
                    if query.needs_segment:
                        source += f"#t={query.start:.2f},{query.end:.2f}"
                    player.set_source(source)

                # Rebuilt only when the view changes. Within a view, the
                # marker moves rather than the picture being redrawn.
                query_view = ui.column().classes("w-full gap-0")
                marker: ui.element | None = None

                async def draw_query_view() -> None:
                    nonlocal marker
                    staged = state.queries.path(query.token)
                    query_view.clear()
                    with query_view:
                        marker = await audio_view(
                            spectrogram_url=(
                                f"/media/query/{query.token}/spectrogram"
                                f"?w={QUERY_WIDTH}&h={QUERY_HEIGHT}"
                            ),
                            audio_path=staged,
                            start=0.0,
                            end=query.duration,
                            span=(query.start, query.end)
                            if query.needs_segment
                            else None,
                            height=QUERY_HEIGHT,
                        )

                await draw_query_view()

                if query.needs_segment:
                    ui.label(
                        t("search.needs_segment", seconds=f"{cap:g}")
                    ).classes("text-xs text-notice")
                    ui.label(t("search.segment_hint", seconds=f"{cap:g}")).classes(
                        "text-xs text-muted"
                    )

                    picker = ui.range(
                        min=0.0,
                        max=round(query.duration, 2),
                        step=0.05,
                        value={"min": query.start, "max": query.end},
                    ).classes("w-full")

                    def on_span(event) -> None:
                        low = float(event.value["min"])
                        high = float(event.value["max"])
                        # Enforce the cap by moving the edge the user did not
                        # just drag, so the selection stays where they aimed it.
                        if high - low > cap:
                            if abs(low - query.start) > abs(high - query.end):
                                high = low + cap
                            else:
                                low = max(0.0, high - cap)
                            picker.set_value({"min": low, "max": high})
                        query.start, query.end = low, high
                        span_label.set_text(
                            t(
                                "search.segment_of",
                                start=f"{low:.2f}",
                                end=f"{high:.2f}",
                                duration=f"{query.duration:.2f}",
                            )
                        )
                        if marker is not None:
                            place_span_marker(
                                marker, (low, high), 0.0, query.duration
                            )

                    def on_span_settled(event) -> None:
                        # The settled value comes with the event: it and the
                        # drag's own updates arrive separately, so reading the
                        # last one to have landed can lag a step behind.
                        span = event.args
                        if isinstance(span, dict) and "min" in span:
                            query.start = float(span["min"])
                            query.end = float(span["max"])
                        play_segment()

                    picker.on_value_change(on_span)
                    # Only once the drag settles: changing the source reloads
                    # the media, which would stutter on every step.
                    picker.on("change", on_span_settled)
                    span_label.set_text(
                        t(
                            "search.segment_of",
                            start=f"{query.start:.2f}",
                            end=f"{query.end:.2f}",
                            duration=f"{query.duration:.2f}",
                        )
                    )
                else:
                    span_label.set_text(
                        t("search.whole_file", duration=f"{query.duration:.2f}")
                    )

                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(t("search.play_query")).classes(
                        "text-xs text-muted"
                    )
                    player = ui.audio(state.query_audio_url(query.token)).classes(
                        "grow"
                    )
                play_segment()

        async def handle_upload(event) -> None:
            # event.file is a NiceGUI FileUpload: .name plus async read()/save().
            # Streamed to disk, since a query may be a long recording from which
            # only a 15 s segment is searched.
            try:
                token, target = state.queries.reserve(event.file.name)
                await event.file.save(target)
                properties = await state.run(audio_io.read_properties, target)
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            except Exception as exc:  # noqa: BLE001
                # Surfaced, not just logged: a silently inert button is how the
                # event.content/event.file API change went unnoticed.
                notify_error(t, exc)
                raise

            query.token = token
            query.name = event.file.name
            query.properties = properties
            query.start = 0.0
            query.end = min(properties.duration_s, cap)
            await draw_query()

        def handle_rejected(_event) -> None:
            """Report a file the browser refused, e.g. wrong type or too large."""
            ui.notify(t("search.no_query"), type="warning", position="top")

        async def stage_observation_sound(observation, sound, on_progress) -> None:
            """Make one of an observation's recordings the query."""
            token, target = state.queries.reserve(f"inat-{observation.id}.wav")
            path = await state.run(
                inaturalist.download_sound, sound, target, on_progress=on_progress
            )
            properties = await state.run(audio_io.read_properties, path)
            # download_sound may have written a .wav beside the name it was
            # given, so the store is told where the file actually is.
            state.queries.replace_path(token, path)

            query.token = token
            # The observation names it, not the staging directory: the token
            # prefix is an implementation detail of where the file sits.
            query.name = f"inat-{observation.id}{path.suffix}"
            query.properties = properties
            query.start = 0.0
            query.end = min(properties.duration_s, cap)
            await draw_query()

        uploader = ui.upload(
            label=t("search.upload"),
            on_upload=handle_upload,
            on_rejected=handle_rejected,
            auto_upload=True,
            max_files=1,
        ).props(f'accept="{ACCEPTED}" flat bordered').classes("w-full")

        if state.settings.inaturalist_enabled:
            inat.fetch_panel(
                t,
                stage=stage_observation_sound,
                after=lambda observation: _fill_filters_from(t, filters, observation),
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
        ui.label(t("app.loading_models_hint")).classes("text-notice text-sm")

    result_map = None
    if maps.enabled():

        def pick_centre(latitude: float, longitude: float) -> None:
            """Make the clicked point the location filter's centre.

            Filled in rather than applied: the panel opens so the narrowing is
            visible, exactly as a fetched observation's coordinates are.
            """
            filters["latitude"].value = round(latitude, 6)
            filters["longitude"].value = round(longitude, 6)
            filters["radius_km"].value = (
                filters["radius_km"].value or OBSERVATION_RADIUS_KM
            )
            filters["panel"].value = True

        result_map = maps.map_card(
            t,
            heading=t("map.heading"),
            hint=t("map.pick_hint"),
            on_pick=pick_centre,
        )

        def show_centre() -> None:
            """Redraw the centre and radius from whatever the filter now holds."""
            result_map.set_focus(
                filters["latitude"].value,
                filters["longitude"].value,
                filters["radius_km"].value,
            )

        # Bound to the inputs rather than called from each place that writes
        # them: the centre is set by clicking the map, by typing, by a fetched
        # observation and by Clear filters, and all four have to agree.
        for name in ("latitude", "longitude", "radius_km"):
            filters[name].on_value_change(show_centre)

    results_area = ui.column().classes("w-full gap-2")

    #: What the results area is currently showing, so a view change can redraw
    #: it without running the search again.
    shown: dict = {"hits": None, "counts": {}, "scored": True}

    async def redraw_views() -> None:
        """Redraw every picture on the page in the chosen view.

        Both toggles do the same thing: one preference, so leaving the other
        half of the page in the old view would just look broken.
        """
        if query.token is not None:
            await draw_query()
        if shown["hits"] is not None:
            await draw_results(
                shown["hits"], shown["counts"], scored=shown["scored"]
            )

    async def run_search() -> None:
        path = (
            None if query.token is None else state.queries.path(query.token)
        )

        query_speed = float(options["speed"].value or DEFAULT_SPEED)
        try:
            search_filters = _build_filters(filters, query_speed=query_speed)
            filter_band = FilteredBand.parse(
                (options["highpass"].value, options["lowpass"].value)
            )
        except ValidationError as exc:
            notify_error(t, exc)
            return
        filter_order = int(options["filter_order"].value or 4)
        top_k = int(options["top_k"].value or state.settings.default_top_k)

        if path is None:
            # The speed filter is implied by the query, not asked for: without
            # a query it would silently narrow a metadata search to one speed.
            search_filters = _build_filters(filters, query_speed=None)
            if search_filters.is_empty:
                ui.notify(t("search.no_query"), type="warning", position="top")
                return

        start, end = query.segment()
        run_button.disable()
        spinner.set_visibility(True)
        results_area.clear()
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
                    query_speed=query_speed,
                    start_time=start,
                    end_time=end,
                    filter_band=filter_band,
                    filter_order=filter_order,
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
        await draw_results(hits, counts, scored=path is not None)

    async def draw_results(hits, counts, *, scored: bool) -> None:
        """Render the result list. Redrawn whole when the view changes.

        Args:
            hits: What the search returned.
            counts: Embedding counts per item, for the export menus.
            scored: Whether the hits came from a query rather than filters.
        """
        shown.update(hits=hits, counts=counts, scored=scored)
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
                # One control for the whole list: the results are redrawn
                # together, so a toggle per card would leave the others stale.
                view_toggle(t, lambda: redraw_views())
            if not scored:
                ui.label(t("search.metadata_only")).classes("text-xs text-muted")
            else:
                ui.label(t("search.compare_hint")).classes("text-xs text-muted")
                ui.label(t("search.marker_hint")).classes("text-xs text-muted")
            for rank, hit in enumerate(hits, start=1):
                await _result_card(t, hit, rank, counts.get(hit.item.item_id, {}))
            if any(h.query_speed != 1.0 for h in hits):
                speeds = sorted({h.query_speed for h in hits if h.query_speed != 1.0})
                ui.label(
                    t(
                        "search.query_speed_note",
                        speed=", ".join(f"{s:g}" for s in speeds),
                    )
                ).classes("text-xs text-notice")
