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
from biotimbre.embeddings import audio as audio_io
from biotimbre.i18n import Translator
from biotimbre.search.filters import SearchFilters
from biotimbre.search.vector_search import FUSIONS, METRICS
from biotimbre.service import SearchHit
from biotimbre.ui.admin import ACCEPTED
from biotimbre.ui.layout import notify_error
from biotimbre.ui.state import get_state

#: Spectrogram size for the query, where the segment is chosen.
QUERY_WIDTH = 860
QUERY_HEIGHT = 180

#: Spectrogram size in a result row.
RESULT_WIDTH = 560
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


def _marked_spectrogram(
    url: str, span: tuple[float, float] | None, start: float, end: float
) -> None:
    """Show a spectrogram with an optional highlighted time span.

    The marker is a positioned overlay rather than something drawn into the PNG,
    so the same cached image serves every span — a result list re-rendered on a
    filter change reuses the images it already has.

    Args:
        url: Image source.
        span: ``(from, to)`` in seconds to highlight, or ``None``.
        start: Time at the image's left edge.
        end: Time at its right edge.
    """
    with ui.element("div").classes("relative w-full"):
        ui.image(url).classes("w-full rounded").props("no-spinner")
        if span is None or end <= start:
            return
        left = max(0.0, min(1.0, (span[0] - start) / (end - start)))
        right = max(0.0, min(1.0, (span[1] - start) / (end - start)))
        if right <= left:
            return
        theme = get_state().settings.theme
        ui.element("div").classes(
            "absolute top-0 bottom-0 border-x-2 pointer-events-none rounded"
        ).style(
            f"left:{left * 100:.2f}%; width:{(right - left) * 100:.2f}%; "
            f"border-color: {theme.marker_border}; background: {theme.marker_fill}"
        )


def _result_card(t: Translator, hit: SearchHit, rank: int) -> None:
    """One search result: score, metadata, marked spectrogram and a player."""
    state = get_state()
    item = hit.item
    span = hit.matched_span

    with ui.card().classes("w-full p-3 gap-2 bg-card"):
        with ui.row().classes("items-start justify-between w-full gap-4"):
            with ui.column().classes("gap-1 grow"):
                with ui.row().classes("items-baseline gap-3"):
                    ui.label(f"{rank}.").classes("text-muted")
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
                for model, score in sorted(hit.per_model.items()):
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
            _marked_spectrogram(
                state.spectrogram_url(
                    item.item_id, width=RESULT_WIDTH, height=RESULT_HEIGHT
                ),
                span,
                0.0,
                item.duration_s,
            )
            with ui.column().classes("grow gap-1"):
                ui.audio(state.audio_url(item.item_id)).classes("w-full")
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

    with ui.expansion(t("search.filters"), icon="filter_alt").classes(
        "w-full"
    ).props("dense"):
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
            inputs["device"] = dense(ui.input(t("search.filter.device")))
            inputs["biotimbre_id"] = dense(
                ui.select(
                    options=[],
                    value=[],
                    label=t("search.filter.biotimbre_id"),
                    multiple=True,
                    new_value_mode="add-unique",
                ).props("use-chips").tooltip(t("search.filter.biotimbre_id_help"))
            )
    return inputs


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


def _build_filters(inputs: dict[str, ui.element]) -> SearchFilters:
    """Assemble :class:`SearchFilters` from the panel's inputs.

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
        device=text("device"),
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

        query_area = ui.column().classes("w-full gap-2")
        with query_area:
            ui.label(t("search.no_query")).classes("text-sm text-muted")

        def draw_query() -> None:
            """Redraw the query preview, player and segment picker."""
            query_area.clear()
            if query.token is None or query.properties is None:
                with query_area:
                    ui.label(t("search.no_query")).classes(
                        "text-sm text-muted"
                    )
                return

            with query_area:
                ui.label(
                    t(
                        "search.chosen",
                        name=query.name,
                        duration=f"{query.duration:.2f}",
                        rate=query.properties.sample_rate,
                    )
                ).classes("text-sm")

                span_label = ui.label().classes("text-xs text-muted")

                image = ui.column().classes("w-full")

                def redraw_marker() -> None:
                    image.clear()
                    with image:
                        _marked_spectrogram(
                            f"/media/query/{query.token}/spectrogram"
                            f"?w={QUERY_WIDTH}&h={QUERY_HEIGHT}",
                            (query.start, query.end)
                            if query.needs_segment
                            else None,
                            0.0,
                            query.duration,
                        )

                redraw_marker()

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
                        redraw_marker()

                    picker.on_value_change(on_span)
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
                    ui.audio(state.query_audio_url(query.token)).classes("grow")

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
            draw_query()

        def handle_rejected(_event) -> None:
            """Report a file the browser refused, e.g. wrong type or too large."""
            ui.notify(t("search.no_query"), type="warning", position="top")

        ui.upload(
            label=t("search.upload"),
            on_upload=handle_upload,
            on_rejected=handle_rejected,
            auto_upload=True,
            max_files=1,
        ).props(f'accept="{ACCEPTED}" flat bordered').classes("w-full")

    filters = _filters_panel(t)
    options = _options_panel(t)

    results_area = ui.column().classes("w-full gap-2")

    async def run_search() -> None:
        if query.token is None:
            ui.notify(t("search.no_query"), type="warning", position="top")
            return
        path = state.queries.path(query.token)
        if path is None:
            ui.notify(t("search.no_query"), type="warning", position="top")
            return

        try:
            search_filters = _build_filters(filters)
        except ValidationError as exc:
            notify_error(t, exc)
            return

        start, end = query.segment()
        run_button.disable()
        spinner.set_visibility(True)
        results_area.clear()
        try:
            hits = await state.run(
                state.service.search,
                path,
                filters=search_filters,
                k=int(options["top_k"].value or state.settings.default_top_k),
                metric=options["metric"].value,
                fusion=options["fusion"].value,
                min_score=options["min_score"].value,
                start_time=start,
                end_time=end,
            )
        except BiotimbreError as exc:
            notify_error(t, exc)
            return
        finally:
            spinner.set_visibility(False)
            run_button.enable()

        with results_area:
            if not hits:
                ui.label(t("search.no_matches")).classes("text-muted")
                return
            with ui.row().classes("items-baseline gap-3"):
                ui.label(t("search.results")).classes("text-lg font-medium")
                ui.label(t("search.result_count", count=len(hits))).classes(
                    "text-xs text-muted"
                )
            ui.label(t("search.compare_hint")).classes("text-xs text-muted")
            ui.label(t("search.marker_hint")).classes("text-xs text-muted")
            for rank, hit in enumerate(hits, start=1):
                _result_card(t, hit, rank)
            if any(h.query_speed != 1.0 for h in hits):
                speeds = sorted({h.query_speed for h in hits if h.query_speed != 1.0})
                ui.label(
                    t(
                        "search.query_speed_note",
                        speed=", ".join(f"{s:g}" for s in speeds),
                    )
                ).classes("text-xs text-notice")

    with ui.row().classes("items-center gap-3"):
        run_button = ui.button(
            t("search.run"), icon="search", on_click=run_search
        ).props("text-color=button-text")
        spinner = ui.row().classes("items-center gap-2")
        with spinner:
            ui.spinner(size="1.2rem")
            ui.label(t("search.running")).classes("text-sm text-muted")
        spinner.set_visibility(False)
        ui.label(t("search.cache_note")).classes("text-xs text-muted")

    if not state.models.ready:
        ui.label(t("app.loading_models_hint")).classes("text-notice text-sm")
