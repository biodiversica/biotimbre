"""Admin page: add, inspect, edit and delete library items.

The list is a table so a large library stays scannable; detail, editing and
deletion happen in dialogs opened from a row. Showing a spectrogram and player per
row would not survive a few hundred items.

Adding runs inference, which takes seconds, so it is offloaded to a worker thread
and the button reports progress rather than the page appearing to hang.
"""

from __future__ import annotations

import asyncio
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from nicegui import app, ui

from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import MAX_SPEED, MIN_SPEED, LibraryItem
from biotimbre.embeddings import audio as audio_io
from biotimbre.i18n import Translator
from biotimbre.models_registry import get_spec
from biotimbre.sources import inaturalist
from biotimbre.ui import inat, maps
from biotimbre.ui.exports import export_button
from biotimbre.ui.sections import foldable_card
from biotimbre.ui.sharing import import_card
from biotimbre.ui.suggestions import library_suggestions
from biotimbre.ui.forms import _blank_to_none, metadata_form
from biotimbre.ui.layout import notify_error
from biotimbre.ui.playback import item_player, play_described, speed_audio
from biotimbre.ui.spectrograms import BandShade, place_span_marker
from biotimbre.ui.visuals import audio_view, view_toggle
from biotimbre.ui.state import get_state
from biotimbre.viz import render_spectrogram

#: Accepted upload extensions, matching what soundfile can decode.
ACCEPTED = ".wav,.flac,.ogg,.aiff,.aif,.w64"

#: Butterworth order used when a submission is filtered on the way in. Matches
#: the search page's default, so a query and a sample are shaped alike.
DEFAULT_FILTER_ORDER = 4

#: Slack when deciding whether a chosen segment is the whole file. The slider
#: works in hundredths of a second; a rounding difference is not a cut.
_SEGMENT_EPSILON = 0.01

#: Spectrogram render size for the detail dialog. Taller than a list thumbnail:
#: this is the one place a sample is meant to be read, not just recognised. The
#: image is scaled to the dialog's width, so it is the ratio that decides how
#: tall it ends up on screen.
DETAIL_WIDTH = 900
DETAIL_HEIGHT = 360

#: On-screen height of the detail view, so a waveform and a spectrogram of the
#: same item occupy the same place in the dialog.
DETAIL_VIEW_HEIGHT = 290

#: On-screen height of the add form's pictures, matching what a spectrogram
#: rendered at 860x170 comes out as across the card.
PREVIEW_VIEW_HEIGHT = 170


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
    """Mutable state of the add form.

    Besides the staged file, this holds what the submitter chose to do to it
    before it is stored: which segment to keep, and what to filter out. Those
    are applied to a copy — :meth:`apply` — and it is the copy that is added,
    so the item's ``sample_window`` and ``filtered_band`` describe the audio
    the library actually holds.

    A multi-channel submission is rewritten even when nothing was asked for:
    the library stores mono, so the copy is what makes that true. So is a
    floating-point one whose selection goes beyond full scale, normalised to
    peak at -1 dBFS so it neither clips on playback nor skews the embeddings.

    Attributes:
        upload: The staged file, or ``None``.
        directory: Staging directory for this form.
        start: Segment start in seconds within the upload.
        end: Segment end in seconds. Equal to the upload's duration means
            "keep it all".
        highpass: Highpass cutoff to apply, in Hz, or ``None``.
        lowpass: Lowpass cutoff to apply, in Hz, or ``None``.
        filter_order: Butterworth order used when a cutoff is set.
        processed: The prepared copy, or ``None`` when the upload is already
            what should be stored — mono, whole, unfiltered, within full scale.
        over_range_peak: The selection's peak when :meth:`apply` found it
            beyond full scale and normalised it, otherwise ``None``.
        normalise: Whether to bring the stored copy's peak to -1 dBFS,
            whatever its level, as asked for on the form.
        normalised_gain_db: The gain that took, when :meth:`apply` normalised
            the copy, otherwise ``None``.
        normalise_note: The line last written into the notes about it, so a
            later change replaces that line rather than adding another.
    """

    upload: Upload | None = None
    directory: Path = field(
        default_factory=lambda: Path(tempfile.mkdtemp(prefix="biotimbre-upload-"))
    )
    start: float = 0.0
    end: float = 0.0
    highpass: float | None = None
    lowpass: float | None = None
    filter_order: int = DEFAULT_FILTER_ORDER
    processed: Path | None = None
    over_range_peak: float | None = None
    normalise: bool = False
    normalised_gain_db: float | None = None
    normalise_note: str | None = None

    @property
    def duration(self) -> float:
        """Full duration of the staged upload."""
        return self.upload.properties.duration_s if self.upload else 0.0

    @property
    def is_cut(self) -> bool:
        """Whether the chosen segment is shorter than the upload."""
        if self.upload is None:
            return False
        return (
            self.start > _SEGMENT_EPSILON
            or self.end < self.duration - _SEGMENT_EPSILON
        )

    @property
    def is_filtered(self) -> bool:
        """Whether a cutoff has been asked for."""
        return self.highpass is not None or self.lowpass is not None

    @property
    def source(self) -> Path | None:
        """The file that would be stored: the processed copy, or the upload."""
        if self.processed is not None:
            return self.processed
        return self.upload.path if self.upload else None

    def apply(self) -> audio_io.AudioProperties:
        """Write the cut/filtered copy, and return what it turned out to be.

        Doing the work up front rather than at save time is what lets the form
        preview exactly what will be stored, and it puts any complaint about an
        impossible cutoff in front of the submitter while they can still change
        it.

        Returns:
            Properties of the file that would be stored.

        Raises:
            ValidationError: If there is no upload, the segment is empty, or a
                cutoff is unusable at this sample rate.
        """
        if self.upload is None:
            raise ValidationError("No audio staged.", "audio")

        self.drop_processed()
        self.normalised_gain_db = None
        start = self.start if self.is_cut else None
        end = self.end if self.is_cut else None
        # Of the selection, not the file: what is kept is what has to be in range.
        self.over_range_peak = audio_io.over_range_peak(
            self.upload.path, start_time=start, end_time=end
        )
        if (
            not self.is_cut
            and not self.is_filtered
            and self.upload.properties.channels == 1
            and self.over_range_peak is None
            and not self.normalise
        ):
            return self.upload.properties

        target = self.directory / f"processed_{self.upload.path.name}"
        properties = audio_io.write_processed(
            self.upload.path,
            target,
            start_time=start,
            end_time=end,
            low_hz=self.highpass,
            high_hz=self.lowpass,
            order=self.filter_order,
        )
        self.processed = target
        if self.normalise:
            # Last, on what is stored: the filter has moved the peak.
            self.normalised_gain_db = audio_io.normalise_file(target)
        return properties

    def drop_processed(self) -> None:
        """Delete the processed copy, if one has been written."""
        if self.processed is not None:
            self.processed.unlink(missing_ok=True)
            self.processed = None

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
        self.start = 0.0
        self.end = properties.duration_s
        self.highpass = None
        self.lowpass = None
        self.filter_order = DEFAULT_FILTER_ORDER
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
        """Delete the staged file and any processed copy of it."""
        self.drop_processed()
        if self.upload is not None and self.upload.path.exists():
            self.upload.path.unlink()
        self.upload = None
        self.start = 0.0
        self.end = 0.0
        self.over_range_peak = None


def _over_range_args(peak: float) -> dict[str, str]:
    """Translation arguments for a selection normalised from ``peak``.

    Shared by the add form and the search page, which warn in the same words.
    """
    return {
        "peak": f"{peak:.3g}",
        "dbfs": f"{20.0 * math.log10(peak):+.1f}",
        "target": f"{audio_io.NORMALISED_PEAK_DBFS:g}",
    }


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
        form.inputs["latitude"].value = observation.latitude
        form.inputs["longitude"].value = observation.longitude
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


def _fill_processing_fields(form, add_state: "AddState") -> None:
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


def _note_normalisation(t: Translator, form, add_state: "AddState") -> None:
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
    settings = get_state().settings
    return render_spectrogram(
        path,
        width=width,
        height=height,
        colors=settings.theme.spectrogram_colors,
        mark_dominant=settings.dominant_frequency,
    ).data_uri()


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
    add_state = AddState()

    #: Route currently serving the staged file, so it can be played before it
    #: is added. Streamed rather than inlined: a sample can be a long recording,
    #: and a data URI would carry the whole thing into the page.
    staged_route: str | None = None

    def stop_serving_staged() -> None:
        nonlocal staged_route
        if staged_route is not None:
            app.remove_route(staged_route)
            staged_route = None

    def serve_staged(path: Path) -> str:
        """Publish the staged file at a fresh URL and return it."""
        nonlocal staged_route
        stop_serving_staged()
        # A new path each time: the browser caches by URL, and two uploads in
        # a row can land on the same filename.
        staged_route = f"/media/staged/{uuid4().hex}"
        return app.add_media_file(local_file=path, url_path=staged_route)

    with foldable_card("add", t("admin.add_heading")):

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
            stop_serving_staged()
            uploader.reset()
            results.clear()
            show_no_file()

        show_no_file()

        #: The player of the file now staged, and its caption, for the Speed
        #: field to steer. One field outlives many uploads, so it is listened
        #: to once and told which player is current, not once per upload.
        listening: dict[str, ui.element | None] = {"player": None, "note": None}

        def follow_speed() -> None:
            """Play the staged file at the speed it will be embedded at."""
            player, note = listening["player"], listening["note"]
            if player is None or player.is_deleted:
                return
            play_described(t, player, note, form.speed)

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
            props = upload.properties
            preview.clear()
            with preview:
                with ui.row().classes("items-center gap-2"):
                    ui.label(
                        t(
                            "admin.file_chosen",
                            name=upload.name,
                            duration=f"{props.duration_s:.2f}",
                            rate=props.sample_rate,
                            channels=props.channels,
                        )
                    ).classes("text-sm")
                    ui.button(
                        t("common.clear_file"), icon="close", on_click=clear_file
                    ).props("flat dense no-caps color=negative")
                # Follows the Speed field below, so it is heard as embedded.
                player = speed_audio(serve_staged(upload.path)).classes(
                    "w-full shrink-0"
                )
                playback_note = ui.label().classes("text-xs text-muted")
                listening["player"] = player
                listening["note"] = playback_note
                player.on("loadedmetadata", lambda _: follow_speed())
                # What the player is currently serving, so a redraw that
                # changes nothing does not retire a URL the browser is using.
                published = {"path": upload.path}
                try:
                    uri = await state.run(_spectrogram_of, upload.path)
                except BiotimbreError as exc:
                    notify_error(t, exc)
                    return

                with ui.row().classes("items-center justify-between w-full"):
                    span_label = ui.label().classes("text-xs text-muted")
                    view_toggle(t, lambda: redraw_views())
                source_view = ui.column().classes("w-full gap-0")
                marker: ui.element | None = None
                # Shows what the high- and low-pass will take away, on the
                # picture of what was submitted — the stored preview below
                # shows the result, this shows the decision.
                shade = BandShade()

                async def draw_source_view() -> None:
                    """Picture the upload, in whichever view is preferred."""
                    nonlocal marker
                    source_view.clear()
                    with source_view:
                        marker = await audio_view(
                            spectrogram_url=uri,
                            audio_path=upload.path,
                            start=0.0,
                            end=props.duration_s,
                            span=(add_state.start, add_state.end)
                            if add_state.is_cut
                            else None,
                            height=PREVIEW_VIEW_HEIGHT,
                            sample_rate=props.sample_rate,
                            band_shade=shade,
                        )

                async def redraw_views() -> None:
                    """Both pictures follow the chosen view, not just one."""
                    await draw_source_view()
                    describe_span()
                    await restage()

                await draw_source_view()
                ui.label(_nyquist_lines(t)).classes("text-xs text-muted")

                ui.label(t("admin.trim_hint")).classes("text-xs text-muted")
                picker = ui.range(
                    min=0.0,
                    max=round(props.duration_s, 2),
                    step=0.01,
                    value={"min": 0.0, "max": round(props.duration_s, 2)},
                ).classes("w-full")

                with ui.row().classes("w-full gap-3 items-start"):
                    # The form's own Speed field, offered here as well: it is
                    # decided while listening, beside the filters, not later
                    # among the metadata. Bound both ways, so it is one value
                    # wherever it is typed, and the player follows either.
                    ui.number(
                        t("field.speed"),
                        step=0.05,
                        min=MIN_SPEED,
                        max=MAX_SPEED,
                    ).props("dense outlined").classes("grow").tooltip(
                        t("field.speed_help")
                    ).bind_value(form.inputs["speed"]).mark("apply-speed")
                    # Marked because the metadata form below carries inputs
                    # of the same name, recording a filter applied elsewhere.
                    highpass = ui.number(
                        t("field.highpass"), value=None, step=100,
                        suffix=t("common.hertz"),
                    ).props("dense outlined").classes("grow").mark("apply-highpass")
                    lowpass = ui.number(
                        t("field.lowpass"), value=None, step=100,
                        suffix=t("common.hertz"),
                    ).props("dense outlined").classes("grow")
                    order = ui.number(
                        t("field.filter_order"),
                        value=DEFAULT_FILTER_ORDER, step=1, min=1,
                    ).props("dense outlined").classes("grow")
                    normalise = ui.checkbox(
                        t("admin.normalise", target=f"{audio_io.NORMALISED_PEAK_DBFS:g}")
                    ).props("dense").classes("self-center").tooltip(
                        t("admin.normalise_help")
                    ).mark("apply-normalise")
                ui.label(t("admin.filter_hint")).classes("text-xs text-muted")

                def shade_band() -> None:
                    """Dim what the cut-offs remove, as they are typed.

                    On every keystroke rather than on change: it only restyles
                    two overlays, and seeing the line move while typing is the
                    point of drawing it.
                    """
                    shade.place(
                        _blank_to_none(highpass.value), _blank_to_none(lowpass.value)
                    )

                highpass.on_value_change(shade_band)
                lowpass.on_value_change(shade_band)

                stored = ui.column().classes("w-full gap-1")

                def describe_span() -> None:
                    """Say where the segment sits, and mark it on the image.

                    Called on every step of a drag, so it does nothing but move
                    a marker and rewrite a label: publishing the audio here
                    retired and re-registered a route per step, which left the
                    browser fetching URLs that had just stopped existing.
                    """
                    span_label.set_text(
                        t(
                            "admin.segment_of",
                            start=f"{add_state.start:.2f}",
                            end=f"{add_state.end:.2f}",
                            duration=f"{add_state.duration:.2f}",
                        )
                    )
                    # Rebuilt whenever the view changes, so never assume one.
                    if marker is not None:
                        place_span_marker(
                            marker,
                            (add_state.start, add_state.end)
                            if add_state.is_cut
                            else None,
                            0.0,
                            add_state.duration,
                        )

                def superseded() -> bool:
                    """Whether this file has been removed or replaced since.

                    "Remove file" can be clicked while a restage is still
                    decoding; carrying on would serve a file that is gone and
                    draw into a preview that has been cleared.
                    """
                    return add_state.upload is not upload

                async def _restage_once() -> None:
                    """Re-apply the choice and show what would be stored."""
                    if superseded():
                        return
                    add_state.highpass = _blank_to_none(highpass.value)
                    add_state.lowpass = _blank_to_none(lowpass.value)
                    add_state.filter_order = int(order.value or DEFAULT_FILTER_ORDER)
                    add_state.normalise = bool(normalise.value)
                    stored.clear()
                    try:
                        result = await state.run(add_state.apply)
                    except BiotimbreError as exc:
                        if not superseded():
                            notify_error(t, exc)
                        return
                    if superseded():
                        return
                    _fill_processing_fields(form, add_state)
                    _note_normalisation(t, form, add_state)
                    # The player plays the copy itself, so what is heard is the
                    # cut, the filter *and* the downmix — a media fragment
                    # could only ever convey the cut.
                    # A rewritten copy keeps its path, so the path alone
                    # cannot say whether the audio changed: republish whenever
                    # there is a copy, and the fresh URL gets past the cache.
                    # Only an untouched upload, already published, is left be.
                    if add_state.processed is not None or (
                        published["path"] != add_state.source
                    ):
                        published["path"] = add_state.source
                        player.set_source(serve_staged(add_state.source))
                    if add_state.processed is None:
                        return
                    # Only when there is a difference to show: an untouched
                    # submission is already pictured above.
                    with stored:
                        if add_state.over_range_peak is not None:
                            ui.label(
                                t(
                                    "admin.over_range",
                                    **_over_range_args(add_state.over_range_peak),
                                )
                            ).classes("text-xs text-notice")
                        ui.label(
                            t(
                                "admin.will_store",
                                duration=f"{result.duration_s:.2f}",
                            )
                        ).classes("text-xs text-muted")
                        try:
                            stored_uri = await state.run(
                                _spectrogram_of, add_state.processed
                            )
                        except BiotimbreError as exc:
                            notify_error(t, exc)
                            return
                        await audio_view(
                            spectrogram_url=stored_uri,
                            audio_path=add_state.processed,
                            start=0.0,
                            end=result.duration_s,
                            span=None,
                            height=PREVIEW_VIEW_HEIGHT,
                            sample_rate=result.sample_rate,
                        )

                # Each run unlinks the previous copy and rewrites one at a
                # fixed path, so two overlapping runs would have one deleting
                # the file the other is still writing. Serialised, and a run a
                # newer choice has already overtaken is dropped rather than
                # spending a decode on a preview nobody will see.
                staging = {"latest": 0}
                staging_lock = asyncio.Lock()

                async def restage() -> None:
                    """Re-apply the choice, where the newest choice wins."""
                    staging["latest"] += 1
                    mine = staging["latest"]
                    async with staging_lock:
                        if mine == staging["latest"]:
                            await _restage_once()

                def on_span(event) -> None:
                    add_state.start = float(event.value["min"])
                    add_state.end = float(event.value["max"])
                    describe_span()

                async def on_span_settled(event) -> None:
                    # The settled value comes from the event, not from what the
                    # drag has managed to report so far: the two arrive
                    # separately, and cutting to a stale span was a real bug.
                    span = event.args
                    if isinstance(span, dict) and "min" in span:
                        add_state.start = float(span["min"])
                        add_state.end = float(span["max"])
                        describe_span()
                    await restage()

                picker.on_value_change(on_span)
                # Cutting and filtering are done on the settled value: each one
                # decodes and rewrites the segment, which is not something to
                # do on every step of a drag.
                picker.on("change", on_span_settled)
                for element in (highpass, lowpass, order):
                    element.on("change", lambda _: restage())
                normalise.on_value_change(lambda _: restage())

                # Wired up and labelled before the first restage rather than
                # after it: the picker is on the page as soon as it is built,
                # and a drag landing during that first decode used to reach an
                # unwired picker and be dropped with nothing to show for it.
                describe_span()

                # Once up front: a stereo submission is already destined for a
                # mono copy, and the preview should be of that copy rather than
                # of something the library will not keep.
                await restage()

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
        form.inputs["speed"].on_value_change(lambda _: follow_speed())
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
            stop_serving_staged()
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


async def _open_detail(t: Translator, item_id: int, on_changed) -> None:
    """Show one item with its spectrogram, player and actions."""
    state = get_state()
    try:
        item = await state.run(state.service.get_item, item_id)
        counts = await state.run(state.service.repo.embedding_counts, item_id)
        label = await state.run(state.service.get_biotimbre_label, item.biotimbre_id)
        # The waveform is drawn from the audio itself, not from a rendered image.
        audio_path, _ = await state.run(state.service.get_audio, item_id)
    except BiotimbreError:
        ui.notify(t("error.not_found"), type="warning", position="top")
        on_changed()
        return

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-3xl p-4 gap-3 bg-card"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(t("admin.detail", item_id=item.item_id)).classes(
                "text-lg font-medium"
            )
            view_toggle(t, lambda: draw_view())
        # The card is a flex column that scrolls, so anything without an
        # intrinsic height is free to shrink away: the player collapsed to
        # nothing and the picture lost most of its height. Neither may shrink.
        visual = ui.column().classes("w-full gap-0 shrink-0")

        async def draw_view() -> None:
            visual.clear()
            with visual:
                await audio_view(
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
