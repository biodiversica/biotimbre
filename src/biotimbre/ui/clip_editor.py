"""A staged recording, and the controls for preparing it.

Shared by the library page, where the prepared copy is what gets stored, and
the search page, where it is what gets searched with. Both offer the same
things — cut, high- and low-pass, normalise, speed — pictured the same way, so
a sample and a query are shaped alike and what is heard is what is used. The
one difference is the search page's cap on how long a query may be, which the
segment picker enforces as it is dragged.
"""

from __future__ import annotations

import asyncio
import math
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from nicegui import app, ui

from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.domain.item import DEFAULT_SPEED, MAX_SPEED, MIN_SPEED
from biotimbre.embeddings import audio as audio_io
from biotimbre.i18n import Translator
from biotimbre.models_registry import get_spec
from biotimbre.ui.forms import _blank_to_none
from biotimbre.ui.layout import notify_error
from biotimbre.ui.playback import play_described, speed_audio
from biotimbre.ui.segment_range import SegmentRange
from biotimbre.ui.spectrograms import BandShade, place_span_marker
from biotimbre.ui.state import get_state
from biotimbre.ui.visuals import audio_view, view_toggle
from biotimbre.viz import render_spectrogram

#: Accepted upload extensions, matching what soundfile can decode.
ACCEPTED = ".wav,.flac,.ogg,.aiff,.aif,.w64"

#: Butterworth order used when a recording is filtered. One default for both
#: pages, so a query and a sample are shaped alike.
DEFAULT_FILTER_ORDER = 4

#: Slack when deciding whether a chosen segment is the whole file. The slider
#: works in hundredths of a second; a rounding difference is not a cut.
_SEGMENT_EPSILON = 0.01

#: Per-browser storage key under which the search page leaves a query for the
#: library page's add form to pick up.
HANDOFF_KEY = "handoff"

#: On-screen height of the editor's pictures, matching what a spectrogram
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
class ClipState:
    """A staged recording, and what is to be done to it before it is used.

    Besides the staged file, this holds what was chosen to do to it: which
    segment to keep, and what to filter out. Those are applied to a copy —
    :meth:`apply` — and it is the copy that is used: added to the library, so
    the item's ``sample_window`` and ``filtered_band`` describe the audio the
    library actually holds, or searched with, so the query is what was heard.

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
        """Full duration of the staged recording."""
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
        impossible cutoff in front of whoever chose it while they can still change
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
    """Translation arguments for a selection normalised from ``peak``."""
    return {
        "peak": f"{peak:.3g}",
        "dbfs": f"{20.0 * math.log10(peak):+.1f}",
        "target": f"{audio_io.NORMALISED_PEAK_DBFS:g}",
    }


def _nyquist_lines(t: Translator) -> str:
    """Describe where each model's limit falls, for a spectrogram caption."""
    limits = ", ".join(
        f"{model_id} {get_spec(model_id).nyquist_hz / 1000:g} kHz"
        for model_id in get_state().settings.active_models
    )
    return f"{t('admin.model_limits')} ({limits})"


def _spectrogram_of(path: Path, *, width: int = 860, height: int = 170) -> str:
    """Render a staged file as an inline data URI.

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


class StagedMedia:
    """Publishes a staged file to the browser, one at a time.

    Streamed rather than inlined: a recording can be long, and a data URI would
    carry the whole thing into the page.
    """

    def __init__(self) -> None:
        self._route: str | None = None

    def serve(self, path: Path) -> str:
        """Publish ``path`` at a fresh URL, retiring the previous one."""
        self.stop()
        # A new path each time: the browser caches by URL, and a rewritten
        # copy, or two uploads in a row, can land on the same filename.
        self._route = f"/media/staged/{uuid4().hex}"
        return app.add_media_file(local_file=path, url_path=self._route)

    def stop(self) -> None:
        """Stop serving whatever was last published."""
        if self._route is not None:
            app.remove_route(self._route)
            self._route = None


@dataclass(frozen=True)
class ClipWording:
    """Translation keys for what the two pages say differently.

    The controls are the same; what the prepared copy is for is not, and the
    words say so — stored on one page, searched with on the other.
    """

    segment_of: str
    trim_hint: str
    filter_hint: str
    result: str
    over_range: str
    normalise_help: str
    speed: str
    speed_help: str
    #: Said when the recording is longer than the cap, if there is one.
    too_long: str | None = None


LIBRARY_WORDING = ClipWording(
    segment_of="admin.segment_of",
    trim_hint="admin.trim_hint",
    filter_hint="admin.filter_hint",
    result="admin.will_store",
    over_range="admin.over_range",
    normalise_help="admin.normalise_help",
    speed="field.speed",
    speed_help="field.speed_help",
)

SEARCH_WORDING = ClipWording(
    segment_of="search.segment_of",
    trim_hint="search.segment_hint",
    filter_hint="search.filter_hint",
    result="search.will_search",
    over_range="search.over_range",
    normalise_help="search.normalise_help",
    speed="search.speed",
    speed_help="search.speed_help",
    too_long="search.needs_segment",
)


@dataclass
class ClipControls:
    """What the page around an editor needs to reach.

    Attributes:
        speed: The speed field. Speed is not applied to the copy — the models
            are told it — so the page reads it from here.
        redraw_views: Redraw both pictures in the reader's preferred view.
        settle: Apply whatever the fields now say, if it has not been applied
            yet, and report whether the prepared copy matches them. To be
            awaited before the copy is used: a cut-off typed and not yet
            committed would otherwise be silently left out.
    """

    speed: ui.number
    redraw_views: Callable[[], Awaitable[None]]
    settle: Callable[[], Awaitable[bool]]


async def clip_editor(
    t: Translator,
    clip: ClipState,
    upload: Upload,
    *,
    media: StagedMedia,
    wording: ClipWording,
    on_clear: Callable[[], object],
    max_span: float | None = None,
    bind_speed: ui.element | None = None,
    speed: float = DEFAULT_SPEED,
    on_applied: Callable[[ClipState], None] | None = None,
    on_view_change: Callable[[], Awaitable[None] | None] | None = None,
    on_ready: Callable[[ClipControls], None] | None = None,
) -> ClipControls | None:
    """Draw the preview and the controls for a staged recording.

    Drawn into the enclosing container, which the caller clears first.

    Args:
        t: Active translator.
        clip: The staged recording and what is to be done to it.
        upload: The upload the editor is for. If ``clip`` moves on to another
            while a restage is in flight, that restage is dropped.
        media: Publishes the audio for the player.
        wording: What the page calls things.
        on_clear: Called by the Remove file button.
        max_span: Longest selection allowed, in seconds, or ``None`` for no
            limit. Applied in the browser as the picker is dragged.
        bind_speed: An element to bind the speed field to, both ways, so the
            same value can be typed elsewhere on the page.
        speed: Starting speed, when not bound.
        on_applied: Called after each successful restage.
        on_view_change: Called after the view toggle has redrawn the editor,
            for the page to redraw anything else it pictures.
        on_ready: Given the controls as soon as they exist, before the first
            restage has finished. The preview of that restage appears while
            this call is still awaiting it, and a page that waited for the
            return value would ignore a click made in that gap.

    Returns:
        The controls, or ``None`` if the recording could not be pictured.
    """
    state = get_state()
    props = upload.properties
    if max_span is not None and clip.end - clip.start > max_span:
        clip.end = clip.start + max_span

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
        ui.button(t("common.clear_file"), icon="close", on_click=on_clear).props(
            "flat dense no-caps color=negative"
        )

    speed_input: ui.number | None = None

    def follow_speed() -> None:
        """Play at the speed the models will hear it at."""
        if speed_input is None or player.is_deleted:
            return
        value = speed_input.value
        play_described(
            t, player, playback_note, float(value) if value else DEFAULT_SPEED
        )

    # Follows the Speed field below, so it is heard as embedded.
    player = speed_audio(media.serve(upload.path)).classes("w-full shrink-0")
    playback_note = ui.label().classes("text-xs text-muted")
    player.on("loadedmetadata", lambda _: follow_speed())
    # What the player is currently serving, so a redraw that changes nothing
    # does not retire a URL the browser is using.
    published = {"path": upload.path}
    try:
        uri = await state.run(_spectrogram_of, upload.path)
    except BiotimbreError as exc:
        notify_error(t, exc)
        return None

    async def toggled() -> None:
        await redraw_views()
        if on_view_change is not None:
            result = on_view_change()
            if result is not None:
                await result

    with ui.row().classes("items-center justify-between w-full"):
        span_label = ui.label().classes("text-xs text-muted")
        view_toggle(t, toggled)
    source_view = ui.column().classes("w-full gap-0")
    marker: ui.element | None = None
    # Shows what the high- and low-pass will take away, on the picture of the
    # recording as it came — the preview below shows the result, this shows
    # the decision.
    shade = BandShade()

    async def draw_source_view() -> None:
        """Picture the recording, in whichever view is preferred."""
        nonlocal marker
        source_view.clear()
        with source_view:
            marker = await audio_view(
                spectrogram_url=uri,
                audio_path=upload.path,
                start=0.0,
                end=props.duration_s,
                span=(clip.start, clip.end) if clip.is_cut else None,
                height=PREVIEW_VIEW_HEIGHT,
                sample_rate=props.sample_rate,
                band_shade=shade,
            )

    await draw_source_view()
    ui.label(_nyquist_lines(t)).classes("text-xs text-muted")

    seconds = f"{max_span:g}" if max_span is not None else ""
    if wording.too_long and max_span is not None and props.duration_s > max_span:
        ui.label(t(wording.too_long, seconds=seconds)).classes("text-xs text-notice")
    ui.label(t(wording.trim_hint, seconds=seconds)).classes("text-xs text-muted")
    duration = round(props.duration_s, 2)
    initial = {"min": round(clip.start, 2), "max": round(clip.end, 2)}
    if max_span is None:
        picker = ui.range(min=0.0, max=duration, step=0.01, value=initial)
    else:
        picker = SegmentRange(
            min=0.0, max=duration, step=0.01, value=initial, max_span=max_span
        )
    picker.classes("w-full")

    with ui.row().classes("w-full gap-3 items-start"):
        # Decided while listening, beside the filters. Bound both ways when
        # the page has a field of its own for it, so it is one value wherever
        # it is typed.
        speed_input = ui.number(
            t(wording.speed),
            value=speed,
            step=0.05,
            min=MIN_SPEED,
            max=MAX_SPEED,
        ).props("dense outlined").classes("grow").tooltip(
            t(wording.speed_help)
        ).mark("apply-speed")
        if bind_speed is not None:
            speed_input.bind_value(bind_speed)
        speed_input.on_value_change(lambda _: follow_speed())
        # Marked because the library's metadata form carries inputs of the
        # same name, recording a filter applied elsewhere.
        # Starting from what the clip already says, which is nothing for a
        # fresh upload and the search page's choices for a handed-over query.
        highpass = ui.number(
            t("field.highpass"), value=clip.highpass, step=100,
            suffix=t("common.hertz"),
        ).props("dense outlined").classes("grow").mark("apply-highpass")
        lowpass = ui.number(
            t("field.lowpass"), value=clip.lowpass, step=100,
            suffix=t("common.hertz"),
        ).props("dense outlined").classes("grow").mark("apply-lowpass")
        order = ui.number(
            t("field.filter_order"), value=clip.filter_order, step=1, min=1,
        ).props("dense outlined").classes("grow").mark("apply-order")
        normalise = ui.checkbox(
            t("admin.normalise", target=f"{audio_io.NORMALISED_PEAK_DBFS:g}"),
            value=clip.normalise,
        ).props("dense").classes("self-center").tooltip(
            t(wording.normalise_help)
        ).mark("apply-normalise")
    ui.label(t(wording.filter_hint)).classes("text-xs text-muted")

    def shade_band() -> None:
        """Dim what the cut-offs remove, as they are typed.

        On every keystroke rather than on change: it only restyles two
        overlays, and seeing the line move while typing is the point of
        drawing it.
        """
        shade.place(_blank_to_none(highpass.value), _blank_to_none(lowpass.value))

    highpass.on_value_change(shade_band)
    lowpass.on_value_change(shade_band)
    shade_band()

    stored = ui.column().classes("w-full gap-1")

    def describe_span() -> None:
        """Say where the segment sits, and mark it on the image.

        Called on every step of a drag, so it does nothing but move a marker
        and rewrite a label: publishing the audio here retired and
        re-registered a route per step, which left the browser fetching URLs
        that had just stopped existing.
        """
        span_label.set_text(
            t(
                wording.segment_of,
                start=f"{clip.start:.2f}",
                end=f"{clip.end:.2f}",
                duration=f"{clip.duration:.2f}",
            )
        )
        # Rebuilt whenever the view changes, so never assume one.
        if marker is not None:
            place_span_marker(
                marker,
                (clip.start, clip.end) if clip.is_cut else None,
                0.0,
                clip.duration,
            )

    def superseded() -> bool:
        """Whether this recording has been removed or replaced since.

        "Remove file" can be clicked while a restage is still decoding;
        carrying on would serve a file that is gone and draw into a preview
        that has been cleared.
        """
        return clip.upload is not upload

    def wanted() -> tuple:
        """What the controls currently ask for."""
        return (
            round(clip.start, 3),
            round(clip.end, 3),
            _blank_to_none(highpass.value),
            _blank_to_none(lowpass.value),
            int(order.value or DEFAULT_FILTER_ORDER),
            bool(normalise.value),
        )

    #: What the prepared copy was last made from, or ``None`` when the last
    #: attempt failed; and what it turned out to be, for redrawing.
    applied: dict = {"wanted": None, "result": None}

    async def draw_stored() -> None:
        """Picture the prepared copy, when it differs from the recording."""
        stored.clear()
        result = applied["result"]
        # Only when there is a difference to show: an untouched recording is
        # already pictured above.
        if result is None or clip.processed is None:
            return
        with stored:
            if clip.over_range_peak is not None:
                ui.label(
                    t(wording.over_range, **_over_range_args(clip.over_range_peak))
                ).classes("text-xs text-notice")
            ui.label(
                t(wording.result, duration=f"{result.duration_s:.2f}")
            ).classes("text-xs text-muted")
            try:
                stored_uri = await state.run(_spectrogram_of, clip.processed)
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            await audio_view(
                spectrogram_url=stored_uri,
                audio_path=clip.processed,
                start=0.0,
                end=result.duration_s,
                span=None,
                height=PREVIEW_VIEW_HEIGHT,
                sample_rate=result.sample_rate,
            )

    async def _restage_once() -> None:
        """Re-apply the choice and show what it makes."""
        if superseded():
            return
        asked = wanted()
        clip.highpass = asked[2]
        clip.lowpass = asked[3]
        clip.filter_order = asked[4]
        clip.normalise = asked[5]
        applied.update(wanted=None, result=None)
        stored.clear()
        try:
            result = await state.run(clip.apply)
        except BiotimbreError as exc:
            if not superseded():
                notify_error(t, exc)
            return
        if superseded():
            return
        applied.update(wanted=asked, result=result)
        if on_applied is not None:
            on_applied(clip)
        # The player plays the copy itself, so what is heard is the cut, the
        # filter *and* the downmix — a media fragment could only ever convey
        # the cut. A rewritten copy keeps its path, so the path alone cannot
        # say whether the audio changed: republish whenever there is a copy,
        # and the fresh URL gets past the cache. Only an untouched upload,
        # already published, is left be.
        if clip.processed is not None or published["path"] != clip.source:
            published["path"] = clip.source
            player.set_source(media.serve(clip.source))
        await draw_stored()

    # Each run unlinks the previous copy and rewrites one at a fixed path, so
    # two overlapping runs would have one deleting the file the other is still
    # writing. Serialised, and a run a newer choice has already overtaken is
    # dropped rather than spending a decode on a preview nobody will see.
    staging = {"latest": 0}
    staging_lock = asyncio.Lock()

    async def restage() -> None:
        """Re-apply the choice, where the newest choice wins."""
        staging["latest"] += 1
        mine = staging["latest"]
        async with staging_lock:
            if mine == staging["latest"]:
                await _restage_once()

    async def settle() -> bool:
        if applied["wanted"] != wanted():
            await restage()
        return not superseded() and applied["wanted"] == wanted()

    async def redraw_views() -> None:
        """Both pictures follow the chosen view, not just one."""
        await draw_source_view()
        describe_span()
        await draw_stored()

    def take_span(low: float, high: float) -> None:
        # The capped picker already slides the selection; this only guards a
        # value set some other way. The edge not just moved is the one that
        # gives.
        if max_span is not None and high - low > max_span:
            if abs(low - clip.start) > abs(high - clip.end):
                high = low + max_span
            else:
                low = max(0.0, high - max_span)
            picker.set_value({"min": low, "max": high})
        clip.start, clip.end = low, high
        describe_span()

    def on_span(event) -> None:
        take_span(float(event.value["min"]), float(event.value["max"]))

    async def on_span_settled(event) -> None:
        # The settled value comes from the event, not from what the drag has
        # managed to report so far: the two arrive separately, and cutting to
        # a stale span was a real bug.
        span = event.args
        if isinstance(span, dict) and "min" in span:
            take_span(float(span["min"]), float(span["max"]))
        await restage()

    picker.on_value_change(on_span)
    # Cutting and filtering are done on the settled value: each one decodes
    # and rewrites the segment, which is not something to do on every step of
    # a drag.
    picker.on("change", on_span_settled)
    for element in (highpass, lowpass, order):
        element.on("change", lambda _: restage())
    normalise.on_value_change(lambda _: restage())

    # Wired up and labelled before the first restage rather than after it: the
    # picker is on the page as soon as it is built, and a drag landing during
    # that first decode used to reach an unwired picker and be dropped with
    # nothing to show for it.
    describe_span()

    controls = ClipControls(
        speed=speed_input, redraw_views=redraw_views, settle=settle
    )
    if on_ready is not None:
        on_ready(controls)

    # Once up front: a stereo recording is already destined for a mono copy,
    # a long one for a cut, and the preview should be of that copy rather than
    # of something that will not be used.
    await restage()
    return controls


@dataclass
class Handoff:
    """A query on its way from the search page to the library's add form.

    The original recording travels, not the prepared copy, together with what
    was chosen for it: the add form prepares it again, so every choice stays
    open to change before anything is stored.

    Attributes:
        name: The recording's filename.
        start: Segment start, in seconds.
        end: Segment end.
        highpass: Highpass cut-off, in Hz, or ``None``.
        lowpass: Lowpass cut-off, in Hz, or ``None``.
        filter_order: Butterworth order.
        normalise: Whether normalising was asked for.
        speed: Speed it was searched at.
        observation: The iNaturalist observation it was fetched from, if any,
            for the add form to take the metadata from.
    """

    name: str
    start: float
    end: float
    highpass: float | None
    lowpass: float | None
    filter_order: int
    normalise: bool
    speed: float
    observation: Any = None


async def hand_over(
    clip: ClipState, *, speed: float, observation: Any = None
) -> str:
    """Leave a copy of a staged recording, and its settings, to be taken over.

    A copy, because the page it came from deletes its own files when it is
    left or cleared.

    Args:
        clip: The staged recording and what was chosen for it.
        speed: Speed chosen for it.
        observation: The observation it was fetched from, if any.

    Returns:
        The token to take it over by.

    Raises:
        ValidationError: If nothing is staged.
    """
    if clip.upload is None:
        raise ValidationError("No audio staged.", "audio")
    state = get_state()
    token, target = state.queries.reserve(clip.upload.name)
    await state.run(shutil.copyfile, clip.upload.path, target)
    state.handoffs[token] = Handoff(
        name=clip.upload.name,
        start=clip.start,
        end=clip.end,
        highpass=clip.highpass,
        lowpass=clip.lowpass,
        filter_order=clip.filter_order,
        normalise=clip.normalise,
        speed=speed,
        observation=observation,
    )
    return token


async def take_over(token: str, clip: ClipState) -> tuple[Upload, Handoff] | None:
    """Stage a handed-over recording in ``clip``, with its settings.

    One use per token: the staged copy and the settings are dropped as they are
    taken.

    Args:
        token: From :func:`hand_over`.
        clip: Where to stage it.

    Returns:
        The staged upload and what came with it, or ``None`` if the token is
        unknown or its file has since been evicted.

    Raises:
        ValidationError: If the file is no longer decodable audio.
    """
    state = get_state()
    handoff = state.handoffs.pop(token, None)
    source = state.queries.path(token)
    if handoff is None or source is None:
        state.queries.discard(token)
        return None
    target = clip.begin(handoff.name)
    try:
        await state.run(shutil.copyfile, source, target)
    finally:
        state.queries.discard(token)
    upload = await state.run(clip.finish, target, handoff.name)
    clip.start = handoff.start
    clip.end = handoff.end
    clip.highpass = handoff.highpass
    clip.lowpass = handoff.lowpass
    clip.filter_order = handoff.filter_order
    clip.normalise = handoff.normalise
    return upload, handoff
