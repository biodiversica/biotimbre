"""Spectrogram images with a movable time-span marker.

Shared by the pages that show a span over an image: a search result's matched
window, and the segment of a submission that will be kept. The marker is a
positioned overlay rather than something drawn into the PNG, so the same cached
image serves every span, and moving it costs two CSS properties rather than a
re-render.

The frequency axis is an overlay too, laid over the image's left edge rather
than beside it: a gutter would make the image narrower than the segment picker
beneath it, and times on the two would no longer line up.
"""

from __future__ import annotations

import math

from nicegui import ui

#: Roughly how far apart, in CSS pixels, frequency labels should be. Closer and
#: they crowd a thumbnail; further and the filter cut-offs have to be guessed.
TICK_SPACING_PX = 28

#: Ticks closer than this fraction of the height to the top edge are dropped:
#: a label there would be cut in half by the edge of the image.
TOP_MARGIN = 0.08


def place_span_marker(
    marker: ui.element,
    span: tuple[float, float] | None,
    start: float,
    end: float,
) -> None:
    """Move a span marker over its spectrogram, or hide it.

    Args:
        marker: The overlay from :func:`marked_spectrogram`.
        span: ``(from, to)`` in seconds to highlight, or ``None`` to hide.
        start: Time at the image's left edge.
        end: Time at its right edge.
    """
    if span is None or end <= start:
        marker.set_visibility(False)
        return
    left = max(0.0, min(1.0, (span[0] - start) / (end - start)))
    right = max(0.0, min(1.0, (span[1] - start) / (end - start)))
    if right <= left:
        marker.set_visibility(False)
        return
    marker.style(f"left:{left * 100:.2f}%; width:{(right - left) * 100:.2f}%")
    marker.set_visibility(True)


def frequency_ticks(max_frequency: float, height: int) -> list[float]:
    """Frequencies, in Hz, worth labelling on a spectrogram's axis.

    Steps of 1, 2 or 5 times a power of ten, so the labels are numbers one can
    type into a filter field as they are.

    Args:
        max_frequency: Frequency at the top edge (the file's Nyquist), in Hz.
        height: Height the image is shown at, in CSS pixels.

    Returns:
        Ascending frequencies strictly above zero and clear of the top edge.
    """
    if max_frequency <= 0 or height <= 0:
        return []
    wanted = max(2, height // TICK_SPACING_PX)
    rough = max_frequency / wanted
    magnitude = 10.0 ** math.floor(math.log10(rough))
    # The nice step nearest the rough one, measured as a ratio: rounding
    # always up would leave a short preview with two labels to go by.
    step = min(
        (m * magnitude for m in (1.0, 2.0, 5.0, 10.0)),
        key=lambda candidate: abs(math.log(candidate / rough)),
    )
    ticks = []
    hz = step
    while hz <= max_frequency * (1.0 - TOP_MARGIN):
        ticks.append(hz)
        hz += step
    return ticks


def _kilohertz(hz: float) -> str:
    """A tick label: ``0.5``, ``2``, ``12.5`` — no trailing zeros."""
    return f"{hz / 1000.0:.2f}".rstrip("0").rstrip(".")


def frequency_axis(max_frequency: float, height: int) -> None:
    """Label frequencies up the left edge of the enclosing image.

    Must be called inside a ``relative`` container the image fills. The top
    label carries the unit, so the axis needs no title taking up room.

    Args:
        max_frequency: Frequency at the image's top edge, in Hz.
        height: Height the image is shown at, in CSS pixels.
    """
    ticks = frequency_ticks(max_frequency, height)
    with ui.element("div").classes(
        "absolute top-0 bottom-0 left-0 pointer-events-none"
    ).mark("frequency-axis"):
        for index, hz in enumerate(ticks):
            text = _kilohertz(hz)
            if index == len(ticks) - 1:
                text += " kHz"
            with ui.element("div").classes("absolute left-0 flex items-center").style(
                f"bottom: {hz / max_frequency * 100:.2f}%; "
                "transform: translateY(50%)"
            ):
                ui.element("div").style(
                    "width: 5px; height: 1px; background: rgba(255,255,255,0.8)"
                )
                ui.label(text).classes("tabular-nums leading-none").style(
                    "font-size: 10px; color: #fff; padding: 1px 3px; "
                    "border-radius: 2px; background: rgba(0,0,0,0.55)"
                )


#: The part of the spectrum a filter will remove is dimmed by this much, not
#: hidden: what is being thrown away is the thing to check before throwing it.
SHADE_FILL = "rgba(0, 0, 0, 0.6)"


class BandShade:
    """Dims the parts of a spectrogram that a high- or low-pass will remove.

    Made by the caller and handed to :func:`marked_spectrogram`, which builds
    it into the picture. It remembers the band, so a spectrogram rebuilt by a
    change of view comes back shaded without being told again; while the
    waveform is shown instead there is nothing to shade, and placing a band
    only records it.
    """

    def __init__(self) -> None:
        self._below: ui.element | None = None
        self._above: ui.element | None = None
        self._max_frequency = 0.0
        self._band: tuple[float | None, float | None] = (None, None)

    def attach(self, max_frequency: float) -> None:
        """Build the shading into the enclosing ``relative`` container."""
        self._max_frequency = max_frequency
        # Each dimmed region is edged on the side of its cut-off, so the line
        # the filter draws is visible, not just the darkness either side of it.
        self._below = ui.element("div").classes(
            "absolute left-0 right-0 bottom-0 pointer-events-none rounded-b"
        ).style(
            f"background: {SHADE_FILL}; border-top: 2px dashed var(--bt-accent)"
        ).mark("band-shade-below")
        self._above = ui.element("div").classes(
            "absolute left-0 right-0 top-0 pointer-events-none rounded-t"
        ).style(
            f"background: {SHADE_FILL}; border-bottom: 2px dashed var(--bt-accent)"
        ).mark("band-shade-above")
        self._draw()

    def detach(self) -> None:
        """Forget the elements, for a view that has no frequency axis."""
        self._below = self._above = None

    def place(self, low: float | None, high: float | None) -> None:
        """Shade below ``low`` and above ``high``, in Hz. ``None`` shades neither.

        Args:
            low: High-pass cut-off, or ``None`` for none.
            high: Low-pass cut-off, or ``None`` for none.
        """
        self._band = (low, high)
        self._draw()

    def _draw(self) -> None:
        if self._below is None or self._above is None or self._max_frequency <= 0:
            return
        low, high = self._band
        below = (
            max(0.0, min(1.0, float(low) / self._max_frequency)) if low else 0.0
        )
        # A low-pass at or above Nyquist removes nothing that is pictured.
        above = (
            max(0.0, min(1.0, 1.0 - float(high) / self._max_frequency))
            if high
            else 0.0
        )
        self._below.style(f"height: {below * 100:.2f}%")
        self._below.set_visibility(below > 0.0)
        self._above.style(f"height: {above * 100:.2f}%")
        self._above.set_visibility(above > 0.0)


def marked_spectrogram(
    url: str,
    span: tuple[float, float] | None,
    start: float,
    end: float,
    *,
    max_frequency: float | None = None,
    height: int | None = None,
    band_shade: BandShade | None = None,
) -> ui.element:
    """Show a spectrogram with an optional highlighted time span.

    The marker is a positioned overlay rather than something drawn into the PNG,
    so the same cached image serves every span — a result list re-rendered on a
    filter change reuses the images it already has.

    Args:
        url: Image source.
        span: ``(from, to)`` in seconds to highlight, or ``None``.
        start: Time at the image's left edge.
        end: Time at its right edge.
        max_frequency: Frequency at the top edge, in Hz, to label the
            frequency axis with. ``None`` draws no axis.
        height: Height the image is shown at, which sets how many frequencies
            are labelled. Needed only with ``max_frequency``.
        band_shade: Shading for a filter's cut-offs, built in here. Needs
            ``max_frequency``, since without it there is no knowing where a
            frequency falls.

    Returns:
        The marker overlay. A caller that changes the span moves it with
        :func:`place_span_marker` instead of re-rendering: rebuilding the
        block makes the browser drop and re-request the PNG, which flashes the
        whole page on every step of a drag.
    """
    # shrink-0: a scrolling flex column (the item dialog) would otherwise
    # squeeze an image that has no intrinsic height down to nothing.
    with ui.element("div").classes("relative w-full shrink-0"):
        ui.image(url).classes("w-full rounded shrink-0").props("no-spinner")
        if band_shade is not None:
            if max_frequency:
                band_shade.attach(max_frequency)
            else:
                band_shade.detach()
        marker = ui.element("div").classes(
            "absolute top-0 bottom-0 border-x-2 pointer-events-none rounded"
        ).style(
            "border-color: var(--bt-marker-border); "
            "background: var(--bt-marker-fill)"
        )
        if max_frequency and height:
            frequency_axis(max_frequency, height)
    place_span_marker(marker, span, start, end)
    return marker
