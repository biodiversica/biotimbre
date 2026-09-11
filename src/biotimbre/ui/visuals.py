"""How a sample is pictured: spectrogram or waveform, and the choice between.

The waveform is inline SVG rather than a rendered image, which is what lets it
take its colours from the theme: a PNG is drawn server-side, where the reader's
light or dark choice is not known — the reason the spectrogram ramp is shared
by both themes. Being vector, it also stretches to whatever width it is given
without another round trip.

Both views carry the same span marker, since both have the same time axis: the
segment being kept, or the window that produced a score, reads the same either
way.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path

from nicegui import app, ui

from biotimbre.i18n import Translator
from biotimbre.ui.spectrograms import marked_spectrogram, place_span_marker
from biotimbre.ui.state import get_state
from biotimbre.viz import Waveform, waveform_envelope

#: The two ways of picturing a sample, and the icon for each.
VIEWS = {"spectrogram": "gradient", "waveform": "graphic_eq"}

#: Per-browser storage key, beside the language and the theme.
_STORAGE_KEY = "audio_view"

#: Columns a waveform is reduced to. One per pixel of a full-width image; more
#: would be drawn onto the same pixel, and the markup is sent per render.
DEFAULT_COLUMNS = 900

#: Fraction of the height the peak envelope may fill, leaving the loudest
#: column just short of the edge rather than clipped against it.
_HEADROOM = 0.96


def preferred_view() -> str:
    """The reader's chosen view, defaulting to the spectrogram."""
    try:
        view = app.storage.user.get(_STORAGE_KEY)
    except RuntimeError:
        view = None
    return view if view in VIEWS else "spectrogram"


def set_preferred_view(view: str) -> None:
    """Remember a chosen view, ignoring anything unrecognised."""
    if view in VIEWS:
        app.storage.user[_STORAGE_KEY] = view


def view_toggle(t: Translator, on_change) -> ui.button:
    """A button that swaps the view and redraws whatever showed the old one.

    Args:
        t: Active translator.
        on_change: Called after the choice changes, to redraw.

    Returns:
        The button.
    """
    offered = "waveform" if preferred_view() == "spectrogram" else "spectrogram"

    async def swap() -> None:
        nonlocal offered
        set_preferred_view(offered)
        offered = "waveform" if offered == "spectrogram" else "spectrogram"
        button.props(f"icon={VIEWS[offered]}")
        tip.set_text(t(f"view.{offered}"))
        # The button outlives the container it redraws, so it says what it
        # offers next rather than what is on screen.
        result = on_change()
        if inspect.isawaitable(result):
            await result

    button = ui.button(icon=VIEWS[offered], on_click=swap).props(
        "flat dense round size=sm"
    )
    with button:
        tip = ui.tooltip(t(f"view.{offered}"))
    return button


@lru_cache(maxsize=64)
def _cached_envelope(
    path: str, fingerprint: tuple[float, int], span: tuple[float | None, float | None],
    columns: int,
) -> Waveform:
    """Envelope for a file, remembered while the file stays as it was.

    Inline SVG is not cached by the browser the way an image route is, so a
    result list redrawn on every toggle would decode every item again. The
    fingerprint is the file's size and mtime, so a restaged upload written to
    the same path is read afresh.
    """
    del fingerprint  # part of the key, not of the computation
    return waveform_envelope(
        path, start_time=span[0], end_time=span[1], columns=columns
    )


def envelope_of(
    path: Path,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    columns: int = DEFAULT_COLUMNS,
) -> Waveform:
    """Envelope for a file, via the cache. Runs the decode, so call it off-loop."""
    stat = Path(path).stat()
    return _cached_envelope(
        str(path), (stat.st_mtime, stat.st_size), (start_time, end_time), columns
    )


def waveform_svg(wave: Waveform, *, height: int) -> str:
    """Draw an envelope as SVG markup.

    Amplitudes are scaled to the loudest column, as the spectrogram's dB range
    is scaled to its loudest bin: a quiet recording is a picture of its own
    signal, not a flat line. The peak envelope is drawn as a filled band and
    the RMS inside it, so both readings are visible at once.

    Args:
        wave: The envelope to draw.
        height: Height in CSS pixels. The width follows the container.

    Returns:
        An ``<svg>`` element, ready for :func:`nicegui.ui.html`.
    """
    scale = wave.peak or 1.0
    columns = wave.columns

    def band(upper, lower) -> str:
        # One polygon: across the top, back along the bottom. Y runs 0..2 with
        # silence at 1, so the viewBox needs no arithmetic per point.
        top = " ".join(
            f"{index},{1 - value / scale * _HEADROOM:.3f}"
            for index, value in enumerate(upper)
        )
        bottom = " ".join(
            f"{index},{1 - value / scale * _HEADROOM:.3f}"
            for index, value in reversed(list(enumerate(lower)))
        )
        return f'<polygon points="{top} {bottom}" />'

    peaks = band(wave.maxima, wave.minima)
    loudness = band(wave.rms, -wave.rms)
    return (
        f'<svg viewBox="0 0 {columns - 1} 2" preserveAspectRatio="none" '
        f'style="width:100%; height:{height}px; display:block; '
        f'background: var(--bt-surface); border-radius: 4px" '
        f'role="img" aria-hidden="true">'
        f'<g fill="var(--bt-primary)" fill-opacity="0.35">{peaks}</g>'
        f'<g fill="var(--bt-primary)">{loudness}</g>'
        f'<line x1="0" y1="1" x2="{columns - 1}" y2="1" stroke="var(--bt-muted)" '
        f'stroke-width="0.01" stroke-opacity="0.6" />'
        f"</svg>"
    )


def marked_waveform(
    wave: Waveform,
    span: tuple[float, float] | None,
    *,
    height: int,
) -> ui.element:
    """A waveform with the same span marker a spectrogram carries.

    Args:
        wave: The envelope to draw.
        span: ``(from, to)`` in seconds to highlight, or ``None``.
        height: Height in CSS pixels.

    Returns:
        The marker overlay, movable with
        :func:`~biotimbre.ui.spectrograms.place_span_marker`.
    """
    with ui.element("div").classes("relative w-full shrink-0"):
        ui.html(waveform_svg(wave, height=height)).classes("w-full shrink-0")
        marker = ui.element("div").classes(
            "absolute top-0 bottom-0 border-x-2 pointer-events-none rounded"
        ).style(
            "border-color: var(--bt-marker-border); "
            "background: var(--bt-marker-fill)"
        )
    place_span_marker(marker, span, wave.start_time, wave.end_time)
    return marker


async def audio_view(
    *,
    spectrogram_url: str,
    audio_path: Path | None,
    start: float,
    end: float,
    span: tuple[float, float] | None,
    height: int,
    columns: int = DEFAULT_COLUMNS,
) -> ui.element:
    """Draw the reader's preferred view of one sample.

    Args:
        spectrogram_url: Image source for the spectrogram view — a route or a
            data URI, as the caller already has it.
        audio_path: The audio itself, for the waveform view. ``None`` or a
            file that has since gone falls back to the spectrogram rather than
            failing: a staged file can be withdrawn while a redraw is in
            flight.
        start: Time at the left edge, in seconds.
        end: Time at the right edge.
        span: ``(from, to)`` to mark, or ``None``.
        height: Height in CSS pixels, so both views sit the same in the page.
        columns: Columns to reduce a waveform to.

    Returns:
        The span marker, whichever view was drawn.
    """
    available = audio_path is not None and Path(audio_path).is_file()
    if preferred_view() == "spectrogram" or not available:
        return marked_spectrogram(spectrogram_url, span, start, end)

    wave = await get_state().run(
        envelope_of, audio_path, start_time=start, end_time=end, columns=columns
    )
    return marked_waveform(wave, span, height=height)
