"""Spectrogram images with a movable time-span marker.

Shared by the pages that show a span over an image: a search result's matched
window, and the segment of a submission that will be kept. The marker is a
positioned overlay rather than something drawn into the PNG, so the same cached
image serves every span, and moving it costs two CSS properties rather than a
re-render.
"""

from __future__ import annotations

from nicegui import ui


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


def marked_spectrogram(
    url: str, span: tuple[float, float] | None, start: float, end: float
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
        marker = ui.element("div").classes(
            "absolute top-0 bottom-0 border-x-2 pointer-events-none rounded"
        ).style(
            "border-color: var(--bt-marker-border); "
            "background: var(--bt-marker-fill)"
        )
    place_span_marker(marker, span, start, end)
    return marker
