"""A range picker whose span is capped in the browser.

Used for the query segment, which may not exceed the 15 s cap (D10). The cap
has to be applied where the drag happens: QRange replays the drag-start value
of the edge not being dragged on every move, so a correction sent from the
server is overwritten by the next mouse event and the selection flickers.
"""

from __future__ import annotations

from nicegui import ui


class SegmentRange(ui.range, component="segment_range.js"):
    """A ``ui.range`` that never spans more than ``max_span``.

    Dragging an edge past the cap carries the other edge with it, so the whole
    selection slides the way the user is pushing; dragging inside the
    selection moves it without changing its length. Events and value are those
    of ``ui.range``: ``update:model-value`` while dragging, ``change`` on release.
    """

    def __init__(
        self,
        *,
        min: float,  # noqa: A002 - mirrors ui.range
        max: float,  # noqa: A002
        step: float,
        value: dict[str, float],
        max_span: float,
    ) -> None:
        super().__init__(min=min, max=max, step=step, value=value)
        # ui.range names q-range explicitly; render this component instead.
        self.tag = self.component.tag
        self._props["max-span"] = max_span
