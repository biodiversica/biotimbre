"""Sections a reader can fold away, and that stay the way they were left.

A page with an upload form, a bundle importer, a map and a table is a long
scroll for someone who came for one of them. Each part folds; the choice is
kept per browser, beside the language and the theme, so a reader who never
imports bundles is not asked to close that card on every visit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from nicegui import app, ui

#: Per-browser storage key: a mapping of section name to open or closed.
_STORAGE_KEY = "open_sections"


def is_open(name: str, default: bool = True) -> bool:
    """Whether a section was last left open, or ``default`` if never touched."""
    try:
        remembered = app.storage.user.get(_STORAGE_KEY) or {}
    except RuntimeError:
        return default
    value = remembered.get(name)
    return value if isinstance(value, bool) else default


def remember(name: str, value: bool) -> None:
    """Keep a section's state for the next visit."""
    remembered = dict(app.storage.user.get(_STORAGE_KEY) or {})
    remembered[name] = bool(value)
    # Reassigned rather than mutated in place: storage persists on assignment,
    # and a nested change would be lost on the next restart.
    app.storage.user[_STORAGE_KEY] = remembered


def foldable(
    name: str,
    heading: str,
    *,
    heading_class: str = "text-lg font-medium",
    default: bool = True,
) -> ui.expansion:
    """An expansion headed by ``heading`` that remembers being opened or closed.

    Args:
        name: Storage key for this section; unique across the app.
        heading: Translated heading, shown whether open or not.
        heading_class: Classes for the heading text.
        default: Whether it starts open the first time.

    Returns:
        The expansion. Enter it to put content inside.
    """
    panel = (
        ui.expansion(heading, value=is_open(name, default))
        .classes("w-full")
        .props(f'dense header-class="{heading_class} px-0"')
        .mark(f"section-{name}")
    )
    panel.on_value_change(lambda event: remember(name, event.value))
    return panel


@contextmanager
def foldable_card(name: str, heading: str, *, default: bool = False) -> Iterator[ui.expansion]:
    """A page card whose body folds away under its heading.

    Used in place of a card with a heading label: content entered here is laid
    out the way a card lays it out, so a card becomes foldable without its
    contents changing.

    Args:
        name: Storage key for this section.
        heading: Translated heading.
        default: Whether it starts open the first time. Closed unless asked:
            a first visit shows the page's headings, and the reader opens the
            one they came for.

    Yields:
        The expansion, for a caller that needs to react to it opening.
    """
    with ui.card().classes("w-full p-4 gap-3 bg-card"):
        panel = foldable(name, heading, default=default)
        with panel, ui.column().classes("w-full gap-3 pt-2"):
            yield panel
