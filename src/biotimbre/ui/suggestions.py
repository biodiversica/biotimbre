"""Offering values already in the library while a field is being typed.

Typing "Dendropsophus minutus" for the fortieth time is tedious, and typing it
slightly differently once is worse: a tag filter then misses that item. So the
taxa, device, microphone and recordist fields offer back what the library
already holds.

**Asked for, never shipped whole.** A library can be very large, so the page
never holds every value. Each keystroke asks the database for a handful of
matches (:meth:`~biotimbre.db.repository.Repository.suggest`), and only those
travel to the browser.

**The newest question wins.** Answers can arrive out of order — a short prefix
matches more and can take longer than the longer one typed after it — so each
answer is dropped unless it is to the last question asked.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from nicegui import ui

from biotimbre.ui.state import get_state

#: Looks up suggestions for one field: what has been typed, to what matches.
Lookup = Callable[[str], Awaitable[list[str]]]

#: Seconds between lookups while typing. The trailing keystroke always gets
#: its lookup, so the last thing typed is what the suggestions answer.
THROTTLE_S = 0.2


async def library_suggestions(field: str, text: str) -> list[str]:
    """Values the open library already holds for a field, off the event loop.

    Args:
        field: ``taxa``, ``device``, ``microphone`` or ``recorded_by``.
        text: What has been typed so far.

    Returns:
        Matching values, best first.
    """
    state = get_state()
    return await state.run(state.service.suggest, field, text)


class _Latest:
    """Tells an answer to the most recent question from one already overtaken."""

    def __init__(self) -> None:
        self._asked = 0

    def ask(self) -> int:
        self._asked += 1
        return self._asked

    def is_current(self, ticket: int) -> bool:
        return ticket == self._asked


def suggest_tags(select: ui.select, lookup: Lookup) -> None:
    """Fill a multiple-choice tag select with matching tags as it is typed in.

    The select keeps its own behaviour: a tag the library has never seen is
    still added by typing it and pressing Enter.

    Args:
        select: A ``multiple`` select with ``new_value_mode`` set.
        lookup: Fetches tags matching the typed text.
    """
    latest = _Latest()

    async def refresh(text: str) -> None:
        ticket = latest.ask()
        found = await lookup(text)
        if not latest.is_current(ticket) or select.is_deleted:
            return
        chosen = list(select.value or [])
        # The chosen tags stay among the options: a value missing from them
        # is silently dropped from the selection.
        options = found + [tag for tag in chosen if tag not in found]
        select.set_options(options, value=chosen)

    async def on_typed(event) -> None:
        await refresh(event.args if isinstance(event.args, str) else "")

    async def on_focus(_event) -> None:
        await refresh("")

    select.on("input-value", on_typed, throttle=THROTTLE_S, leading_events=False)
    select.on("focus", on_focus)


def suggest_text(field: ui.input, lookup: Lookup) -> ui.menu:
    """Show matching earlier values under a text input, to pick with a click.

    The input stays a plain input: what is typed is the value, with no Enter
    needed to commit it, and a value not in the library is simply typed.

    Args:
        field: The input to suggest for.
        lookup: Fetches values matching the typed text.

    Returns:
        The menu holding the suggestions.
    """
    latest = _Latest()
    # Typing only counts while the field has focus. Otherwise a value written
    # by code — an iNaturalist fetch, the form being reset — would pop the
    # menu open over a field nobody is in.
    focus = {"in": False}

    with field:
        menu = ui.menu().props(
            "no-parent-event no-focus no-refocus fit max-height=280px"
        )

    def pick(value: str) -> None:
        field.value = value
        menu.close()

    async def refresh(text: str) -> None:
        ticket = latest.ask()
        found = await lookup(text)
        if not latest.is_current(ticket) or field.is_deleted:
            return
        # Offering back exactly what is already typed is noise.
        found = [value for value in found if value != text]
        menu.clear()
        if not found or not focus["in"]:
            menu.close()
            return
        with menu:
            for value in found:
                ui.menu_item(value, on_click=lambda _, value=value: pick(value))
        menu.open()

    async def on_focus(_event) -> None:
        focus["in"] = True
        await refresh(field.value or "")

    def on_blur(_event) -> None:
        # The menu is not closed here: a click on a suggestion blurs the field
        # first, and closing now would swallow the click. A click anywhere
        # else closes the menu by itself.
        focus["in"] = False

    async def on_typed(_event) -> None:
        if focus["in"]:
            await refresh(field.value or "")

    field.on("focus", on_focus)
    field.on("blur", on_blur)
    # What the browser sends as the text changes, so typing alone triggers it.
    field.on("update:value", on_typed, throttle=THROTTLE_S, leading_events=False)
    return menu
