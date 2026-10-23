"""Choosing a library when the interface was started without one.

Started without ``--data-dir``, the interface asks rather than opening or
quietly making a library wherever it happened to be launched from. Creating a
library and opening one are separate choices, because they are separate
intentions: someone who means to open last season's library and mistypes its
path should be told it is not there, not handed a new, empty one.

A configuration file's ``data_dir`` does not skip the question; it is offered
first (:attr:`~biotimbre.ui.state.AppState.suggested_library`).

The folder listing is read on the server, because a browser cannot report a
path — a file input gives names, not locations. See
:func:`biotimbre.ui.routes.register_browser` for what that exposes and for how
long.
"""

from __future__ import annotations

import re
from pathlib import Path

from nicegui import app, ui

from biotimbre.domain.errors import BiotimbreError
from biotimbre.i18n import Translator
from biotimbre.ui.browser import DirectoryBrowser
from biotimbre.ui.layout import notify_error
from biotimbre.ui.routes import holds_library, unregister_browser
from biotimbre.ui.state import get_state

#: Per-browser storage key for libraries opened before, newest first.
_RECENT_KEY = "recent_libraries"

#: How many to remember. A working set, not a history.
MAX_RECENT = 6

#: Names Windows will not give a file or folder, whatever the extension.
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def recent_libraries() -> list[str]:
    """Libraries opened from this browser before, newest first."""
    try:
        remembered = app.storage.user.get(_RECENT_KEY)
    except RuntimeError:
        return []
    if not isinstance(remembered, list):
        return []
    return [str(entry) for entry in remembered][:MAX_RECENT]


def remember_library(path: str | Path) -> None:
    """Put a library at the head of this browser's list."""
    entry = str(path)
    kept = [other for other in recent_libraries() if other != entry]
    try:
        app.storage.user[_RECENT_KEY] = [entry, *kept][:MAX_RECENT]
    except RuntimeError:
        pass  # a library that cannot be remembered is still worth opening


def library_folder_name(name: str) -> str:
    """The folder a new library called ``name`` is created in.

    Lower-cased, with every run of characters that is not a letter, digit or
    underscore turned into one hyphen: what is left is a valid name on every
    system the interface runs on, and still reads as the library's name.
    Accented letters are letters, so "Cigarras da Mata Atlântica" keeps them.

    Args:
        name: The library's name as typed.

    Returns:
        A folder name, never empty.
    """
    folder = re.sub(r"\W+", "-", name.strip().lower()).strip("-_")
    if not folder:
        return "library"
    if folder.upper() in _RESERVED_NAMES:
        return f"{folder}-library"
    return folder


def new_library_problem(target: Path) -> str | None:
    """Why a new library cannot be created at ``target``, if it cannot.

    Args:
        target: The folder the library would be created as.

    Returns:
        A catalogue key under ``setup.`` naming the problem, or ``None`` when
        the folder is absent or empty.
    """
    if holds_library(target):
        return "already_library"
    try:
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            return "not_empty"
    except OSError:
        return "not_empty"
    return None


def render(t: Translator) -> None:
    """Draw the chooser."""
    state = get_state()
    suggested = state.suggested_library
    remembered = recent_libraries()
    cwd = Path.cwd()

    # What to offer first: a library the configuration names, or the last one
    # opened here, is most likely what was meant; with neither, a new one.
    if suggested is not None and holds_library(suggested):
        opening, start = True, suggested
    elif remembered:
        opening, start = True, Path(remembered[0])
    else:
        opening, start = False, cwd
    # A configured data_dir with nothing in it yet is a library to create
    # exactly there: its folder becomes the name, its parent the location.
    if suggested is not None and not holds_library(suggested):
        location, name = suggested.parent, suggested.name
    else:
        location, name = cwd, ""

    async def finish(target: Path, name: str | None) -> bool:
        """Open (creating if needed) ``target``, then go to the library."""
        try:
            await state.run(state.open_library, target, name=name)
        except BiotimbreError as exc:
            notify_error(t, exc)
            return False
        except OSError as exc:
            ui.notify(str(exc), type="warning", position="top")
            return False
        remember_library(state.settings.data_dir)
        # The listing exists for the moment before a library is open.
        unregister_browser(app)
        ui.navigate.to("/")
        return True

    with ui.card().classes("w-full max-w-2xl p-4 gap-3 bg-card"):
        ui.label(t("setup.heading")).classes("text-lg font-medium")
        ui.label(t("setup.explain")).classes("text-sm text-muted")

        with ui.tabs().classes("w-full").props("align=left no-caps") as tabs:
            create_tab = ui.tab("create", label=t("setup.tab_create"), icon="add")
            open_tab = ui.tab(
                "open", label=t("setup.tab_open"), icon="folder_open"
            ).mark("setup-tab-open")
        create_tab.mark("setup-tab-create")

        with ui.tab_panels(tabs, value="open" if opening else "create").classes(
            "w-full bg-card"
        ):
            with ui.tab_panel(create_tab).classes("p-0 gap-3"):
                _create_panel(t, finish, location=location, name=name)
            with ui.tab_panel(open_tab).classes("p-0 gap-3"):
                _open_panel(t, finish, start=start, remembered=remembered)


def _create_panel(t: Translator, finish, *, location: Path, name: str) -> None:
    """Name a library and say where it goes; it gets a folder of its own."""
    name_input = ui.input(t("setup.name"), value=name).props(
        "dense outlined"
    ).classes("w-full").mark("setup-name")
    with ui.row().classes("items-center gap-2 w-full no-wrap"):
        location_input = ui.input(
            t("setup.location"), value=str(location)
        ).props("dense outlined").classes("grow").mark("setup-location")
        browse = ui.button(t("setup.browse"), icon="folder_open").props(
            "flat dense no-caps"
        )
    ui.label(t("setup.location_hint")).classes("text-xs text-muted")

    browser = DirectoryBrowser(
        t, lambda data: location_input.set_value(data["path"]), closable=True
    )
    browser.hide()
    browse.on_click(
        lambda: browser.show((location_input.value or "").strip() or None)
    )

    preview = ui.label().classes("text-sm font-mono break-all").mark(
        "setup-creates"
    )

    def target() -> Path | None:
        typed = (name_input.value or "").strip()
        place = (location_input.value or "").strip()
        if not typed or not place:
            return None
        return Path(place).expanduser() / library_folder_name(typed)

    def update_preview() -> None:
        where = target()
        preview.set_text(t("setup.creates", path=str(where)) if where else "")

    name_input.on_value_change(lambda _: update_preview())
    location_input.on_value_change(lambda _: update_preview())
    update_preview()

    async def create() -> None:
        typed = (name_input.value or "").strip()
        where = target()
        if not typed or where is None:
            ui.notify(t("setup.name_required"), type="warning", position="top")
            return
        problem = await get_state().run(new_library_problem, where)
        if problem:
            ui.notify(
                t(f"setup.{problem}", path=str(where)),
                type="warning",
                position="top",
            )
            return
        button.disable()
        try:
            await finish(where, typed)
        finally:
            button.enable()

    with ui.row().classes("w-full justify-end"):
        button = ui.button(
            t("setup.create"), icon="add", on_click=create
        ).props("text-color=button-text").mark("setup-create")


def _open_panel(
    t: Translator, finish, *, start: Path, remembered: list[str]
) -> None:
    """Find a library that already exists; nothing is created from here."""
    with ui.row().classes("items-center gap-2 w-full no-wrap"):
        path_input = ui.input(t("setup.path"), value=str(start)).props(
            "dense outlined"
        ).classes("grow").mark("setup-path")

    browser = DirectoryBrowser(t, lambda data: path_input.set_value(data["path"]))

    async def go_to_typed() -> None:
        """Follow a path typed into the field, so it browses as well as tells."""
        await browser.show((path_input.value or "").strip() or None)

    path_input.on("keydown.enter", lambda _: go_to_typed())

    if remembered:
        ui.label(t("setup.recent")).classes("text-xs text-muted mt-2")
        for entry in remembered:
            with ui.row().classes("items-center gap-2"):
                ui.icon("history", size="1.1rem").classes("text-muted")
                ui.button(
                    entry, on_click=lambda p=entry: browser.show(p)
                ).props("flat dense no-caps").classes("font-mono")

    async def open_existing() -> None:
        typed = (path_input.value or "").strip()
        if not typed:
            ui.notify(t("setup.no_path"), type="warning", position="top")
            return
        where = Path(typed).expanduser()
        if not await get_state().run(holds_library, where):
            ui.notify(
                t("setup.not_a_library", path=str(where)),
                type="warning",
                position="top",
            )
            return
        button.disable()
        try:
            await finish(where, None)
        finally:
            button.enable()

    with ui.row().classes("w-full justify-end"):
        button = ui.button(
            t("setup.open"), icon="folder_open", on_click=open_existing
        ).props("text-color=button-text").mark("setup-open")

    ui.timer(0.1, lambda: browser.show(str(start)), once=True)
