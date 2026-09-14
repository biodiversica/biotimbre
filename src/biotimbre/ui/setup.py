"""First run: choosing where the library lives.

Started with no ``--data-dir`` and no configured one, the interface asks rather
than quietly making a library in whatever directory it happened to be launched
from. Everything else is unchanged: a configured library opens straight away
and never sees this page.

The picker reads directories from the server, because a browser cannot report a
path — a file input gives names, not locations. See
:func:`biotimbre.ui.routes.register_browser` for what that exposes and for how
long.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import app, ui

from biotimbre.domain.errors import BiotimbreError
from biotimbre.i18n import Translator
from biotimbre.ui.layout import notify_error
from biotimbre.ui.routes import list_directory, unregister_browser
from biotimbre.ui.state import get_state

#: Per-browser storage key for libraries opened before, newest first.
_RECENT_KEY = "recent_libraries"

#: How many to remember. A working set, not a history.
MAX_RECENT = 6


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


def render(t: Translator) -> None:
    """Draw the chooser."""
    state = get_state()
    chosen = {"path": str(Path.cwd())}

    with ui.card().classes("w-full max-w-2xl p-4 gap-3 bg-card"):
        ui.label(t("setup.heading")).classes("text-lg font-medium")
        ui.label(t("setup.explain")).classes("text-sm text-muted")

        path_input = ui.input(
            t("setup.path"), value=chosen["path"]
        ).props("dense outlined").classes("w-full")
        name_input = ui.input(t("setup.name")).props("dense outlined").classes(
            "w-full"
        )
        ui.label(t("setup.name_hint")).classes("text-xs text-muted")

        listing = ui.column().classes("w-full gap-1")

        async def go_to_typed() -> None:
            """Follow a path typed into the field, so it browses as well as tells."""
            await show((path_input.value or "").strip() or None)

        async def show(path: str | None) -> None:
            """Draw one directory's contents."""
            listing.clear()
            try:
                data = await state.run(list_directory, path)
            except Exception as exc:  # noqa: BLE001 — a bad path is user input
                with listing:
                    ui.label(str(exc)).classes("text-sm text-notice")
                return

            path_input.value = data["path"]
            with listing:
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    ui.label(data["path"]).classes("text-sm font-mono grow")
                    if data["parent"]:
                        ui.button(
                            t("setup.parent"),
                            icon="arrow_upward",
                            on_click=lambda p=data["parent"]: show(p),
                        ).props("flat dense no-caps")
                    ui.button(
                        t("setup.home"),
                        icon="home",
                        on_click=lambda: show(None),
                    ).props("flat dense no-caps")

                if not data["directories"]:
                    ui.label(t("setup.empty")).classes("text-xs text-muted")
                with ui.column().classes(
                    "w-full gap-0 max-h-64 overflow-auto rounded bg-surface p-1"
                ):
                    for entry in data["directories"]:
                        with ui.row().classes(
                            "items-center gap-2 w-full px-2 py-1 rounded"
                        ):
                            ui.icon(
                                "folder_special" if entry["is_library"] else "folder",
                                size="1.1rem",
                            ).classes(
                                "text-link" if entry["is_library"] else "text-muted"
                            )
                            ui.button(
                                entry["name"],
                                on_click=lambda p=entry["path"]: show(p),
                            ).props("flat dense no-caps").classes("grow justify-start")
                            if entry["is_library"]:
                                ui.label(t("setup.is_library")).classes(
                                    "text-xs text-muted"
                                )

        path_input.on("change", lambda _: go_to_typed())
        path_input.on("keydown.enter", lambda _: go_to_typed())

        async def create_folder() -> None:
            """Make a subdirectory of the one being looked at."""
            name = (new_folder.value or "").strip()
            if not name:
                return
            target = Path(path_input.value or ".").expanduser() / Path(name).name
            try:
                target.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                ui.notify(str(exc), type="warning", position="top")
                return
            new_folder.value = ""
            await show(str(target))

        with ui.row().classes("items-center gap-2 w-full"):
            new_folder = ui.input(t("setup.new_folder")).props(
                "dense outlined"
            ).classes("grow")
            ui.button(
                t("setup.create"), icon="create_new_folder", on_click=create_folder
            ).props("flat dense no-caps")

        async def open_library() -> None:
            """Open the chosen directory and go to the library."""
            target = (path_input.value or "").strip()
            if not target:
                ui.notify(t("setup.no_path"), type="warning", position="top")
                return
            open_button.disable()
            try:
                await state.run(
                    state.open_library,
                    Path(target).expanduser(),
                    name=(name_input.value or "").strip() or None,
                )
            except BiotimbreError as exc:
                notify_error(t, exc)
                return
            except OSError as exc:
                ui.notify(str(exc), type="warning", position="top")
                return
            finally:
                open_button.enable()

            remember_library(state.settings.data_dir)
            # The listing exists for the moment before a library is open.
            unregister_browser(app)
            ui.navigate.to("/")

        with ui.row().classes("items-center gap-3 mt-2"):
            open_button = ui.button(
                t("setup.open"), icon="folder_open", on_click=open_library
            ).props("text-color=button-text")

        remembered = recent_libraries()
        if remembered:
            ui.label(t("setup.recent")).classes("text-xs text-muted mt-2")
            for entry in remembered:
                with ui.row().classes("items-center gap-2"):
                    ui.icon("history", size="1.1rem").classes("text-muted")
                    ui.button(
                        entry, on_click=lambda p=entry: show(p)
                    ).props("flat dense no-caps").classes("font-mono")

    ui.timer(0.1, lambda: show(chosen["path"]), once=True)
