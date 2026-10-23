"""A folder listing read on the server, for the fields that take a library's path.

A browser cannot report a path — a file input gives names, not locations — so
every place the interface asks where a library is, or where one should go,
lists directories from the server instead. This is that listing, drawn once
here rather than once per field: the library chooser uses it to open and to
create, and the importer to name another library.

It calls :func:`~biotimbre.ui.routes.list_directory` directly, on the server.
The chooser's HTTP route (:func:`~biotimbre.ui.routes.register_browser`) is a
separate matter, and withdrawn as soon as a library is open.
"""

from __future__ import annotations

from typing import Callable

from nicegui import ui

from biotimbre.i18n import Translator
from biotimbre.ui.routes import list_directory
from biotimbre.ui.state import get_state


class DirectoryBrowser:
    """Subdirectories of one folder, with the ones holding a library marked.

    Args:
        t: Active translator.
        on_navigate: Called with :func:`~biotimbre.ui.routes.list_directory`'s
            answer each time a folder is shown, so the field it serves can
            follow — every folder, or only libraries, as that field needs.
        closable: Offer a button that hides the listing again, for a listing
            opened on demand rather than shown from the start.
    """

    def __init__(
        self,
        t: Translator,
        on_navigate: Callable[[dict], None],
        *,
        closable: bool = False,
    ):
        self._t = t
        self._on_navigate = on_navigate
        self._closable = closable
        self.container = ui.column().classes("w-full gap-1")

    def hide(self) -> None:
        self.container.set_visibility(False)

    async def show(self, path: str | None) -> None:
        """Draw one folder's contents. ``None`` starts at the reader's home."""
        t = self._t
        self.container.clear()
        self.container.set_visibility(True)
        try:
            data = await get_state().run(list_directory, path)
        except Exception as exc:  # noqa: BLE001 — a bad path is user input
            with self.container:
                ui.label(str(exc)).classes("text-sm text-notice")
            return

        self._on_navigate(data)
        with self.container:
            with ui.row().classes("items-center gap-2 flex-wrap w-full"):
                ui.label(data["path"]).classes("text-sm font-mono grow")
                if data["parent"]:
                    ui.button(
                        t("setup.parent"),
                        icon="arrow_upward",
                        on_click=lambda p=data["parent"]: self.show(p),
                    ).props("flat dense no-caps")
                ui.button(
                    t("setup.home"), icon="home", on_click=lambda: self.show(None)
                ).props("flat dense no-caps")
                if self._closable:
                    ui.button(icon="close", on_click=self.hide).props(
                        "flat dense round"
                    )

            if not data["directories"]:
                ui.label(t("setup.empty")).classes("text-xs text-muted")
            with ui.column().classes(
                "w-full gap-0 max-h-64 overflow-auto rounded bg-surface p-1"
            ):
                for entry in data["directories"]:
                    with ui.row().classes("items-center gap-2 w-full px-2 py-1"):
                        ui.icon(
                            "folder_special" if entry["is_library"] else "folder",
                            size="1.1rem",
                        ).classes(
                            "text-link" if entry["is_library"] else "text-muted"
                        )
                        ui.button(
                            entry["name"],
                            on_click=lambda p=entry["path"]: self.show(p),
                        ).props("flat dense no-caps align=left").classes("grow")
                        if entry["is_library"]:
                            ui.label(t("setup.is_library")).classes(
                                "text-xs text-muted"
                            )
