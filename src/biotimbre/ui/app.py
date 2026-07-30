"""NiceGUI application: page registration and server startup.

The two pages proper arrive with phases 6 and 7. What is here now is the frame
they hang from — shared layout, working navigation and language switching, the
media routes, and models warmed in the background — plus enough content on each
page to prove those parts function end to end.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from nicegui import app, ui

from biotimbre.config import Settings, load_settings
from biotimbre.ui import routes
from biotimbre.ui.layout import page
from biotimbre.ui.state import AppState, get_state, set_state

#: Default bind address. Loopback, because this is a local library: exposing it
#: is a deliberate act, not a default (PLAN.md D5/phase 9).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


@ui.page("/")
async def admin_page() -> None:
    """Library page: add, inspect, edit and delete items."""
    from biotimbre.ui import admin

    with page("admin") as t:
        await admin.render(t)


@ui.page("/search")
async def search_page() -> None:
    """Search page. Query input and results arrive in phase 7."""
    with page("search") as t:
        ui.label(t("search.heading")).classes("text-xl font-medium")
        ui.label(
            t("search.upload_hint", seconds=f"{get_state().settings.max_query_duration_s:g}")
        ).classes("text-sm text-slate-600")
        ui.label(t("search.filters_hint")).classes("text-xs text-slate-500")

        state = get_state()
        if not state.models.ready:
            ui.label(t("app.loading_models_hint")).classes("text-amber-700 text-sm")


def create_app(settings: Settings | None = None) -> AppState:
    """Build application state, register routes, and start warming models.

    Args:
        settings: Resolved settings. ``None`` loads them from the usual places.

    Returns:
        The installed :class:`~biotimbre.ui.state.AppState`.
    """
    state = AppState(settings)
    set_state(state)
    routes.register(app)

    # Warm in the background so the server answers immediately; a first run has
    # 70 MB to download and the page should say so rather than appear to hang.
    app.on_startup(state.start_warming)
    app.on_shutdown(state.close)
    return state


def run(
    settings: Settings | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    show: bool = True,
    reload: bool = False,
) -> None:
    """Start the interface.

    Args:
        settings: Resolved settings. ``None`` loads them from the usual places.
        host: Bind address. Defaults to loopback.
        port: Bind port.
        show: Open a browser window on start.
        reload: Enable NiceGUI's auto-reload (development only).
    """
    create_app(settings if settings is not None else load_settings())
    ui.run(
        host=host,
        port=port,
        title="Biotimbre Library",
        favicon="🦗",
        show=show,
        reload=reload,
        # Per-browser storage holds only the chosen language. Generated per run
        # rather than committed, since nothing here needs to survive a restart.
        storage_secret=secrets.token_urlsafe(32),
        dark=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
