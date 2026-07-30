"""NiceGUI application: page registration and server startup.

The two pages live in :mod:`biotimbre.ui.admin` and :mod:`biotimbre.ui.search`;
this module wires them to routes and starts the server. Both are imported lazily
inside their page function so importing this module does not drag in the whole
interface.
"""

from __future__ import annotations

import secrets

from nicegui import app, ui

from biotimbre.config import Settings, load_settings
from biotimbre.ui import routes
from biotimbre.ui.layout import page
from biotimbre.ui.state import AppState, set_state

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
    """Search page: query, filters, and ranked results with players."""
    from biotimbre.ui import search

    with page("search") as t:
        await search.render(t)


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
