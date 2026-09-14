"""NiceGUI application: page registration and server startup.

The two pages live in :mod:`biotimbre.ui.admin` and :mod:`biotimbre.ui.search`;
this module wires them to routes and starts the server. Both are imported lazily
inside their page function so importing this module does not drag in the whole
interface.
"""

from __future__ import annotations

import secrets
import socket
import threading
import time
import webbrowser

from nicegui import app, ui

from biotimbre.config import Settings, load_settings
from biotimbre.ui import routes
from biotimbre.ui.layout import page
from biotimbre.ui.state import AppState, get_state, set_state

#: Default bind address. Loopback, because this is a local library: exposing it
#: is a deliberate act, not a default (PLAN.md D5/phase 9).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

#: How long the browser thread waits for the server to answer before giving up.
BROWSER_TIMEOUT = 30.0


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """Poll until something accepts connections on ``host:port``.

    Args:
        host: Address to connect to.
        port: Port to connect to.
        timeout: Seconds to keep trying.

    Returns:
        ``True`` once the port answers, ``False`` if the timeout ran out.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def open_browser(host: str, port: int, timeout: float = BROWSER_TIMEOUT) -> None:
    """Open the interface in exactly one browser, once the server answers.

    :func:`webbrowser.open` walks its list of launchers until one *reports*
    success, and a launcher that has already exited when its return code is
    read reports failure even though it did open a tab. On a loaded machine
    -- model warm-up, say -- that is how one call opens three tabs, one per
    launcher tried (xdg-open, gio, gvfs-open). Asking for a single controller
    and driving it directly launches one browser, once.

    Args:
        host: Bind address the server was given.
        port: Bind port the server was given.
        timeout: Seconds to wait for the port before giving up.
    """
    target = "127.0.0.1" if host == "0.0.0.0" else host
    if not _wait_for_port(target, port, timeout):
        return
    try:
        browser = webbrowser.get()
    except webbrowser.Error:
        return
    browser.open(f"http://{target}:{port}/")


@ui.page("/")
async def admin_page() -> None:
    """Library page: add, inspect, edit and delete items."""
    from biotimbre.ui import admin, setup

    with page("admin") as t:
        if not get_state().is_open:
            setup.render(t)
            return
        await admin.render(t)


@ui.page("/search")
async def search_page() -> None:
    """Search page: query, filters, and ranked results with players."""
    from biotimbre.ui import search, setup

    with page("search") as t:
        if not get_state().is_open:
            setup.render(t)
            return
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
    if not state.is_open:
        # Only while there is a library to choose: the listing is withdrawn as
        # soon as one is open (biotimbre.ui.setup).
        routes.register_browser(app)

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
    if show:
        # Opened here rather than by NiceGUI's ``show``, which goes through
        # ``webbrowser.open`` and can open a tab per launcher it tries.
        threading.Thread(
            target=open_browser, args=(host, port), daemon=True
        ).start()
    ui.run(
        host=host,
        port=port,
        title="Biotimbre Library",
        favicon="🦗",
        show=False,
        reload=reload,
        # Per-browser storage holds only the chosen language. Generated per run
        # rather than committed, since nothing here needs to survive a restart.
        storage_secret=secrets.token_urlsafe(32),
        # Auto until a page says otherwise: every page installs its own
        # ui.dark_mode from the reader's stored choice (biotimbre.ui.theming),
        # and following the system is the least wrong thing to show first.
        dark=None,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
