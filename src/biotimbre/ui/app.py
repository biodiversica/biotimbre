"""NiceGUI application: page registration and server startup.

The two pages live in :mod:`biotimbre.ui.admin` and :mod:`biotimbre.ui.search`;
this module wires them to routes and starts the server. Both are imported lazily
inside :func:`_shell` so importing this module does not drag in the whole
interface. Either route serves both pages in one browser page, so switching
between them keeps what was on each.

**Search-only runs** (:attr:`Settings.search_only`, ``gui --search-only``) serve
the search page and nothing else, for people who should be able to query a
library without being able to change it. The library page is not hidden but
withdrawn — ``/`` redirects, so the page is never built and none of its handlers
exist to be reached over the socket — and the library chooser is never mounted,
since choosing a library is how a reader would otherwise get a writable one.
The database itself is opened read-only (:meth:`AppState.open_library`), which is
what makes this a guarantee rather than a convention.
"""

from __future__ import annotations

import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Mapping
from pathlib import Path

from fastapi import Response
from fastapi.responses import RedirectResponse
from nicegui import app, ui
from nicegui.storage import Storage

from biotimbre.config import Settings, load_settings
from biotimbre.ui import routes
from biotimbre.ui.layout import current_views, page
from biotimbre.ui.state import AppState, get_state, set_state

#: Default bind address. Loopback, because this is a local library: exposing it
#: is a deliberate act, not a default (PLAN.md D5/phase 9).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

#: How long the browser thread waits for the server to answer before giving up.
BROWSER_TIMEOUT = 30.0


def user_state_dir() -> Path:
    """Where the interface keeps what it remembers per person, across runs.

    Per user rather than per working directory: the same person starting the
    interface from the launcher one day and from a terminal elsewhere the next
    should find their language, theme and recent libraries both times.

    Returns:
        ``$XDG_STATE_HOME/biotimbre`` (``~/.local/state/biotimbre``) on Linux,
        ``~/Library/Application Support/biotimbre`` on macOS,
        ``%LOCALAPPDATA%\\biotimbre`` on Windows. Not created here.
    """
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        state = os.environ.get("XDG_STATE_HOME")
        base = Path(state) if state else Path.home() / ".local" / "state"
    return base / "biotimbre"


def storage_secret(directory: Path) -> str:
    """The key that signs each browser's storage cookie, made once and kept.

    A key generated per run gives every browser a new identity on every start,
    so everything it remembered — language, theme, recent libraries — is
    orphaned in a file nobody reads again.

    Args:
        directory: Where to keep it. Created if absent.

    Returns:
        The stored key, or a fresh one if it could not be read or written; the
        interface then forgets on restart, as it used to, but still starts.
    """
    path = directory / "storage_secret"
    try:
        kept = path.read_text(encoding="ascii").strip()
        if kept:
            return kept
    except (OSError, UnicodeDecodeError):
        pass

    secret = secrets.token_urlsafe(32)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(secret, encoding="ascii")
        path.chmod(0o600)
    except OSError:
        pass
    return secret


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
async def admin_page() -> Response | None:
    """Library page: add, inspect, edit and delete items.

    Returns:
        A redirect to the search page on a search-only run, where this page
        does not exist; ``None`` otherwise, having rendered it.
    """
    if get_state().settings.search_only:
        # Returned rather than rendered: NiceGUI hands a Response straight back,
        # so the page is never built and its handlers are never registered.
        return RedirectResponse("/search")

    await _shell("admin")
    return None


@ui.page("/search")
async def search_page() -> None:
    """Search page: query, filters, and ranked results with players."""
    await _shell("search")


async def _shell(active: str) -> None:
    """Render the app with ``active`` showing, the other view kept beside it.

    Both pages are panes of the one browser page (:class:`~biotimbre.ui.layout.
    Views`), so moving between them keeps what was on each: typed fields, a
    staged recording, results. Each is built the first time it is shown.

    Args:
        active: ``"admin"`` or ``"search"``.
    """
    from biotimbre.ui import admin, search, setup

    with page(active) as t:
        state = get_state()
        if not state.is_open:
            if state.settings.search_only:
                # The chooser opens libraries, which is exactly what this run
                # is not for. Saying so is all that is left to do.
                ui.label(t("setup.search_only_missing")).classes("text-notice")
                return
            setup.render(t)
            return
        views = current_views()
        if not state.settings.search_only:
            views.add("admin", lambda: admin.render(t))
        views.add("search", lambda: search.render(t))
        await views.show(active, push=False)


def create_app(
    settings: Settings | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> AppState:
    """Build application state, register routes, and start warming models.

    Args:
        settings: Resolved settings. ``None`` loads them from the usual places.
        overrides: Settings fields given on the command line. Passed on so
            they survive the reader opening a different library, which re-reads
            the configuration.

    Returns:
        The installed :class:`~biotimbre.ui.state.AppState`.
    """
    state = AppState(settings, overrides=overrides)
    set_state(state)
    routes.register(app)
    if not state.is_open and not state.settings.search_only:
        # Only while there is a library to choose: the listing is withdrawn as
        # soon as one is open (biotimbre.ui.setup), and never mounted at all on
        # a search-only run, where no library may be chosen and the route would
        # be the one place a request could name a path on the server.
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
    overrides: Mapping[str, object] | None = None,
) -> None:
    """Start the interface.

    Args:
        settings: Resolved settings. ``None`` loads them from the usual places.
        host: Bind address. Defaults to loopback.
        port: Bind port.
        show: Open a browser window on start.
        reload: Enable NiceGUI's auto-reload (development only).
        overrides: Settings fields given on the command line, which outrank
            any config file the reader later opens a library beside.
    """
    create_app(
        settings if settings is not None else load_settings(),
        overrides=overrides,
    )
    state_dir = user_state_dir()
    # NiceGUI otherwise keeps it in ./.nicegui, i.e. wherever it was started.
    Storage.path = state_dir / "browser"
    if show:
        # Opened here rather than by NiceGUI's ``show``, which goes through
        # ``webbrowser.open`` and can open a tab per launcher it tries.
        threading.Thread(
            target=open_browser, args=(host, port), daemon=True
        ).start()
    ui.run(
        host=host,
        port=port,
        # Only a fallback: every page sets its own title to the library's name
        # (biotimbre.ui.layout.page).
        title="biotimbre",
        favicon="🦗",
        show=False,
        reload=reload,
        # Per-browser storage holds interface preferences and recent libraries,
        # which have to survive a restart; see storage_secret().
        storage_secret=storage_secret(state_dir),
        # Auto until a page says otherwise: every page installs its own
        # ui.dark_mode from the reader's stored choice (biotimbre.ui.theming),
        # and following the system is the least wrong thing to show first.
        dark=None,
    )


if __name__ in {"__main__", "__mp_main__"}:
    run()
