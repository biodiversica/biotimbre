"""Shared page frame: header, navigation, language switcher, model status.

The language and the theme are held per browser session rather than in the
settings file, so two people using the same local instance can each read it in
their own language and their own theme, and switching does not rewrite
configuration.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import Any

from nicegui import app, ui

from biotimbre.domain.errors import BiotimbreError
from biotimbre.i18n import DEFAULT_LANGUAGE, LANGUAGE_NAMES, LANGUAGES, Translator
from biotimbre.ui.state import get_state
from biotimbre.ui.theming import (
    THEME_ICONS,
    THEME_MODES,
    apply_mode_script,
    quasar_dark,
    set_theme_mode,
    theme_css,
    theme_mode,
    theme_variables,
)

_LANGUAGE_KEY = "language"

#: User storage key of the token a reloading tab left its work under.
_CARRY_KEY = "carry"

log = logging.getLogger(__name__)

#: Where each view lives. Both are served by the same shell, so either URL
#: opens the app on that view, and reload or a bookmark lands where it was.
VIEW_PATHS = {"admin": "/", "search": "/search"}

#: Client storage key of the :class:`Views` for the open tab.
_VIEWS_KEY = "views"

#: The footnote at the bottom of every page. Not translated: it is a name.
FOOTNOTE_TEXT = "~/biodiversica"
FOOTNOTE_URL = "https://biodiversica.xyz/"

#: The footnote link takes the header's text colours, as its bar takes the
#: header's background: dimmed like an unselected tab, full on hover.
_FOOTNOTE_CSS = """
.bt-footnote { color: var(--bt-header-inactive-text); }
.bt-footnote:hover { color: var(--bt-header-text); }
"""

#: One typeface for the whole interface: the system's own monospace, so
#: nothing is downloaded and the page works offline. Quasar's controls inherit
#: it from the body; Leaflet names a font of its own for the map's labels.
_FONT_CSS = """
:root {
  --bt-font: ui-monospace, SFMono-Regular, Menlo, Consolas, "DejaVu Sans Mono",
             "Liberation Mono", monospace;
}
body, .leaflet-container { font-family: var(--bt-font); }
"""


def translator() -> Translator:
    """Translator for the current browser session."""
    try:
        language = app.storage.user.get(_LANGUAGE_KEY)
    except Exception:
        # Storage is unavailable outside a request (e.g. during startup).
        language = None
    if language not in LANGUAGES:
        language = get_state().settings.language
    return Translator(language)


async def set_language(language: str) -> None:
    """Persist the session's language and reload so every label re-renders.

    Every label is translated as the page is built, so a new language means a
    new page. What is on the old one -- a loaded recording, typed fields,
    results -- is carried over rather than lost with it.
    """
    if language not in LANGUAGES:
        return
    views = current_views()
    if views is not None:
        await views.carry()
    app.storage.user[_LANGUAGE_KEY] = language
    ui.navigate.reload()


def _take_carried() -> dict[str, Any]:
    """What the previous page of this browser left before reloading, if any."""
    try:
        token = app.storage.user.pop(_CARRY_KEY, None)
    except RuntimeError:
        # Storage is unavailable outside a request.
        return {}
    if token is None:
        return {}
    return get_state().carried.pop(token, {})


def _model_status(t: Translator) -> None:
    """Say how the models are doing in a passing notice rather than in the header.

    Ready by the time the page opens is the usual case, and says nothing.
    Otherwise a notice with a spinner stays while they load, then turns into
    one saying they are ready, which leaves by itself, or into the error,
    which stays until it is closed: without the models nothing can be
    searched. One notice changed in place rather than one replacing another,
    so the reader sees it turn rather than flicker.
    """
    status = get_state().models
    if status.ready:
        return
    notice = ui.notification(position="top", timeout=None)
    if status.failed:
        _show_failure(t, notice, status.failed)
        return
    notice.props["options"]["caption"] = t("app.loading_models_hint")
    notice.spinner = True
    notice.message = t("app.loading_models")

    # Poll until warm-up finishes. Only the notice changes: a reload would
    # throw away whatever the reader has started on meanwhile.
    def poll() -> None:
        status = get_state().models
        if not (status.ready or status.failed):
            return
        timer.cancel()
        notice.spinner = False
        if status.failed:
            _show_failure(t, notice, status.failed)
            return
        notice.props["options"]["caption"] = ", ".join(
            f"{model} {dim}d" for model, dim in sorted(status.dimensions.items())
        )
        notice.type = "positive"
        notice.timeout = 5
        notice.message = t("app.models_ready")

    timer = ui.timer(1.5, poll)


def _show_failure(t: Translator, notice: ui.notification, error: str) -> None:
    """Turn ``notice`` into the models' failure, kept until it is closed."""
    notice.props["options"].pop("caption", None)
    notice.type = "negative"
    notice.multi_line = True
    notice.close_button = True
    notice.message = t("error.generic", message=error)


def _theme_toggle(t: Translator, mode: str, dark_mode: ui.dark_mode) -> None:
    """The header control that cycles system → light → dark.

    Switching needs no reload: the colours are variables the browser resolves
    itself, so only ``data-theme`` and Quasar's dark flag have to move.

    Args:
        t: Active translator.
        mode: The mode the page was rendered with.
        dark_mode: The page's Quasar dark-mode element.
    """
    current = mode

    def cycle() -> None:
        nonlocal current
        current = THEME_MODES[(THEME_MODES.index(current) + 1) % len(THEME_MODES)]
        set_theme_mode(current)
        dark_mode.value = quasar_dark(current)
        if current == "system":
            ui.run_javascript("delete document.documentElement.dataset.theme;")
        else:
            ui.run_javascript(
                f'document.documentElement.dataset.theme = "{current}";'
            )
        button.props(f"icon={THEME_ICONS[current]}")
        tip.set_text(t(f"app.theme.{current}"))

    button = ui.button(icon=THEME_ICONS[mode], color=None, on_click=cycle).props(
        "flat dense round"
    ).style("color: var(--bt-header-text)")
    with button:
        tip = ui.tooltip(t(f"app.theme.{mode}"))


class Views:
    """The pages of one browser tab, built once and kept while it is open.

    Navigating between ``/`` and ``/search`` used to load a fresh page, which
    threw away everything on the one being left: typed fields, a staged
    recording, search results. Both views are now panes of one page instead,
    and switching only changes which is shown. A view is built the first time
    it is shown rather than up front, so opening the app costs one page, not
    two. The address follows along, so a reload, a bookmark and the browser's
    back button all still mean what they did.

    Args:
        active: The view the tab was opened on.
    """

    def __init__(self, active: str) -> None:
        self.active = active
        self._builders: dict[str, Callable[[], Awaitable[None]]] = {}
        self._panes: dict[str, ui.element] = {}
        self._built: set[str] = set()
        self._hooks: dict[str, list[Callable[[], None]]] = {}
        self._tabs: dict[str, ui.button] = {}
        self._keepers: dict[str, Callable[[], Awaitable[Any]]] = {}
        #: Work the previous page left, by view, until each view is built.
        self._carried = _take_carried()

    def add(self, name: str, build: Callable[[], Awaitable[None]]) -> None:
        """Give a view a pane in the current container, to build when first shown.

        Args:
            name: A key of :data:`VIEW_PATHS`.
            build: Renders the view into the pane.
        """
        self._builders[name] = build
        self._panes[name] = ui.column().classes("w-full gap-4")
        self._panes[name].set_visibility(False)

    def add_tab(self, name: str, tab: ui.button) -> None:
        """Register the header tab for ``name``, so it follows the switches."""
        self._tabs[name] = tab
        _style_tab(tab, name == self.active)

    async def keep(
        self,
        name: str,
        snapshot: Callable[[], Awaitable[Any]],
        restore: Callable[[Any], Awaitable[None]],
    ) -> None:
        """Keep a view's work across a reload for a change of language.

        Called once the view is built: if the page before this one left work
        for it, it is restored now.

        Args:
            name: A key of :data:`VIEW_PATHS`.
            snapshot: Returns what to carry over, or ``None`` for nothing.
            restore: Puts back what ``snapshot`` returned, on the new page.
        """
        self._keepers[name] = snapshot
        carried = self._carried.pop(name, None)
        if carried is not None:
            await restore(carried)

    async def carry(self) -> None:
        """Leave every view's work for the page that replaces this one.

        A view not yet built holds on to what the page before left for it,
        so a second change of language does not drop it either.
        """
        carried = dict(self._carried)
        for name, snapshot in self._keepers.items():
            try:
                kept = await snapshot()
            except BiotimbreError:
                # A view that cannot be kept is rebuilt empty, as it always
                # was; the change of language goes ahead regardless.
                log.exception("could not keep the %s view's work", name)
                continue
            if kept is not None:
                carried[name] = kept
        if carried:
            token = secrets.token_urlsafe(16)
            get_state().carried[token] = carried
            app.storage.user[_CARRY_KEY] = token

    def when_shown(self, name: str, hook: Callable[[], None]) -> None:
        """Run ``hook`` every time ``name`` is switched to after being built."""
        self._hooks.setdefault(name, []).append(hook)

    async def show(self, name: str, *, push: bool = True) -> None:
        """Switch to a view, building it if this is its first time.

        A view this tab does not hold -- the library chooser is up, say --
        is navigated to the old way.

        Args:
            name: A key of :data:`VIEW_PATHS`.
            push: Record the switch in the browser history. Off when the
                switch is the history moving, or the page's first view.
        """
        if name not in self._builders:
            ui.navigate.to(VIEW_PATHS[name])
            return
        previous, self.active = self.active, name
        for other, pane in self._panes.items():
            pane.set_visibility(other == name)
        for tab_name, tab in self._tabs.items():
            _style_tab(tab, tab_name == name)
        if push and previous != name:
            ui.run_javascript(
                f'history.pushState({{}}, "", "{VIEW_PATHS[name]}");'
            )
        if name not in self._built:
            self._built.add(name)
            with self._panes[name]:
                await self._builders[name]()
            return
        for hook in self._hooks.get(name, ()):
            with self._panes[name]:
                hook()


def current_views() -> Views | None:
    """The :class:`Views` of the tab being handled, if it has any."""
    try:
        return app.storage.client.get(_VIEWS_KEY)
    except RuntimeError:
        return None


def _style_tab(tab: ui.button, is_active: bool) -> None:
    """Colour a header tab as the one shown, or not."""
    text_color = (
        "var(--bt-header-text)" if is_active else "var(--bt-header-inactive-text)"
    )
    background = "var(--bt-header-active-bg)" if is_active else "transparent"
    tab.style(f"color: {text_color}; background: {background}")


@contextmanager
def page(active: str):
    """Wrap page content in the shared frame.

    Args:
        active: Which nav entry to highlight, ``"admin"`` or ``"search"``. On a
            search-only run there is only the one, and the library tab is left
            out rather than drawn dead.

    Yields:
        The translator for this session, so callers need not fetch it again.
    """
    t = translator()
    settings = get_state().settings
    mode = theme_mode()
    views = Views(active)
    app.storage.client[_VIEWS_KEY] = views

    # Both palettes reach the page as one set of variables; which half of each
    # is used comes from color-scheme alone. The script pins the reader's
    # choice before the first paint — NiceGUI builds the body afterwards, so
    # leaving it to the app would flash the other theme.
    ui.add_head_html(theme_css(settings.theme, settings.dark_theme))
    ui.add_head_html(apply_mode_script(mode))
    ui.add_css(_FONT_CSS)
    # Quasar's own dark flag, which restyles its components from the inside.
    dark_mode = ui.dark_mode(value=quasar_dark(mode))
    # Quasar's brand colours point at those variables rather than at literals,
    # which is what lets bg-card, text-muted and friends follow the theme.
    ui.colors(dark_page="var(--bt-page-bg)", **theme_variables())
    ui.query("body").style("background: var(--bt-page-bg); color: var(--bt-text)")

    state = get_state()
    identity = state.service.identity() if state.is_open else None
    library_name = (
        (identity.name or t("app.unnamed_library"))
        if identity is not None
        else t("setup.no_library")
    )
    # The browser tab says which library, like the header does: with two open
    # side by side, "Library" twice tells them apart no better than nothing.
    ui.page_title(library_name)

    # Pinned to the window like the footer. No ``relative`` class: Quasar
    # already positions the header (fixed), and ``relative`` would undo the
    # pinning.
    #
    # Two lines: the title on its own, centred; below it, on the left, the
    # library's name and the tabs, and on the right the settings. The name
    # gives way first: it is cut short as the window narrows, down to a
    # minimum, and only below that does the line wrap, the settings going to
    # a line of their own.
    with ui.header().classes("px-4 py-2 gap-1 shadow-sm").style(
        "background: var(--bt-header-bg); flex-direction: column;"
        " align-items: stretch"
    ):
        ui.label(t("app.title")).classes(
            "self-center text-base font-medium font-mono"
        ).style("color: var(--bt-header-text)")

        with ui.row().classes("w-full items-center justify-between gap-4"):
            # Starting from nothing rather than from its content, so wrapping
            # waits until the name is down to its minimum. Wrapping itself, on
            # a phone, which puts the tabs below the name.
            with ui.row().classes("items-center gap-x-6 gap-y-1").style(
                "flex: 1 1 0"
            ):
                # Only once a library is open: before that there is no name to
                # show, and the page below is where one is chosen.
                if identity is not None:
                    name_part = ui.row(wrap=False).classes("items-baseline gap-2")
                    with name_part.style(
                        # Also from nothing, up to its content, so the tabs follow
                        # it rather than the end of the line.
                        "flex: 1 1 0; min-width: 8rem; max-width: max-content"
                    ):
                        ui.label(library_name).classes(
                            "text-sm font-medium truncate"
                        ).style("color: var(--bt-header-text); min-width: 0").tooltip(
                            library_name
                        )
                        ui.label(f"#{identity.uuid[:8]}").classes(
                            "text-xs font-mono shrink-0"
                        ).style("color: var(--bt-header-inactive-text)").tooltip(
                            identity.uuid
                        )
                        if not settings.search_only:
                            # Beside the name it changes. Not on a search-only
                            # run, which serves one library and no other.
                            from biotimbre.ui.setup import confirm_leave

                            ui.button(
                                icon="swap_horiz",
                                on_click=lambda: confirm_leave(t, library_name),
                            ).props("flat dense round size=sm").style(
                                "color: var(--bt-header-inactive-text)"
                            ).classes("shrink-0").tooltip(
                                t("setup.switch")
                            ).mark("switch-library")
                        if settings.search_only:
                            # Beside the library's name, because that is the
                            # question this answers: which library, and in what
                            # capacity. Without it the only evidence is a
                            # missing tab, which reads as a bug.
                            with ui.row(wrap=False).classes(
                                "items-center gap-1 shrink-0"
                            ).style("color: var(--bt-header-inactive-text)"):
                                ui.icon("lock", size="0.9rem")
                                ui.label(t("app.search_only")).classes(
                                    "text-xs"
                                ).tooltip(t("app.search_only_hint"))

                with ui.row(wrap=False).classes("items-center gap-1 shrink-0"):
                    names = list(VIEW_PATHS)
                    if settings.search_only:
                        # The library page is not served on this run, so a tab
                        # pointing at it would be a link to a redirect.
                        names.remove("admin")
                    icons = {"admin": "library_music", "search": "search"}
                    for name in names:
                        tab = ui.button(
                            t(f"app.nav.{name}"),
                            icon=icons[name],
                            color=None,
                            on_click=lambda name=name: views.show(name),
                        ).props("flat dense no-caps")
                        views.add_tab(name, tab)

            # Pushed right, on a line of its own too.
            with ui.row(wrap=False).classes("items-center gap-4 shrink-0").style(
                "margin-left: auto"
            ):
                _theme_toggle(t, mode, dark_mode)
                ui.select(
                    {code: LANGUAGE_NAMES[code] for code in LANGUAGES},
                    value=t.language or DEFAULT_LANGUAGE,
                    on_change=lambda event: set_language(event.value),
                ).props(
                    # The header is dark, so the field and its popup are too. On a
                    # dark popup Quasar marks the current option with the primary
                    # colour, a dark green like the popup's own background, which
                    # all but disappears; accent reads against both. ``color``
                    # keeps the focus ring off primary green as well.
                    'dense outlined dark options-dense color=white'
                    ' options-selected-class="text-accent"'
                ).style(
                    # Wide enough for the longest name, plus the field's padding
                    # and dropdown arrow.
                    f"width: calc({max(map(len, LANGUAGE_NAMES.values()))}ch + 3.5rem)"
                ).tooltip(
                    t("app.language")
                )

    # Back and forward move between views without leaving the page,
    # so they have to be told apart from a real navigation.
    ui.add_body_html(
        "<script>window.addEventListener('popstate', () =>"
        " emitEvent('bt_view', location.pathname));</script>"
    )
    paths = {path: name for name, path in VIEW_PATHS.items()}
    ui.on(
        "bt_view",
        lambda event: views.show(paths.get(event.args, "search"), push=False),
    )

    _model_status(t)

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        yield t

    # Pinned to the window rather than following the content, which jumps
    # whenever a section folds or unfolds. Quasar reserves the footer's height
    # below the page, so nothing is hidden behind it. In the header's colours,
    # so the page is framed top and bottom alike.
    # A class rather than an inline colour, which could not change on hover.
    ui.add_css(_FOOTNOTE_CSS)
    with ui.footer().classes("justify-center py-1").style(
        "background: var(--bt-header-bg)"
    ):
        ui.link(FOOTNOTE_TEXT, FOOTNOTE_URL, new_tab=True).classes(
            "bt-footnote text-xs font-mono no-underline hover:underline"
        ).mark("footnote")


def notify_error(t: Translator, exc: Exception) -> None:
    """Show an error to the user, using field information when present.

    A :class:`~biotimbre.domain.errors.ValidationError` names the offending field,
    which makes a far more useful message than a bare string.

    Args:
        t: Active translator.
        exc: The exception to report.
    """
    from biotimbre.domain.errors import ValidationError

    if isinstance(exc, ValidationError) and exc.field:
        message = t(
            "error.validation",
            field=exc.field,
            message=str(exc).split(": ", 1)[-1],
        )
    else:
        message = t("error.generic", message=str(exc))
    ui.notify(message, type="negative", position="top", timeout=6000)
