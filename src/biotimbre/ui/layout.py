"""Shared page frame: header, navigation, language switcher, model status.

The language and the theme are held per browser session rather than in the
settings file, so two people using the same local instance can each read it in
their own language and their own theme, and switching does not rewrite
configuration.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import contextmanager

from nicegui import app, ui

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


def set_language(language: str) -> None:
    """Persist the session's language and reload so every label re-renders."""
    if language in LANGUAGES:
        app.storage.user[_LANGUAGE_KEY] = language
        ui.navigate.reload()


@ui.refreshable
def _model_badge(t: Translator) -> None:
    """Show whether models are ready, loading, or failed."""
    status = get_state().models

    if status.failed:
        with ui.row().classes("items-center gap-1").style(
            "color: var(--bt-header-status-failed)"
        ):
            ui.icon("error_outline", size="1.1rem")
            ui.label(t("error.generic", message=status.failed)).classes(
                "text-xs max-w-64 truncate"
            ).tooltip(status.failed)
        return

    if status.ready:
        with ui.row().classes("items-center gap-1").style(
            "color: var(--bt-header-status-ready)"
        ):
            ui.icon("check_circle_outline", size="1.1rem")
            summary = ", ".join(
                f"{model} {dim}d" for model, dim in sorted(status.dimensions.items())
            )
            ui.label(t("app.models_ready")).classes("text-xs").tooltip(summary)
        return

    with ui.row().classes("items-center gap-2").style(
        "color: var(--bt-header-status-loading)"
    ):
        ui.spinner(size="1rem")
        ui.label(t("app.loading_models")).classes("text-xs").tooltip(
            t("app.loading_models_hint")
        )
    # Poll until warm-up finishes, then redraw the badge. Only the badge: a
    # reload would throw away whatever the reader has started on meanwhile.

    def poll() -> None:
        status = get_state().models
        if status.ready or status.failed:
            _model_badge.refresh()

    ui.timer(1.5, poll)


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
    # Quasar's own dark flag, which restyles its components from the inside.
    dark_mode = ui.dark_mode(value=quasar_dark(mode))
    # Quasar's brand colours point at those variables rather than at literals,
    # which is what lets bg-card, text-muted and friends follow the theme.
    ui.colors(dark_page="var(--bt-page-bg)", **theme_variables())
    ui.query("body").style("background: var(--bt-page-bg); color: var(--bt-text)")

    # Pinned to the window like the footer. No ``relative`` class: Quasar
    # already positions the header (fixed), which is what the centred name
    # below anchors to, and ``relative`` would undo the pinning.
    with ui.header().classes(
        "items-center justify-between px-4 py-2 shadow-sm"
    ).style("background: var(--bt-header-bg)"):
        state = get_state()
        identity = state.service.identity() if state.is_open else None
        with ui.row().classes(
            "absolute left-1/2 items-baseline gap-2"
        ).style("transform: translateX(-50%)"):
            ui.label(
                (identity.name or t("app.unnamed_library"))
                if identity is not None
                else t("setup.no_library")
            ).classes("text-sm font-medium").style("color: var(--bt-header-text)")
            if identity is not None:
                ui.label(f"#{identity.uuid[:8]}").classes(
                    "text-xs font-mono"
                ).style("color: var(--bt-header-inactive-text)").tooltip(
                    identity.uuid
                )
            if settings.search_only:
                # Beside the library's name, because that is the question this
                # answers: which library, and in what capacity. Without it the
                # only evidence is a missing tab, which reads as a bug.
                with ui.row().classes("items-center gap-1").style(
                    "color: var(--bt-header-inactive-text)"
                ):
                    ui.icon("lock", size="0.9rem")
                    ui.label(t("app.search_only")).classes("text-xs").tooltip(
                        t("app.search_only_hint")
                    )

        with ui.row().classes("items-center gap-6"):
            ui.label(t("app.title")).classes("text-lg font-medium").style(
                "color: var(--bt-header-text)"
            )
            with ui.row().classes("gap-1"):
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

        with ui.row().classes("items-center gap-4"):
            _model_badge(t)
            _theme_toggle(t, mode, dark_mode)
            ui.select(
                {code: LANGUAGE_NAMES[code] for code in LANGUAGES},
                value=t.language or DEFAULT_LANGUAGE,
                on_change=lambda event: set_language(event.value),
            ).props(
                # The header is dark, so the field and its popup are too. On a
                # dark popup Quasar marks the current option with the primary
                # colour, which is the same green as the header and all but
                # disappears; accent reads against both. ``color`` keeps the
                # focus ring off primary green as well.
                'dense outlined dark options-dense color=white'
                ' options-selected-class="text-accent"'
            ).classes("w-40").tooltip(
                t("app.language")
            )

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
