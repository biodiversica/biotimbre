"""Shared page frame: header, navigation, language switcher, model status.

The language is held per browser session rather than in the settings file, so two
people using the same local instance can each read it in their own language, and
switching does not rewrite configuration.
"""

from __future__ import annotations

from contextlib import contextmanager

from nicegui import app, ui

from biotimbre.config import Theme
from biotimbre.i18n import DEFAULT_LANGUAGE, LANGUAGE_NAMES, LANGUAGES, Translator
from biotimbre.ui.state import get_state

_LANGUAGE_KEY = "language"


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


def _model_badge(t: Translator, theme: Theme) -> None:
    """Show whether models are ready, loading, or failed."""
    status = get_state().models

    if status.failed:
        with ui.row().classes("items-center gap-1").style(
            f"color: {theme.header_status_failed}"
        ):
            ui.icon("error_outline", size="1.1rem")
            ui.label(t("error.generic", message=status.failed)).classes(
                "text-xs max-w-64 truncate"
            ).tooltip(status.failed)
        return

    if status.ready:
        with ui.row().classes("items-center gap-1").style(
            f"color: {theme.header_status_ready}"
        ):
            ui.icon("check_circle_outline", size="1.1rem")
            summary = ", ".join(
                f"{model} {dim}d" for model, dim in sorted(status.dimensions.items())
            )
            ui.label(t("app.models_ready")).classes("text-xs").tooltip(summary)
        return

    with ui.row().classes("items-center gap-2").style(
        f"color: {theme.header_status_loading}"
    ):
        ui.spinner(size="1rem")
        ui.label(t("app.loading_models")).classes("text-xs").tooltip(
            t("app.loading_models_hint")
        )
    # Poll until warm-up finishes, then redraw the header.
    ui.timer(1.5, lambda: ui.navigate.reload() if get_state().models.ready else None,
             once=False)


@contextmanager
def page(active: str):
    """Wrap page content in the shared frame.

    Args:
        active: Which nav entry to highlight, ``"admin"`` or ``"search"``.

    Yields:
        The translator for this session, so callers need not fetch it again.
    """
    t = translator()
    theme = get_state().settings.theme
    ui.colors(
        primary=theme.primary,
        secondary=theme.secondary,
        accent=theme.accent,
        dark=theme.dark,
        positive=theme.positive,
        negative=theme.negative,
        info=theme.info,
        warning=theme.warning,
        button_text=theme.button_text,
        card=theme.card,
        link=theme.link,
        muted=theme.muted,
        notice=theme.notice,
        surface=theme.surface,
        chip=theme.chip,
    )
    ui.query("body").style(f"background: {theme.page_bg}; color: {theme.text}")

    with ui.header().classes(
        "items-center justify-between px-4 py-2 shadow-sm"
    ).style(f"background: {theme.header_bg}"):
        with ui.row().classes("items-center gap-6"):
            ui.label(t("app.title")).classes("text-lg font-medium").style(
                f"color: {theme.header_text}"
            )
            with ui.row().classes("gap-1"):
                for name, target, icon in (
                    ("admin", "/", "library_music"),
                    ("search", "/search", "search"),
                ):
                    is_active = name == active
                    text_color = theme.header_text if is_active else (
                        theme.header_inactive_text
                    )
                    background = (
                        f"background: {theme.header_active_bg};" if is_active else ""
                    )
                    ui.button(
                        t(f"app.nav.{name}"),
                        icon=icon,
                        color=None,
                        on_click=lambda target=target: ui.navigate.to(target),
                    ).props("flat dense no-caps").style(
                        f"color: {text_color}; {background}"
                    )

        with ui.row().classes("items-center gap-4"):
            _model_badge(t, theme)
            ui.select(
                {code: LANGUAGE_NAMES[code] for code in LANGUAGES},
                value=t.language or DEFAULT_LANGUAGE,
                on_change=lambda event: set_language(event.value),
            ).props("dense outlined dark options-dense").classes("w-40").tooltip(
                t("app.language")
            )

    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        yield t


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
