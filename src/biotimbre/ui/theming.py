"""Light and dark palettes, and the three-state choice between them.

Each colour is declared once, as ``light-dark(<light>, <dark>)`` against a
``color-scheme`` — so the two palettes cannot drift apart, and native controls
that no stylesheet reaches (scrollbars, the audio player, date pickers) follow
the theme too.

The theme follows the operating system until the reader picks one, at which
point ``data-theme`` on ``<html>`` pins it. "System" is a real third state, not
the absence of a choice: without it there is no way back to following the OS
once a theme has been pinned.

Quasar has its own dark flag, which restyles its components from the inside;
:func:`quasar_dark` maps the same choice onto it, so both halves of the page
agree.
"""

from __future__ import annotations

from dataclasses import fields

from nicegui import app

from biotimbre.config import Theme

#: The three states, in the order the header control cycles through them.
THEME_MODES = ("system", "light", "dark")

#: Icon per mode. Material names, as Quasar expects.
THEME_ICONS = {
    "system": "brightness_auto",
    "light": "light_mode",
    "dark": "dark_mode",
}

#: Per-browser storage key. Beside the language, for the same reason: two
#: people using one local instance each get their own.
_STORAGE_KEY = "theme"

#: Theme fields that are not colours and so have no CSS variable.
_NON_COLOUR_FIELDS = frozenset({"spectrogram_colors"})


def _variable(name: str) -> str:
    """CSS custom property name for a theme field."""
    return f"--bt-{name.replace('_', '-')}"


def theme_mode() -> str:
    """The reader's chosen mode, defaulting to following the system."""
    try:
        mode = app.storage.user.get(_STORAGE_KEY)
    except RuntimeError:
        # No request context (a test rendering a fragment on its own).
        mode = None
    return mode if mode in THEME_MODES else "system"


def set_theme_mode(mode: str) -> None:
    """Remember a chosen mode, ignoring anything unrecognised."""
    if mode in THEME_MODES:
        app.storage.user[_STORAGE_KEY] = mode


def quasar_dark(mode: str) -> bool | None:
    """Quasar's dark flag for a mode: ``None`` is its own "follow the system"."""
    return {"light": False, "dark": True}.get(mode)


def theme_variables() -> dict[str, str]:
    """Quasar brand colours, as references to the theme's own variables.

    ``ui.colors`` writes whatever it is given into ``--q-*``; giving it a
    ``var()`` reference rather than a literal is what lets Quasar's own classes
    — ``bg-card``, ``text-muted``, ``text-notice`` — change with the theme
    without the page being re-rendered.
    """
    return {
        field.name: f"var({_variable(field.name)})"
        for field in fields(Theme)
        if field.name not in _NON_COLOUR_FIELDS
    }


def theme_css(light: Theme, dark: Theme) -> str:
    """The stylesheet that defines every colour for both themes.

    Args:
        light: Palette used when the theme resolves to light.
        dark: Palette used when it resolves to dark.

    Returns:
        A ``<style>`` block: the tokens on ``:root``, then the two rules that
        let ``data-theme`` pin one scheme.
    """
    tokens = "\n".join(
        f"  {_variable(field.name)}: "
        f"light-dark({getattr(light, field.name)}, {getattr(dark, field.name)});"
        for field in fields(Theme)
        if field.name not in _NON_COLOUR_FIELDS
    )
    return (
        "<style>\n"
        ":root {\n"
        "  color-scheme: light dark;\n"
        f"{tokens}\n"
        "}\n"
        ':root[data-theme="light"] { color-scheme: light; }\n'
        ':root[data-theme="dark"] { color-scheme: dark; }\n'
        "</style>"
    )


def apply_mode_script(mode: str) -> str:
    """Inline script that pins the mode before the first paint.

    NiceGUI builds the body after the page loads, so waiting for the app to
    start would show one frame of the system theme to a reader who pinned the
    other one.

    Args:
        mode: One of :data:`THEME_MODES`.

    Returns:
        A ``<script>`` block for the page head.
    """
    if mode == "system":
        statement = "delete document.documentElement.dataset.theme;"
    else:
        statement = f'document.documentElement.dataset.theme = "{mode}";'
    return f"<script>{statement}</script>"
