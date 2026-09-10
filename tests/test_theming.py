"""Theme tokens and the three-state light/dark choice."""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from biotimbre.config import DEFAULT_DARK_THEME, Theme  # noqa: E402
from biotimbre.ui.theming import (  # noqa: E402
    THEME_MODES,
    apply_mode_script,
    quasar_dark,
    theme_css,
    theme_variables,
)


class TestThemeCss:
    def test_declares_both_palettes_once_per_colour(self):
        css = theme_css(Theme(), DEFAULT_DARK_THEME)
        assert "color-scheme: light dark;" in css
        assert (
            f"--bt-page-bg: light-dark({Theme().page_bg}, "
            f"{DEFAULT_DARK_THEME.page_bg});" in css
        )
        assert "--bt-header-inactive-text: light-dark(" in css

    def test_lets_a_choice_pin_the_scheme(self):
        css = theme_css(Theme(), DEFAULT_DARK_THEME)
        assert ':root[data-theme="light"] { color-scheme: light; }' in css
        assert ':root[data-theme="dark"] { color-scheme: dark; }' in css

    def test_leaves_the_spectrogram_ramp_alone(self):
        # Spectrograms are rendered server-side, where the reader's theme is
        # not known; the ramp is not a CSS colour at all.
        assert "spectrogram" not in theme_css(Theme(), DEFAULT_DARK_THEME)


class TestThemeVariables:
    def test_every_colour_is_a_variable_reference(self):
        variables = theme_variables()
        assert variables["page_bg"] == "var(--bt-page-bg)"
        assert variables["header_inactive_text"] == "var(--bt-header-inactive-text)"
        # Literals here would freeze Quasar's brand colours to one palette.
        assert all(value.startswith("var(--bt-") for value in variables.values())

    def test_skips_what_is_not_a_colour(self):
        assert "spectrogram_colors" not in theme_variables()


class TestModes:
    def test_quasar_dark_maps_the_three_states(self):
        assert quasar_dark("light") is False
        assert quasar_dark("dark") is True
        # Auto: Quasar follows the system, as the CSS does.
        assert quasar_dark("system") is None

    def test_every_mode_has_a_quasar_setting(self):
        assert {quasar_dark(mode) for mode in THEME_MODES} == {True, False, None}

    def test_pinning_sets_the_attribute_before_the_first_paint(self):
        assert 'dataset.theme = "dark"' in apply_mode_script("dark")

    def test_following_the_system_removes_it(self):
        assert "delete document.documentElement.dataset.theme" in apply_mode_script(
            "system"
        )
