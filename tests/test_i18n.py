"""Translation lookup, and completeness of the Portuguese catalogue."""

from __future__ import annotations

import pytest

from biotimbre.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    LANGUAGES,
    Translator,
    catalogue,
    extra_keys,
    missing_keys,
)


class TestCatalogues:
    def test_both_languages_are_available(self):
        assert set(LANGUAGES) == {"en", "pt_BR"}

    def test_every_language_has_a_display_name(self):
        assert set(LANGUAGE_NAMES) == set(LANGUAGES)

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            catalogue("fr")

    def test_nested_json_is_flattened_to_dotted_keys(self):
        assert catalogue("en")["app.nav.search"] == "Search"

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_catalogue_is_not_empty(self, language):
        assert len(catalogue(language)) > 50


class TestCompleteness:
    """These are the tests that make a half-translated UI a failure, not a
    surprise found by eye at review time."""

    @pytest.mark.parametrize(
        "language", [lang for lang in LANGUAGES if lang != DEFAULT_LANGUAGE]
    )
    def test_translation_covers_every_english_key(self, language):
        missing = missing_keys(language)
        assert not missing, (
            f"{language} is missing {len(missing)} key(s): "
            f"{', '.join(missing[:10])}"
        )

    @pytest.mark.parametrize(
        "language", [lang for lang in LANGUAGES if lang != DEFAULT_LANGUAGE]
    )
    def test_translation_has_no_stale_keys(self, language):
        # Left-behind keys from a rename would otherwise go unnoticed.
        extra = extra_keys(language)
        assert not extra, (
            f"{language} has {len(extra)} key(s) English does not: "
            f"{', '.join(extra[:10])}"
        )

    @pytest.mark.parametrize(
        "language", [lang for lang in LANGUAGES if lang != DEFAULT_LANGUAGE]
    )
    def test_placeholders_match_english(self, language):
        import re

        pattern = re.compile(r"\{(\w+)\}")
        english = catalogue(DEFAULT_LANGUAGE)
        translated = catalogue(language)
        for key, source in english.items():
            expected = set(pattern.findall(source))
            actual = set(pattern.findall(translated.get(key, "")))
            # A translation with different placeholders would either format wrong
            # or silently drop a value the user needs to see.
            assert expected == actual, (
                f"{language}:{key} has placeholders {sorted(actual)}, "
                f"English has {sorted(expected)}"
            )

    def test_english_is_trivially_complete(self):
        assert missing_keys("en") == []


class TestTranslator:
    def test_translates(self):
        assert Translator("en")("app.nav.search") == "Search"
        assert Translator("pt_BR")("app.nav.search") == "Busca"

    def test_substitutes_fields(self):
        text = Translator("en")("admin.saved", item_id=7, biotimbre_id="00A3")
        assert "7" in text
        assert "00A3" in text

    def test_missing_key_returns_the_key(self):
        # A visible key tells a translator what to add; raising would take the
        # page down over a cosmetic gap.
        assert Translator("en")("no.such.key") == "no.such.key"

    def test_falls_back_to_english_for_an_untranslated_key(self, monkeypatch):
        import biotimbre.i18n as module

        monkeypatch.setattr(
            module, "catalogue",
            lambda lang: {"only.in.english": "Fallback"} if lang == "en" else {},
        )
        assert module.Translator("pt_BR")("only.in.english") == "Fallback"

    def test_unknown_language_falls_back_rather_than_raising(self):
        # A stale setting must not stop the app from starting.
        assert Translator("klingon").language == DEFAULT_LANGUAGE

    def test_wrong_placeholders_do_not_crash(self, monkeypatch):
        import biotimbre.i18n as module

        monkeypatch.setattr(
            module, "catalogue", lambda lang: {"k": "needs {absent}"}
        )
        assert module.Translator("en")("k", present=1) == "needs {absent}"

    def test_switch_changes_language(self):
        assert Translator("en").switch("pt_BR")("app.nav.admin") == "Biblioteca"
