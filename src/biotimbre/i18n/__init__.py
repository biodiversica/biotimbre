"""Translation lookup for the user interface.

JSON catalogues rather than gettext (PLAN.md D7): two languages need no compile
step, and the files port directly to i18next if a JS frontend ever happens.

Keys are dotted and namespaced by area — ``admin.field.date``, ``search.button.run``
— so a catalogue reads as an outline of the UI.

A missing key returns the key itself rather than raising. A half-translated
interface is bad; a crashed one is worse, and the visible key tells a translator
exactly what to add. :func:`missing_keys` makes those gaps assertable in tests
instead of leaving them to be spotted by eye.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

#: Language used when none is set, and the fallback for any key a translation
#: lacks. English is complete by construction: it is the source of the keys.
DEFAULT_LANGUAGE = "en"

#: Languages with a catalogue file in this package.
LANGUAGES: tuple[str, ...] = ("en", "pt_BR")

#: Human-readable names, shown in the language switcher in the language itself.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "pt_BR": "Português (BR)",
}


@lru_cache(maxsize=None)
def catalogue(language: str) -> dict[str, str]:
    """Load and flatten a language's catalogue.

    Nested JSON objects are flattened to dotted keys, so the file can be
    organised by area while lookup stays a single dict access.

    Args:
        language: Language code, one of :data:`LANGUAGES`.

    Returns:
        Mapping of dotted key to translated string.

    Raises:
        ValueError: If the language has no catalogue.
    """
    if language not in LANGUAGES:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Available: {', '.join(LANGUAGES)}."
        )
    raw = json.loads(
        resources.files("biotimbre.i18n")
        .joinpath(f"{language}.json")
        .read_text("utf-8")
    )
    return _flatten(raw)


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten nested catalogue objects into dotted keys."""
    flat: dict[str, str] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = str(value)
    return flat


class Translator:
    """Resolves keys for one language, falling back to English.

    Args:
        language: Language code. Unknown codes fall back to
            :data:`DEFAULT_LANGUAGE` rather than raising, so a stale setting
            cannot prevent the app from starting.
    """

    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language if language in LANGUAGES else DEFAULT_LANGUAGE

    def __call__(self, key: str, **fields: object) -> str:
        """Translate a key, substituting any named fields.

        Args:
            key: Dotted catalogue key.
            **fields: Values for ``{placeholders}`` in the string.

        Returns:
            The translated string, or the key itself if it is absent from both
            the active catalogue and English.
        """
        text = catalogue(self.language).get(key)
        if text is None:
            text = catalogue(DEFAULT_LANGUAGE).get(key, key)
        if not fields:
            return text
        try:
            return text.format(**fields)
        except (KeyError, IndexError):
            # A translation with the wrong placeholders must not crash a page.
            return text

    def switch(self, language: str) -> "Translator":
        """Return a translator for another language."""
        return Translator(language)


def missing_keys(language: str) -> list[str]:
    """Keys present in English but absent from another language.

    Args:
        language: Language to check.

    Returns:
        Sorted missing keys; empty when the translation is complete.
    """
    english = set(catalogue(DEFAULT_LANGUAGE))
    return sorted(english - set(catalogue(language)))


def extra_keys(language: str) -> list[str]:
    """Keys in a language that English does not have.

    These are usually renamed or deleted keys left behind, which would otherwise
    go unnoticed.

    Args:
        language: Language to check.

    Returns:
        Sorted extra keys; empty when the translation matches English.
    """
    english = set(catalogue(DEFAULT_LANGUAGE))
    return sorted(set(catalogue(language)) - english)


__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "LANGUAGE_NAMES",
    "Translator",
    "catalogue",
    "extra_keys",
    "missing_keys",
]
