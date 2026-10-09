"""Earlier values offered back while metadata is typed."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.i18n import Translator  # noqa: E402
from biotimbre.ui.forms import metadata_form  # noqa: E402
from biotimbre.ui.suggestions import suggest_tags, suggest_text  # noqa: E402

LIBRARY = {
    "device": ["AudioMoth 1.2", "Song Meter SM4", "Zoom H5"],
    "taxa": ["anura", "boana faber", "cicadidae"],
}


def lookup_in(field: str, asked: list[str] | None = None):
    """A lookup over LIBRARY, matching the way the repository does."""

    async def lookup(text: str) -> list[str]:
        if asked is not None:
            asked.append(text)
        needle = text.casefold()
        return [value for value in LIBRARY[field] if needle in value.casefold()]

    return lookup


def menu_texts(menu: ui.menu) -> list[str]:
    return [
        section.text
        for section in menu.descendants()
        if isinstance(section, ui.item_section)
    ]


def click_suggestion(user: User, text: str) -> None:
    """Click the menu item showing ``text``; its label is in a child section."""
    items = user.find(kind=ui.menu_item)
    items.elements = {
        item
        for item in items.elements
        if any(
            isinstance(section, ui.item_section) and section.text == text
            for section in item.descendants()
        )
    }
    assert items.elements, text
    items.click()


async def settle(condition, attempts: int = 50) -> None:
    """Let the lookups, which run as background tasks, finish."""
    for _ in range(attempts):
        if condition():
            return
        await asyncio.sleep(0.02)


class TestText:
    async def _field(self, user: User, path: str, asked=None):
        holder: dict = {}

        @ui.page(path)
        def page() -> None:
            holder["field"] = ui.input("Device")
            holder["menu"] = suggest_text(holder["field"], lookup_in("device", asked))

        await user.open(path)
        return holder["field"], holder["menu"]

    async def test_focusing_an_empty_field_lists_what_the_library_has(
        self, user: User
    ):
        field, menu = await self._field(user, "/textfocus")
        user.find(ui.input).trigger("focus")
        await user.should_see("Song Meter SM4")
        assert menu.value is True
        assert menu_texts(menu) == LIBRARY["device"]

    async def test_typing_narrows_the_list(self, user: User):
        field, menu = await self._field(user, "/texttype")
        user.find(ui.input).trigger("focus")
        await user.should_see("Zoom H5")

        # As the browser sends it: the event carries the new text.
        user.find(ui.input).trigger("update:value", "moth")
        await settle(lambda: menu_texts(menu) == ["AudioMoth 1.2"])
        assert field.value == "moth"
        assert menu_texts(menu) == ["AudioMoth 1.2"]

    async def test_a_click_fills_the_field(self, user: User):
        field, menu = await self._field(user, "/textpick")
        user.find(ui.input).trigger("focus")
        await user.should_see("AudioMoth 1.2")

        click_suggestion(user, "AudioMoth 1.2")
        assert field.value == "AudioMoth 1.2"
        assert menu.value is False

    async def test_a_value_written_by_code_opens_nothing(self, user: User):
        # An iNaturalist fetch fills fields nobody is typing in; a menu
        # springing open over each would be a nuisance.
        asked: list[str] = []
        field, menu = await self._field(user, "/textcode", asked)
        # The field never had focus, so a change reaching it is not typing.
        user.find(ui.input).trigger("update:value", "Zoom")
        await asyncio.sleep(0.1)
        assert field.value == "Zoom"
        assert asked == []
        assert menu.value is False

    async def test_an_exact_match_is_not_offered_back(self, user: User):
        field, menu = await self._field(user, "/textexact")
        field.value = "Zoom H5"
        user.find(ui.input).trigger("focus")
        await user.should_see("Zoom H5")  # the field itself
        assert menu_texts(menu) == []
        assert menu.value is False


class TestTags:
    async def test_typing_offers_matching_tags_and_keeps_the_chosen(
        self, user: User
    ):
        holder: dict = {}

        @ui.page("/tags")
        def page() -> None:
            holder["select"] = ui.select(
                [], value=[], multiple=True, new_value_mode="add-unique"
            )
            suggest_tags(holder["select"], lookup_in("taxa"))

        await user.open("/tags")
        select = holder["select"]
        select.set_options(["my own tag"], value=["my own tag"])

        # NiceGUI files the listener under the camel-cased event name.
        user.find(ui.select).trigger("inputValue", "an")
        await settle(lambda: "anura" in select.options)
        assert select.options == ["anura", "boana faber", "my own tag"]
        # A chosen tag the library has never seen stays chosen.
        assert select.value == ["my own tag"]


class TestForm:
    async def test_the_metadata_form_suggests_for_each_field(self, user: User):
        asked: list[tuple[str, str]] = []

        async def suggest(field: str, text: str) -> list[str]:
            asked.append((field, text))
            return []

        @ui.page("/suggestform")
        def page() -> None:
            form = metadata_form(Translator("en"), suggest=suggest)
            for name in ("device", "microphone", "recorded_by", "taxa"):
                form.inputs[name].mark(f"field-{name}")

        await user.open("/suggestform")
        for name in ("device", "microphone", "recorded_by", "taxa"):
            user.find(marker=f"field-{name}").trigger("focus")
        await settle(lambda: len(asked) == 4)
        assert {field for field, _ in asked} == {
            "device",
            "microphone",
            "recorded_by",
            "taxa",
        }
