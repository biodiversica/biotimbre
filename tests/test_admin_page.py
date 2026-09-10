"""Admin page behaviour.

Two layers, both browser-free:

* :class:`~biotimbre.ui.forms.MetadataForm` on its own — draft construction and
  the mapping from a validation error back onto the offending input.
* the rendered page, driven through NiceGUI's simulated ``user`` fixture.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import app as nicegui_app  # noqa: E402
from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.domain.errors import ValidationError  # noqa: E402
from biotimbre.domain.item import ItemDraft, LibraryItem  # noqa: E402
from biotimbre.i18n import LANGUAGE_NAMES, Translator  # noqa: E402
from biotimbre.ui.admin import AddState  # noqa: E402
from biotimbre.ui.forms import metadata_form  # noqa: E402
from biotimbre.ui.state import AppState, get_state, set_state  # noqa: E402
from conftest import EXAMPLES_DIR  # noqa: E402

GUYVIR = EXAMPLES_DIR / "GUYVIR.wav"


@pytest.fixture
def installed_state(settings, shared_pool):
    """Install app state backed by a temporary library."""
    state = AppState(settings)
    state.pool = shared_pool
    state.service._pool = shared_pool
    state.service._owns_pool = False

    previous = None
    try:
        previous = get_state()
    except RuntimeError:
        pass
    set_state(state)
    yield state
    state.close()
    if previous is not None:
        set_state(previous)


class TestAddState:
    def test_stages_an_upload_and_reads_properties(self, tmp_path):
        add_state = AddState()
        upload = add_state.replace("GUYVIR.wav", GUYVIR.read_bytes())
        assert upload.path.is_file()
        assert upload.properties.sample_rate == 32_000
        assert upload.properties.channels == 1
        add_state.discard()

    def test_rejects_undecodable_bytes_at_upload_time(self):
        add_state = AddState()
        # Better to fail here than after the user has filled in the whole form.
        with pytest.raises(ValidationError):
            add_state.replace("bad.wav", b"not audio")

    def test_replacing_discards_the_previous_file(self):
        add_state = AddState()
        first = add_state.replace("a.wav", GUYVIR.read_bytes())
        first_path = first.path
        add_state.replace("b.wav", GUYVIR.read_bytes())
        assert not first_path.exists() or first_path.name == "b.wav"
        add_state.discard()

    def test_discard_removes_the_file(self):
        add_state = AddState()
        upload = add_state.replace("a.wav", GUYVIR.read_bytes())
        path = upload.path
        add_state.discard()
        assert not path.exists()
        assert add_state.upload is None

    def test_only_the_basename_is_used(self):
        # An upload name is attacker-controlled in a public deployment; a path
        # must not escape the staging directory.
        add_state = AddState()
        upload = add_state.replace("../../escape.wav", GUYVIR.read_bytes())
        assert upload.path.parent == add_state.directory
        assert upload.path.name == "escape.wav"
        add_state.discard()


@pytest.mark.usefixtures("installed_state")
class TestMetadataForm:
    """The form itself, rendered inside a throwaway page context."""

    async def _form(self, user: User, item: LibraryItem | None = None):
        holder: dict = {}

        @ui.page("/form")
        def form_page():
            holder["form"] = metadata_form(Translator("en"), item)

        await user.open("/form")
        return holder["form"]

    async def test_builds_a_draft_from_filled_inputs(self, user: User):
        form = await self._form(user)
        form.inputs["date"].value = "2021-03-05"
        form.inputs["time"].value = "22:10:00"
        form.inputs["latitude"].value = -26.9
        form.inputs["longitude"].value = -48.7
        form.inputs["window_start"].value = 12.0
        form.inputs["window_end"].value = 17.0

        draft = form.draft()
        assert isinstance(draft, ItemDraft)
        assert draft.date == "2021-03-05"
        assert draft.location.latitude == -26.9
        assert draft.sample_window.end == 17.0
        # An untouched biotimbre field means "allocate a new one".
        assert draft.biotimbre_id is None

    async def test_blank_window_builds_a_draft_without_one(self, user: User):
        form = await self._form(user)
        form.inputs["date"].value = "2021-03-05"
        form.inputs["time"].value = "22:10:00"
        form.inputs["latitude"].value = -26.9
        form.inputs["longitude"].value = -48.7

        assert form.draft().sample_window is None

    async def test_half_a_window_is_reported_on_its_input(self, user: User):
        form = await self._form(user)
        form.inputs["date"].value = "2021-03-05"
        form.inputs["time"].value = "22:10:00"
        form.inputs["latitude"].value = -26.9
        form.inputs["longitude"].value = -48.7
        form.inputs["window_start"].value = 12.0

        with pytest.raises(ValidationError) as exc:
            form.draft()
        assert exc.value.field == "sample_window.end"
        assert form.show_error(exc.value) is True

    async def test_missing_location_is_reported_on_the_latitude_input(
        self, user: User
    ):
        form = await self._form(user)
        form.inputs["date"].value = "2021-03-05"
        form.inputs["time"].value = "22:10:00"
        form.inputs["window_start"].value = 0.0
        form.inputs["window_end"].value = 5.0

        with pytest.raises(ValidationError) as exc:
            form.draft()
        # The point of ValidationError.field: mark the input, not a bare toast.
        assert form.show_error(exc.value) is True

    async def test_bad_date_is_mapped_to_the_date_input(self, user: User):
        form = await self._form(user)
        form.inputs["date"].value = "not-a-date"
        form.inputs["time"].value = "22:10:00"
        form.inputs["latitude"].value = -26.9
        form.inputs["longitude"].value = -48.7
        form.inputs["window_start"].value = 0.0
        form.inputs["window_end"].value = 5.0

        with pytest.raises(ValidationError) as exc:
            form.draft()
        assert exc.value.field == "date"
        assert form.show_error(exc.value) is True

    async def test_inverted_window_is_mapped_to_an_input(self, user: User):
        form = await self._form(user)
        form.inputs["date"].value = "2021-03-05"
        form.inputs["time"].value = "22:10:00"
        form.inputs["latitude"].value = -26.9
        form.inputs["longitude"].value = -48.7
        form.inputs["window_start"].value = 17.0
        form.inputs["window_end"].value = 12.0

        with pytest.raises(ValidationError) as exc:
            form.draft()
        assert form.show_error(exc.value) is True

    async def test_open_ended_band_is_accepted(self, user: User):
        form = await self._form(user)
        form.inputs["date"].value = "2021-03-05"
        form.inputs["time"].value = "22:10:00"
        form.inputs["latitude"].value = -26.9
        form.inputs["longitude"].value = -48.7
        form.inputs["window_start"].value = 0.0
        form.inputs["window_end"].value = 5.0
        form.inputs["highpass"].value = 2000
        form.inputs["lowpass"].value = None

        draft = form.draft()
        # Matches the sidecar format's "2000,-".
        assert draft.filtered_band.low == 2000.0
        assert draft.filtered_band.high is None

    async def test_blank_optional_text_becomes_none(self, user: User):
        form = await self._form(user)
        form.inputs["date"].value = "2021-03-05"
        form.inputs["time"].value = "22:10:00"
        form.inputs["latitude"].value = -26.9
        form.inputs["longitude"].value = -48.7
        form.inputs["window_start"].value = 0.0
        form.inputs["window_end"].value = 5.0
        form.inputs["device"].value = "   "
        form.inputs["notes"].value = ""

        draft = form.draft()
        assert draft.device is None
        assert draft.notes is None

    async def test_out_of_range_speed_is_mapped(self, user: User):
        form = await self._form(user)
        form.inputs["date"].value = "2021-03-05"
        form.inputs["time"].value = "22:10:00"
        form.inputs["latitude"].value = -26.9
        form.inputs["longitude"].value = -48.7
        form.inputs["window_start"].value = 0.0
        form.inputs["window_end"].value = 5.0
        form.inputs["speed"].value = 500.0

        with pytest.raises(ValidationError) as exc:
            form.draft()
        assert exc.value.field == "speed"
        assert form.show_error(exc.value) is True

    async def test_clear_errors_is_safe_before_any_error(self, user: User):
        form = await self._form(user)
        form.clear_errors()

    async def test_unmapped_field_reports_not_handled(self, user: User):
        form = await self._form(user)
        # Something with no matching input must fall back to a notification
        # rather than silently vanishing.
        assert form.show_error(ValidationError("boom", "audio")) is False


@pytest.mark.usefixtures("installed_state")
class TestEditPrefill:
    async def test_form_prefills_from_an_item(self, user: User, installed_state):
        from biotimbre.io.sidecar import draft_from_sidecar

        result = installed_state.call(
            installed_state.service.add_item,
            GUYVIR,
            draft_from_sidecar(EXAMPLES_DIR / "GUYVIR.txt"),
            compute_embeddings=False,
        )
        item = result.item

        holder: dict = {}

        @ui.page("/edit")
        def edit_page():
            holder["form"] = metadata_form(
                Translator("en"), item, allow_new_biotimbre=False
            )

        await user.open("/edit")
        form = holder["form"]

        assert form.inputs["date"].value == item.date
        assert form.inputs["latitude"].value == item.location.latitude
        assert form.inputs["device"].value == "Audiomoth"
        # The open-ended band round-trips as an empty upper bound.
        assert form.inputs["highpass"].value == 2000.0
        assert form.inputs["lowpass"].value is None
        assert form.inputs["biotimbre_id"].value == item.biotimbre_id

    async def test_blanking_biotimbre_id_on_edit_is_rejected(
        self, user: User, installed_state
    ):
        # update_item() treats a blank biotimbre_id as "unchanged" (an edit
        # never allocates), so a blank field here must be rejected up front —
        # otherwise saving silently no-ops and looks like nothing happened.
        from biotimbre.io.sidecar import draft_from_sidecar

        result = installed_state.call(
            installed_state.service.add_item,
            GUYVIR,
            draft_from_sidecar(EXAMPLES_DIR / "GUYVIR.txt"),
            compute_embeddings=False,
        )
        item = result.item

        holder: dict = {}

        @ui.page("/edit2")
        def edit_page():
            holder["form"] = metadata_form(
                Translator("en"), item, allow_new_biotimbre=False
            )

        await user.open("/edit2")
        form = holder["form"]
        form.inputs["biotimbre_id"].value = ""

        with pytest.raises(ValidationError) as exc:
            form.draft()
        assert exc.value.field == "biotimbre_id"
        assert form.show_error(exc.value) is True


@pytest.mark.usefixtures("installed_state")
class TestAdminPageRenders:
    async def test_empty_library_invites_adding(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401  (registers pages)

        await user.open("/")
        await user.should_see("Library items")
        await user.should_see("Add a sample")
        await user.should_see("The library is empty. Add a sample to begin.")

    async def test_shows_required_and_optional_sections(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/")
        await user.should_see("Required")
        await user.should_see("Optional")
        await user.should_see("Add to library")

    async def test_lists_an_existing_item(self, user: User, installed_state):
        from biotimbre.io.sidecar import draft_from_sidecar
        from biotimbre.ui import app as ui_app  # noqa: F401

        result = installed_state.call(
            installed_state.service.add_item,
            GUYVIR,
            draft_from_sidecar(EXAMPLES_DIR / "ZAMTYM.txt"),
            compute_embeddings=False,
        )
        await user.open("/")

        # The count label is real text; the rows are not. A ui.table carries its
        # rows as element props rather than child text nodes, so they have to be
        # asserted on the element rather than through should_see.
        await user.should_see("1 item(s)")
        table = next(iter(user.find(ui.table).elements))
        assert len(table.rows) == 1
        assert table.rows[0]["taxa"] == "cicadidae"
        # The table shows a truncated ID; the tooltip carries the full UUID.
        assert table.rows[0]["biotimbre_id"] == result.item.biotimbre_id[:8]

    async def test_a_chosen_file_can_be_taken_back(self, user: User):
        from nicegui.elements.upload_files import SmallFileUpload

        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/")
        await user.should_see("No file chosen yet.")

        uploader = next(iter(user.find(ui.upload).elements))
        await uploader.handle_uploads(
            [SmallFileUpload("GUYVIR.wav", "audio/wav", GUYVIR.read_bytes())]
        )
        await user.should_see("GUYVIR.wav")
        await user.should_see("Remove file")

        # The staged file is playable before it is added, streamed from a
        # route of its own rather than inlined into the page.
        player = next(iter(user.find(ui.audio).elements))
        staged_url = player._props["src"]
        assert staged_url.startswith("/media/staged/")
        assert any(route.path == staged_url for route in nicegui_app.routes)

        # Until this existed the file was stuck until the page was reloaded.
        user.find("Remove file").click()
        await user.should_see("No file chosen yet.")
        assert not any(route.path == staged_url for route in nicegui_app.routes)

        # And it really is gone: adding now asks for a file rather than
        # storing the one that was taken back.
        user.find("Add to library").click()
        await user.should_see("Choose an audio file first.")

    async def test_list_columns_identify_rather_than_describe(
        self, user: User, installed_state
    ):
        from biotimbre.io.sidecar import draft_from_sidecar
        from biotimbre.ui import app as ui_app  # noqa: F401

        installed_state.call(
            installed_state.service.add_item,
            GUYVIR,
            draft_from_sidecar(EXAMPLES_DIR / "ZAMTYM.txt"),
            compute_embeddings=False,
        )
        await user.open("/")
        await user.should_see("1 item(s)")

        # Length and speed belong to the item, not to picking it out of a list.
        table = next(iter(user.find(ui.table).elements))
        assert [column["name"] for column in table.columns] == [
            "item_id",
            "biotimbre_id",
            "biotimbre_label",
            "taxa",
            "date",
        ]

    async def test_detail_reports_a_speed_shifted_item(self, user: User, installed_state):
        import dataclasses

        from biotimbre.io.sidecar import draft_from_sidecar
        from biotimbre.ui import app as ui_app  # noqa: F401

        draft = dataclasses.replace(
            draft_from_sidecar(EXAMPLES_DIR / "GUYVIR.txt"), speed=0.25
        ).validated()
        item = installed_state.call(
            installed_state.service.add_item,
            GUYVIR,
            draft,
            compute_embeddings=False,
        ).item
        await user.open("/")
        # should_see first: it awaits the page's async render, whereas find()
        # inspects whatever is present at that instant.
        await user.should_see("1 item(s)")
        user.find(ui.table).trigger("rowClick", args=[None, {"item_id": item.item_id}])

        await user.should_see("Speed")
        await user.should_see("0.25")

    async def test_translates_to_portuguese(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401

        installed = get_state()
        installed.settings = type(installed.settings)(
            **{**installed.settings.__dict__, "language": "pt_BR"}
        )
        await user.open("/")
        await user.should_see("Itens da biblioteca")
        await user.should_see("Adicionar uma amostra")


@pytest.mark.usefixtures("installed_state")
class TestLanguageSelect:
    """The header select sits on the dark header, where primary is invisible."""

    async def test_marks_the_current_language_readably(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/")
        # Other pages leave selects in the registry, so pick the one offering
        # the languages rather than the first one found.
        select = next(
            element
            for element in user.find(ui.select).elements
            if any(
                option.get("label") in LANGUAGE_NAMES.values()
                for option in element._props.get("options", [])
            )
        )
        # Without this the current option keeps Quasar's default, the primary
        # colour, which is the header's own green on a dark popup.
        assert select._props["options-selected-class"] == "text-accent"
        assert select._props["color"] == "white"


@pytest.mark.usefixtures("installed_state")
class TestThemeControl:
    """The header control that chooses light, dark, or the system's own."""

    @staticmethod
    def _toggle(user: User):
        from biotimbre.ui.theming import THEME_ICONS

        return next(
            element
            for element in user.find(ui.button).elements
            if element._props.get("icon") in THEME_ICONS.values()
        )

    async def test_starts_out_following_the_system(self, user: User):
        from nicegui import app as nicegui_app

        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/")
        assert self._toggle(user)._props["icon"] == "brightness_auto"
        # Nothing stored: "system" is the default, not a pinned choice.
        assert "theme" not in nicegui_app.storage.user

    async def test_cycles_system_light_dark_and_back(self, user: User):
        from nicegui import app as nicegui_app
        from nicegui.testing.user_interaction import UserInteraction

        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/")
        button = self._toggle(user)

        for expected, icon in (
            ("light", "light_mode"),
            ("dark", "dark_mode"),
            ("system", "brightness_auto"),
        ):
            UserInteraction(user, {button}, None).click()
            assert nicegui_app.storage.user["theme"] == expected
            assert button._props["icon"] == icon

    async def test_the_page_carries_both_palettes(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/")
        head = "".join(user.client._head_html)
        # One declaration per colour, so the palettes cannot drift apart.
        assert "color-scheme: light dark;" in head
        assert "--bt-page-bg: light-dark(" in head


@pytest.mark.usefixtures("installed_state")
class TestItemDetail:
    """The dialog opened from a row, and what a change to it leaves behind."""

    @staticmethod
    def _add(state, **overrides):
        import dataclasses

        from biotimbre.io.sidecar import draft_from_sidecar

        draft = draft_from_sidecar(EXAMPLES_DIR / "ZAMTYM.txt")
        if overrides:
            draft = dataclasses.replace(draft, **overrides)
        return state.call(
            state.service.add_item, GUYVIR, draft, compute_embeddings=False
        )

    async def test_shows_the_link_and_the_timestamps(self, user: User, installed_state):
        from biotimbre.ui import app as ui_app  # noqa: F401

        item = self._add(
            installed_state, link_to_full_recording="https://example.org/rec"
        ).item
        await user.open("/")
        await user.should_see("1 item(s)")
        user.find(ui.table).trigger("rowClick", args=[None, {"item_id": item.item_id}])

        await user.should_see("Link to full recording")
        await user.should_see("https://example.org/rec")
        await user.should_see("Created")
        await user.should_see("Updated")

        # The dialog's card is a scrolling flex column: without ``shrink-0``
        # the player collapses to nothing and the spectrogram loses most of
        # its height as soon as the content overflows.
        image = next(iter(user.find(ui.image).elements))
        audio = next(iter(user.find(ui.audio).elements))
        assert "shrink-0" in image._classes
        assert "shrink-0" in audio._classes

    async def test_reads_the_band_and_window_with_dashes(
        self, user: User, installed_state
    ):
        from biotimbre.ui import app as ui_app  # noqa: F401

        # ZAMTYM.txt: filtered_band 2000,10000 and sample_window 5,10.
        item = self._add(installed_state).item
        await user.open("/")
        await user.should_see("1 item(s)")
        user.find(ui.table).trigger("rowClick", args=[None, {"item_id": item.item_id}])

        await user.should_see("2000 - 10000")
        await user.should_see("5 - 10 s")

    async def test_names_the_link_field_even_when_unset(self, user: User, installed_state):
        from biotimbre.ui import app as ui_app  # noqa: F401

        item = self._add(installed_state).item
        await user.open("/")
        await user.should_see("1 item(s)")
        user.find(ui.table).trigger("rowClick", args=[None, {"item_id": item.item_id}])

        await user.should_see("Link to full recording")

    async def test_list_survives_an_edit(self, user: User, installed_state):
        import asyncio

        from biotimbre.ui import app as ui_app  # noqa: F401

        item = self._add(installed_state).item
        await user.open("/")
        await user.should_see("1 item(s)")

        user.find(ui.table).trigger("rowClick", args=[None, {"item_id": item.item_id}])
        await user.should_see("Edit")
        user.find("Edit").click()
        await user.should_see("Save")
        user.find("Save").click()

        # The redraw runs from a timer, and saving used to create that timer
        # inside the very subtree the redraw clears: deleting the timer
        # cancelled the redraw at its next read, leaving the page with no list
        # at all until it was reloaded. The live element registry is the oracle
        # here — ``should_see`` still matches elements that have been deleted.
        for _ in range(40):
            await asyncio.sleep(0.05)
            tables = [
                element
                for element in user.client.elements.values()
                if isinstance(element, ui.table)
            ]
            if tables:
                break
        assert tables, "the item list never came back after saving an edit"
