"""Choosing a library when none was given."""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.config import load_settings  # noqa: E402
from biotimbre.domain.errors import BiotimbreError, NotFoundError  # noqa: E402
from biotimbre.ui.routes import list_directory  # noqa: E402
from biotimbre.ui.state import AppState, get_state, set_state  # noqa: E402


class TestListing:
    def test_lists_subdirectories_only(self, tmp_path):
        (tmp_path / "a_dir").mkdir()
        (tmp_path / "b_dir").mkdir()
        (tmp_path / "a_file.wav").write_bytes(b"")
        listing = list_directory(str(tmp_path))
        names = [entry["name"] for entry in listing["directories"]]
        # A library is a directory; listing files would answer a question
        # nobody asked.
        assert names == ["a_dir", "b_dir"]
        assert listing["path"] == str(tmp_path)
        assert listing["parent"] == str(tmp_path.parent)

    def test_skips_hidden_entries(self, tmp_path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "shown").mkdir()
        assert [e["name"] for e in list_directory(str(tmp_path))["directories"]] == [
            "shown"
        ]

    def test_flags_a_directory_that_already_holds_a_library(self, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()
        (existing / "library.db").write_bytes(b"")
        entry = list_directory(str(tmp_path))["directories"][0]
        assert entry["is_library"] is True

    def test_can_leave_home(self, tmp_path):
        # Libraries live on external drives and mounted archives; confining the
        # picker to $HOME would put them out of reach.
        listing = list_directory("/")
        assert listing["path"] == "/"
        assert listing["parent"] is None

    def test_rejects_what_is_not_a_directory(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("x")
        with pytest.raises(NotFoundError, match="No such directory"):
            list_directory(str(target))

    def test_reports_an_unreadable_directory(self, tmp_path):
        import os

        blocked = tmp_path / "blocked"
        blocked.mkdir()
        os.chmod(blocked, 0o000)
        try:
            with pytest.raises(BiotimbreError):
                list_directory(str(blocked))
        finally:
            os.chmod(blocked, 0o755)


class TestOpening:
    @pytest.fixture
    def unset(self, tmp_path, monkeypatch, shared_pool):
        """State started with no library chosen."""
        monkeypatch.chdir(tmp_path)
        settings = load_settings()
        assert settings.data_dir_chosen is False
        state = AppState(settings)
        state.pool = shared_pool

        previous = None
        try:
            previous = get_state()
        except RuntimeError:
            pass
        set_state(state)
        yield state
        state.close_library()
        if previous is not None:
            set_state(previous)

    def test_starts_with_no_library_and_creates_nothing(self, unset, tmp_path):
        assert unset.is_open is False
        # Nothing has been written where the interface happened to start.
        assert not (tmp_path / "data").exists()
        with pytest.raises(RuntimeError, match="No library is open"):
            unset.service  # noqa: B018

    def test_opening_creates_and_names_the_library(self, unset, tmp_path):
        target = tmp_path / "libraries" / "cicadas"
        unset.open_library(target, name="Cicadas")

        assert unset.is_open is True
        assert unset.settings.data_dir == target
        assert (target / "library.db").is_file()
        assert unset.service.identity().name == "Cicadas"

    def test_opening_an_existing_library_keeps_its_name(self, unset, tmp_path):
        target = tmp_path / "existing"
        unset.open_library(target, name="First name")
        unset.close_library()

        unset.open_library(target, name="Second name")
        # Opening is not renaming.
        assert unset.service.identity().name == "First name"

    def test_opening_replaces_the_one_before(self, unset, tmp_path):
        unset.open_library(tmp_path / "one", name="One")
        first = unset.service.identity().uuid
        unset.open_library(tmp_path / "two", name="Two")

        assert unset.service.identity().uuid != first
        assert unset.settings.data_dir == tmp_path / "two"


@pytest.mark.usefixtures("shared_pool")
class TestSetupPage:
    async def test_the_page_asks_instead_of_showing_a_library(
        self, user: User, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        state = AppState(load_settings())
        set_state(state)

        try:
            await user.open("/")
            await user.should_see("Choose a library")
            await user.should_see("no library open")
        finally:
            state.close_library()

    async def test_opening_from_the_page_reaches_the_library(
        self, user: User, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        state = AppState(load_settings())
        set_state(state)

        try:
            await user.open("/")
            await user.should_see("Choose a library")

            inputs = {
                element._props.get("label"): element
                for element in user.find(ui.input).elements
            }
            inputs["Library directory"].value = str(tmp_path / "chosen")
            inputs["Library name"].value = "Chosen"
            user.find("Open library").click()
            await user.should_see("Library items", retries=200)

            assert state.is_open is True
            assert state.service.identity().name == "Chosen"
        finally:
            state.close_library()
