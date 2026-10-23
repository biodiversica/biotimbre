"""Choosing a library when none was given."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.config import load_settings  # noqa: E402
from biotimbre.domain.errors import BiotimbreError, NotFoundError  # noqa: E402
from biotimbre.service import LibraryService  # noqa: E402
from biotimbre.ui.routes import list_directory  # noqa: E402
from biotimbre.ui.setup import library_folder_name, new_library_problem  # noqa: E402
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
        # The filesystem root: "/" here, the current drive's root on Windows.
        assert listing["path"] == str(Path("/").resolve())
        assert listing["parent"] is None

    def test_rejects_what_is_not_a_directory(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("x")
        with pytest.raises(NotFoundError, match="No such directory"):
            list_directory(str(target))

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="chmod cannot take away read access on Windows",
    )
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


class TestNewLibraryFolder:
    def test_the_folder_is_named_after_the_library(self):
        assert library_folder_name("Atlantic Forest Cicadas") == (
            "atlantic-forest-cicadas"
        )

    def test_what_a_filesystem_might_refuse_becomes_hyphens(self):
        assert library_folder_name('  Bats: São Paulo / 2024?  ') == (
            "bats-são-paulo-2024"
        )

    def test_a_name_with_nothing_usable_still_gets_a_folder(self):
        assert library_folder_name("!!!") == "library"

    def test_a_name_windows_reserves_is_avoided(self):
        assert library_folder_name("Con") == "con-library"

    def test_an_absent_or_empty_folder_is_fine(self, tmp_path):
        assert new_library_problem(tmp_path / "absent") is None
        (tmp_path / "empty").mkdir()
        assert new_library_problem(tmp_path / "empty") is None

    def test_a_folder_in_use_is_refused(self, tmp_path):
        (tmp_path / "used").mkdir()
        (tmp_path / "used" / "notes.txt").write_text("x")
        assert new_library_problem(tmp_path / "used") == "not_empty"

    def test_a_library_is_refused(self, tmp_path):
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "library.db").write_bytes(b"")
        assert new_library_problem(tmp_path / "lib") == "already_library"


def make_library(directory: Path, name: str) -> Path:
    """An existing library, as an earlier run would have left it."""
    service = LibraryService(load_settings(data_dir=directory))
    service.rename(name)
    service.close()
    return directory


def _field(user: User, marker: str):
    return next(iter(user.find(marker=marker).elements))


@pytest.mark.usefixtures("shared_pool")
class TestSetupPage:
    @pytest.fixture
    def chooser(self, tmp_path, monkeypatch):
        """Install state started without a library, optionally suggesting one."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIOTIMBRE_CONFIG", raising=False)
        states = []

        def make(suggested: Path | None = None) -> AppState:
            state = AppState(load_settings(), suggested_library=suggested)
            set_state(state)
            states.append(state)
            return state

        yield make
        for state in states:
            state.close_library()

    async def test_the_page_asks_instead_of_showing_a_library(
        self, user: User, chooser
    ):
        chooser()
        await user.open("/")
        await user.should_see("Choose a library")
        await user.should_see("Create a new library")
        await user.should_see("Open an existing library")
        await user.should_see("no library open")
        assert user.client.title == "no library open"

    async def test_with_nothing_to_open_it_offers_to_create(
        self, user: User, chooser
    ):
        chooser()
        await user.open("/")
        panels = next(iter(user.find(ui.tab_panels).elements))
        assert panels.value == "create"

    async def test_creating_makes_a_named_folder_and_opens_it(
        self, user: User, chooser, tmp_path
    ):
        state = chooser()
        await user.open("/")
        _field(user, "setup-name").value = "Atlantic Forest Cicadas"
        _field(user, "setup-location").value = str(tmp_path / "libraries")
        target = tmp_path / "libraries" / "atlantic-forest-cicadas"
        await user.should_see(f"Creates {target}")

        user.find(marker="setup-create").click()
        await user.should_see("Library items", retries=200)

        assert state.is_open is True
        assert state.settings.data_dir == target
        assert state.service.identity().name == "Atlantic Forest Cicadas"

    async def test_creating_needs_a_name(self, user: User, chooser):
        state = chooser()
        await user.open("/")
        user.find(marker="setup-create").click()
        await user.should_see("Give the library a name.")
        assert state.is_open is False

    async def test_creating_refuses_a_folder_in_use(
        self, user: User, chooser, tmp_path
    ):
        state = chooser()
        (tmp_path / "cicadas").mkdir()
        (tmp_path / "cicadas" / "field-notes.txt").write_text("x")

        await user.open("/")
        _field(user, "setup-name").value = "Cicadas"
        _field(user, "setup-location").value = str(tmp_path)
        user.find(marker="setup-create").click()

        await user.should_see("already exists and is not empty")
        assert state.is_open is False
        assert not (tmp_path / "cicadas" / "library.db").exists()

    async def test_creating_where_a_library_is_says_to_open_it(
        self, user: User, chooser, tmp_path
    ):
        state = chooser()
        make_library(tmp_path / "cicadas", "Cicadas")

        await user.open("/")
        _field(user, "setup-name").value = "Cicadas"
        _field(user, "setup-location").value = str(tmp_path)
        user.find(marker="setup-create").click()

        await user.should_see("already holds a library")
        assert state.is_open is False

    async def test_opening_an_existing_library(self, user: User, chooser, tmp_path):
        state = chooser()
        existing = make_library(tmp_path / "existing", "Existing")

        await user.open("/")
        _field(user, "setup-path").value = str(existing)
        user.find(marker="setup-open").click()
        await user.should_see("Library items", retries=200)

        assert state.is_open is True
        assert state.service.identity().name == "Existing"

    async def test_opening_a_folder_without_a_library_creates_nothing(
        self, user: User, chooser, tmp_path
    ):
        state = chooser()
        (tmp_path / "nothing").mkdir()

        await user.open("/")
        _field(user, "setup-path").value = str(tmp_path / "nothing")
        user.find(marker="setup-open").click()

        await user.should_see("There is no library in")
        assert state.is_open is False
        assert not (tmp_path / "nothing" / "library.db").exists()

    async def test_a_configured_library_is_offered_to_open(
        self, user: User, chooser, tmp_path
    ):
        existing = make_library(tmp_path / "configured", "Configured")
        state = chooser(suggested=existing)

        await user.open("/")
        panels = next(iter(user.find(ui.tab_panels).elements))
        assert panels.value == "open"
        assert _field(user, "setup-path").value == str(existing)
        # Offered, not opened.
        assert state.is_open is False

    async def test_a_configured_folder_not_made_yet_is_offered_to_create(
        self, user: User, chooser, tmp_path
    ):
        chooser(suggested=tmp_path / "data")

        await user.open("/")
        panels = next(iter(user.find(ui.tab_panels).elements))
        assert panels.value == "create"
        assert _field(user, "setup-name").value == "data"
        assert _field(user, "setup-location").value == str(tmp_path)
        await user.should_see(f"Creates {tmp_path / 'data'}")
