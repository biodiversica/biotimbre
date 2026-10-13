"""Search-only runs: one page, and a library that cannot be written to.

``gui --search-only`` is the mode for handing other people a way to query a
library without handing them the library. Three things have to hold together
for that to mean anything: the library page must not be served, the chooser
must not be mounted — it is how a reader would otherwise open a writable
library — and the database must be opened read-only, which is what turns a
convention into a guarantee.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.config import Settings, load_settings
from biotimbre.db.connection import (  # noqa: E402
    SCHEMA_VERSION,
    initialize,
    open_read_only,
)
from biotimbre.domain.errors import BiotimbreError  # noqa: E402
from conftest import EXAMPLES_DIR  # noqa: E402

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"


def make_library(directory: Path) -> Path:
    """Create an empty but valid library, and close it again."""
    directory.mkdir(parents=True, exist_ok=True)
    connection = initialize(directory / "library.db")
    connection.close()
    return directory


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A closed, empty library on disk for a search-only run to open."""
    return make_library(tmp_path / "shared")


def search_only_settings(directory: Path) -> Settings:
    """Settings a search-only run would resolve for ``directory``."""
    return Settings(
        data_dir=directory,
        db_path=directory / "library.db",
        audio_dir=directory / "audio",
        search_only=True,
    )


class TestSettings:
    def test_it_is_off_unless_asked_for(self, tmp_path):
        assert load_settings(data_dir=tmp_path).search_only is False

    def test_the_config_file_can_turn_it_on(self, tmp_path):
        config = tmp_path / "biotimbre.toml"
        config.write_text("[ui]\nsearch_only = true\n")
        assert load_settings(config, data_dir=tmp_path).search_only is True


class TestOpeningReadOnly:
    def test_an_existing_library_opens(self, library):
        connection = open_read_only(library / "library.db")
        try:
            assert connection.execute("SELECT count(*) FROM items").fetchone()[0] == 0
        finally:
            connection.close()

    def test_a_write_is_refused_by_the_database_itself(self, library):
        connection = open_read_only(library / "library.db")
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute(
                    "INSERT INTO schema_meta (key, value) VALUES ('x', 'y')"
                )
        finally:
            connection.close()

    def test_a_missing_library_is_not_created(self, tmp_path):
        # The whole point of the mode is that this run adds nothing, so a
        # mistyped path has to be an error rather than a new empty library.
        missing = tmp_path / "nowhere" / "library.db"
        with pytest.raises(BiotimbreError, match="No library database"):
            open_read_only(missing)
        assert not missing.exists()

    def test_a_file_that_is_not_a_library_is_refused(self, tmp_path):
        stray = tmp_path / "library.db"
        stray.write_bytes(b"not a database")
        with pytest.raises(BiotimbreError):
            open_read_only(stray)

    def test_an_old_schema_is_refused_rather_than_migrated(self, library):
        # Migrating writes, and this connection has no writes to give. Saying
        # so beats failing later, in the middle of somebody's search.
        connection = initialize(library / "library.db")
        connection.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION - 1),),
        )
        connection.commit()
        connection.close()

        with pytest.raises(BiotimbreError, match="--search-only"):
            open_read_only(library / "library.db")


class TestAppState:
    def test_the_library_opens_read_only(self, library, shared_pool):
        from biotimbre.ui.state import AppState

        state = AppState(search_only_settings(library))
        try:
            assert state.is_open
            assert state.service.read_only is True
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                state.service.rename("renamed")
        finally:
            state.close()

    def test_it_creates_nothing_on_the_way_in(self, library, shared_pool):
        from biotimbre.ui.state import AppState

        # A normal open calls ensure_dirs; a search-only one must not, because
        # a run that cannot add audio has no use for a directory to put it in.
        state = AppState(search_only_settings(library))
        try:
            assert not (library / "audio").exists()
        finally:
            state.close()

    def test_a_missing_library_is_reported_not_made(self, tmp_path):
        from biotimbre.ui.state import AppState

        with pytest.raises(BiotimbreError, match="No library database"):
            AppState(search_only_settings(tmp_path / "absent"))


@pytest.mark.models
class TestSearching:
    def test_a_search_runs_against_a_read_only_library(self, tmp_path, shared_pool):
        from biotimbre.service import LibraryService
        from biotimbre.ui.state import AppState

        directory = tmp_path / "shared"
        directory.mkdir()
        built = Settings(
            data_dir=directory,
            db_path=directory / "library.db",
            audio_dir=directory / "audio",
        )
        with LibraryService(built, pool=shared_pool) as service:
            results, failures = service.import_directory(EXAMPLES_DIR)
            assert not failures and results

        state = AppState(search_only_settings(directory))
        state.pool = shared_pool
        state.service._pool = shared_pool
        state.service._owns_pool = False
        try:
            # Searching registers models on a writable library, which is a
            # write on the read path: a read-only run has to get past it.
            hits = state.call(state.service.search, GUYVIR_WAV, k=2)
            assert hits and hits[0].score > 0.0
        finally:
            state.close()


@pytest.fixture
def installed_state(library, shared_pool):
    """A search-only run over ``library``, installed for the pages to find."""
    from biotimbre.ui.state import AppState, get_state, set_state

    state = AppState(search_only_settings(library))
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


class TestPages:
    async def test_the_library_page_is_not_served(
        self, user: User, installed_state
    ):
        # A redirect rather than a hidden tab: the page is never built, so none
        # of its handlers exist to be reached over the socket afterwards.
        response = await user.http_client.get("/", follow_redirects=False)
        assert response.status_code in (302, 303, 307)
        assert response.headers["location"] == "/search"

    async def test_the_root_lands_on_the_search_page(
        self, user: User, installed_state
    ):
        await user.open("/")
        await user.should_see("Find similar samples")

    async def test_there_is_no_tab_back_to_the_library(
        self, user: User, installed_state
    ):
        await user.open("/search")
        await user.should_see("Find similar samples")

        labels = {
            element._props.get("label") for element in user.find(ui.button).elements
        }
        assert "Library" not in labels
        assert "Search" in labels

    async def test_the_header_says_what_kind_of_run_this_is(
        self, user: User, installed_state
    ):
        # Otherwise the only evidence is a missing tab, which reads as a bug.
        await user.open("/search")
        await user.should_see("Search only")

    async def test_a_query_cannot_be_taken_to_the_library(
        self, user: User, installed_state
    ):
        from nicegui.elements.upload_files import SmallFileUpload

        await user.open("/search")
        uploader = next(iter(user.find(ui.upload).elements))
        await uploader.handle_uploads(
            [SmallFileUpload("GUYVIR.wav", "audio/wav", GUYVIR_WAV.read_bytes())]
        )
        await user.should_see("Query speed", retries=60)
        # There is no add form to take it to.
        await user.should_not_see("Add to library…")

    async def test_a_normal_run_still_has_both_pages(
        self, user: User, settings, shared_pool
    ):
        from biotimbre.ui.state import AppState, get_state, set_state

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
        try:
            response = await user.http_client.get("/", follow_redirects=False)
            assert response.status_code == 200
        finally:
            state.close()
            if previous is not None:
                set_state(previous)


class TestTheChooser:
    """The one route that lets a request name a path on the server."""

    def test_it_is_not_mounted_when_there_is_no_library_to_search(
        self, tmp_path, shared_pool, monkeypatch
    ):
        from nicegui import app as nicegui_app

        from biotimbre.ui import app as ui_app
        from biotimbre.ui.routes import BROWSE_ROUTE
        from biotimbre.ui.state import get_state, set_state

        # Nothing chosen, so a normal run would offer the chooser. A
        # search-only run may not open a library at all, so it does not.
        settings = Settings(
            data_dir=tmp_path / "data",
            db_path=tmp_path / "data" / "library.db",
            audio_dir=tmp_path / "data" / "audio",
            data_dir_chosen=False,
            search_only=True,
        )
        monkeypatch.setattr(nicegui_app, "on_startup", lambda *a, **k: None)
        monkeypatch.setattr(nicegui_app, "on_shutdown", lambda *a, **k: None)

        previous = None
        try:
            previous = get_state()
        except RuntimeError:
            pass
        state = ui_app.create_app(settings)
        try:
            assert not state.is_open
            paths = {getattr(route, "path", None) for route in nicegui_app.routes}
            assert BROWSE_ROUTE not in paths
        finally:
            state.close()
            if previous is not None:
                set_state(previous)


class TestStickiness:
    """Whatever asked for the mode, a reload of the configuration keeps it."""

    def test_opening_another_library_stays_read_only(
        self, library, tmp_path, shared_pool, monkeypatch
    ):
        from biotimbre.ui.state import AppState

        # A config file that says nothing about the mode, found from the
        # working directory the way open_library finds one.
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        (workdir / "biotimbre.toml").write_text(
            '[ui]\nlanguage = "en"\n', "utf-8"
        )
        monkeypatch.chdir(workdir)
        other = make_library(tmp_path / "other")

        state = AppState(search_only_settings(library))
        state.pool = shared_pool
        try:
            state.open_library(other)
            assert state.settings.search_only is True
            assert state.service.read_only is True
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                state.service.rename("renamed")
        finally:
            state.close()


class TestCommandLine:
    def test_the_flag_reaches_the_interface(self, tmp_path, monkeypatch):
        from biotimbre.cli import main

        seen = {}

        def fake_run(settings, **kwargs):
            seen["settings"] = settings
            seen["overrides"] = kwargs.get("overrides")

        import biotimbre.ui.app as ui_app

        monkeypatch.setattr(ui_app, "run", fake_run)
        directory = str(make_library(tmp_path / "shared"))
        assert main(["--data-dir", directory, "gui", "--search-only"]) == 0
        assert seen["settings"].search_only is True
        # As an override too, so opening a library cannot undo it.
        assert seen["overrides"]["search_only"] is True

    def test_it_refuses_to_start_without_a_library(
        self, tmp_path, monkeypatch, capsys
    ):
        from biotimbre.cli import main

        import biotimbre.ui.app as ui_app

        # Somewhere with no biotimbre.toml, so nothing supplies a library.
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIOTIMBRE_CONFIG", raising=False)
        monkeypatch.setattr(
            ui_app, "run", lambda *a, **k: pytest.fail("should not have started")
        )
        # With no library the interface falls back to the chooser, which is
        # precisely what this mode withholds; there would be nothing to serve.
        assert main(["gui", "--search-only"]) == 2
        assert "--search-only needs a library" in capsys.readouterr().err

    def test_without_the_flag_nothing_changes(self, tmp_path, monkeypatch):
        from biotimbre.cli import main

        seen = {}

        import biotimbre.ui.app as ui_app

        monkeypatch.setattr(ui_app, "run", lambda s, **k: seen.update(settings=s))
        assert main(["--data-dir", str(tmp_path / "fresh"), "gui"]) == 0
        assert seen["settings"].search_only is False
