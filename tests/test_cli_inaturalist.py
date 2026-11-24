"""The command line's iNaturalist fetch: add --inat and search --inat.

iNaturalist is not reached: the observation and the download are replaced by
stand-ins, so what is tested is what the commands make of them.
"""

from __future__ import annotations

import shutil

import pytest

from biotimbre.cli import main
from biotimbre.sources import inaturalist
from biotimbre.sources.inaturalist import Observation, Sound
from conftest import EXAMPLES_DIR

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"

FIRST = Sound(id=1, url="https://x/1.wav", license_code="cc-by",
              attribution="(c) Ana, some rights reserved (CC BY)")
SECOND = Sound(id=2, url="https://x/2.wav", license_code="cc0",
               attribution="Ana, no rights reserved (CC0)")

OBSERVATION = Observation(
    id=123,
    url="https://www.inaturalist.org/observations/123",
    date="2024-02-03",
    time="21:15:00",
    latitude=-26.9,
    longitude=-48.7,
    taxon="Guyalna bonaerensis",
    quality_grade="research",
    observer="Ana",
    sounds=(FIRST, SECOND),
)


@pytest.fixture
def run(tmp_path, monkeypatch):
    from biotimbre.config import ENV_CONFIG

    monkeypatch.delenv(ENV_CONFIG, raising=False)
    monkeypatch.chdir(tmp_path)
    data_dir = str(tmp_path / "library")

    def invoke(*argv: str) -> int:
        return main(["--data-dir", data_dir, *argv])

    return invoke


@pytest.fixture
def observation(monkeypatch):
    """Serve an observation, and record which recording was downloaded."""
    served = {"observation": OBSERVATION, "downloaded": []}

    def fetch(observation_id, **_):
        assert observation_id == served["observation"].id
        return served["observation"]

    def download(sound, destination, **_):
        served["downloaded"].append(sound.id)
        shutil.copy(GUYVIR_WAV, destination)
        return destination

    monkeypatch.setattr(inaturalist, "fetch_observation", fetch)
    monkeypatch.setattr(inaturalist, "download_sound", download)
    return served


def _show(run, capsys, item_id: int = 1) -> str:
    capsys.readouterr()
    run("show", str(item_id))
    return capsys.readouterr().out


class TestAdd:
    def test_the_observation_fills_the_metadata(self, run, capsys, observation):
        assert run("add", "--inat", "123", "--no-embeddings") == 0
        shown = _show(run, capsys)
        assert "2024-02-03" in shown and "21:15:00" in shown
        assert "-26.9,-48.7" in shown
        assert "guyalna bonaerensis" in shown
        assert "recorded_by    Ana" in shown
        assert "https://www.inaturalist.org/observations/123" in shown
        assert "iNaturalist observation 123 — (c) Ana, some rights reserved" in shown

    def test_options_replace_what_it_says(self, run, capsys, observation):
        assert run("add", "--inat", "123", "--taxa", "cicadidae", "--date",
                   "2024-02-04", "--notes", "by the river", "--no-embeddings") == 0
        shown = _show(run, capsys)
        assert "cicadidae" in shown and "guyalna" not in shown
        assert "2024-02-04" in shown
        # Notes given are kept, and the credit still added.
        assert "by the river" in shown
        assert "iNaturalist observation 123" in shown

    def test_the_credit_is_for_the_recording_used(self, run, capsys, observation):
        assert run("add", "--inat", "123", "--inat-sound", "2",
                   "--no-embeddings") == 0
        assert observation["downloaded"] == [2]
        shown = _show(run, capsys)
        assert "CC0" in shown
        assert "CC BY)" not in shown

    def test_several_recordings_are_listed(self, run, capsys, observation):
        run("add", "--inat", "123", "--no-embeddings")
        out = capsys.readouterr().out
        assert "* recording 1: cc-by" in out
        assert "  recording 2: cc0" in out

    def test_an_obscured_location_must_be_given(self, run, capsys, observation):
        import dataclasses

        observation["observation"] = dataclasses.replace(
            OBSERVATION, latitude=None, longitude=None, obscured=True
        )
        assert run("add", "--inat", "123", "--no-embeddings") == 2
        err = capsys.readouterr().err
        assert "--latitude (iNaturalist hides this observation's location)" in err

        assert run("add", "--inat", "123", "--latitude", "-26", "--longitude",
                   "-48", "--no-embeddings") == 0

    def test_a_recording_number_out_of_range(self, run, capsys, observation):
        assert run("add", "--inat", "123", "--inat-sound", "3") == 2
        assert "--inat-sound must be 1 to 2" in capsys.readouterr().err

    def test_reserved_rights_are_warned_of(self, run, capsys, observation):
        import dataclasses

        observation["observation"] = dataclasses.replace(
            OBSERVATION, sounds=(dataclasses.replace(FIRST, license_code=None),)
        )
        run("add", "--inat", "123", "--no-embeddings")
        assert "all rights are reserved" in capsys.readouterr().err

    def test_off_for_an_offline_library(self, run, capsys, observation, tmp_path):
        (tmp_path / "biotimbre.toml").write_text(
            "[sources.inaturalist]\nenabled = false\n", encoding="utf-8"
        )
        assert run("add", "--inat", "123") == 2
        assert "turned off for this library" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["add"], "Give AUDIO, the recording to add, or --inat"),
            (["add", "x.wav", "--inat", "123"], "not both"),
            (["add", "--inat", "123", "--sidecar", "x.txt"], "two sources of metadata"),
            (["add", "x.wav", "--inat-sound", "2"], "--inat-sound needs --inat"),
        ],
    )
    def test_conflicting_sources_are_refused(self, run, capsys, argv, message):
        assert run(*argv) == 2
        assert message in capsys.readouterr().err


class TestSearchFilters:
    """--inat-filters, checked by a search with no query: none is needed to
    see which filters were built, and it needs no models."""

    def _filters(self, monkeypatch, *argv):
        from biotimbre import cli

        captured = {}
        original = cli._search_filters

        def spy(args, observation=None, **kwargs):
            captured["filters"] = original(args, observation, **kwargs)
            return captured["filters"]

        monkeypatch.setattr(cli, "_search_filters", spy)
        # The query is there; the search is stopped before it embeds anything.
        monkeypatch.setattr(
            cli.LibraryService, "search", lambda self, *a, **k: []
        )
        main(["--data-dir", "lib", "search", "--inat", "123", *argv])
        return captured["filters"]

    def test_without_the_flag_only_the_query_comes_from_it(
        self, monkeypatch, tmp_path, observation
    ):
        monkeypatch.chdir(tmp_path)
        filters = self._filters(monkeypatch)
        assert filters.taxa == () and filters.radius_km is None

    def test_with_it_its_taxon_and_place_narrow_the_search(
        self, monkeypatch, tmp_path, observation, capsys
    ):
        monkeypatch.chdir(tmp_path)
        filters = self._filters(monkeypatch, "--inat-filters")
        assert filters.taxa == ("guyalna bonaerensis",)
        assert (filters.latitude, filters.longitude, filters.radius_km) == (
            -26.9, -48.7, 25.0
        )
        assert "Filters from the observation" in capsys.readouterr().out

    def test_options_given_win(self, monkeypatch, tmp_path, observation):
        monkeypatch.chdir(tmp_path)
        filters = self._filters(
            monkeypatch, "--inat-filters", "--taxa", "cicadidae", "--radius-km", "5"
        )
        assert filters.taxa == ("cicadidae",)
        assert filters.radius_km == 5.0

    def test_an_obscured_location_is_not_a_filter(
        self, monkeypatch, tmp_path, observation, capsys
    ):
        import dataclasses

        monkeypatch.chdir(tmp_path)
        observation["observation"] = dataclasses.replace(
            OBSERVATION, latitude=None, longitude=None, obscured=True
        )
        filters = self._filters(monkeypatch, "--inat-filters")
        assert filters.radius_km is None
        assert "not used as a filter" in capsys.readouterr().err


class TestCredit:
    def test_credit_for_names_the_recording_used(self):
        assert "CC0" in OBSERVATION.credit_for(SECOND)
        assert OBSERVATION.credit == OBSERVATION.credit_for(FIRST)

    def test_the_add_form_replaces_the_line_for_another_recording(self):
        pytest.importorskip("nicegui")
        from types import SimpleNamespace

        from biotimbre.ui.admin import _credit_recording

        notes = SimpleNamespace(value="my own note")
        form = SimpleNamespace(inputs={"notes": notes})
        _credit_recording(form, OBSERVATION, FIRST)
        _credit_recording(form, OBSERVATION, SECOND)
        assert notes.value == "my own note\n" + OBSERVATION.credit_for(SECOND)
