"""Adding items from a CSV table, and pacing iNaturalist while doing it."""

from __future__ import annotations

import shutil

import pytest

from biotimbre.cli import main
from biotimbre.config import Settings
from biotimbre.domain.errors import BiotimbreError, ValidationError
from biotimbre.service import LibraryService
from biotimbre.sources import inaturalist
from biotimbre.sources.inaturalist import Observation, Sound, Throttle, Throttled
from biotimbre.table_import import import_table, read_table, resolve_file
from conftest import EXAMPLES_DIR

HEADER = "file,date,time,latitude,longitude,taxa"


@pytest.fixture
def folder(tmp_path):
    """A folder of recordings, as a table would sit beside."""
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    for name in ("GUYVIR.wav", "ZAMTYM.wav"):
        shutil.copy(EXAMPLES_DIR / name, recordings / name)
    return recordings


@pytest.fixture
def service(tmp_path):
    data_dir = tmp_path / "library"
    settings = Settings(
        data_dir=data_dir, db_path=data_dir / "library.db", audio_dir=data_dir / "audio"
    )
    with LibraryService(settings) as service:
        yield service


def _table(folder, text: str, name: str = "table.csv"):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def _import(service, path, **kwargs):
    kwargs.setdefault("compute_embeddings", False)
    return import_table(service, path, **kwargs)


class TestReading:
    def test_columns_and_rows(self, folder):
        rows = read_table(_table(folder, f"{HEADER}\nGUYVIR.wav,2020-01-12,09:40:21,-26,-48,cicadidae\n"))
        assert len(rows) == 1
        assert rows[0].line == 2
        assert rows[0].fields["taxa"] == "cicadidae"

    def test_an_unknown_column_is_refused(self, folder):
        with pytest.raises(ValidationError, match="unknown column.*'locaton'"):
            read_table(_table(folder, "file,locaton\nx.wav,1\n"))

    def test_a_table_needs_file_or_inat(self, folder):
        with pytest.raises(ValidationError, match="needs a 'file' column"):
            read_table(_table(folder, "date,time\n2020-01-01,10:00:00\n"))

    def test_empty_rows_are_left_out_and_lines_counted_across_quoted_breaks(
        self, folder
    ):
        rows = read_table(_table(
            folder,
            'file,notes\nGUYVIR.wav,"two\nlines"\n,\nZAMTYM.wav,\n',
        ))
        assert [(row.line, row.fields["file"]) for row in rows] == [
            (2, "GUYVIR.wav"), (5, "ZAMTYM.wav"),
        ]
        assert rows[0].fields["notes"] == "two\nlines"

    def test_a_semicolon_table_with_decimal_commas(self, folder):
        rows = read_table(_table(
            folder,
            "\ufefffile;latitude;longitude;trim\nGUYVIR.wav;-26,9;-48,7;1,4\n",
        ))
        fields = rows[0].fields
        assert (fields["latitude"], fields["longitude"]) == ("-26.9", "-48.7")
        # A pair keeps its comma: it separates the two values.
        assert fields["trim"] == "1,4"

    def test_a_tab_separated_table(self, folder):
        rows = read_table(_table(folder, "file\ttaxa\nGUYVIR.wav\tcicadidae\n"))
        assert rows[0].fields == {"file": "GUYVIR.wav", "taxa": "cicadidae"}


class TestPaths:
    def test_a_bare_name_is_beside_the_table(self, folder):
        assert resolve_file("GUYVIR.wav", folder) == folder / "GUYVIR.wav"

    def test_a_relative_path_is_from_the_table(self, folder):
        sub = folder / "dawn"
        sub.mkdir()
        shutil.copy(folder / "GUYVIR.wav", sub / "a.wav")
        assert resolve_file("dawn/a.wav", folder) == sub / "a.wav"

    def test_an_absolute_path_is_used_as_given(self, folder, tmp_path):
        assert resolve_file(str(folder / "ZAMTYM.wav"), tmp_path) == folder / "ZAMTYM.wav"

    def test_a_missing_file_says_where_it_was_looked_for(self, folder):
        with pytest.raises(ValidationError, match="looked for it in .*recordings"):
            resolve_file("nope.wav", folder)

    @pytest.mark.skipif(
        __import__("os").name == "nt", reason="a Windows path is at home there"
    )
    def test_a_windows_path_is_named_as_one(self, folder):
        with pytest.raises(ValidationError, match="is a Windows path"):
            resolve_file(r"C:\Users\me\GUYVIR.wav", folder)

    @pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX separators")
    def test_a_relative_windows_path_still_finds_the_file(self, folder):
        sub = folder / "dawn"
        sub.mkdir()
        shutil.copy(folder / "GUYVIR.wav", sub / "a.wav")
        assert resolve_file(r"dawn\a.wav", folder) == sub / "a.wav"


class TestFileRows:
    def test_each_row_is_added_and_bad_ones_reported(self, service, folder):
        path = _table(folder, (
            "file,date,time,latitude,longitude,trim,normalise\n"
            'GUYVIR.wav,2020-01-12,09:40:21,-26,-48,"1,4",yes\n'
            "missing.wav,2020-01-12,09:40:21,-26,-48,,\n"
            "ZAMTYM.wav,2020-01-12,,-26,-48,,\n"
        ))
        report = _import(service, path)
        assert [row.line for row, _ in report.added] == [2]
        failed = {row.line: message for row, message in report.failed}
        assert "No such recording" in failed[3]
        assert "time" in failed[4]

        item = report.added[0][1].item
        assert item.duration_s == pytest.approx(3.0, abs=0.01)
        assert "Normalised" in item.notes

    def test_running_it_again_adds_nothing_twice(self, service, folder):
        path = _table(folder, (
            "file,date,time,latitude,longitude,apply_highpass\n"
            "GUYVIR.wav,2020-01-12,09:40:21,-26,-48,500\n"
            "ZAMTYM.wav,2020-01-12,09:40:21,-26,-48,\n"
        ))
        assert len(_import(service, path).added) == 2
        again = _import(service, path)
        assert again.added == []
        assert [ids for _, ids in again.skipped] == [(1,), (2,)]

    def test_keep_duplicates_stores_them_again(self, service, folder):
        path = _table(folder, f"{HEADER}\nGUYVIR.wav,2020-01-12,09:40:21,-26,-48,\n")
        _import(service, path)
        assert len(_import(service, path, skip_duplicates=False).added) == 1

    @pytest.mark.parametrize(
        "row, message",
        [
            ("GUYVIR.wav,123,2020-01-12,09:40:21,-26,-48", "not both"),
            (",,2020-01-12,09:40:21,-26,-48", "one of the two"),
        ],
    )
    def test_a_row_needs_one_source(self, service, folder, row, message):
        path = _table(folder, f"file,inat,date,time,latitude,longitude\n{row}\n")
        (row_failed, text), = _import(service, path).failed
        assert message in text


# ── iNaturalist rows ──────────────────────────────────────────────────────────

FIRST = Sound(id=1, url="https://x/1.wav", license_code="cc-by",
              attribution="(c) Ana, some rights reserved (CC BY)")
OBSERVATION = Observation(
    id=123, url="https://www.inaturalist.org/observations/123",
    date="2024-02-03", time="21:15:00", latitude=-26.9, longitude=-48.7,
    taxon="Guyalna bonaerensis", quality_grade="research", observer="Ana",
    sounds=(FIRST,),
)


class FakeThrottle(Throttle):
    """Paces nothing: what is tested here is what the import asks of it."""

    def __init__(self):
        super().__init__(sleep=lambda s: None)
        self.calls: list[bool] = []

    def call(self, request, *, media=False):
        self.calls.append(media)
        return request()


@pytest.fixture
def inat(monkeypatch):
    served = {"observations": {123: OBSERVATION}, "batches": [], "downloads": 0}

    def fetch_observations(ids, **_):
        ids = list(ids)
        served["batches"].append(ids)
        return {i: served["observations"][i] for i in ids if i in served["observations"]}

    def download_sound(sound, destination, *, on_progress=None, **_):
        served["downloads"] += 1
        shutil.copy(EXAMPLES_DIR / "GUYVIR.wav", destination)
        if on_progress:
            on_progress(1000, 1000)
        return destination

    monkeypatch.setattr(inaturalist, "fetch_observations", fetch_observations)
    monkeypatch.setattr(inaturalist, "download_sound", download_sound)
    return served


class TestInatRows:
    def test_the_observation_fills_what_the_row_leaves_empty(self, service, folder, inat):
        path = _table(folder, "inat,taxa\n123,cicadidae\n")
        throttle = FakeThrottle()
        report = _import(service, path, throttle=throttle)
        (row, result), = report.added
        item = result.item
        assert item.taxa == ("cicadidae",)  # the row's
        assert (item.date, item.recorded_by) == ("2024-02-03", "Ana")
        assert "iNaturalist observation 123" in item.notes
        # Downloads go through the throttle as media.
        assert throttle.calls == [True]

    def test_observations_are_asked_for_together(self, service, folder, inat):
        path = _table(folder, "inat\n123\nhttps://www.inaturalist.org/observations/456\n123\n")
        _import(service, path, throttle=FakeThrottle())
        assert inat["batches"] == [[123, 456]]

    def test_a_missing_observation_fails_its_row_only(self, service, folder, inat):
        path = _table(folder, "inat\n456\n123\n")
        report = _import(service, path, throttle=FakeThrottle())
        assert len(report.added) == 1
        (row, message), = report.failed
        assert "no observation 456" in message

    def test_an_obscured_location_must_be_in_the_row(self, service, folder, inat):
        import dataclasses

        inat["observations"][123] = dataclasses.replace(
            OBSERVATION, latitude=None, longitude=None, obscured=True
        )
        report = _import(service, _table(folder, "inat\n123\n"), throttle=FakeThrottle())
        (row, message), = report.failed
        assert "iNaturalist hides this observation's location" in message

        given = _table(folder, "inat,latitude,longitude\n123,-26,-48\n", "b.csv")
        assert len(_import(service, given, throttle=FakeThrottle()).added) == 1

    def test_caveats_become_warnings(self, service, folder, inat):
        import dataclasses

        inat["observations"][123] = dataclasses.replace(OBSERVATION, quality_grade="needs_id")
        (row, result), = _import(
            service, _table(folder, "inat\n123\n"), throttle=FakeThrottle()
        ).added
        assert any("not research grade" in w for w in result.warnings)

    def test_off_for_an_offline_library(self, folder, tmp_path, inat):
        import dataclasses

        data_dir = tmp_path / "offline"
        settings = Settings(
            data_dir=data_dir, db_path=data_dir / "library.db",
            audio_dir=data_dir / "audio", inaturalist_enabled=False,
        )
        with LibraryService(settings) as service:
            report = _import(service, _table(folder, "inat\n123\n"))
        (row, message), = report.failed
        assert "turned off" in message
        assert inat["batches"] == []


# ── the throttle ──────────────────────────────────────────────────────────────


class Clock:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestThrottle:
    def test_requests_are_a_second_apart(self):
        clock = Clock()
        throttle = Throttle(clock=clock, sleep=clock.sleep)
        for _ in range(3):
            throttle.call(lambda: None)
        assert clock.slept == [1.0, 1.0]

    def test_a_throttled_request_waits_what_it_is_told(self):
        clock = Clock()
        waits: list[tuple[float, str]] = []
        throttle = Throttle(clock=clock, sleep=clock.sleep,
                            on_wait=lambda s, why: waits.append((s, why)))
        answers = iter([Throttled(30.0), "ok"])

        def request():
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        assert throttle.call(request) == "ok"
        assert waits == [(30.0, "iNaturalist asked to slow down")]

    def test_without_a_retry_after_it_backs_off_doubling(self):
        clock = Clock()
        throttle = Throttle(clock=clock, sleep=clock.sleep)
        tries = {"n": 0}

        def request():
            tries["n"] += 1
            if tries["n"] <= 3:
                raise Throttled()
            return "ok"

        throttle.call(request)
        backoffs = [s for s in clock.slept if s > 1.0]
        assert backoffs == [2.0, 4.0, 8.0]

    def test_it_gives_up_eventually(self):
        clock = Clock()
        throttle = Throttle(clock=clock, sleep=clock.sleep)

        def request():
            raise Throttled(1.0)

        with pytest.raises(BiotimbreError, match="kept asking to slow down"):
            throttle.call(request)

    def test_downloads_wait_for_room_under_the_hourly_limit(self):
        clock = Clock()
        waits: list[str] = []
        throttle = Throttle(clock=clock, sleep=clock.sleep, media_bytes_per_hour=100,
                            on_wait=lambda s, why: waits.append(why))
        throttle.call(lambda: None, media=True)
        throttle.record_media(100)
        throttle.call(lambda: None, media=True)
        assert "the hourly media limit was reached" in waits
        assert clock.now >= 3600.0


class TestCommand:
    def test_dry_run_changes_nothing(self, folder, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIOTIMBRE_CONFIG", raising=False)
        path = _table(folder, f"{HEADER}\nGUYVIR.wav,2020-01-12,09:40:21,-26,-48,\nnope.wav,,,,,\n")
        assert main(["--data-dir", "lib", "import-table", str(path), "--dry-run"]) == 1
        out = capsys.readouterr().out
        assert "line    2  ok" in out
        assert "line    3  FAILED  nope.wav" in out
        assert main(["--data-dir", "lib", "list"]) == 0
        assert "empty" in capsys.readouterr().out

    def test_the_import_reports_each_row(self, folder, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIOTIMBRE_CONFIG", raising=False)
        path = _table(folder, f"{HEADER}\nGUYVIR.wav,2020-01-12,09:40:21,-26,-48,\n")
        argv = ["--data-dir", "lib", "import-table", str(path), "--no-embeddings"]
        assert main(argv) == 0
        assert "line    2  added   item 1  GUYVIR.wav" in capsys.readouterr().out
        assert main(argv) == 0
        assert "skipped GUYVIR.wav: already item(s) 1" in capsys.readouterr().out
