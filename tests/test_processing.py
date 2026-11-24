"""Processing a recording before it is stored, from the CLI and sidecars."""

from __future__ import annotations

import shutil

import pytest

from biotimbre.cli import main
from biotimbre.config import Settings
from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import FilteredBand, ItemDraft, Location, SampleWindow
from biotimbre.embeddings import audio as audio_io
from biotimbre.io.sidecar import parse_sidecar, processing_from_fields
from biotimbre.processing import Processing
from biotimbre.service import LibraryService
from conftest import EXAMPLES_DIR

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"


def _draft(**fields) -> ItemDraft:
    return ItemDraft(
        date="2020-01-12", time="09:40:21", location=Location(-26.0, -48.0), **fields
    ).validated()


class TestDescribe:
    def test_a_trim_of_a_whole_recording_becomes_the_sample_window(self):
        described = Processing(trim=(1.0, 4.0)).describe(_draft(), None)
        assert described.sample_window == SampleWindow(1.0, 4.0)

    def test_a_trim_of_an_excerpt_shifts_its_sample_window(self):
        draft = _draft(sample_window=SampleWindow(100.0, 130.0))
        described = Processing(trim=(5.0, 10.0)).describe(draft, None)
        assert described.sample_window == SampleWindow(105.0, 110.0)

    def test_a_filter_narrows_the_recorded_band(self):
        draft = _draft(filtered_band=FilteredBand(2000.0, 10000.0))
        described = Processing(highpass=3000.0, lowpass=12000.0).describe(draft, None)
        assert described.filtered_band == FilteredBand(3000.0, 10000.0)
        assert (described.filter_type, described.filter_order) == ("Butterworth", 4)

    def test_a_named_filter_type_is_kept(self):
        draft = _draft(filter_type="Chebyshev", filter_order=8)
        described = Processing(highpass=500.0).describe(draft, None)
        assert (described.filter_type, described.filter_order) == ("Chebyshev", 8)

    def test_normalisation_adds_a_line_to_the_notes(self):
        described = Processing(normalise=True).describe(_draft(notes="dawn"), 3.25)
        assert described.notes == (
            "dawn\nNormalised to peak at -1 dBFS (gain +3.2 dB)."
        )

    def test_silence_is_not_said_to_be_normalised(self):
        assert Processing(normalise=True).describe(_draft(), None).notes is None

    @pytest.mark.parametrize(
        "fields, message",
        [
            ({"trim": (4.0, 1.0)}, "end after it starts"),
            ({"trim": (-1.0, 1.0)}, "before 0"),
            ({"highpass": 5000.0, "lowpass": 1000.0}, "must be below"),
            ({"highpass": 100.0, "filter_order": 40}, "between 1 and 16"),
        ],
    )
    def test_impossible_processing_is_refused(self, fields, message):
        with pytest.raises(ValidationError, match=message):
            Processing(**fields)


class TestSidecar:
    def test_processing_lines_are_read(self):
        fields = parse_sidecar(
            "date 2020-01-01\ntime 10:00:00\nlocation 1,2\ntrim 0.5,3.5\n"
            "apply_highpass 1500\napply_filter_order 6\nnormalise yes\n"
        )
        assert processing_from_fields(fields) == Processing(
            trim=(0.5, 3.5), highpass=1500.0, filter_order=6, normalise=True
        )

    def test_a_sidecar_without_them_asks_for_nothing(self):
        fields = parse_sidecar("date 2020-01-01\ntime 10:00:00\nlocation 1,2\n")
        assert processing_from_fields(fields).is_empty

    @pytest.mark.parametrize(
        "line, message",
        [
            ("trim 5", "START,END"),
            ("apply_lowpass high", "must be a number"),
            ("normalise maybe", "yes or no"),
            ("apply_filter_order 4.5", "whole number"),
        ],
    )
    def test_an_unreadable_line_is_refused(self, line, message):
        with pytest.raises(ValidationError, match=message):
            processing_from_fields(parse_sidecar(line))


@pytest.fixture
def service(tmp_path):
    data_dir = tmp_path / "library"
    settings = Settings(
        data_dir=data_dir, db_path=data_dir / "library.db", audio_dir=data_dir / "audio"
    )
    with LibraryService(settings) as service:
        yield service


class TestStored:
    def test_the_processed_copy_is_what_is_stored(self, service):
        result = service.add_item(
            GUYVIR_WAV,
            _draft(),
            compute_embeddings=False,
            processing=Processing(trim=(1.0, 4.0), highpass=1500.0, normalise=True),
        )
        item = result.item
        assert item.duration_s == pytest.approx(3.0, abs=0.01)
        assert item.sample_window == SampleWindow(1.0, 4.0)
        assert item.filtered_band == FilteredBand(1500.0, None)
        assert "Normalised to peak at -1 dBFS" in item.notes
        stored = service.settings.absolute_audio_path(item.audio_path)
        peak = audio_io.sample_peak(audio_io.load_mono(stored)[0])
        assert peak == pytest.approx(10 ** (-1 / 20), rel=0.01)

    def test_a_trim_past_the_end_is_refused(self, service):
        with pytest.raises(ValidationError, match="only 5.00 s long"):
            service.add_item(
                GUYVIR_WAV,
                _draft(),
                compute_embeddings=False,
                processing=Processing(trim=(1.0, 60.0)),
            )

    def test_command_line_processing_wins_over_the_sidecar(self, service, tmp_path):
        audio = tmp_path / "clip.wav"
        shutil.copy(GUYVIR_WAV, audio)
        (tmp_path / "clip.txt").write_text(
            "date 2020-01-01\ntime 10:00:00\nlocation 1,2\n"
            "trim 0,4\napply_filter_order 6\napply_highpass 500\n",
            encoding="utf-8",
        )
        item = service.add_from_sidecar(
            audio,
            compute_embeddings=False,
            processing_overrides={"trim": (1.0, 2.0), "filter_order": 4},
        ).item
        assert item.sample_window == SampleWindow(1.0, 2.0)
        # Not given on the command line: the sidecar's line stands.
        assert item.filtered_band == FilteredBand(500.0, None)
        assert item.filter_order == 4


# ── the commands ──────────────────────────────────────────────────────────────


@pytest.fixture
def run(tmp_path, monkeypatch):
    from biotimbre.config import ENV_CONFIG

    monkeypatch.delenv(ENV_CONFIG, raising=False)
    monkeypatch.chdir(tmp_path)
    data_dir = str(tmp_path / "library")

    def invoke(*argv: str) -> int:
        return main(["--data-dir", data_dir, *argv])

    return invoke


def _add(run, *extra: str) -> None:
    assert run(
        "add", str(GUYVIR_WAV), "--date", "2020-01-12", "--time", "09:40:21",
        "--latitude", "-26", "--longitude", "-48", "--taxa", "cicadidae",
        "--device", "Audiomoth", "--no-embeddings", *extra,
    ) == 0


class TestEditCommand:
    def test_changes_only_what_is_given_and_reports_it(self, run, capsys):
        _add(run)
        capsys.readouterr()
        assert run("edit", "1", "--latitude", "-27", "--notes", "fence post") == 0
        out = capsys.readouterr().out
        assert "location  -26.0,-48.0 -> -27.0,-48.0" in out
        assert "notes     - -> fence post" in out

        run("show", "1")
        shown = capsys.readouterr().out
        assert "Audiomoth" in shown  # kept
        assert "-27.0,-48.0" in shown

    def test_clear_empties_fields(self, run, capsys):
        _add(run)
        capsys.readouterr()
        assert run("edit", "1", "--clear", "device", "taxa") == 0
        out = capsys.readouterr().out
        assert "device  Audiomoth -> -" in out
        assert "taxa    cicadidae -> -" in out

    def test_one_end_of_the_window_keeps_the_other(self, run, capsys):
        _add(run, "--window-start", "10", "--window-end", "15")
        capsys.readouterr()
        assert run("edit", "1", "--window-end", "16") == 0
        assert "sample_window  10,15 -> 10,16" in capsys.readouterr().out

    def test_nothing_to_change_is_an_error(self, run, capsys):
        _add(run)
        assert run("edit", "1") == 2
        assert "Nothing to change" in capsys.readouterr().err

    def test_setting_and_clearing_a_field_at_once_is_an_error(self, run, capsys):
        _add(run)
        assert run("edit", "1", "--device", "X", "--clear", "device") == 2
        assert "both given" in capsys.readouterr().err


class TestFilterOnlySearch:
    def test_lists_matching_items_unscored(self, run, capsys):
        _add(run)
        capsys.readouterr()
        assert run("search", "--taxa", "cicadidae") == 0
        out = capsys.readouterr().out
        assert "1. item 1" in out
        assert "score" not in out

    def test_reports_when_nothing_matches(self, run, capsys):
        _add(run)
        capsys.readouterr()
        assert run("search", "--taxa", "orthoptera") == 0
        assert "No items match these filters." in capsys.readouterr().out

    def test_query_options_need_a_query(self, run, capsys):
        _add(run)
        assert run("search", "--metric", "dot", "--apply-highpass", "100") == 2
        err = capsys.readouterr().err
        assert "--metric, --apply-highpass only apply to a query recording" in err

    def test_add_with_processing_flags(self, run, capsys):
        _add(run, "--trim", "1", "4", "--apply-highpass", "1500", "--normalise")
        capsys.readouterr()
        run("show", "1")
        out = capsys.readouterr().out
        assert "sample_window  1,4" in out
        assert "Butterworth" in out
        assert "Normalised to peak" in out
