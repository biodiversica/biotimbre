"""CLI argument handling and output.

Kept to the paths that do not need models, plus one end-to-end run.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from biotimbre.cli import build_parser, main
from conftest import EXAMPLES_DIR

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"


@pytest.fixture
def data_dir(tmp_path):
    """A fresh library directory for CLI invocations."""
    return str(tmp_path / "library")


@pytest.fixture(autouse=True)
def _share_sessions(monkeypatch, request):
    """Give CLI invocations the session-scoped ONNX pool.

    A real CLI run is one process, so building its own pool is correct in
    production. In tests it means every `run()` reloads both graphs — about 2 s
    each, and a workflow test makes four invocations. Injecting the shared pool
    leaves what these tests actually cover (argument parsing, output, exit codes)
    untouched.
    """
    if "models" not in request.keywords:
        return
    from biotimbre import cli

    pool = request.getfixturevalue("shared_pool")
    original = cli.LibraryService

    def factory(settings, **kwargs):
        kwargs.setdefault("pool", pool)
        return original(settings, **kwargs)

    monkeypatch.setattr(cli, "LibraryService", factory)


def run(data_dir: str, *argv: str) -> int:
    """Invoke the CLI with a data directory, returning the exit code."""
    return main(["--data-dir", data_dir, *argv])


class TestArgumentParsing:
    def test_negative_coordinates_parse(self):
        # argparse treats a value starting with '-' as an option unless it looks
        # like a number, so "--location -26,-48" is impossible. Every southern
        # hemisphere latitude is negative, hence scalar options.
        args = build_parser().parse_args(
            [
                "add", "x.wav",
                "--date", "2020-01-01", "--time", "09:00:00",
                "--latitude", "-26.9", "--longitude", "-48.7",
                "--window-start", "0", "--window-end", "5",
            ]
        )
        assert args.latitude == -26.9
        assert args.longitude == -48.7

    def test_negative_search_coordinates_parse(self):
        args = build_parser().parse_args(
            ["search", "q.wav", "--latitude", "-26", "--longitude", "-48",
             "--radius-km", "50"]
        )
        assert (args.latitude, args.longitude, args.radius_km) == (-26.0, -48.0, 50.0)

    def test_speed_defaults_to_one(self):
        args = build_parser().parse_args(["add", "x.wav"])
        assert args.speed == 1.0

    def test_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestMetadataOnlyCommands:
    def test_init_without_models(self, data_dir, capsys):
        assert run(data_dir, "init", "--skip-models", "--name", "Test Library") == 0
        assert "not registered" in capsys.readouterr().out

    def test_list_on_empty_library(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        assert run(data_dir, "list") == 0
        assert "empty" in capsys.readouterr().out

    def test_add_reports_missing_options(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        # --date given but the rest omitted: must name what is missing rather
        # than falling through to the sidecar path.
        code = run(
            data_dir, "add", str(GUYVIR_WAV), "--date", "2020-01-01",
            "--no-embeddings",
        )
        assert code == 2
        assert "--latitude" in capsys.readouterr().err

    def test_add_from_sidecar_and_list(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        assert run(data_dir, "add", str(GUYVIR_WAV), "--no-embeddings") == 0
        capsys.readouterr()
        assert run(data_dir, "list") == 0
        out = capsys.readouterr().out
        assert "unknown" in out

    def test_add_without_a_sample_window(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        code = run(
            data_dir, "add", str(GUYVIR_WAV),
            "--date", "2021-03-05", "--time", "22:10:00",
            "--latitude", "-26.9", "--longitude", "-48.7",
            "--no-embeddings",
        )
        assert code == 0
        capsys.readouterr()
        assert run(data_dir, "show", "1") == 0
        assert "sample_window  -" in capsys.readouterr().out

    def test_add_with_explicit_flags(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        code = run(
            data_dir, "add", str(GUYVIR_WAV),
            "--date", "2021-03-05", "--time", "22:10:00",
            "--latitude", "-26.9", "--longitude", "-48.7",
            "--window-start", "12", "--window-end", "17",
            "--taxa", "orthoptera", "insecta",
            "--highpass", "8000", "--lowpass", "40000",
            "--speed", "0.25", "--no-embeddings",
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "insecta, orthoptera" in out
        assert "speed    0.25" in out

    def test_nyquist_advisory_appears(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        run(
            data_dir, "add", str(GUYVIR_WAV),
            "--date", "2021-03-05", "--time", "22:10:00",
            "--latitude", "-26.9", "--longitude", "-48.7",
            "--window-start", "12", "--window-end", "17",
            "--lowpass", "40000", "--no-embeddings",
        )
        out = capsys.readouterr().out
        assert "warning" in out
        assert "Consider speed" in out

    def test_no_advisory_when_speed_compensates(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        run(
            data_dir, "add", str(GUYVIR_WAV),
            "--date", "2021-03-05", "--time", "22:10:00",
            "--latitude", "-26.9", "--longitude", "-48.7",
            "--window-start", "12", "--window-end", "17",
            "--lowpass", "40000", "--speed", "0.25", "--no-embeddings",
        )
        # 40 kHz x 0.25 = 10 kHz, inside both models' range.
        assert "warning" not in capsys.readouterr().out

    def test_show_renders_every_field(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        run(data_dir, "add", str(GUYVIR_WAV), "--no-embeddings")
        capsys.readouterr()
        assert run(data_dir, "show", "1") == 0
        out = capsys.readouterr().out
        for label in (
            "item ID", "biotimbre ID", "taxa", "filtered_band",
            "microphone", "recorded_by", "filter_type", "filter_order",
            "temperature", "humidity",
            "speed", "sample_window", "sha256", "embeddings",
        ):
            assert label in out
        # The open-ended band from the real sidecar round-trips as "2000,-".
        assert "2000,-" in out

    def test_delete_requires_confirmation_or_yes(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        run(data_dir, "add", str(GUYVIR_WAV), "--no-embeddings")
        capsys.readouterr()
        assert run(data_dir, "delete", "1", "--yes") == 0
        assert run(data_dir, "list") == 0
        assert "empty" in capsys.readouterr().out

    def test_stats_on_populated_library(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        run(data_dir, "add", str(GUYVIR_WAV), "--no-embeddings")
        capsys.readouterr()
        assert run(data_dir, "stats") == 0
        out = capsys.readouterr().out
        assert "items       1" in out

    def test_import_dir_reports_failures_and_exits_nonzero(
        self, data_dir, tmp_path, capsys
    ):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        source = tmp_path / "mixed"
        source.mkdir()
        shutil.copy2(GUYVIR_WAV, source / "good.wav")
        shutil.copy2(EXAMPLES_DIR / "GUYVIR.txt", source / "good.txt")
        shutil.copy2(GUYVIR_WAV, source / "orphan.wav")
        capsys.readouterr()

        code = run(data_dir, "import-dir", str(source), "--no-embeddings")
        captured = capsys.readouterr()
        assert code == 1
        assert "Added 1 item(s)." in captured.out
        assert "orphan.wav" in captured.err

    def test_unknown_item_is_a_clean_error(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        assert run(data_dir, "show", "999") == 2
        assert "error:" in capsys.readouterr().err

    def test_invalid_metadata_is_a_clean_error(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test Library")
        code = run(
            data_dir, "add", str(GUYVIR_WAV),
            "--date", "not-a-date", "--time", "09:00:00",
            "--latitude", "-26", "--longitude", "-48",
            "--window-start", "0", "--window-end", "5", "--no-embeddings",
        )
        assert code == 2
        assert "date" in capsys.readouterr().err


@pytest.mark.models
class TestEndToEndCli:
    def test_full_workflow(self, data_dir, tmp_path, capsys):
        assert run(data_dir, "init", "--name", "Test Library") == 0
        assert "1536 dimensions" in capsys.readouterr().out

        assert run(data_dir, "import-dir", str(EXAMPLES_DIR)) == 0
        out = capsys.readouterr().out
        assert "perch_v2       1 window(s)" in out
        assert "birdnet_v2.4   3 window(s)" in out

        assert run(data_dir, "search", str(GUYVIR_WAV), "-k", "2") == 0
        search_out = capsys.readouterr().out
        # The query is itself in the library, so it must top the ranking at ~1.0.
        assert "1. score +1.0000" in search_out
        assert "matched" in search_out
        assert "perch_v2=" in search_out

        assert run(data_dir, "models") == 0
        assert "dimensions    1536" in capsys.readouterr().out

    def test_export_npy_reports_the_path_it_wrote(self, data_dir, tmp_path, capsys):
        run(data_dir, "init", "--name", "Test Library")
        run(data_dir, "add", str(GUYVIR_WAV))
        capsys.readouterr()

        target = tmp_path / "vectors"  # no suffix, does not exist yet
        assert run(data_dir, "export-npy", "1", "perch_v2", str(target)) == 0
        reported = capsys.readouterr().out.strip().removeprefix("Wrote ")

        # The extension is chosen by the contents; the reported path must be
        # the real file either way.
        from pathlib import Path

        assert Path(reported).is_file()
        assert Path(reported).suffix == ".npy"
        # Perch takes the whole sample in one window: one vector, one file.
        assert np.load(reported).shape == (1536,)

    def test_export_npy_zips_a_model_with_several_windows(
        self, data_dir, tmp_path, capsys
    ):
        import zipfile

        run(data_dir, "init", "--name", "Test Library")
        run(data_dir, "add", str(GUYVIR_WAV))
        capsys.readouterr()

        assert run(data_dir, "export-npy", "1", "birdnet_v2.4", str(tmp_path)) == 0
        from pathlib import Path

        reported = Path(capsys.readouterr().out.strip().removeprefix("Wrote "))
        assert reported.name == "item_1_birdnet_v2.4.zip"
        with zipfile.ZipFile(reported) as archive:
            assert len(archive.namelist()) == 3

    def test_speed_shifted_item_found_by_native_query(self, data_dir, capsys):
        run(data_dir, "init", "--name", "Test Library")
        run(
            data_dir, "add", str(GUYVIR_WAV),
            "--date", "2021-03-05", "--time", "22:10:00",
            "--latitude", "-26.9", "--longitude", "-48.7",
            "--window-start", "12", "--window-end", "17",
            "--speed", "0.5",
        )
        capsys.readouterr()

        assert run(data_dir, "search", str(GUYVIR_WAV), "-k", "1") == 0
        out = capsys.readouterr().out
        assert "indexed at speed 0.5" in out
        assert "1. score +1.0000" in out
