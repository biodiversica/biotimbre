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


class TestBundleCommands:
    """Sharing between libraries, from the command line.

    Two data directories, as two people would have. No models: the metadata,
    the audio and the decisions are all exercisable without them.
    """

    @pytest.fixture
    def two_libraries(self, tmp_path):
        """A sender and a receiver, both named and both empty."""
        sender = str(tmp_path / "sender")
        receiver = str(tmp_path / "receiver")
        assert run(sender, "init", "--skip-models", "--name", "Sender") == 0
        assert run(receiver, "init", "--skip-models", "--name", "Receiver") == 0
        return sender, receiver

    def add_item(
        self, data_dir: str, audio=GUYVIR_WAV, *, taxa: str | None = None
    ) -> None:
        """Add one item from explicit flags, so taxa can be chosen per test."""
        argv = [
            "add", str(audio),
            "--date", "2020-01-12", "--time", "09:40:21",
            "--latitude", "-26", "--longitude", "-48",
            "--no-embeddings",
        ]
        if taxa:
            argv += ["--taxa", taxa]
        assert run(data_dir, *argv) == 0

    def test_export_writes_a_file(self, two_libraries, tmp_path, capsys):
        sender, _ = two_libraries
        self.add_item(sender)
        out = tmp_path / "share"
        assert run(sender, "export-bundle", "1", "-o", str(out)) == 0

        assert out.with_suffix(".btz").is_file()
        assert "1 item(s)" in capsys.readouterr().out

    def test_export_needs_an_output(self, two_libraries):
        sender, _ = two_libraries
        with pytest.raises(SystemExit):
            build_parser().parse_args(["export-bundle", "1"])

    def test_export_takes_several_items(self, two_libraries, tmp_path, capsys):
        sender, _ = two_libraries
        self.add_item(sender)
        self.add_item(sender, EXAMPLES_DIR / "ZAMTYM.wav")
        out = tmp_path / "share.btz"
        assert run(sender, "export-bundle", "1", "2", "-o", str(out)) == 0
        assert "2 item(s)" in capsys.readouterr().out

    def test_export_reports_an_unknown_item(self, two_libraries, tmp_path, capsys):
        sender, _ = two_libraries
        assert run(sender, "export-bundle", "99", "-o", str(tmp_path / "x.btz")) == 2
        assert "error:" in capsys.readouterr().err

    def test_the_round_trip(self, two_libraries, tmp_path, capsys):
        sender, receiver = two_libraries
        self.add_item(sender, taxa="cicadidae")
        out = tmp_path / "share.btz"
        run(sender, "export-bundle", "1", "-o", str(out))
        capsys.readouterr()

        assert run(receiver, "import-bundle", str(out), "--no-embeddings") == 0
        printed = capsys.readouterr().out
        assert "Bundle from Sender" in printed
        assert "added item 1" in printed
        assert "Added 1, skipped 0." in printed

        # And it really is there.
        assert run(receiver, "list") == 0
        assert "cicadidae" in capsys.readouterr().out

    def test_a_dry_run_changes_nothing(self, two_libraries, tmp_path, capsys):
        sender, receiver = two_libraries
        self.add_item(sender)
        out = tmp_path / "share.btz"
        run(sender, "export-bundle", "1", "-o", str(out))
        capsys.readouterr()

        assert run(receiver, "import-bundle", str(out), "--dry-run") == 0
        printed = capsys.readouterr().out
        assert "Would add 1" in printed
        assert "Nothing was changed." in printed

        assert run(receiver, "list") == 0
        assert "Library is empty." in capsys.readouterr().out

    def test_a_second_import_skips(self, two_libraries, tmp_path, capsys):
        sender, receiver = two_libraries
        self.add_item(sender)
        out = tmp_path / "share.btz"
        run(sender, "export-bundle", "1", "-o", str(out))
        run(receiver, "import-bundle", str(out), "--no-embeddings")
        capsys.readouterr()

        assert run(receiver, "import-bundle", str(out), "--no-embeddings") == 0
        printed = capsys.readouterr().out
        assert "Added 0, skipped 1." in printed
        assert "already in this library" in printed

    def test_a_species_conflict_is_explained(self, two_libraries, tmp_path, capsys):
        sender, receiver = two_libraries
        self.add_item(sender, taxa="guyalna bonaerensis")
        out = tmp_path / "share.btz"
        run(sender, "export-bundle", "1", "-o", str(out))
        self.add_item(
            receiver, EXAMPLES_DIR / "ZAMTYM.wav", taxa="guyalna bonaerensis"
        )
        capsys.readouterr()

        assert run(receiver, "import-bundle", str(out), "--no-embeddings") == 0
        printed = capsys.readouterr().out
        assert "warning:" in printed
        assert "already has a biotimbre here" in printed

    def test_the_sender_biotimbre_can_be_kept(self, two_libraries, tmp_path, capsys):
        sender, receiver = two_libraries
        self.add_item(sender, taxa="guyalna bonaerensis")
        out = tmp_path / "share.btz"
        run(sender, "export-bundle", "1", "-o", str(out))
        self.add_item(
            receiver, EXAMPLES_DIR / "ZAMTYM.wav", taxa="guyalna bonaerensis"
        )
        capsys.readouterr()

        assert run(receiver, "import-bundle", str(out), "--no-embeddings",
                   "--species", "keep-incoming") == 0
        assert "two biotimbres" in capsys.readouterr().out

    def test_a_duplicate_can_be_forced_in(self, two_libraries, tmp_path, capsys):
        sender, receiver = two_libraries
        self.add_item(sender)
        out = tmp_path / "share.btz"
        run(sender, "export-bundle", "1", "-o", str(out))
        self.add_item(receiver)
        capsys.readouterr()

        assert run(receiver, "import-bundle", str(out), "--no-embeddings",
                   "--duplicates", "add") == 0
        assert "Added 1" in capsys.readouterr().out

    def test_a_bad_bundle_is_reported(self, two_libraries, tmp_path, capsys):
        _, receiver = two_libraries
        broken = tmp_path / "broken.btz"
        broken.write_bytes(b"not a zip")
        assert run(receiver, "import-bundle", str(broken)) == 2
        assert "error:" in capsys.readouterr().err

    def test_import_dir_points_at_the_bundle_command(
        self, two_libraries, tmp_path, capsys
    ):
        sender, receiver = two_libraries
        self.add_item(sender)
        out = tmp_path / "share.btz"
        run(sender, "export-bundle", "1", "-o", str(out))
        capsys.readouterr()

        # A bundle is one file, not a directory of them.
        assert run(receiver, "import-dir", str(out)) == 2
        assert "import-bundle" in capsys.readouterr().err

    def test_species_choices_are_constrained(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["import-bundle", "x.btz", "--species", "whatever"]
            )

    def test_policy_flags_reach_the_policy(self):
        args = build_parser().parse_args(
            ["import-bundle", "x.btz", "--distrust-embeddings",
             "--duplicates", "add", "--species", "keep-incoming"]
        )
        assert args.distrust_embeddings is True
        assert args.duplicates == "add"
        assert args.species == "keep-incoming"


class TestSidecarAddDefersEmbedding:
    """`add --sidecar --no-embeddings` used to embed anyway.

    ``cmd_add`` takes a different branch for a sidecar, and that branch called
    ``add_from_sidecar``, which had no way to defer. The flag was accepted and
    silently ignored, so a metadata-only import ran inference on every file.
    """

    def test_the_flag_is_honoured(self, data_dir, capsys):
        assert run(data_dir, "init", "--skip-models", "--name", "Test") == 0
        assert run(
            data_dir,
            "add", str(GUYVIR_WAV),
            "--sidecar", str(EXAMPLES_DIR / "GUYVIR.txt"),
            "--no-embeddings",
        ) == 0
        capsys.readouterr()

        assert run(data_dir, "show", "1") == 0
        assert "embeddings" in capsys.readouterr().out.replace("\n", " ")

    def test_nothing_was_embedded(self, data_dir, capsys):
        run(data_dir, "init", "--skip-models", "--name", "Test")
        run(
            data_dir,
            "add", str(GUYVIR_WAV),
            "--sidecar", str(EXAMPLES_DIR / "GUYVIR.txt"),
            "--no-embeddings",
        )
        capsys.readouterr()
        run(data_dir, "show", "1")
        line = [
            row
            for row in capsys.readouterr().out.splitlines()
            if row.startswith("embeddings")
        ]
        assert line and line[0].split()[-1] == "none"


class TestGuiMapFlag:
    """``gui --no-map``: refusing the map for one run, without a config file."""

    def test_the_flag_defaults_to_keeping_the_map(self):
        args = build_parser().parse_args(["gui"])
        assert args.no_map is False

    def test_the_flag_is_accepted(self):
        assert build_parser().parse_args(["gui", "--no-map"]).no_map is True

    def test_it_reaches_the_interface_as_a_setting(self, data_dir, monkeypatch):
        seen = {}

        def fake_run(settings, **kwargs):
            seen["settings"] = settings

        import biotimbre.ui.app as ui_app

        monkeypatch.setattr(ui_app, "run", fake_run)
        assert main(["--data-dir", data_dir, "gui", "--no-map"]) == 0
        assert seen["settings"].map_enabled is False

    def test_without_the_flag_the_map_stays_on(self, data_dir, monkeypatch):
        seen = {}

        def fake_run(settings, **kwargs):
            seen["settings"] = settings

        import biotimbre.ui.app as ui_app

        monkeypatch.setattr(ui_app, "run", fake_run)
        assert main(["--data-dir", data_dir, "gui"]) == 0
        assert seen["settings"].map_enabled is True
