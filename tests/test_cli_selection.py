"""Selecting items on the command line: search --csv, export-bundle --items-from."""

from __future__ import annotations

import csv
import zipfile

import pytest

from biotimbre.cli import main
from biotimbre.domain.errors import ValidationError
from biotimbre.io.table import read_item_ids
from conftest import EXAMPLES_DIR


@pytest.fixture
def run(tmp_path, monkeypatch):
    from biotimbre.config import ENV_CONFIG

    monkeypatch.delenv(ENV_CONFIG, raising=False)
    monkeypatch.chdir(tmp_path)

    def invoke(*argv: str) -> int:
        return main(["--data-dir", str(tmp_path / "library"), *argv])

    return invoke


def _add_three(run):
    for name, taxon in (("GUYVIR", "cicadidae"), ("ZAMTYM", "cicadidae"), ("GUYVIR", "orthoptera")):
        assert run("add", str(EXAMPLES_DIR / f"{name}.wav"), "--date", "2020-01-12",
                   "--time", "09:40:21", "--latitude", "-26", "--longitude", "-48",
                   "--taxa", taxon, "--no-embeddings") == 0


class TestReadItemIds:
    def test_reads_the_column_in_order_once_each(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_text("rank,item_id,taxa\n1,7,a\n2,3,b\n3,7,c\n4,,d\n", encoding="utf-8")
        assert read_item_ids(path) == [7, 3]

    def test_a_semicolon_table(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_text("﻿item_id;taxa\n5;a\n", encoding="utf-8")
        assert read_item_ids(path) == [5]

    def test_a_table_without_the_column(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_text("file\nx.wav\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="no item_id column"):
            read_item_ids(path)

    def test_a_value_that_is_not_an_id(self, tmp_path):
        path = tmp_path / "t.csv"
        path.write_text("item_id\n5\nfive\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="line 3: item_id 'five'"):
            read_item_ids(path)


class TestSearchCsv:
    def test_writes_the_results_it_prints(self, run, tmp_path, capsys):
        _add_three(run)
        capsys.readouterr()
        out = tmp_path / "hits.csv"
        assert run("search", "--taxa", "cicadidae", "--csv", str(out)) == 0
        assert "Wrote 2 result(s)" in capsys.readouterr().out
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert sorted(int(row["item_id"]) for row in rows) == [1, 2]

    def test_a_folder_gets_the_default_name(self, run, tmp_path):
        _add_three(run)
        folder = tmp_path / "exports"
        folder.mkdir()
        assert run("search", "--csv", str(folder)) == 0
        assert (folder / "biotimbre_search_results.csv").is_file()


class TestBundleFromCsv:
    def test_the_rows_kept_are_the_items_exported(self, run, tmp_path, capsys):
        _add_three(run)
        hits = tmp_path / "hits.csv"
        run("search", "--csv", str(hits))
        # Checked in a spreadsheet: item 2's row deleted.
        with hits.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, [r for r in reader if r["item_id"] != "2"]
        with hits.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        capsys.readouterr()

        bundle = tmp_path / "checked.btz"
        assert run("export-bundle", "--items-from", str(hits), "-o", str(bundle)) == 0
        assert "2 item(s)" in capsys.readouterr().out
        with zipfile.ZipFile(bundle) as archive:
            audio = [n for n in archive.namelist() if n.endswith("audio.wav")]
        assert len(audio) == 2

    def test_ids_and_a_csv_together(self, run, tmp_path, capsys):
        _add_three(run)
        table = tmp_path / "one.csv"
        table.write_text("item_id\n1\n", encoding="utf-8")
        capsys.readouterr()
        assert run("export-bundle", "3", "1", "--items-from", str(table),
                   "-o", str(tmp_path / "b.btz")) == 0
        assert "2 item(s)" in capsys.readouterr().out

    def test_nothing_to_export_is_an_error(self, run, capsys):
        _add_three(run)
        assert run("export-bundle", "-o", "b.btz") == 2
        assert "No items to export" in capsys.readouterr().err


class TestResultLayout:
    """Search results as blocks: a rule, the header, labelled rows."""

    def test_each_result_is_ruled_off_with_labelled_rows(self, run, tmp_path, capsys):
        _add_three(run)
        capsys.readouterr()
        assert run("search", "--taxa", "orthoptera") == 0
        lines = capsys.readouterr().out.splitlines()
        assert set(lines[0]) == {"─"} and set(lines[-1]) == {"─"}
        assert lines[1].startswith("  1. item 3  biotimbre ")
        labels = [line.split()[0] for line in lines[2:-1]]
        assert labels == ["taxa", "recorded", "location", "audio"]

    def test_the_audio_row_is_the_stored_file(self, run, tmp_path, capsys):
        from pathlib import Path

        _add_three(run)
        capsys.readouterr()
        run("search", "--taxa", "orthoptera")
        audio = next(l for l in capsys.readouterr().out.splitlines() if "audio" in l)
        path = Path(audio.split(None, 1)[1])
        assert path.is_absolute() and path.is_file()

    def test_in_a_terminal_it_is_a_link_to_the_file(self):
        import io
        from pathlib import Path
        from types import SimpleNamespace

        from rich.console import Console

        from biotimbre.cli import _audio_link

        settings = SimpleNamespace(absolute_audio_path=lambda p: Path("/lib") / p)
        item = SimpleNamespace(audio_path="audio/ab/1.wav")
        buffer = io.StringIO()
        # legacy_windows=False: on a Windows runner Rich otherwise takes the
        # console for the classic one, which it drives through the Win32 API
        # and which cannot show links at all, so it leaves them out. Windows
        # Terminal and VS Code are not legacy, and do get them.
        Console(
            file=buffer, force_terminal=True, legacy_windows=False, width=200
        ).print(_audio_link(settings, item))
        # OSC 8: what terminals that support links turn into a clickable one.
        assert "\x1b]8;" in buffer.getvalue()
        assert Path("/lib/audio/ab/1.wav").resolve().as_uri() in buffer.getvalue()
