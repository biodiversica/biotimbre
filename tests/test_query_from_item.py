"""Searching with a library item's own recording: search --item."""

from __future__ import annotations

import pytest

from biotimbre.cli import main
from conftest import EXAMPLES_DIR


@pytest.fixture
def run(tmp_path, monkeypatch):
    from biotimbre.config import ENV_CONFIG

    monkeypatch.delenv(ENV_CONFIG, raising=False)
    monkeypatch.chdir(tmp_path)

    def invoke(*argv: str) -> int:
        return main(["--data-dir", str(tmp_path / "library"), *argv])

    return invoke


def _add(run, name: str, *extra: str) -> None:
    assert run("add", str(EXAMPLES_DIR / f"{name}.wav"), "--date", "2020-01-12",
               "--time", "09:40:21", "--latitude", "-26", "--longitude", "-48",
               *extra) == 0


class TestArguments:
    @pytest.mark.parametrize(
        "argv, message",
        [
            (["search", "q.wav", "--item", "1"], "Give one query: AUDIO or --item"),
            (["search", "--inat", "5", "--item", "1"], "--inat or --item"),
            (["search", "--include-self"], "--include-self needs --item"),
            (["search", "--item", "1", "--speed", "1"], "--speed is for a search"),
        ],
    )
    def test_conflicts_are_refused(self, run, capsys, argv, message):
        assert run(*argv) == 2
        assert message in capsys.readouterr().err

    def test_an_unknown_item(self, run, capsys):
        _add(run, "GUYVIR", "--no-embeddings")
        assert run("search", "--item", "99") == 2
        assert "No item with id 99" in capsys.readouterr().err


@pytest.mark.models
class TestSearch:
    def test_the_item_finds_the_rest_but_not_itself(self, run, capsys):
        _add(run, "GUYVIR")
        _add(run, "ZAMTYM")
        capsys.readouterr()
        assert run("search", "--item", "1") == 0
        out = capsys.readouterr().out
        assert out.startswith("Query: item 1 (unknown, 2020-01-12 09:40:21)")
        assert "item 2" in out
        assert "item 1 " not in out.split("\n", 1)[1]

    def test_include_self_puts_it_first(self, run, capsys):
        _add(run, "GUYVIR")
        _add(run, "ZAMTYM")
        capsys.readouterr()
        assert run("search", "--item", "1", "--include-self") == 0
        first_hit = capsys.readouterr().out.split("\n")[1]
        assert "score +1.0000  item 1" in first_hit
