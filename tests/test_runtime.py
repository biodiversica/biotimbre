"""The onnxruntime load check: a crash in a child process instead of this one."""

from __future__ import annotations

import subprocess

import pytest

from biotimbre.domain.errors import EmbeddingError
from biotimbre.embeddings import runtime
from biotimbre.embeddings.onnx_backend import OnnxEmbedder
from biotimbre.embeddings.runtime import VC_REDIST_URL, onnxruntime_problem


@pytest.fixture(autouse=True)
def _fresh_check():
    onnxruntime_problem.cache_clear()
    yield
    onnxruntime_problem.cache_clear()


def _on_windows(monkeypatch, outcome):
    """Pretend to be on Windows, with the child process doing ``outcome``."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(cmd, outcome, "", "")

    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    return calls


class TestCheck:
    def test_not_asked_off_windows(self, monkeypatch):
        monkeypatch.setattr(runtime.sys, "platform", "linux")

        def refuse(*args, **kwargs):
            raise AssertionError("no child process off Windows")

        monkeypatch.setattr(runtime.subprocess, "run", refuse)
        assert onnxruntime_problem() is None

    def test_a_clean_import_is_no_problem(self, monkeypatch):
        calls = _on_windows(monkeypatch, 0)
        assert onnxruntime_problem() is None
        assert calls and calls[0][1:] == ["-c", "import onnxruntime"]

    def test_an_access_violation_names_the_redistributable(self, monkeypatch):
        # How Windows reports 0xC0000005 as a signed exit code.
        _on_windows(monkeypatch, -1073741819)
        problem = onnxruntime_problem()
        assert "0xC0000005" in problem
        assert VC_REDIST_URL in problem

    def test_an_import_error_is_a_problem_too(self, monkeypatch):
        _on_windows(monkeypatch, 1)
        assert "exits with 1" in onnxruntime_problem()

    @pytest.mark.parametrize(
        "failure", [OSError("no python"), subprocess.TimeoutExpired("python", 1)]
    )
    def test_a_check_that_cannot_run_leaves_it_to_the_import(
        self, monkeypatch, failure
    ):
        _on_windows(monkeypatch, failure)
        assert onnxruntime_problem() is None

    def test_asked_once_per_process(self, monkeypatch):
        calls = _on_windows(monkeypatch, 0)
        onnxruntime_problem()
        onnxruntime_problem()
        assert len(calls) == 1


def test_the_backend_refuses_before_importing(monkeypatch, tmp_path):
    _on_windows(monkeypatch, -1073741819)
    embedder = OnnxEmbedder(tmp_path / "model.onnx", model_id="perch_v2")
    with pytest.raises(EmbeddingError, match="Visual C\\+\\+ Redistributable"):
        embedder.embedding_dim(160_000)


def test_gui_warns_on_the_console_and_still_starts(
    monkeypatch, tmp_path, capsys
):
    import biotimbre.ui.app as ui_app
    from biotimbre.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BIOTIMBRE_CONFIG", raising=False)
    _on_windows(monkeypatch, -1073741819)
    started = []
    monkeypatch.setattr(ui_app, "run", lambda settings, **kw: started.append(1))

    assert main(["gui"]) == 0
    assert started
    assert "warning: onnxruntime" in capsys.readouterr().err
