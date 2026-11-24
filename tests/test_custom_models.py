"""User-defined models: their definition, download, memory and input types.

Nothing here needs a real model file. URL downloads are served by a local HTTP
server, and the ONNX session is replaced by a stand-in wherever a graph would
be needed, so these run offline in milliseconds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.server
import threading
from pathlib import Path

import numpy as np
import pytest

from biotimbre.config import Settings, load_settings
from biotimbre.domain.errors import EmbeddingError, ValidationError
from biotimbre.domain.item import ItemDraft, Location
from biotimbre.embeddings import onnx_backend
from biotimbre.embeddings.onnx_backend import OnnxEmbedder
from biotimbre.models_registry import (
    ModelSpec,
    get_spec,
    resolve_url_file,
    sha256_revision,
    spec_from_row,
    url_cache_path,
)
from biotimbre.service import LibraryService, with_library_models
from conftest import EXAMPLES_DIR

HF_SPEC = ModelSpec(
    model_id="my_hf",
    repo_id="someone/some-onnx",
    filename="onnx/model.onnx",
    revision="abc123",
    sample_rate=16_000,
    window_duration_s=2.0,
)

URL_SPEC = ModelSpec(
    model_id="my_url",
    url="https://example.org/models/encoder%20v1.onnx",
    sample_rate=22_050,
    window_duration_s=1.5,
    window_overlap=0.25,
)


# ── the definition ────────────────────────────────────────────────────────────


class TestModelSpec:
    def test_a_url_names_its_file_after_the_last_path_segment(self):
        assert URL_SPEC.filename == "encoder v1.onnx"
        assert URL_SPEC.source == "https://example.org/models/encoder%20v1.onnx"

    def test_a_repository_is_a_hugging_face_source(self):
        assert HF_SPEC.source == "hf://someone/some-onnx"

    @pytest.mark.parametrize(
        "changes, message",
        [
            ({"repo_id": None}, "exactly one of repo_id or url"),
            ({"url": "https://example.org/x.onnx"}, "exactly one of repo_id or url"),
            ({"model_id": "My Model"}, "Invalid model id"),
            ({"sample_rate": 0}, "sample_rate must be positive"),
            ({"sample_rate": 16000.5}, "whole number"),
            ({"window_duration_s": 0}, "window_duration_s must be positive"),
            ({"window_overlap": 0.99}, "window_overlap"),
            ({"filename": ""}, "needs filename"),
        ],
    )
    def test_an_unusable_definition_is_refused(self, changes, message):
        fields = {
            "model_id": "m",
            "repo_id": "a/b",
            "filename": "m.onnx",
            "sample_rate": 16_000,
            "window_duration_s": 1.0,
            **changes,
        }
        with pytest.raises(ValueError, match=message):
            ModelSpec(**fields)

    def test_a_url_must_be_http(self):
        with pytest.raises(ValueError, match="http"):
            ModelSpec(
                model_id="m",
                url="file:///home/me/model.onnx",
                sample_rate=16_000,
                window_duration_s=1.0,
            )

    def test_a_url_model_is_pinned_only_by_digest(self):
        with pytest.raises(ValueError, match="sha256"):
            dataclasses.replace(URL_SPEC, revision="main")

    @pytest.mark.parametrize("spec", [HF_SPEC, URL_SPEC], ids=["hf", "url"])
    def test_a_registered_row_gives_the_same_spec_back(self, spec):
        row = {
            "model_id": spec.model_id,
            "source": spec.source,
            "filename": spec.filename,
            "revision": spec.revision,
            "sample_rate": spec.sample_rate,
            "window_duration_s": spec.window_duration_s,
            "window_overlap": spec.window_overlap,
        }
        assert spec_from_row(row).recipe == spec.recipe

    def test_an_unknown_id_names_where_to_define_one(self):
        with pytest.raises(KeyError, match=r"\[models\.custom"):
            get_spec("nope")


# ── the config file ───────────────────────────────────────────────────────────


def _settings(tmp_path: Path, toml: str) -> Settings:
    config = tmp_path / "biotimbre.toml"
    config.write_text(toml, encoding="utf-8")
    return load_settings(config)


HF_TOML = """
[models.custom.my_hf]
repo_id = "someone/some-onnx"
filename = "onnx/model.onnx"
revision = "abc123"
sample_rate = 16000
window_duration_s = 2.0
"""


class TestConfig:
    def test_custom_models_are_read(self, tmp_path):
        settings = _settings(tmp_path, HF_TOML)
        assert settings.custom_models["my_hf"].recipe == HF_SPEC.recipe
        assert settings.model_spec("my_hf") == settings.custom_models["my_hf"]

    def test_by_default_a_custom_model_is_active(self, tmp_path):
        settings = _settings(tmp_path, HF_TOML)
        assert settings.active_models == ("perch_v2", "birdnet_v2.4", "my_hf")
        assert not settings.active_models_chosen

    def test_an_explicit_active_list_is_used_as_given(self, tmp_path):
        settings = _settings(tmp_path, '[models]\nactive = ["perch_v2"]\n' + HF_TOML)
        assert settings.active_models == ("perch_v2",)
        assert settings.active_models_chosen

    def test_a_url_model_with_a_digest(self, tmp_path):
        digest = "AB" * 32
        settings = _settings(
            tmp_path,
            f"""
[models.custom.my_url]
url = "https://example.org/m.onnx"
sha256 = "{digest}"
sample_rate = 32000
window_duration_s = 5
""",
        )
        spec = settings.custom_models["my_url"]
        assert spec.revision == "sha256:" + digest.lower()
        assert spec.filename == "m.onnx"

    @pytest.mark.parametrize(
        "table, message",
        [
            ('path = "model.onnx"\nsample_rate = 1\nwindow_duration_s = 1',
             "local model files are not supported"),
            ('repo_id = "a/b"\nfilename = "m.onnx"\nwindow_duration_s = 1',
             "needs sample_rate"),
            ('repo_id = "a/b"\nfilename = "m.onnx"\nsample_rate = 1\n'
             'window_duration_s = 1\noutput = "x"', "Unknown"),
            ('url = "https://e.org/m.onnx"\nrevision = "main"\nsample_rate = 1\n'
             'window_duration_s = 1', "pinned with sha256"),
            ('repo_id = "a/b"\nfilename = "m.onnx"\nsha256 = "00"\nsample_rate = 1\n'
             'window_duration_s = 1', "pinned with revision"),
            ('url = "https://e.org/m.onnx"\nsha256 = "xyz"\nsample_rate = 1\n'
             'window_duration_s = 1', "64 hexadecimal"),
        ],
    )
    def test_an_unusable_table_is_refused(self, tmp_path, table, message):
        with pytest.raises(ValueError, match=message):
            _settings(tmp_path, f"[models.custom.mine]\n{table}\n")

    def test_a_built_in_id_cannot_be_redefined(self, tmp_path):
        with pytest.raises(ValueError, match="built-in"):
            _settings(
                tmp_path,
                '[models.custom.perch_v2]\nrepo_id = "a/b"\nfilename = "m.onnx"\n'
                "sample_rate = 1\nwindow_duration_s = 1\n",
            )


# ── downloading from a URL ────────────────────────────────────────────────────


@pytest.fixture
def served(tmp_path):
    """Serve files over local HTTP, counting the requests made."""
    root = tmp_path / "www"
    root.mkdir()
    requests: list[str] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self):
            requests.append(self.path)
            super().do_GET()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield root, base, requests
    server.shutdown()
    server.server_close()


def _url_spec(base: str, **changes) -> ModelSpec:
    return ModelSpec(
        model_id="my_url",
        url=f"{base}/model.onnx",
        sample_rate=16_000,
        window_duration_s=1.0,
        **changes,
    )


class TestUrlDownload:
    def test_downloads_once_and_records_the_digest(self, served, tmp_path):
        root, base, requests = served
        (root / "model.onnx").write_bytes(b"graph bytes")
        spec = _url_spec(base)
        cache = tmp_path / "cache"

        path, digest = resolve_url_file(spec, cache)
        assert path.read_bytes() == b"graph bytes"
        assert digest == "sha256:" + hashlib.sha256(b"graph bytes").hexdigest()
        assert path == url_cache_path(spec, cache)

        # Cache first: the second resolution does not reach the server.
        assert resolve_url_file(spec, cache) == (path, digest)
        assert len(requests) == 1

    def test_a_download_that_differs_from_the_pin_is_discarded(
        self, served, tmp_path
    ):
        root, base, _ = served
        (root / "model.onnx").write_bytes(b"someone replaced this")
        spec = _url_spec(base, revision=sha256_revision("0" * 64))

        with pytest.raises(RuntimeError, match="has changed"):
            resolve_url_file(spec, tmp_path)
        assert not url_cache_path(spec, tmp_path).exists()

    def test_a_cached_file_that_differs_from_the_pin_is_fetched_again(
        self, served, tmp_path
    ):
        root, base, requests = served
        (root / "model.onnx").write_bytes(b"old")
        resolve_url_file(_url_spec(base), tmp_path)

        (root / "model.onnx").write_bytes(b"new")
        pinned = _url_spec(
            base, revision=sha256_revision(hashlib.sha256(b"new").hexdigest())
        )
        path, _ = resolve_url_file(pinned, tmp_path)
        assert path.read_bytes() == b"new"
        assert len(requests) == 2

    def test_an_unreachable_url_names_the_model(self, served, tmp_path):
        _, base, _ = served
        with pytest.raises(RuntimeError, match="my_url"):
            resolve_url_file(_url_spec(base), tmp_path)  # 404: nothing served


# ── the library remembers ─────────────────────────────────────────────────────


class _FakeExtractor:
    def __init__(
        self, spec: ModelSpec, dim: int = 8, outputs=("embedding",), shape=None
    ):
        self.spec = spec
        self._dim = dim
        self.output_names = tuple(outputs)
        self._shape = tuple(shape) if shape is not None else (1, dim)

    def output_shape(self) -> tuple[int, ...]:
        return self._shape

    def embedding_dim(self) -> int:
        return self._dim


class _FakePool:
    """Serves specs with no graph behind them."""

    def __init__(
        self, *specs: ModelSpec, outputs=("embedding",), dim: int = 8, shape=None
    ):
        self._extractors = {
            spec.model_id: _FakeExtractor(spec, dim, outputs=outputs, shape=shape)
            for spec in specs
        }

    def get(self, model_id: str) -> _FakeExtractor:
        return self._extractors[model_id]

    def close(self) -> None:
        pass


def _library(tmp_path: Path, **settings) -> Settings:
    data_dir = tmp_path / "library"
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "library.db",
        audio_dir=data_dir / "audio",
        **settings,
    )


def _register(settings: Settings, spec: ModelSpec, *, with_vectors: bool) -> None:
    """Register ``spec`` in the library, optionally storing a vector under it."""
    with LibraryService(settings, pool=_FakePool(spec)) as service:
        service.ensure_models_registered([spec.model_id])
        if with_vectors:
            item = service.add_item(
                EXAMPLES_DIR / "GUYVIR.wav",
                ItemDraft(
                    date="2020-01-12",
                    time="09:40:21",
                    location=Location(-26.0, -48.0),
                ),
                compute_embeddings=False,
            ).item
            with service._transaction():
                service.repo.set_embeddings(
                    item.item_id,
                    spec.model_id,
                    [(0.0, 1.0, np.ones(8, dtype=np.float32))],
                )


class TestLibraryMemory:
    def test_a_registered_model_works_without_the_config(self, tmp_path):
        configured = _library(tmp_path, custom_models={"my_hf": HF_SPEC})
        _register(configured, HF_SPEC, with_vectors=False)

        # Opened again with no definition of it anywhere.
        with LibraryService(_library(tmp_path), pool=_FakePool()) as service:
            assert service.settings.model_spec("my_hf").recipe == HF_SPEC.recipe
            assert "my_hf" in service.settings.active_models

    def test_an_explicit_active_list_leaves_it_out(self, tmp_path):
        _register(
            _library(tmp_path, custom_models={"my_hf": HF_SPEC}),
            HF_SPEC,
            with_vectors=False,
        )
        settings = _library(
            tmp_path, active_models=("perch_v2",), active_models_chosen=True
        )
        with LibraryService(settings, pool=_FakePool()) as service:
            assert service.settings.active_models == ("perch_v2",)
            # Still known, for exporting the vectors it already made.
            assert service.settings.model_spec("my_hf").recipe == HF_SPEC.recipe

    def test_an_unpinned_url_keeps_the_digest_the_library_recorded(self):
        pinned = dataclasses.replace(URL_SPEC, revision=sha256_revision("ab" * 32))
        row = {
            "model_id": "my_url",
            "source": pinned.source,
            "filename": pinned.filename,
            "revision": pinned.revision,
            "sample_rate": pinned.sample_rate,
            "window_duration_s": pinned.window_duration_s,
            "window_overlap": pinned.window_overlap,
        }
        settings = Settings(
            data_dir=Path("x"),
            db_path=Path("x/library.db"),
            audio_dir=Path("x/audio"),
            custom_models={"my_url": URL_SPEC},
        )
        merged = with_library_models(settings, [row])
        assert merged.custom_models["my_url"].revision == pinned.revision

    def test_a_changed_definition_is_refused_once_vectors_exist(self, tmp_path):
        settings = _library(tmp_path, custom_models={"my_hf": HF_SPEC})
        _register(settings, HF_SPEC, with_vectors=True)

        changed = dataclasses.replace(HF_SPEC, window_duration_s=3.0)
        with LibraryService(
            _library(tmp_path, custom_models={"my_hf": changed}),
            pool=_FakePool(changed),
        ) as service:
            with pytest.raises(ValidationError, match="window_duration_s"):
                service.ensure_models_registered(["my_hf"])

    def test_without_vectors_a_definition_may_change(self, tmp_path):
        _register(
            _library(tmp_path, custom_models={"my_hf": HF_SPEC}),
            HF_SPEC,
            with_vectors=False,
        )
        changed = dataclasses.replace(HF_SPEC, window_duration_s=3.0)
        settings = _library(tmp_path, custom_models={"my_hf": changed})
        with LibraryService(settings, pool=_FakePool(changed)) as service:
            service.ensure_models_registered(["my_hf"])
            assert service.repo.get_model("my_hf")["window_duration_s"] == 3.0

    def test_a_model_with_several_outputs_is_noted_once(self, tmp_path):
        settings = _library(tmp_path, custom_models={"my_hf": HF_SPEC})
        pool = _FakePool(HF_SPEC, outputs=("logits", "embedding"))
        with LibraryService(settings, pool=pool) as service:
            service.ensure_models_registered(["my_hf"])
            (notice,) = service.pop_notices()
            assert "2 outputs (logits, embedding)" in notice
            assert "'logits'" in notice
            # Already registered: not said again on every later command.
            service.ensure_models_registered(["my_hf"])
            assert service.pop_notices() == []

    def test_a_very_wide_output_is_noted_as_possible_class_scores(self, tmp_path):
        settings = _library(tmp_path, custom_models={"my_hf": HF_SPEC})
        pool = _FakePool(HF_SPEC, dim=6522)
        with LibraryService(settings, pool=pool) as service:
            service.ensure_models_registered(["my_hf"])
            (notice,) = service.pop_notices()
            assert "6522 values per window" in notice
            assert "class scores" in notice

    def test_an_ordinary_width_is_not_noted(self, tmp_path):
        from biotimbre.service import SUSPICIOUS_EMBEDDING_DIM

        settings = _library(tmp_path, custom_models={"my_hf": HF_SPEC})
        pool = _FakePool(HF_SPEC, dim=SUSPICIOUS_EMBEDDING_DIM)
        with LibraryService(settings, pool=pool) as service:
            service.ensure_models_registered(["my_hf"])
            assert service.pop_notices() == []

    @pytest.mark.parametrize(
        "shape, averaged",
        [((1, 1280, 5, 3), "over 1280 x 5 into a vector of the last axis, 3"),
         ((1, 10, 512), "over 10 into a vector of the last axis, 512")],
    )
    def test_an_output_that_is_not_a_vector_is_noted(
        self, tmp_path, shape, averaged
    ):
        settings = _library(tmp_path, custom_models={"my_hf": HF_SPEC})
        pool = _FakePool(HF_SPEC, dim=shape[-1], shape=shape)
        with LibraryService(settings, pool=pool) as service:
            service.ensure_models_registered(["my_hf"])
            (notice,) = service.pop_notices()
            assert "array per window rather than one vector" in notice
            assert averaged in notice

    def test_unit_axes_alone_are_not_noted(self, tmp_path):
        settings = _library(tmp_path, custom_models={"my_hf": HF_SPEC})
        pool = _FakePool(HF_SPEC, shape=(1, 1, 1, 8))
        with LibraryService(settings, pool=pool) as service:
            service.ensure_models_registered(["my_hf"])
            assert service.pop_notices() == []

    def test_a_single_output_model_is_not_noted(self, tmp_path):
        settings = _library(tmp_path, custom_models={"my_hf": HF_SPEC})
        with LibraryService(settings, pool=_FakePool(HF_SPEC)) as service:
            service.ensure_models_registered(["my_hf"])
            assert service.pop_notices() == []

    def test_the_cli_prints_the_note(self, tmp_path, monkeypatch, capsys):
        from biotimbre import cli

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("BIOTIMBRE_CONFIG", raising=False)
        (tmp_path / "biotimbre.toml").write_text(
            HF_TOML + '[models]\nactive = ["my_hf"]\n', encoding="utf-8"
        )
        original = cli.LibraryService
        monkeypatch.setattr(
            cli,
            "LibraryService",
            lambda settings: original(
                settings,
                pool=_FakePool(
                    settings.model_spec("my_hf"), outputs=("logits", "embedding")
                ),
            ),
        )
        assert cli.main(["--data-dir", str(tmp_path / "lib"), "init", "--name", "x"]) == 0
        err = capsys.readouterr().err
        assert "warning: model 'my_hf' has 2 outputs" in err

    def test_an_unknown_active_model_is_a_clean_error(self, tmp_path):
        settings = _library(
            tmp_path, active_models=("nope",), active_models_chosen=True
        )
        with LibraryService(settings) as service:
            with pytest.raises(ValidationError, match="Unknown model id 'nope'"):
                service.pool


# ── input types ───────────────────────────────────────────────────────────────


class _Node:
    def __init__(self, name, type_, shape):
        self.name, self.type, self.shape = name, type_, shape


class _FakeSession:
    """An inference session whose input type is whatever the test says."""

    def __init__(self, input_type: str, output: np.ndarray | None = None):
        self._input_type = input_type
        self._output = output
        self.fed: list[np.ndarray] = []

    def get_inputs(self):
        return [_Node("audio", self._input_type, ["batch", 16])]

    def get_outputs(self):
        return [_Node("embedding", "tensor(float16)", ["batch", 4])]

    def run(self, names, feeds):
        batch = feeds["audio"]
        self.fed.append(batch)
        if self._output is not None:
            return [self._output]
        return [np.ones((batch.shape[0], 4), dtype=np.float16)]


def _embedder(monkeypatch, session: _FakeSession) -> OnnxEmbedder:
    import onnxruntime

    monkeypatch.setattr(
        onnxruntime, "InferenceSession", lambda *args, **kwargs: session
    )
    monkeypatch.setattr(onnx_backend, "onnxruntime_problem", lambda: None)
    return OnnxEmbedder("model.onnx", model_id="fp16_model")


class TestInputType:
    @pytest.mark.parametrize(
        "declared, fed",
        [
            ("tensor(float16)", np.float16),
            ("tensor(float)", np.float32),
            ("tensor(double)", np.float64),
        ],
    )
    def test_audio_is_fed_as_the_graph_declares(self, monkeypatch, declared, fed):
        session = _FakeSession(declared)
        embedder = _embedder(monkeypatch, session)
        vectors = embedder.run(np.zeros((2, 16), dtype=np.float32))
        assert session.fed[0].dtype == fed
        # Whatever went in, float32 comes out: it is what the library stores.
        assert vectors.dtype == np.float32
        assert vectors.shape == (2, 4)

    def test_the_raw_output_shape_is_measured(self, monkeypatch):
        spatial = np.ones((1, 1280, 5, 3), dtype=np.float32)
        embedder = _embedder(monkeypatch, _FakeSession("tensor(float)", spatial))
        assert embedder.output_shape(16) == (1, 1280, 5, 3)
        # What the note warns about: the last axis is kept as the features.
        assert embedder.run(np.zeros((1, 16), dtype=np.float32)).shape == (1, 3)

    def test_an_input_that_is_not_audio_is_refused(self, monkeypatch):
        embedder = _embedder(monkeypatch, _FakeSession("tensor(int64)"))
        with pytest.raises(EmbeddingError, match="tensor\\(int64\\)"):
            embedder.run(np.zeros((1, 16), dtype=np.float32))

    def test_an_overflowed_output_is_refused(self, monkeypatch):
        overflow = np.array([[np.inf, 1, 1, 1]], dtype=np.float16)
        embedder = _embedder(monkeypatch, _FakeSession("tensor(float16)", overflow))
        with pytest.raises(EmbeddingError, match="non-finite"):
            embedder.run(np.zeros((1, 16), dtype=np.float32))
