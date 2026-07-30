"""Application state and media routes.

Exercises the server side of the interface without a browser: state lifecycle,
thread-safe access, and the HTTP routes the pages depend on.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from biotimbre.ui import routes  # noqa: E402
from biotimbre.ui.state import AppState, get_state, set_state  # noqa: E402
from conftest import EXAMPLES_DIR  # noqa: E402


@pytest.fixture
def state(settings, shared_pool):
    """App state over an empty temporary library, sharing the ONNX sessions."""
    app_state = AppState(settings)
    # Swap in the session-scoped pool so tests do not each pay ~2 s of session
    # setup; the service was built with the state's own pool.
    app_state.pool = shared_pool
    app_state.service._pool = shared_pool
    app_state.service._owns_pool = False

    previous = None
    try:
        previous = get_state()
    except RuntimeError:
        pass
    set_state(app_state)
    yield app_state
    app_state.close()
    if previous is not None:
        set_state(previous)


@pytest.fixture
def client(state):
    """Test client over just the media routes."""
    app = FastAPI()
    routes.register(app)
    return TestClient(app)


@pytest.fixture
def populated(state):
    """State with both examples ingested, embeddings included."""
    state.warm_models()
    state.call(state.service.import_directory, EXAMPLES_DIR)
    return state


class TestStateLifecycle:
    def test_opens_a_thread_capable_connection(self, state):
        # A web handler and the inference worker do not share a thread; sqlite3
        # would otherwise raise when the second one touches the connection.
        result: list = []

        def touch():
            result.append(state.call(state.service.repo.count_items))

        worker = threading.Thread(target=touch)
        worker.start()
        worker.join()
        assert result == [0]

    def test_models_start_not_ready(self, state):
        assert state.models.ready is False
        assert state.models.failed is None

    def test_serialises_concurrent_access(self, state):
        # Ten threads hammering one SQLite connection must not corrupt or raise.
        errors: list[Exception] = []

        def work():
            try:
                for _ in range(20):
                    state.call(state.service.repo.count_items)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=work) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []

    def test_run_offloads_to_a_thread(self, state):
        import asyncio

        # Inference takes seconds; on the event loop it would freeze every page.
        calling_thread = threading.current_thread().name
        worker_thread = asyncio.run(
            state.run(lambda: threading.current_thread().name)
        )
        assert worker_thread != calling_thread

    def test_urls_are_item_scoped(self, state):
        # A request can only ever name an item id, never a path.
        assert state.audio_url(7) == "/media/audio/7"
        assert "spectrogram/7" in state.spectrogram_url(7)

    def test_spectrogram_url_carries_options(self, state):
        url = state.spectrogram_url(3, width=450, height=130, start=1.0, end=4.0)
        assert "w=450" in url and "h=130" in url
        assert "start=1.0" in url and "end=4.0" in url

    def test_get_state_before_init_is_a_clear_error(self):
        import biotimbre.ui.state as module

        saved = module._state
        module._state = None
        try:
            with pytest.raises(RuntimeError, match="not initialised"):
                get_state()
        finally:
            module._state = saved


@pytest.mark.models
class TestModelWarmUp:
    def test_warming_reports_dimensions(self, state):
        state.warm_models()
        assert state.models.ready is True
        assert state.models.dimensions == {"perch_v2": 1536, "birdnet_v2.4": 1024}

    def test_warming_twice_is_a_no_op(self, state):
        state.warm_models()
        state.warm_models()
        assert state.models.ready is True

    def test_failure_is_recorded_not_raised(self, state, monkeypatch):
        # A model that will not load must leave metadata browsing usable rather
        # than taking the whole interface down.
        def explode():
            raise RuntimeError("no model for you")

        monkeypatch.setattr(state.service, "ensure_models_registered", explode)
        state.warm_models()
        assert state.models.ready is False
        assert "no model for you" in state.models.failed

    def test_background_warming_completes(self, state):
        thread = state.start_warming()
        thread.join(timeout=180)
        assert not thread.is_alive()
        assert state.models.ready is True


class TestStatusRoute:
    def test_reports_readiness_and_item_count(self, client):
        payload = client.get("/media/status").json()
        assert payload["ready"] is False
        assert payload["items"] == 0


@pytest.mark.models
class TestMediaRoutes:
    def test_serves_stored_audio_bytes(self, client, populated):
        response = client.get("/media/audio/1")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        # Byte-for-byte what was stored (D1b).
        assert response.content == (EXAMPLES_DIR / "GUYVIR.wav").read_bytes()

    def test_audio_is_cacheable_and_named(self, client, populated):
        response = client.get("/media/audio/1")
        assert "immutable" in response.headers["cache-control"]
        assert "biotimbre_item_1" in response.headers["content-disposition"]

    def test_missing_item_is_404(self, client, populated):
        assert client.get("/media/audio/999").status_code == 404

    def test_serves_a_spectrogram_png(self, client, populated):
        response = client.get("/media/spectrogram/1")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")

    def test_spectrogram_honours_size(self, client, populated):
        import io

        from PIL import Image

        response = client.get("/media/spectrogram/1?w=450&h=130")
        assert Image.open(io.BytesIO(response.content)).size == (450, 130)

    def test_spectrogram_size_is_clamped(self, client, populated):
        import io

        from PIL import Image

        # A crafted query must not be able to ask for an enormous render.
        response = client.get("/media/spectrogram/1?w=99999&h=99999")
        width, height = Image.open(io.BytesIO(response.content)).size
        assert width <= routes.MAX_WIDTH
        assert height <= routes.MAX_HEIGHT

    def test_spectrogram_honours_a_span(self, client, populated):
        first = client.get("/media/spectrogram/1?start=0&end=2").content
        second = client.get("/media/spectrogram/1?start=3&end=5").content
        assert first != second

    def test_spectrogram_of_missing_item_is_404(self, client, populated):
        assert client.get("/media/spectrogram/999").status_code == 404

    def test_status_reflects_a_populated_library(self, client, populated):
        payload = client.get("/media/status").json()
        assert payload["items"] == 2
        assert payload["ready"] is True
