"""Search page behaviour: query staging, the segment cap, and results."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.i18n import Translator  # noqa: E402
from biotimbre.ui import routes  # noqa: E402
from biotimbre.ui.search import QueryState, _build_filters, _filters_panel  # noqa: E402
from biotimbre.ui.state import AppState, QueryStore, get_state, set_state  # noqa: E402
from conftest import EXAMPLES_DIR  # noqa: E402

GUYVIR = EXAMPLES_DIR / "GUYVIR.wav"


@pytest.fixture
def installed_state(settings, shared_pool):
    """Install app state backed by a temporary library."""
    state = AppState(settings)
    state.pool = shared_pool
    state.service._pool = shared_pool
    state.service._owns_pool = False

    previous = None
    try:
        previous = get_state()
    except RuntimeError:
        pass
    set_state(state)
    yield state
    state.close()
    if previous is not None:
        set_state(previous)


@pytest.fixture
def long_wav(tmp_path):
    """A 40 s recording, well over the 15 s cap."""
    samples, rate = sf.read(GUYVIR, dtype="float32")
    path = tmp_path / "long.wav"
    sf.write(path, np.tile(samples, 8), rate)
    return path


class TestQueryStore:
    def test_stages_and_resolves_by_token(self):
        store = QueryStore()
        token = store.add("q.wav", GUYVIR.read_bytes())
        assert store.path(token) is not None
        assert store.path(token).read_bytes() == GUYVIR.read_bytes()
        store.clear()

    def test_unknown_token_resolves_to_none(self):
        store = QueryStore()
        # A 404 rather than an opportunity to name a path.
        assert store.path("not-a-real-token") is None
        store.clear()

    def test_tokens_are_unguessable_and_distinct(self):
        store = QueryStore()
        first = store.add("a.wav", GUYVIR.read_bytes())
        second = store.add("a.wav", GUYVIR.read_bytes())
        assert first != second
        assert len(first) > 16
        store.clear()

    def test_upload_name_cannot_escape_the_directory(self):
        store = QueryStore()
        token = store.add("../../escape.wav", GUYVIR.read_bytes())
        path = store.path(token)
        assert path is not None
        assert "escape.wav" in path.name
        assert path.parent == store._directory
        store.clear()

    def test_oldest_entry_is_evicted(self):
        store = QueryStore()
        tokens = [
            store.add(f"q{i}.wav", GUYVIR.read_bytes())
            for i in range(QueryStore.MAX_ENTRIES + 2)
        ]
        # A long session must not accumulate uploads without bound.
        assert store.path(tokens[0]) is None
        assert store.path(tokens[-1]) is not None
        store.clear()

    def test_clear_removes_every_file(self):
        store = QueryStore()
        token = store.add("q.wav", GUYVIR.read_bytes())
        path = store.path(token)
        store.clear()
        assert not path.exists()


class TestQueryStateSegment:
    def _state(self, path, installed_state) -> QueryState:
        from biotimbre.embeddings.audio import read_properties

        query = QueryState()
        query.properties = read_properties(path)
        cap = installed_state.settings.max_query_duration_s
        query.start = 0.0
        query.end = min(query.properties.duration_s, cap)
        return query

    def test_short_file_is_searched_whole(self, installed_state):
        query = self._state(GUYVIR, installed_state)
        assert query.needs_segment is False
        # None/None keeps behaviour and the cache key identical to the CLI.
        assert query.segment() == (None, None)

    def test_long_file_requires_a_segment(self, installed_state, long_wav):
        query = self._state(long_wav, installed_state)
        assert query.needs_segment is True
        start, end = query.segment()
        assert (start, end) == (0.0, 15.0)

    def test_segment_defaults_to_the_cap(self, installed_state, long_wav):
        query = self._state(long_wav, installed_state)
        assert query.end - query.start == pytest.approx(15.0)

    def test_moved_segment_is_reported(self, installed_state, long_wav):
        query = self._state(long_wav, installed_state)
        query.start, query.end = 20.0, 32.0
        assert query.segment() == (20.0, 32.0)


@pytest.mark.usefixtures("installed_state")
class TestFilterPanel:
    async def _panel(self, user: User):
        holder: dict = {}

        @ui.page("/filters")
        def filters_page():
            holder["inputs"] = _filters_panel(Translator("en"))

        await user.open("/filters")
        return holder["inputs"]

    async def test_empty_panel_matches_everything(self, user: User):
        inputs = await self._panel(user)
        assert _build_filters(inputs).is_empty is True

    async def test_taxa_are_collected(self, user: User):
        inputs = await self._panel(user)
        inputs["taxa"].value = ["Cicadidae", "insecta"]
        filters = _build_filters(inputs)
        # Normalisation is the domain's job, and it happens on construction.
        assert filters.taxa == ("cicadidae", "insecta")

    async def test_taxa_all_switch(self, user: User):
        inputs = await self._panel(user)
        inputs["taxa"].value = ["cicadidae"]
        inputs["taxa_all"].value = True
        assert _build_filters(inputs).taxa_match_all is True

    async def test_blank_text_is_not_a_filter(self, user: User):
        inputs = await self._panel(user)
        inputs["device"].value = "   "
        inputs["date_from"].value = ""
        assert _build_filters(inputs).is_empty is True

    async def test_location_is_assembled(self, user: User):
        inputs = await self._panel(user)
        inputs["latitude"].value = -26.0
        inputs["longitude"].value = -48.0
        inputs["radius_km"].value = 50.0
        filters = _build_filters(inputs)
        assert filters.has_location is True

    async def test_partial_location_is_rejected(self, user: User):
        from biotimbre.domain.errors import ValidationError

        inputs = await self._panel(user)
        inputs["latitude"].value = -26.0
        # Latitude without a radius cannot mean anything; the domain says so.
        with pytest.raises(ValidationError, match="together"):
            _build_filters(inputs)

    async def test_dates_are_canonicalised(self, user: User):
        inputs = await self._panel(user)
        inputs["date_from"].value = "2020-1-5"
        assert _build_filters(inputs).date_from == "2020-01-05"


class TestQueryRoutes:
    @pytest.fixture
    def client(self, installed_state):
        app = FastAPI()
        routes.register(app)
        return TestClient(app)

    def test_serves_a_staged_query(self, client, installed_state):
        token = installed_state.queries.add("q.wav", GUYVIR.read_bytes())
        response = client.get(f"/media/query/{token}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == GUYVIR.read_bytes()

    def test_unknown_token_is_404(self, client):
        assert client.get("/media/query/bogus").status_code == 404

    def test_query_audio_is_not_publicly_cacheable(self, client, installed_state):
        # An uploaded query is the user's own material, not library content.
        token = installed_state.queries.add("q.wav", GUYVIR.read_bytes())
        cache = client.get(f"/media/query/{token}").headers["cache-control"]
        assert "private" in cache

    def test_serves_a_query_spectrogram(self, client, installed_state):
        token = installed_state.queries.add("q.wav", GUYVIR.read_bytes())
        response = client.get(f"/media/query/{token}/spectrogram?w=450&h=120")
        assert response.status_code == 200
        assert response.content.startswith(b"\x89PNG")

    def test_query_spectrogram_of_unknown_token_is_404(self, client):
        assert client.get("/media/query/bogus/spectrogram").status_code == 404

    def test_query_spectrogram_honours_a_span(self, client, installed_state):
        token = installed_state.queries.add("q.wav", GUYVIR.read_bytes())
        whole = client.get(f"/media/query/{token}/spectrogram").content
        part = client.get(
            f"/media/query/{token}/spectrogram?start=0&end=2"
        ).content
        assert whole != part


@pytest.mark.usefixtures("installed_state")
class TestSearchPageRenders:
    async def test_renders_without_a_query(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/search")
        await user.should_see("Find similar samples")
        await user.should_see("Choose a recording to search with.")
        await user.should_see("Search")

    async def test_mentions_the_query_limit(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/search")
        # The cap has to be visible before a long file is rejected.
        await user.should_see("Up to 15 s is searched")

    async def test_shows_filter_and_option_panels(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401

        await user.open("/search")
        await user.should_see("Filters")
        await user.should_see("Options")

    async def test_translates_to_portuguese(self, user: User):
        from biotimbre.ui import app as ui_app  # noqa: F401

        installed = get_state()
        installed.settings = type(installed.settings)(
            **{**installed.settings.__dict__, "language": "pt_BR"}
        )
        await user.open("/search")
        await user.should_see("Encontrar amostras semelhantes")
        await user.should_see("Buscar")


@pytest.mark.models
@pytest.mark.usefixtures("installed_state")
class TestSearchEndToEnd:
    async def test_finds_a_stored_item(self, user: User, installed_state):
        installed_state.warm_models()
        installed_state.call(
            installed_state.service.import_directory, EXAMPLES_DIR
        )
        hits = installed_state.call(installed_state.service.search, GUYVIR, k=2)
        # The page renders SearchHit objects; this checks the data behind it.
        assert hits
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)
        assert hits[0].matched_span is not None
        start, end = hits[0].matched_span
        # Clamped, so the marker overlay can be positioned without going off.
        assert 0.0 <= start < end <= hits[0].item.duration_s

    async def test_long_query_needs_a_segment_and_then_works(
        self, installed_state, long_wav
    ):
        from biotimbre.domain.errors import ValidationError

        installed_state.warm_models()
        installed_state.call(
            installed_state.service.import_directory, EXAMPLES_DIR
        )
        with pytest.raises(ValidationError, match="limited to 15 s"):
            installed_state.call(installed_state.service.search, long_wav, k=1)

        hits = installed_state.call(
            installed_state.service.search,
            long_wav,
            k=1,
            start_time=0.0,
            end_time=5.0,
        )
        assert hits
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)
