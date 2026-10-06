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
from biotimbre.ui import maps, routes  # noqa: E402
from biotimbre.ui.search import (  # noqa: E402
    OBSERVATION_RADIUS_KM,
    QueryState,
    _build_filters,
    _filters_panel,
)
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

    async def test_empty_panel_only_filters_by_query_speed(self, user: User):
        # The query speed filter is always applied — an item embedded at a
        # different speed is not comparable — so "empty" means every other
        # field is unset, not that SearchFilters.is_empty is True.
        inputs = await self._panel(user)
        filters = _build_filters(inputs, query_speed=1.0)
        assert filters.speeds == (1.0,)
        assert filters.taxa == ()
        assert filters.biotimbre_ids == ()
        assert filters.date_from is None
        assert filters.date_to is None
        assert filters.has_location is False

    async def test_taxa_are_collected(self, user: User):
        inputs = await self._panel(user)
        inputs["taxa"].value = ["Cicadidae", "insecta"]
        filters = _build_filters(inputs, query_speed=1.0)
        # Normalisation is the domain's job, and it happens on construction.
        assert filters.taxa == ("cicadidae", "insecta")

    async def test_taxa_all_switch(self, user: User):
        inputs = await self._panel(user)
        inputs["taxa"].value = ["cicadidae"]
        inputs["taxa_all"].value = True
        assert _build_filters(inputs, query_speed=1.0).taxa_match_all is True

    async def test_blank_text_is_not_a_filter(self, user: User):
        inputs = await self._panel(user)
        inputs["date_from"].value = ""
        assert _build_filters(inputs, query_speed=1.0).date_from is None

    async def test_location_is_assembled(self, user: User):
        inputs = await self._panel(user)
        inputs["latitude"].value = -26.0
        inputs["longitude"].value = -48.0
        inputs["radius_km"].value = 50.0
        filters = _build_filters(inputs, query_speed=1.0)
        assert filters.has_location is True

    async def test_partial_location_is_rejected(self, user: User):
        from biotimbre.domain.errors import ValidationError

        inputs = await self._panel(user)
        inputs["latitude"].value = -26.0
        # Latitude without a radius cannot mean anything; the domain says so.
        with pytest.raises(ValidationError, match="together"):
            _build_filters(inputs, query_speed=1.0)

    async def test_dates_are_canonicalised(self, user: User):
        inputs = await self._panel(user)
        inputs["date_from"].value = "2020-1-5"
        assert _build_filters(inputs, query_speed=1.0).date_from == "2020-01-05"

    async def test_query_speed_becomes_a_speed_filter(self, user: User):
        inputs = await self._panel(user)
        assert _build_filters(inputs, query_speed=0.25).speeds == (0.25,)

    async def test_clearing_empties_every_filter(self, user: User):
        from biotimbre.ui.search import clear_filters

        inputs = await self._panel(user)
        inputs["taxa"].set_options(["cicadidae"], value=["cicadidae"])
        inputs["taxa_all"].value = True
        inputs["date_from"].value = "2020-01-01"
        inputs["date_to"].value = "2020-12-31"
        inputs["latitude"].value = -26.0
        inputs["longitude"].value = -48.0
        inputs["radius_km"].value = 50.0
        inputs["biotimbre_id"].set_options(["x"], value=["x"])

        clear_filters(inputs)

        # Nothing left to narrow the search with.
        assert _build_filters(inputs, query_speed=None).is_empty is True
        # The tags typed in are still offered, only unchosen.
        assert inputs["taxa"].options == ["cicadidae"]

    async def test_clearing_an_empty_panel_changes_nothing(self, user: User):
        from biotimbre.ui.search import clear_filters

        inputs = await self._panel(user)
        clear_filters(inputs)
        assert _build_filters(inputs, query_speed=None).is_empty is True

    async def test_without_a_query_no_speed_is_imposed(self, user: User):
        # A metadata-only search has no query to embed, so the speed it would
        # have been embedded at must not narrow the library.
        inputs = await self._panel(user)
        filters = _build_filters(inputs, query_speed=None)
        assert filters.speeds == ()
        assert filters.is_empty is True

        inputs["taxa"].value = ["cicadidae"]
        assert _build_filters(inputs, query_speed=None).is_empty is False


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
        await user.open("/search")
        await user.should_see("Find similar samples")
        await user.should_see("Choose a recording to search with")
        await user.should_see("Search")

    async def test_mentions_the_query_limit(self, user: User):
        await user.open("/search")
        # The cap has to be visible before a long file is rejected.
        await user.should_see("Up to 15 s is searched")

    async def test_shows_filter_and_option_panels(self, user: User):
        await user.open("/search")
        await user.should_see("Filters")
        await user.should_see("Options")

    async def test_shows_the_dsp_filter_controls(self, user: User):
        await user.open("/search")
        await user.should_see("Query speed")
        await user.should_see("Highpass cutoff")
        await user.should_see("Lowpass cutoff")
        await user.should_see("Filter order")

    async def test_translates_to_portuguese(self, user: User):
        installed = get_state()
        installed.settings = type(installed.settings)(
            **{**installed.settings.__dict__, "language": "pt_BR"}
        )
        await user.open("/search")
        await user.should_see("Encontrar amostras semelhantes")
        await user.should_see("Buscar")


@pytest.mark.models
@pytest.mark.usefixtures("installed_state")
class TestQueryRemoval:
    """Taking back a chosen recording, as the library page allows."""

    async def test_a_chosen_recording_can_be_taken_back(
        self, user: User, installed_state
    ):
        from nicegui.elements.upload_files import SmallFileUpload

        await user.open("/search")
        await user.should_see("Query recording")

        uploader = next(iter(user.find(ui.upload).elements))
        await uploader.handle_uploads(
            [SmallFileUpload("GUYVIR.wav", "audio/wav", GUYVIR.read_bytes())]
        )
        await user.should_see("GUYVIR.wav")
        await user.should_see("Remove file")
        staged = next(iter(user.find(ui.audio).elements))._props["src"]
        token = staged.split("/media/query/")[1].split("/")[0]
        assert installed_state.queries.path(token) is not None

        user.find("Remove file").click()

        # Gone from the page and from the staging directory, not merely hidden.
        # The filename is the thing that disappears; the uploader's own label
        # says "Choose a recording" whether or not one is staged.
        await user.should_not_see("GUYVIR.wav", retries=60)
        assert installed_state.queries.path(token) is None
        # find() insists on a match, so absence is read off the live registry.
        assert not [
            element
            for element in user.client.elements.values()
            if isinstance(element, ui.audio)
        ]


class TestSegmentPicker:
    """Choosing a window in a long recording, without the page thrashing."""

    @staticmethod
    async def _stage(user: User, path):
        from nicegui.elements.upload_files import SmallFileUpload

        await user.open("/search")
        await user.should_see("Query recording")
        uploader = next(iter(user.find(ui.upload).elements))
        await uploader.handle_uploads(
            [SmallFileUpload(path.name, "audio/wav", path.read_bytes())]
        )
        await user.should_see(path.name)

    async def test_moving_the_span_keeps_the_same_spectrogram_element(
        self, user: User, installed_state, long_wav
    ):
        await self._stage(user, long_wav)

        image = next(iter(user.find(ui.image).elements))
        marker = next(
            element
            for element in user.client.elements.values()
            if "pointer-events-none" in element._classes
        )
        before = marker._style.get("left")

        picker = next(iter(user.find(ui.range).elements))
        picker.value = {"min": 6.0, "max": 18.0}

        # Rebuilding the block would make the browser re-request the PNG, which
        # flashed the page on every step of a drag. Only the marker moves.
        assert next(iter(user.find(ui.image).elements)) is image
        assert marker._style.get("left") != before
        assert marker._style["left"] == "15.00%"

    async def test_the_player_is_limited_to_the_chosen_window(
        self, user: User, installed_state, long_wav
    ):
        await self._stage(user, long_wav)

        player = next(iter(user.find(ui.audio).elements))
        assert player._props["src"].endswith("#t=0.00,15.00")

        picker = next(iter(user.find(ui.range).elements))
        picker.value = {"min": 6.0, "max": 18.0}
        # The source follows the settled drag, not every step of it.
        user.find(ui.range).trigger("change")
        assert player._props["src"].endswith("#t=6.00,18.00")

    async def test_the_cap_is_applied_in_the_browser(
        self, user: User, installed_state, long_wav
    ):
        from biotimbre.ui.segment_range import SegmentRange

        await self._stage(user, long_wav)

        # QRange replays the undragged edge on every move, so a cap enforced
        # only by the server made the selection flicker once it was reached.
        picker = next(iter(user.find(ui.range).elements))
        assert isinstance(picker, SegmentRange)
        assert picker.tag == SegmentRange.component.tag
        assert (
            picker._props["max-span"]
            == installed_state.settings.max_query_duration_s
        )

    async def test_a_short_recording_plays_whole(self, user: User, installed_state):
        await self._stage(user, GUYVIR)

        player = next(iter(user.find(ui.audio).elements))
        assert "#t=" not in player._props["src"]


class TestMetadataOnlySearch:
    """Searching the library without a recording to search with."""

    @staticmethod
    async def _open(user: User, installed_state):
        installed_state.call(
            installed_state.service.import_directory,
            EXAMPLES_DIR,
            compute_embeddings=False,
        )
        await user.open("/search")
        await user.should_see("Query recording")

    @staticmethod
    def _run_search(user: User) -> None:
        """Click the page's Search button, not the header's Search tab.

        ``user.find("Search")`` picks the lowest element id, which is the
        navigation button in the header — clicking that just reloads the page.
        """
        from nicegui.testing.user_interaction import UserInteraction

        button = next(
            element
            for element in user.find(ui.button).elements
            if element._props.get("label") == "Search"
            and "flat" not in element._props
        )
        UserInteraction(user, {button}, None).click()

    @staticmethod
    def _taxa_input(user: User):
        return next(
            element
            for element in user.find(ui.select).elements
            if element._props.get("label") == "Taxa"
        )

    async def test_taxa_alone_finds_items(self, user: User, installed_state):
        await self._open(user, installed_state)

        # Only ZAMTYM declares taxa; no recording is staged.
        self._taxa_input(user).value = ["cicadidae"]
        self._run_search(user)

        await user.should_see("1 match(es)")
        # Nothing was compared, so the results say so instead of ranking.
        await user.should_see("not ranked")

    async def test_nothing_at_all_asks_for_a_file_or_a_filter(
        self, user: User, installed_state
    ):
        await self._open(user, installed_state)

        self._run_search(user)
        await user.should_see("set a filter")

    async def test_a_taxon_nothing_carries_finds_nothing(
        self, user: User, installed_state
    ):
        await self._open(user, installed_state)

        self._taxa_input(user).value = ["mammalia"]
        self._run_search(user)
        await user.should_see("No matches")


class TestResultExport:
    """A hit can be taken away as a file, like an item in the library list."""

    async def test_a_result_offers_the_audio_and_its_embeddings(
        self, user: User, installed_state
    ):
        from biotimbre.models_registry import ModelSpec
        from conftest import unit_vectors

        installed_state.call(
            installed_state.service.import_directory,
            EXAMPLES_DIR,
            compute_embeddings=False,
        )
        spec = ModelSpec(
            model_id="fake_model",
            repo_id="test/fake",
            filename="fake.onnx",
            sample_rate=32_000,
            window_duration_s=5.0,
            window_overlap=0.5,
            description="test double",
        )
        installed_state.call(installed_state.service.repo.register_model, spec, 4)
        item_id = installed_state.call(
            installed_state.service.list_items
        )[0].item_id
        installed_state.call(
            installed_state.service.repo.set_embeddings,
            item_id,
            "fake_model",
            [(0.0, 5.0, unit_vectors(1, 4)[0])],
        )

        await user.open("/search")
        await user.should_see("Query recording")
        taxa = next(
            element
            for element in user.find(ui.select).elements
            if element._props.get("label") == "Taxa"
        )
        taxa.value = ["cicadidae"]
        from nicegui.testing.user_interaction import UserInteraction

        button = next(
            element
            for element in user.find(ui.button).elements
            if element._props.get("label") == "Search"
            and "flat" not in element._props
        )
        UserInteraction(user, {button}, None).click()

        await user.should_see("1 match(es)")
        await user.should_see("Export")
        await user.should_see("Audio file")
        await user.should_see("fake_model embeddings (.npy)")


class TestObservationFetch:
    """Taking the query, and the filters, from an iNaturalist observation."""

    @staticmethod
    def _observation():
        from biotimbre.sources.inaturalist import Observation, Sound

        return Observation(
            id=400139850,
            url="https://www.inaturalist.org/observations/400139850",
            date="2026-09-14",
            time="17:59:27",
            latitude=-28.82,
            longitude=153.28,
            taxon="Guyalna bonaerensis",
            taxon_rank="species",
            quality_grade="needs_id",
            observer="Paul Montgomery",
            sounds=(
                Sound(
                    id=2158429,
                    url="https://static.inaturalist.org/sounds/2158429.m4a",
                    content_type="audio/mp4",
                    license_code="cc-by-nc",
                    attribution="(c) Paul Montgomery, some rights reserved (CC BY-NC)",
                ),
            ),
        )

    @staticmethod
    def _stub(monkeypatch, observation):
        """Answer the fetch without a network, with GUYVIR as the recording."""
        from biotimbre.sources import inaturalist

        def fake_fetch(observation_id, **kwargs):
            return observation

        def fake_download(sound, destination, **kwargs):
            from pathlib import Path

            target = Path(destination).with_suffix(".wav")
            target.write_bytes(GUYVIR.read_bytes())
            return target

        monkeypatch.setattr(inaturalist, "fetch_observation", fake_fetch)
        monkeypatch.setattr(inaturalist, "download_sound", fake_download)

    async def _fetch(self, user: User, monkeypatch, observation=None):
        from nicegui.testing.user_interaction import UserInteraction

        observation = observation or self._observation()
        self._stub(monkeypatch, observation)

        await user.open("/search")
        await user.should_see("Query recording")

        field = next(
            element
            for element in user.find(ui.input).elements
            if element._props.get("label", "").startswith("iNaturalist")
        )
        field.value = str(observation.id)
        button = next(
            element
            for element in user.find(ui.button).elements
            if element._props.get("label") == "Fetch"
        )
        UserInteraction(user, {button}, None).click()
        return observation

    async def test_the_recording_becomes_the_query(
        self, user: User, installed_state, monkeypatch
    ):
        observation = await self._fetch(user, monkeypatch)
        await user.should_see(f"inat-{observation.id}.wav", retries=200)

        # Staged like any other query, so the rest of the page is unchanged.
        player = next(iter(user.find(ui.audio).elements))
        assert "/media/query/" in player._props["src"]

    async def test_what_the_observation_knows_fills_the_filters(
        self, user: User, installed_state, monkeypatch
    ):
        await self._fetch(user, monkeypatch)
        await user.should_see("Observation 400139850", retries=200)

        inputs = {
            element._props.get("label"): element
            for element in [
                *user.find(ui.input).elements,
                *user.find(ui.number).elements,
                *user.find(ui.select).elements,
            ]
        }
        # Offered as well as chosen: a value the select has never heard of
        # renders as an empty field.
        assert inputs["Taxa"].value == ["Guyalna bonaerensis"]
        assert "Guyalna bonaerensis" in inputs["Taxa"].options
        # Not the date: an observation's day says when that one recording was
        # made, not when the animal sings.
        assert not inputs["From date"].value
        assert not inputs["To date"].value
        assert inputs["Centre latitude"].value == pytest.approx(-28.82)
        assert inputs["Radius"].value == 25.0
        # Filled, not applied behind the reader's back: the panel is opened so
        # the narrowing is visible and can be cleared.
        panel = next(
            element
            for element in user.find(ui.expansion).elements
            if element.text == "Filters"
        )
        assert panel.value is True

    async def test_it_says_what_to_know_before_trusting_it(
        self, user: User, installed_state, monkeypatch
    ):
        await self._fetch(user, monkeypatch)
        await user.should_see("Not research grade", retries=200)
        await user.should_see("Paul Montgomery")

    async def test_a_hidden_location_is_withheld_and_said(
        self, user: User, installed_state, monkeypatch
    ):
        import dataclasses

        hidden = dataclasses.replace(
            self._observation(), latitude=None, longitude=None, obscured=True
        )
        await self._fetch(user, monkeypatch, hidden)
        await user.should_see("hides this observation's location", retries=200)

        inputs = {
            element._props.get("label"): element
            for element in user.find(ui.number).elements
        }
        assert inputs["Centre latitude"].value is None

    async def test_an_observation_without_sound_is_reported(
        self, user: User, installed_state, monkeypatch
    ):
        import dataclasses

        silent = dataclasses.replace(self._observation(), sounds=())
        await self._fetch(user, monkeypatch, silent)
        await user.should_see("has no recordings", retries=200)

    async def test_the_switch_hides_it(
        self, user: User, installed_state, monkeypatch
    ):
        import dataclasses

        installed_state.settings = dataclasses.replace(
            installed_state.settings, inaturalist_enabled=False
        )
        await user.open("/search")
        await user.should_see("Query recording")
        assert not [
            element
            for element in user.find(ui.input).elements
            if element._props.get("label", "").startswith("iNaturalist")
        ]


class TestResultMap:
    """One map above the results, and the location filter made visible on it."""

    @staticmethod
    def _store(state, *locations):
        from biotimbre.domain.item import ItemDraft, Location, SampleWindow
        from conftest import add_item

        for index, (latitude, longitude) in enumerate(locations):
            draft = ItemDraft(
                date="2020-01-14",
                time="09:40:21",
                location=Location(latitude, longitude),
                sample_window=SampleWindow(5.0, 10.0),
                taxa=("cicadidae",),
            ).validated()
            add_item(state.service.repo, draft, sha256=f"{index:064d}")

    @staticmethod
    def _map(user: User):
        return list(user.find(ui.leaflet).elements)[0]

    @staticmethod
    def _run_search(user: User) -> None:
        """Click the page's Search button, not the header's Search tab."""
        from nicegui.testing.user_interaction import UserInteraction

        button = next(
            element
            for element in user.find(ui.button).elements
            if element._props.get("label") == "Search"
            and "flat" not in element._props
        )
        UserInteraction(user, {button}, None).click()

    @staticmethod
    def _click_map(element, latitude: float, longitude: float) -> None:
        """Deliver a Leaflet map-click, as the browser would."""
        # NiceGUI camel-cases event names on the way in, so the listener for
        # "map-click" is filed under "mapClick".
        listener = next(
            listener
            for listener in element._event_listeners.values()
            if listener.type == "mapClick"
        )
        element._handle_event(
            {
                "listener_id": listener.id,
                "args": {"latlng": {"lat": latitude, "lng": longitude}},
            }
        )

    @staticmethod
    def _numbers(user: User):
        return {
            element._props.get("label"): element
            for element in user.find(ui.number).elements
        }

    @staticmethod
    def _markers(element):
        return [
            layer
            for layer in element.layers
            if layer.to_dict()["type"] == "circleMarker"
        ]

    async def test_one_map_for_the_page_not_one_per_result(
        self, user: User, installed_state
    ):
        self._store(installed_state, (-26.3, -48.85), (-25.4, -49.2))
        await user.open("/search")
        await user.should_see("Find similar samples")

        elements = list(user.find(ui.leaflet).elements)
        assert len(elements) == 1

    async def test_results_are_plotted(self, user: User, installed_state):
        self._store(installed_state, (-26.3, -48.85), (-25.4, -49.2))
        await user.open("/search")
        await user.should_see("Find similar samples")

        # A metadata-only search, so no models are needed: a filter alone is
        # enough, and a search with no filter and no query is refused.
        dates = {
            element._props.get("label"): element
            for element in user.find(ui.input).elements
        }
        dates["From date"].value = "2000-01-01"

        self._run_search(user)
        await user.should_see("2 match(es)")

        markers = self._markers(self._map(user))
        assert {tuple(layer.to_dict()["args"][0]) for layer in markers} == {
            (-26.3, -48.85),
            (-25.4, -49.2),
        }

    async def test_the_location_filter_draws_a_centre_and_a_radius(
        self, user: User, installed_state
    ):
        await user.open("/search")
        await user.should_see("Find similar samples")

        # Typing into the filter is enough: the map follows the inputs rather
        # than each of the four places that write them.
        numbers = {
            element._props.get("label"): element
            for element in user.find(ui.number).elements
        }
        numbers["Centre latitude"].value = -26.3
        numbers["Centre longitude"].value = -48.85
        numbers["Radius"].value = 25.0

        element = self._map(user)
        circles = [
            layer for layer in element.layers if layer.to_dict()["type"] == "circle"
        ]
        assert len(circles) == 1
        assert circles[0].to_dict()["args"][0] == [-26.3, -48.85]
        assert circles[0].to_dict()["args"][1]["radius"] == 25_000.0

    async def test_clicking_the_map_fills_the_centre_and_zooms_to_it(
        self, user: User, installed_state
    ):
        from test_maps import visible_height_km

        await user.open("/search")
        await user.should_see("Find similar samples")

        element = self._map(user)
        self._click_map(element, -26.3, -48.85)

        numbers = self._numbers(user)
        assert numbers["Centre latitude"].value == -26.3
        assert numbers["Centre longitude"].value == -48.85
        # A click is a click on a region, not on a point: the default radius
        # is assumed and the map goes to the scale that shows it.
        assert numbers["Radius"].value == OBSERVATION_RADIUS_KM
        assert element.center == pytest.approx((-26.3, -48.85))
        shown = visible_height_km(element.zoom, -26.3)
        assert 2 * OBSERVATION_RADIUS_KM <= shown <= 200.0

    async def test_a_new_radius_reframes_the_map(
        self, user: User, installed_state
    ):
        await user.open("/search")
        await user.should_see("Find similar samples")

        element = self._map(user)
        self._click_map(element, -26.3, -48.85)
        near = element.zoom

        # The filter is the thing the map is drawing, so widening it there is
        # enough; nothing has to be clicked again.
        self._numbers(user)["Radius"].value = 200.0
        assert element.zoom < near

    async def test_a_kept_radius_survives_a_second_click(
        self, user: User, installed_state
    ):
        await user.open("/search")
        await user.should_see("Find similar samples")

        element = self._map(user)
        self._numbers(user)["Radius"].value = 5.0
        self._click_map(element, -26.3, -48.85)

        # A radius the curator chose is not overwritten by the default, and
        # the framing follows the radius that is actually in force.
        assert self._numbers(user)["Radius"].value == 5.0
        assert element.zoom > maps.fit_view(maps.radius_box(-26.3, -48.85, 25.0))[1]

    async def test_clearing_the_filters_removes_the_centre(
        self, user: User, installed_state
    ):
        await user.open("/search")
        await user.should_see("Find similar samples")

        numbers = {
            element._props.get("label"): element
            for element in user.find(ui.number).elements
        }
        numbers["Centre latitude"].value = -26.3
        numbers["Centre longitude"].value = -48.85
        numbers["Radius"].value = 25.0

        user.find("Clear filters").click()
        element = self._map(user)
        assert not [
            layer for layer in element.layers if layer.to_dict()["type"] == "circle"
        ]

    async def test_the_switch_hides_it(self, user: User, installed_state):
        import dataclasses

        installed_state.settings = dataclasses.replace(
            installed_state.settings, map_enabled=False
        )
        self._store(installed_state, (-26.3, -48.85))
        await user.open("/search")
        await user.should_see("Find similar samples")

        # The coordinate filters are untouched by the switch: only the picture
        # goes away, not the ability to search on location.
        labels = {
            element._props.get("label") for element in user.find(ui.number).elements
        }
        assert {"Centre latitude", "Centre longitude", "Radius"} <= labels
        assert not [
            element
            for element in user.client.elements.values()
            if isinstance(element, ui.leaflet)
        ]


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
