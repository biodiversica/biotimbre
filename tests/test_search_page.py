"""Search page behaviour: query staging, the segment cap, and results."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from nicegui import app as nicegui_app  # noqa: E402
from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.i18n import Translator  # noqa: E402
from biotimbre.ui import maps, routes  # noqa: E402
from biotimbre.ui.coordinates import CoordinatesInput  # noqa: E402
from biotimbre.ui.search import (  # noqa: E402
    OBSERVATION_RADIUS_KM,
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
        inputs["location"].value = "-26.0, -48.0"
        inputs["radius_km"].value = 50.0
        filters = _build_filters(inputs, query_speed=1.0)
        assert filters.has_location is True
        assert (filters.latitude, filters.longitude) == (-26.0, -48.0)

    async def test_partial_location_is_rejected(self, user: User):
        from biotimbre.domain.errors import ValidationError

        inputs = await self._panel(user)
        inputs["location"].value = "-26.0, -48.0"
        # A centre without a radius cannot mean anything; the domain says so.
        with pytest.raises(ValidationError, match="together"):
            _build_filters(inputs, query_speed=1.0)

    async def test_an_unreadable_centre_is_rejected(self, user: User):
        from biotimbre.domain.errors import ValidationError

        inputs = await self._panel(user)
        inputs["location"].value = "-26.0"
        inputs["radius_km"].value = 50.0
        with pytest.raises(ValidationError, match="latitude, longitude"):
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
        inputs["location"].value = "-26.0, -48.0"
        inputs["radius_km"].value = 50.0
        inputs["biotimbre_id"].set_options(["x"], value=["x"])

        clear_filters(inputs)

        # Nothing left to narrow the search with.
        assert _build_filters(inputs, query_speed=None).is_empty is True
        # The tags typed in are still offered, only unchosen.
        assert inputs["taxa"].options == ["cicadidae"]

    async def test_the_taxa_filter_offers_the_library_s_tags(
        self, user: User, installed_state
    ):
        import asyncio
        import dataclasses

        from biotimbre.io.sidecar import draft_from_sidecar

        draft = draft_from_sidecar(EXAMPLES_DIR / "GUYVIR.txt")
        installed_state.call(
            installed_state.service.add_item,
            GUYVIR,
            dataclasses.replace(draft, taxa=("boana faber", "anura")),
            compute_embeddings=False,
        )
        inputs = await self._panel(user)

        # NiceGUI files the listener under the camel-cased event name.
        user.find(marker="filter-taxa").trigger("inputValue", "fab")
        for _ in range(100):
            if inputs["taxa"].options:
                break
            await asyncio.sleep(0.02)
        assert inputs["taxa"].options == ["boana faber"]

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

    def test_serves_a_selection_normalised(self, client, installed_state, tmp_path):
        import io

        samples = np.full(32_000, 0.5, dtype=np.float32)
        samples[16_000:] = 4.0
        hot = tmp_path / "hot.wav"
        sf.write(hot, samples, 16_000, subtype="FLOAT")
        token = installed_state.queries.add("hot.wav", hot.read_bytes())

        response = client.get(f"/media/query/{token}/normalised?start=1&end=2")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        from scipy.io import wavfile

        rate, played = wavfile.read(io.BytesIO(response.content))
        # The selection alone, brought within full scale.
        assert rate == 16_000
        assert played.size == 16_000
        assert np.abs(played).max() == pytest.approx(10 ** (-1 / 20))

    def test_normalised_selection_of_unknown_token_is_404(self, client):
        assert client.get("/media/query/bogus/normalised").status_code == 404

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

    async def test_shows_the_dsp_filter_controls_with_the_query(self, user: User):
        from nicegui.elements.upload_files import SmallFileUpload

        await user.open("/search")
        # They shape the query, so they arrive with it, as on the library page.
        await user.should_not_see("Highpass cutoff")
        uploader = next(iter(user.find(ui.upload).elements))
        await uploader.handle_uploads(
            [SmallFileUpload("GUYVIR.wav", "audio/wav", GUYVIR.read_bytes())]
        )
        await user.should_see("Query speed", retries=60)
        await user.should_see("Highpass cutoff")
        await user.should_see("Lowpass cutoff")
        await user.should_see("Filter order")
        await user.should_see("Normalise to -1 dBFS")

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
        assert staged.startswith("/media/staged/")

        user.find("Remove file").click()

        # Gone from the page and no longer served, not merely hidden. The
        # filename is the thing that disappears; the uploader's own label says
        # "Choose a recording" whether or not one is staged.
        await user.should_not_see("GUYVIR.wav", retries=60)
        assert not any(
            getattr(route, "path", None) == staged
            for route in nicegui_app.routes
        )
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
        # The controls arrive with the spectrogram, a moment after the name.
        await user.should_see("Query speed", retries=60)

    async def test_moving_the_span_keeps_the_same_spectrogram_element(
        self, user: User, installed_state, long_wav
    ):
        await self._stage(user, long_wav)
        await user.should_see("What will be searched with (15.00 s)", retries=60)

        image = next(iter(user.find(ui.image).elements))
        marker = next(
            element
            for element in user.client.elements.values()
            if "border-x-2" in element._classes
        )
        before = marker._style.get("left")

        picker = next(iter(user.find(ui.range).elements))
        picker.value = {"min": 6.0, "max": 18.0}

        # Rebuilding the block would make the browser re-request the PNG, which
        # flashed the page on every step of a drag. Only the marker moves.
        assert next(iter(user.find(ui.image).elements)) is image
        assert marker._style.get("left") != before
        assert marker._style["left"] == "15.00%"

    async def test_the_player_plays_the_chosen_window(
        self, user: User, installed_state, long_wav
    ):
        await self._stage(user, long_wav)
        # Cut to the cap at once: the whole file is never what is searched.
        await user.should_see("What will be searched with (15.00 s)", retries=60)
        player = next(iter(user.find(ui.audio).elements))
        first = player._props["src"]

        picker = next(iter(user.find(ui.range).elements))
        picker.value = {"min": 6.0, "max": 18.0}
        # The copy follows the settled drag, not every step of it, and the
        # player follows the copy.
        user.find(ui.range).trigger("change", args={"min": 6.0, "max": 18.0})
        await user.should_see("What will be searched with (12.00 s)", retries=60)
        assert player._props["src"] != first

    async def test_the_player_follows_the_query_speed(
        self, user: User, installed_state, long_wav
    ):
        await self._stage(user, long_wav)

        player = next(iter(user.find(ui.audio).elements))
        # One speed, the option's: the browser's own menu keeps the pitch.
        assert player._props["controlslist"] == "noplaybackrate"
        speed = next(
            element
            for element in user.find(ui.number).elements
            if element._props.get("label") == "Query speed"
        )
        speed.value = 0.25
        await user.should_see("Playing at speed 0.25")

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

    async def test_a_selection_beyond_full_scale_is_flagged_and_played_normalised(
        self, user: User, installed_state, tmp_path
    ):
        samples, rate = sf.read(GUYVIR, dtype="float32")
        hot = tmp_path / "hot.wav"
        sf.write(hot, samples * 10, rate, subtype="FLOAT")
        await self._stage(user, hot)

        await user.should_see("beyond full scale", retries=60)
        # What is played is the prepared copy, normalised, not the file.
        await user.should_see("What will be searched with", retries=60)

    async def test_a_selection_within_full_scale_is_not_flagged(
        self, user: User, installed_state, long_wav
    ):
        await self._stage(user, long_wav)
        await user.should_not_see("beyond full scale")

    async def test_a_short_recording_plays_whole(self, user: User, installed_state):
        await self._stage(user, GUYVIR)

        player = next(iter(user.find(ui.audio).elements))
        assert "#t=" not in player._props["src"]


@pytest.mark.usefixtures("installed_state")
class TestSearchesWhatIsHeard:
    """The query is searched as the editor prepared it, not as it was chosen."""

    @staticmethod
    async def _stage_and_capture(user: User, installed_state, monkeypatch, path):
        from nicegui.elements.upload_files import SmallFileUpload

        calls: list[dict] = []

        def fake_search(audio_path, **kwargs):
            import soundfile as sf

            calls.append(
                {"path": audio_path, "info": sf.info(str(audio_path)), **kwargs}
            )
            return []

        monkeypatch.setattr(installed_state.service, "search", fake_search)
        await user.open("/search")
        await user.should_see("Query recording")
        uploader = next(iter(user.find(ui.upload).elements))
        await uploader.handle_uploads(
            [SmallFileUpload(path.name, "audio/wav", path.read_bytes())]
        )
        await user.should_see(path.name)
        return calls

    @staticmethod
    def _run_search(user: User) -> None:
        TestMetadataOnlySearch._run_search(user)

    async def test_a_long_recording_is_searched_as_the_cut(
        self, user: User, installed_state, monkeypatch, long_wav
    ):
        calls = await self._stage_and_capture(
            user, installed_state, monkeypatch, long_wav
        )
        await user.should_see("What will be searched with (15.00 s)", retries=60)
        self._run_search(user)
        await user.should_see("No matches", retries=200)

        (call,) = calls
        # The prepared copy, whole: no segment or filter left to apply.
        assert call["path"] != long_wav
        assert call["info"].duration == pytest.approx(15.0, abs=0.01)
        assert "start_time" not in call and "filter_band" not in call

    async def test_a_cut_off_still_being_typed_is_applied_first(
        self, user: User, installed_state, monkeypatch
    ):
        calls = await self._stage_and_capture(
            user, installed_state, monkeypatch, GUYVIR
        )
        await user.should_see("0.00–5.00 s of 5.00 s", retries=60)
        # Typed, but never committed with a change event.
        highpass = next(iter(user.find(marker="apply-highpass").elements))
        highpass.value = 2000
        speed = next(iter(user.find(marker="apply-speed").elements))
        speed.value = 0.5
        self._run_search(user)
        await user.should_see("No matches", retries=200)

        (call,) = calls
        assert call["path"] != GUYVIR
        assert call["query_speed"] == 0.5
        await user.should_see("What will be searched with (5.00 s)")


@pytest.mark.usefixtures("installed_state")
class TestTakingAQueryToTheLibrary:
    """A query found to be new, brought to the add form as it was searched."""

    @staticmethod
    def _click(user: User, marker: str) -> None:
        from nicegui.testing.user_interaction import UserInteraction

        UserInteraction(user, set(user.find(marker=marker).elements), None).click()

    @staticmethod
    def _field(user: User, marker: str):
        return next(iter(user.find(marker=marker).elements))

    async def test_the_settings_arrive_in_the_add_form(
        self, user: User, installed_state, long_wav
    ):
        await TestSegmentPicker._stage(user, long_wav)
        await user.should_see("What will be searched with (15.00 s)", retries=60)

        picker = next(iter(user.find(ui.range).elements))
        picker.value = {"min": 6.0, "max": 18.0}
        user.find(ui.range).trigger("change", args={"min": 6.0, "max": 18.0})
        await user.should_see("What will be searched with (12.00 s)", retries=60)
        # Typed and not committed: carried over all the same.
        self._field(user, "apply-highpass").value = 2000
        self._field(user, "apply-speed").value = 0.5

        self._click(user, "to-library")

        await user.should_see("Brought over from Search", retries=200)
        await user.should_see("Keeping 6.00–18.00 s of 40.00 s", retries=60)
        await user.should_see("What will be stored (12.00 s)", retries=60)
        assert self._field(user, "apply-highpass").value == 2000
        assert self._field(user, "apply-speed").value == 0.5
        assert self._field(user, "apply-normalise").value is False
        # Staged, not added: the metadata and Save are still to come.
        assert installed_state.call(installed_state.service.stats)["items"] == 0
        # The staged copy was taken, not left behind.
        assert not installed_state.handoffs

    async def test_an_unknown_handoff_is_nothing_to_take(self, installed_state):
        from biotimbre.ui.clip_editor import ClipState, take_over

        clip = ClipState()
        assert await take_over("bogus", clip) is None
        assert clip.upload is None

    async def test_an_evicted_file_is_nothing_to_take(self, installed_state):
        from biotimbre.ui.clip_editor import ClipState, hand_over, take_over

        source = ClipState()
        target = source.begin("GUYVIR.wav")
        target.write_bytes(GUYVIR.read_bytes())
        source.finish(target, "GUYVIR.wav")
        token = await hand_over(source, speed=1.0)
        installed_state.queries.discard(token)

        assert await take_over(token, ClipState()) is None
        assert token not in installed_state.handoffs


@pytest.mark.usefixtures("installed_state")
class TestMovingBetweenPages:
    """Both pages live in one browser page, so leaving one keeps what was on it."""

    @staticmethod
    def _tab(user: User, label: str) -> None:
        from nicegui.testing.user_interaction import UserInteraction

        tab = next(
            element
            for element in user.find(ui.button).elements
            if element._props.get("label") == label and "flat" in element._props
        )
        UserInteraction(user, {tab}, None).click()

    @staticmethod
    def _visible(element) -> bool:
        while element is not None:
            if not element.visible:
                return False
            element = element.parent_slot.parent if element.parent_slot else None
        return True

    async def test_a_staged_query_and_its_settings_survive_a_visit(
        self, user: User, long_wav
    ):
        await TestSegmentPicker._stage(user, long_wav)
        highpass = next(iter(user.find(marker="apply-highpass").elements))
        highpass.value = 2000

        self._tab(user, "Library")
        await user.should_see("Add a sample")
        assert not self._visible(highpass)

        self._tab(user, "Search")
        await user.should_see(long_wav.name)
        assert self._visible(highpass)
        # The same element, not a new page's: nothing was rebuilt.
        assert next(iter(user.find(marker="apply-highpass").elements)) is highpass
        assert highpass.value == 2000

    async def test_a_query_reaches_a_library_page_already_open(
        self, user: User, long_wav
    ):
        from nicegui.elements.upload_files import SmallFileUpload
        from nicegui.testing.user_interaction import UserInteraction

        await user.open("/")
        await user.should_see("Add a sample")
        self._tab(user, "Search")
        await user.should_see("Query recording")
        uploader = next(
            element
            for element in user.find(ui.upload).elements
            if self._visible(element)
        )
        await uploader.handle_uploads(
            [SmallFileUpload(long_wav.name, "audio/wav", long_wav.read_bytes())]
        )
        await user.should_see("What will be searched with (15.00 s)", retries=60)

        UserInteraction(
            user, set(user.find(marker="to-library").elements), None
        ).click()

        await user.should_see("Brought over from Search", retries=200)
        await user.should_see("What will be stored (15.00 s)", retries=60)


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

    async def test_a_slowed_result_plays_either_way(
        self, user: User, installed_state
    ):
        import dataclasses

        from nicegui.testing.user_interaction import UserInteraction

        from biotimbre.io.sidecar import draft_from_sidecar

        draft = dataclasses.replace(
            draft_from_sidecar(EXAMPLES_DIR / "ZAMTYM.txt"), speed=0.25
        ).validated()
        installed_state.call(
            installed_state.service.add_item,
            EXAMPLES_DIR / "ZAMTYM.wav",
            draft,
            compute_embeddings=False,
        )
        await user.open("/search")
        await user.should_see("Query recording")
        self._taxa_input(user).value = ["cicadidae"]
        self._run_search(user)
        await user.should_see("1 match(es)")

        player = next(iter(user.find(ui.audio).elements))
        # The rate is applied once the browser has the audio.
        UserInteraction(user, {player}, None).trigger("loadedmetadata")
        await user.should_see("Playing at speed 0.25")

        # Or as it was recorded.
        heard_at = next(iter(user.find(ui.toggle).elements))
        heard_at.value = 1.0
        await user.should_not_see("Playing at speed 0.25")

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
        assert "/media/staged/" in player._props["src"]

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
        assert inputs["Centre (latitude, longitude)"].coordinates == pytest.approx((-28.82, 153.28))
        assert inputs["Radius"].value == 25.0
        # Filled, not applied behind the reader's back: the panel is opened so
        # the narrowing is visible and can be cleared.
        panel = next(
            element
            for element in user.find(ui.expansion).elements
            if element.text == "Filters"
        )
        assert panel.value is True

    async def test_taken_to_the_library_it_brings_the_metadata(
        self, user: User, installed_state, monkeypatch
    ):
        from nicegui.testing.user_interaction import UserInteraction

        observation = await self._fetch(user, monkeypatch)
        await user.should_see(f"inat-{observation.id}.wav", retries=200)
        await user.should_see("Query speed", retries=60)
        UserInteraction(
            user, set(user.find(marker="to-library").elements), None
        ).click()

        await user.should_see("Brought over from Search", retries=200)
        link = next(
            element
            for element in user.find(ui.input).elements
            if element._props.get("label") == "Link to full recording"
        )
        assert link.value == observation.url

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
            for element in user.find(ui.input).elements
        }
        assert not inputs["Centre (latitude, longitude)"].value

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
    def _centre(user: User):
        return next(iter(user.find(CoordinatesInput).elements))

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
        self._centre(user).value = "-26.3, -48.85"
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
        assert self._centre(user).value == "-26.3, -48.85"
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
        self._centre(user).value = "-26.3, -48.85"
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
            element._props.get("label")
            for element in [
                *user.find(ui.input).elements,
                *user.find(ui.number).elements,
            ]
        }
        assert {"Centre (latitude, longitude)", "Radius"} <= labels
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


class TestResultViews:
    """Each result chooses between spectrogram and waveform on its own."""

    async def test_switching_one_result_leaves_the_others_alone(
        self, user: User, installed_state
    ):
        from nicegui.testing.user_interaction import UserInteraction

        from biotimbre.io.sidecar import draft_from_sidecar

        # Real audio, since a waveform is drawn from the file itself.
        for name in ("GUYVIR", "ZAMTYM"):
            installed_state.call(
                installed_state.service.add_item,
                EXAMPLES_DIR / f"{name}.wav",
                draft_from_sidecar(EXAMPLES_DIR / f"{name}.txt"),
                compute_embeddings=False,
            )
        await user.open("/search")
        await user.should_see("Find similar samples")
        dates = {
            element._props.get("label"): element
            for element in user.find(ui.input).elements
        }
        dates["From date"].value = "1900-01-01"
        TestResultMap._run_search(user)
        await user.should_see("2 match(es)")

        toggles = list(user.find(marker="result-view").elements)
        assert len(toggles) == 2
        UserInteraction(user, {toggles[0]}, None).click()
        await user.should_see(ui.html, retries=60)

        waveforms = [
            element
            for element in user.client.elements.values()
            if isinstance(element, ui.html) and "<polygon" in element.content
        ]
        assert len(waveforms) == 1
        # Each button offers the view its own card is not showing.
        assert toggles[0]._props["icon"] == "gradient"
        assert toggles[1]._props["icon"] == "graphic_eq"


class TestResultDetails:
    """A result opens the library's detail dialog, without its edit actions."""

    @staticmethod
    async def _search(user: User, state):
        from biotimbre.io.sidecar import draft_from_sidecar

        item = state.call(
            state.service.add_item,
            EXAMPLES_DIR / "GUYVIR.wav",
            draft_from_sidecar(EXAMPLES_DIR / "GUYVIR.txt"),
            compute_embeddings=False,
        ).item
        await user.open("/search")
        await user.should_see("Find similar samples")
        dates = {
            element._props.get("label"): element
            for element in user.find(ui.input).elements
        }
        dates["From date"].value = "1900-01-01"
        TestResultMap._run_search(user)
        await user.should_see("1 match(es)")
        return item

    @staticmethod
    def _labels(user: User) -> set[str]:
        return {
            element._props.get("label") for element in user.find(ui.button).elements
        }

    async def test_details_shows_everything_but_offers_no_edit(
        self, user: User, installed_state
    ):
        from nicegui.testing.user_interaction import UserInteraction

        item = await self._search(user, installed_state)
        UserInteraction(
            user, set(user.find(marker="result-details").elements), None
        ).click()

        await user.should_see(f"Item {item.item_id}")
        await user.should_see("Sample rate")
        await user.should_see(item.biotimbre_id)
        # Changing the library is the library page's business.
        assert "Edit" not in self._labels(user)
        assert "Delete" not in self._labels(user)
        assert "Close" in self._labels(user)

    async def test_a_dot_s_link_opens_the_same_dialog(
        self, user: User, installed_state
    ):
        from nicegui.testing.user_interaction import UserInteraction

        item = await self._search(user, installed_state)
        leaflet = next(iter(user.find(ui.leaflet).elements))
        UserInteraction(user, {user.client.layout}, None).trigger(
            f"btMapOpen{leaflet.id}", args=item.item_id
        )
        await user.should_see("Sample rate")
        assert "Edit" not in self._labels(user)


class TestResultsCsv:
    async def test_the_shown_results_export_as_csv(
        self, user: User, installed_state
    ):
        import csv
        import io

        from nicegui.testing.user_interaction import UserInteraction

        # Two items, one of them outside the filter: only what was shown goes.
        TestResultMap._store(installed_state, (-26.3, -48.85), (-25.4, -49.2))
        await user.open("/search")
        await user.should_see("Find similar samples")
        TestResultMap._centre(user).value = "-26.3, -48.85"
        TestResultMap._numbers(user)["Radius"].value = 10.0
        TestResultMap._run_search(user)
        await user.should_see("1 match(es)")

        UserInteraction(
            user, set(user.find(marker="export-results-csv").elements), None
        ).click()
        response = await user.download.next(timeout=10.0)

        rows = list(csv.DictReader(io.StringIO(response.content.decode("utf-8"))))
        assert len(rows) == 1
        assert rows[0]["rank"] == "1"
        assert (rows[0]["latitude"], rows[0]["longitude"]) == ("-26.3", "-48.85")
        # Measured from the filter's centre, as the card says.
        assert float(rows[0]["distance_km"]) == pytest.approx(0.0, abs=0.01)
