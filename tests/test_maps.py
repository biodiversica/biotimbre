"""The location map: what it plots, how it frames it, and its off switch."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("nicegui", reason="needs the 'gui' extra")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

from biotimbre.config import DEFAULT_TILE_URL  # noqa: E402
from biotimbre.ui import maps  # noqa: E402
from biotimbre.ui.state import AppState, get_state, set_state  # noqa: E402


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


def leaflets(user: User):
    """Every Leaflet element on the page the user is looking at."""
    return list(user.find(ui.leaflet).elements)


class TestFitView:
    def test_no_points_shows_the_whole_world(self):
        assert maps.fit_view([]) == (maps.WORLD_VIEW, maps.WORLD_ZOOM)

    def test_one_point_is_centred_on_itself(self):
        centre, zoom = maps.fit_view([(-26.3, -48.85)])
        assert centre == pytest.approx((-26.3, -48.85))
        # Nothing to fit, so the zoom is the chosen one rather than computed.
        assert zoom == maps.SINGLE_POINT_ZOOM

    def test_repeated_coordinates_count_as_one_point(self):
        # Ten recordings from the same clearing have no extent between them;
        # fitting a zero-width box would divide by zero.
        centre, zoom = maps.fit_view([(-26.3, -48.85)] * 10)
        assert centre == pytest.approx((-26.3, -48.85))
        assert zoom == maps.SINGLE_POINT_ZOOM

    def test_centres_between_two_points(self):
        centre, _ = maps.fit_view([(-10.0, -50.0), (-30.0, -40.0)])
        assert centre == pytest.approx((-20.0, -45.0))

    def test_a_wider_spread_zooms_further_out(self):
        _, near = maps.fit_view([(-26.0, -48.0), (-26.2, -48.2)])
        _, far = maps.fit_view([(-10.0, -70.0), (-40.0, -35.0)])
        assert far < near

    def test_a_pole_to_pole_span_is_lifted_to_the_world_zoom(self):
        # Unclamped this computes 0, which Leaflet will not accept. A
        # hemisphere does not reach the clamp, so it cannot stand in here.
        _, zoom = maps.fit_view([(-85.0, 0.0), (85.0, 0.1)])
        assert zoom == maps.WORLD_ZOOM

    def test_close_points_are_not_zoomed_to_the_street(self):
        # Two recordings metres apart should not fill the view with a street
        # map; a little context is the more honest picture.
        _, zoom = maps.fit_view([(-26.3000, -48.8500), (-26.3001, -48.8501)])
        assert zoom == maps.FIT_MAX_ZOOM

    def test_latitude_uses_mercator_not_degrees(self):
        # The same degree span is taller in pixels near the pole, so it must
        # not be framed identically to one on the equator.
        _, equator = maps.fit_view([(0.0, 0.0), (10.0, 0.0)])
        _, polar = maps.fit_view([(70.0, 0.0), (80.0, 0.0)])
        assert polar < equator


class TestTileLayer:
    async def test_the_configured_tiles_replace_nicegui_s_default(
        self, user: User, installed_state
    ):
        # NiceGUI's Leaflet installs an OSM layer of its own, at a deprecated
        # hostname and ignoring the configured URL. Only ours should survive.
        @ui.page("/onlymap")
        def page() -> None:
            maps.LocationMap()

        await user.open("/onlymap")
        layers = leaflets(user)[0].layers
        tiles = [
            layer for layer in layers if layer.to_dict()["type"] == "tileLayer"
        ]
        assert len(tiles) == 1
        assert tiles[0].url_template == DEFAULT_TILE_URL
        assert "OpenStreetMap" in tiles[0].options["attribution"]

    async def test_a_custom_tile_server_is_used(self, user: User, installed_state):
        installed_state.settings = dataclasses.replace(
            installed_state.settings,
            map_tile_url="http://localhost:8088/{z}/{x}/{y}.png",
            map_attribution="Local tiles",
            map_max_zoom=14,
        )

        @ui.page("/customtiles")
        def page() -> None:
            maps.LocationMap()

        await user.open("/customtiles")
        tiles = [
            layer
            for layer in leaflets(user)[0].layers
            if layer.to_dict()["type"] == "tileLayer"
        ]
        assert tiles[0].url_template == "http://localhost:8088/{z}/{x}/{y}.png"
        assert tiles[0].options == {"attribution": "Local tiles", "maxZoom": 14}


class TestPoints:
    async def test_points_are_drawn_and_replaced_not_accumulated(
        self, user: User, installed_state
    ):
        holder = {}

        @ui.page("/points")
        def page() -> None:
            holder["map"] = maps.LocationMap()

        await user.open("/points")
        location_map = holder["map"]

        location_map.show(
            [maps.MapPoint(-26.0, -48.0, "one"), maps.MapPoint(-25.0, -47.0, "two")]
        )
        assert len(location_map._points) == 2

        # A second search replaces the first one's dots rather than piling on.
        location_map.show([maps.MapPoint(-26.0, -48.0, "one")])
        assert len(location_map._points) == 1
        markers = [
            layer
            for layer in location_map.element.layers
            if layer.to_dict()["type"] == "circleMarker"
        ]
        assert len(markers) == 1
        assert markers[0].to_dict()["args"][1]["className"] == maps.POINT_CLASS

    async def test_showing_points_moves_the_view_to_them(
        self, user: User, installed_state
    ):
        holder = {}

        @ui.page("/refit")
        def page() -> None:
            holder["map"] = maps.LocationMap()

        await user.open("/refit")
        location_map = holder["map"]
        assert location_map.element.center == maps.WORLD_VIEW

        location_map.show([maps.MapPoint(-26.3, -48.85, "")])
        assert location_map.element.center == pytest.approx((-26.3, -48.85))
        assert location_map.element.zoom == maps.SINGLE_POINT_ZOOM

    async def test_a_label_becomes_an_escaped_popup(
        self, user: User, installed_state
    ):
        holder = {}

        @ui.page("/popup")
        def page() -> None:
            holder["map"] = maps.LocationMap()

        await user.open("/popup")
        location_map = holder["map"]
        location_map._ready = False
        location_map.show([maps.MapPoint(0.0, 0.0, '<img src=x onerror="bad()">')])

        # Labels are built from curator-typed metadata, so they are escaped
        # rather than handed to Leaflet as HTML.
        assert location_map._pending_popups
        _, content = location_map._pending_popups[0]
        assert "<img" not in content
        assert "&lt;img" in content


class TestFocus:
    async def test_a_centre_and_radius_draw_a_dot_and_a_circle(
        self, user: User, installed_state
    ):
        holder = {}

        @ui.page("/focus")
        def page() -> None:
            holder["map"] = maps.LocationMap()

        await user.open("/focus")
        location_map = holder["map"]

        location_map.set_focus(-26.3, -48.85, 25.0)
        types = [layer.to_dict()["type"] for layer in location_map._focus]
        assert types == ["circle", "circleMarker"]
        # Leaflet circles are metres; the filter is kilometres.
        assert location_map._focus[0].to_dict()["args"][1]["radius"] == 25_000.0

    async def test_a_centre_without_a_radius_draws_only_the_dot(
        self, user: User, installed_state
    ):
        holder = {}

        @ui.page("/halffocus")
        def page() -> None:
            holder["map"] = maps.LocationMap()

        await user.open("/halffocus")
        location_map = holder["map"]

        location_map.set_focus(-26.3, -48.85, None)
        assert [layer.to_dict()["type"] for layer in location_map._focus] == [
            "circleMarker"
        ]

    async def test_clearing_the_centre_removes_both(
        self, user: User, installed_state
    ):
        holder = {}

        @ui.page("/clearfocus")
        def page() -> None:
            holder["map"] = maps.LocationMap()

        await user.open("/clearfocus")
        location_map = holder["map"]

        location_map.set_focus(-26.3, -48.85, 25.0)
        location_map.set_focus(None, None, None)
        assert location_map._focus == []
        assert not [
            layer
            for layer in location_map.element.layers
            if layer.to_dict()["type"] == "circle"
        ]
