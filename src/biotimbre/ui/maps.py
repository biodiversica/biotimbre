"""The location map: one per page, never one per row.

Every item carries coordinates — the columns are ``NOT NULL`` — so a library is
always plottable, and "where did these come from" is a question the item list
answers badly and a map answers at a glance.

**One map per page.** A mini-map per result row would mean one Leaflet
instance, one DOM subtree and one set of tile requests *per row*, which is how
a twenty-result page becomes slow. A single map showing the whole result set is
both cheaper and more informative: the spread of the matches is the thing worth
seeing, and each row still prints its own coordinates.

**Leaflet is local, tiles are not.** NiceGUI vendors Leaflet itself, so nothing
is fetched from a CDN. The basemap tiles are the exception, and the only reason
this module can reach the network at all: see :attr:`Settings.map_enabled`.

Points are drawn as ``circleMarker`` rather than Leaflet's default pin. They
are plain SVG, so they need no marker images, stay cheap in quantity, and
can be themed from the stylesheet — a pin is a fixed-colour PNG that looks
equally wrong against both palettes.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from nicegui import ui

from biotimbre.i18n import Translator
from biotimbre.ui.coordinates import CoordinatesInput
from biotimbre.ui.state import get_state

#: Height of the map, in pixels. One big map per page, so it can afford to be
#: big enough to read.
DEFAULT_HEIGHT = 420

#: Assumed map width when choosing a zoom. The page is capped at ``max-w-6xl``
#: (1152 px) minus its padding; only the zoom estimate depends on it, and being
#: a little conservative errs towards showing slightly more than asked.
ASSUMED_WIDTH = 1100

#: Leaflet tile size, which fixes the relationship between zoom and pixels.
TILE_SIZE = 256

#: View used when there is nothing to show: the whole world, roughly centred on
#: land rather than on the empty Pacific that (0, 0) gives.
WORLD_VIEW = (20.0, 10.0)
WORLD_ZOOM = 1

#: Zoom used for a single point, which has no extent to fit. Close enough to
#: place a recording in its valley, wide enough not to imply GPS precision the
#: coordinates may not have.
SINGLE_POINT_ZOOM = 11

#: Never zoom past this when fitting several points. Filling the view with a
#: street map because two recordings came from the same clearing is a worse
#: picture than a little surrounding context.
FIT_MAX_ZOOM = 14

#: Height of the map that picks a recording's coordinates. It sits inside a
#: form, between fields, so it is smaller than a page's map.
PICKER_HEIGHT = 300

#: Latitude beyond which Web Mercator stops being defined.
MERCATOR_LIMIT = 85.05112878

#: Kilometres per degree of latitude. A sphere, not the ellipsoid: the error is
#: a few parts in a thousand, and this only decides a framing.
KM_PER_DEGREE = 111.32

#: How much wider than the radius circle its frame is drawn. A circle flush
#: against the edges of the map reads as something cut off rather than as the
#: area being searched.
FOCUS_PADDING = 1.25

#: Floor on ``cos(latitude)`` when converting kilometres to degrees of
#: longitude. At the poles the true factor is zero and the conversion diverges;
#: this caps the box at a few thousand kilometres wide instead.
MIN_COSINE = 0.01

#: Point and radius styling. Colours live in the stylesheet rather than in the
#: layer options because Leaflet writes them as SVG presentation attributes,
#: where ``var(--bt-primary)`` does not resolve; a class does.
POINT_CLASS = "bt-map-point"
FOCUS_CLASS = "bt-map-focus"
RADIUS_CLASS = "bt-map-radius"

_MAP_CSS = """
.bt-map-point {
  stroke: var(--bt-card);
  fill: var(--bt-primary);
  fill-opacity: 0.85;
}
.bt-map-focus {
  stroke: var(--bt-accent);
  fill: var(--bt-accent);
  fill-opacity: 0.9;
}
.bt-map-radius {
  stroke: var(--bt-accent);
  fill: var(--bt-accent);
  fill-opacity: 0.12;
}
"""


#: Lets the map have ctrl+wheel and the page have every other wheel. A map in a
#: scrolling page that eats the wheel is a nuisance, but with Leaflet's wheel
#: zoom simply off, ctrl+wheel went to the browser and zoomed the whole page.
#: Captured on the container, ahead of Leaflet's own listener: a plain wheel is
#: stopped before Leaflet sees it and left to scroll the page, and a ctrl+wheel
#: — which is also what a trackpad pinch sends — is kept from the browser and
#: let through to zoom the map. ``{id}`` is the leaflet element's id.
CTRL_WHEEL_ONLY = """
const el = getHtmlElement({id});
if (el && !el.dataset.ctrlWheel) {{
    el.dataset.ctrlWheel = "1";
    el.addEventListener("wheel", (event) => {{
        if (event.ctrlKey) event.preventDefault();
        else event.stopImmediatePropagation();
    }}, {{ capture: true, passive: false }});
}}
"""


@dataclass(frozen=True)
class MapPoint:
    """One plotted recording.

    Attributes:
        latitude: Degrees north.
        longitude: Degrees east.
        label: Popup text. Plain text, escaped before it reaches the browser.
        item_id: The library item plotted, for a map that can open it.
    """

    latitude: float
    longitude: float
    label: str = ""
    item_id: int | None = None


def enabled() -> bool:
    """Whether the location map is switched on for this library."""
    return get_state().settings.map_enabled


def _mercator_y(latitude: float) -> float:
    """Web Mercator northing, normalised to ``[0, 1]`` from north to south."""
    clamped = max(-MERCATOR_LIMIT, min(MERCATOR_LIMIT, latitude))
    sin = math.sin(math.radians(clamped))
    return 0.5 - math.log((1.0 + sin) / (1.0 - sin)) / (4.0 * math.pi)


def fit_view(
    points: Sequence[tuple[float, float]],
    *,
    width: int = ASSUMED_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> tuple[tuple[float, float], int]:
    """Centre and zoom that bring every point into view.

    Computed here rather than by asking Leaflet to ``fitBounds`` because the
    map is not initialised when a page first renders: a method call at that
    moment is dropped, and the reader watches the world map jump once the
    socket connects. A centre and a zoom are plain props, so they are correct
    in the very first frame.

    Args:
        points: ``(latitude, longitude)`` pairs. Empty gives the world view.
        width: Map width in pixels.
        height: Map height in pixels.

    Returns:
        The centre as ``(latitude, longitude)``, and an integer zoom.
    """
    if not points:
        return WORLD_VIEW, WORLD_ZOOM

    latitudes = [lat for lat, _ in points]
    longitudes = [lon for _, lon in points]
    centre = (
        (min(latitudes) + max(latitudes)) / 2.0,
        (min(longitudes) + max(longitudes)) / 2.0,
    )

    # Normalised spans: longitude is linear in Mercator, latitude is not.
    span_x = (max(longitudes) - min(longitudes)) / 360.0
    span_y = abs(_mercator_y(max(latitudes)) - _mercator_y(min(latitudes)))
    if span_x <= 0.0 and span_y <= 0.0:
        # Every point is in the same place, so there is no extent to fit.
        return centre, SINGLE_POINT_ZOOM

    # The world is TILE_SIZE * 2**zoom pixels across, so a span fits when
    # span * TILE_SIZE * 2**zoom <= the pixels available for it.
    candidates = [
        math.log2(pixels / (TILE_SIZE * span))
        for span, pixels in ((span_x, width), (span_y, height))
        if span > 0.0
    ]
    zoom = int(math.floor(min(candidates)))
    return centre, max(WORLD_ZOOM, min(FIT_MAX_ZOOM, zoom))


def radius_box(
    latitude: float, longitude: float, radius_km: float
) -> list[tuple[float, float]]:
    """Corners of a box enclosing the circle of ``radius_km`` about a centre.

    Framing a circle is framing its bounding box, and :func:`fit_view` already
    knows how to frame a set of points. The box is padded so the circle sits
    inside the map rather than against its edges.

    Args:
        latitude: Centre latitude in degrees.
        longitude: Centre longitude in degrees.
        radius_km: Radius in kilometres.

    Returns:
        The south-west and north-east corners as ``(latitude, longitude)``.
    """
    reach = abs(radius_km) * FOCUS_PADDING
    span_lat = reach / KM_PER_DEGREE
    # A degree of longitude shrinks towards the poles, so the same radius is
    # more degrees wide the further north or south the centre is.
    cosine = max(MIN_COSINE, math.cos(math.radians(latitude)))
    span_lon = min(180.0, reach / (KM_PER_DEGREE * cosine))
    return [
        (max(-90.0, latitude - span_lat), longitude - span_lon),
        (min(90.0, latitude + span_lat), longitude + span_lon),
    ]


def wrap_longitude(longitude: float) -> float:
    """Bring a longitude back into ``[-180, 180)``.

    With ``worldCopyJump`` the world repeats sideways, and a click on one of
    the copies reports a longitude like 311.5 — the right place, but one the
    location validation rightly refuses.
    """
    return (longitude + 180.0) % 360.0 - 180.0


class LocationMap:
    """A single map, redrawable with a new set of points.

    Args:
        on_pick: Called with ``(latitude, longitude)`` when the map is clicked.
            ``None`` leaves the map read-only.
        on_open: Called with a point's ``item_id`` when the link in its popup
            is followed. ``None`` leaves popups as labels alone.
        open_label: Translated text of that link.
    """

    def __init__(
        self,
        *,
        height: int = DEFAULT_HEIGHT,
        on_pick: Callable[[float, float], None] | None = None,
        on_open: Callable[[int], object] | None = None,
        open_label: str = "",
    ) -> None:
        settings = get_state().settings
        ui.add_css(_MAP_CSS)

        self._height = height
        self._points: list = []
        self._focus: list = []
        # The corners the focus contributes to a fit, so that a later redraw of
        # the points keeps the searched area whole rather than cropping it.
        self._focus_bounds: list[tuple[float, float]] = []
        # Layers bound to a popup before the map came up, which have to wait:
        # a method call on an uninitialised map is silently discarded.
        self._pending_popups: list[tuple[object, str]] = []
        self._ready = False

        self.element = ui.leaflet(
            center=WORLD_VIEW,
            zoom=WORLD_ZOOM,
            options={
                # On, but only for ctrl+wheel: see CTRL_WHEEL_ONLY.
                "scrollWheelZoom": True,
                "worldCopyJump": True,
            },
        ).classes("w-full rounded").style(f"height: {height}px")

        # NiceGUI's Leaflet installs its own OSM layer in the constructor, at a
        # deprecated hostname and with no regard for the configured one. It has
        # not been sent to the browser yet, so dropping it costs nothing.
        self.element.clear_layers()
        self.element.tile_layer(
            url_template=settings.map_tile_url,
            options={
                "attribution": settings.map_attribution,
                "maxZoom": settings.map_max_zoom,
            },
        )

        # Popup content is HTML handed to Leaflet, not a NiceGUI element, so a
        # link in it cannot carry a Python handler. It emits a page event
        # instead, named after this map so two maps never answer each other.
        self._open_label = open_label
        self._open_event = f"btMapOpen{self.element.id}" if on_open else None
        if on_open is not None:
            ui.on(self._open_event, lambda event: on_open(int(event.args)))

        # Registered after the element's own handler, so by the time this runs
        # the map counts as initialised and the calls go through.
        self.element.on("init", self._on_init)
        if on_pick is not None:
            self.element.on(
                "map-click",
                lambda event: on_pick(
                    float(event.args["latlng"]["lat"]),
                    wrap_longitude(float(event.args["latlng"]["lng"])),
                ),
            )

    def _on_init(self) -> None:
        """Bind the popups queued while the map was still coming up."""
        self._ready = True
        self.element.client.run_javascript(
            CTRL_WHEEL_ONLY.format(id=self.element.id)
        )
        for layer, html in self._pending_popups:
            layer.run_method("bindPopup", html)
        self._pending_popups.clear()

    def _popup(self, layer, point: MapPoint) -> None:
        """Attach a popup, now or once the map is up.

        Leaflet takes popup content as HTML, and a label is assembled from
        library metadata — taxon names, filenames, whatever a curator typed.
        It is escaped rather than trusted. The link to open the item is built
        from an integer and the map's own event name, never from metadata.
        """
        content = html.escape(point.label)
        if self._open_event is not None and point.item_id is not None:
            link = (
                f'<a href="#" class="text-link" onclick="emitEvent('
                f"'{self._open_event}', {int(point.item_id)}); return false;\">"
                f"{html.escape(self._open_label)}</a>"
            )
            content = f"{content}<br>{link}" if content else link
        if not content:
            return
        if self._ready:
            layer.run_method("bindPopup", content)
        else:
            self._pending_popups.append((layer, content))

    def _move_to(self, points: Sequence[tuple[float, float]]) -> None:
        """Frame ``points``, which may be markers, focus corners or both."""
        centre, zoom = fit_view(points, height=self._height)
        # The bindable attributes rather than set_center/set_zoom: those
        # write the props behind the binding's back, leaving anything
        # bound to the view stale.
        self.element.center = centre
        self.element.zoom = zoom
        # Props alone only reach a map that has yet to be built. For one
        # already on screen, this is what turns them into a setView.
        self.element.update()

    def show(self, points: Sequence[MapPoint], *, refit: bool = True) -> None:
        """Replace the plotted points.

        Args:
            points: What to draw.
            refit: Whether to move the view to cover them, and the searched
                area if one is set. ``False`` leaves the reader where they had
                panned to.
        """
        for layer in self._points:
            self.element.remove_layer(layer)
        self._points.clear()

        with self.element:
            for point in points:
                layer = self.element.generic_layer(
                    name="circleMarker",
                    args=[
                        [point.latitude, point.longitude],
                        {
                            "radius": 6,
                            "weight": 2,
                            "className": POINT_CLASS,
                        },
                    ],
                )
                self._points.append(layer)
                self._popup(layer, point)

        if refit:
            # The focus goes into the fit as well: a search narrowed to a
            # radius would otherwise frame its hits and cut the circle that
            # explains why those are the hits.
            self._move_to(
                [(point.latitude, point.longitude) for point in points]
                + self._focus_bounds
            )

    def set_focus(
        self,
        latitude: float | None,
        longitude: float | None,
        radius_km: float | None,
        *,
        refit: bool = False,
    ) -> None:
        """Draw the search centre and its radius, or clear them.

        This is the location filter made visible: the same three numbers the
        filter panel holds, as a dot and a circle.

        Args:
            latitude: Centre latitude, or ``None`` for no centre.
            longitude: Centre longitude, or ``None`` for no centre.
            radius_km: Radius in kilometres. ``None`` or zero draws the centre
                alone, which is what a half-filled filter deserves.
            refit: Whether to frame the searched area. A circle drawn at the
                world zoom is a few pixels wide and tells the reader nothing,
                so setting a centre is also a request to go and look at it.
                Clearing the centre never moves the view: there is nothing to
                look at, and yanking the map back would lose the reader's place.
        """
        for layer in self._focus:
            self.element.remove_layer(layer)
        self._focus.clear()
        self._focus_bounds = []

        if latitude is None or longitude is None:
            return

        self._focus_bounds = (
            radius_box(latitude, longitude, float(radius_km))
            if radius_km
            else [(latitude, longitude)]
        )

        with self.element:
            if radius_km:
                self._focus.append(
                    self.element.generic_layer(
                        name="circle",
                        args=[
                            [latitude, longitude],
                            {
                                "radius": float(radius_km) * 1000.0,
                                "weight": 1,
                                "className": RADIUS_CLASS,
                            },
                        ],
                    )
                )
            self._focus.append(
                self.element.generic_layer(
                    name="circleMarker",
                    args=[
                        [latitude, longitude],
                        {"radius": 5, "weight": 2, "className": FOCUS_CLASS},
                    ],
                )
            )

        if refit:
            self._move_to(self._focus_bounds)


def map_card(
    t: Translator,
    *,
    heading: str,
    hint: str = "",
    height: int = DEFAULT_HEIGHT,
    on_pick: Callable[[float, float], None] | None = None,
    on_open: Callable[[int], object] | None = None,
    open_label: str = "",
) -> LocationMap:
    """Render a titled card holding one map, and return it.

    Args:
        t: Active translator.
        heading: Translated card heading.
        hint: Translated line under it, if there is anything to say.
        height: Map height in pixels.
        on_pick: Passed through to :class:`LocationMap`.
        on_open: Passed through to :class:`LocationMap`.
        open_label: Passed through to :class:`LocationMap`.

    Returns:
        The map, ready to be given points.
    """
    with ui.card().classes("w-full p-4 gap-2 bg-card"):
        ui.label(heading).classes("text-sm font-medium")
        if hint:
            ui.label(hint).classes("text-xs text-muted")
        return LocationMap(
            height=height, on_pick=on_pick, on_open=on_open, open_label=open_label
        )


def location_picker(
    t: Translator,
    coordinates: CoordinatesInput,
    *,
    height: int = PICKER_HEIGHT,
) -> ui.expansion:
    """A collapsed map that fills in a coordinates input when clicked.

    An option beside the input, not a replacement for it: typing, pasting
    and an iNaturalist fetch still write the field, and the map follows.

    The map is built the first time the panel opens. Most additions never need
    it, so tiles are not fetched for nothing — and Leaflet sizes itself when it
    is built, which inside a closed panel means sizing itself to nothing.

    Args:
        t: Active translator.
        coordinates: The input to fill.
        height: Map height in pixels.

    Returns:
        The expansion panel holding the map.
    """
    panel = ui.expansion(t("map.pick_location"), icon="place").props(
        "dense"
    ).classes("w-full")
    holder: dict = {"map": None, "clicked": False}

    def show_pick() -> None:
        """Draw the current coordinates as a dot.

        A typed or fetched location is framed, since it may be anywhere. A
        clicked one is not: the reader is already looking at it, perhaps
        zoomed in closer than a refit would go, and is likely to click again
        to refine it.
        """
        picker = holder["map"]
        if picker is not None:
            latitude, longitude = coordinates.coordinates or (None, None)
            picker.set_focus(
                latitude, longitude, None, refit=not holder["clicked"]
            )

    def pick(lat: float, lon: float) -> None:
        holder["clicked"] = True
        try:
            coordinates.coordinates = (lat, lon)
        finally:
            holder["clicked"] = False

    def build(event) -> None:
        if not event.value or holder["map"] is not None:
            return
        with panel:
            ui.label(t("map.pick_location_hint")).classes("text-xs text-muted")
            holder["map"] = LocationMap(height=height, on_pick=pick)
        show_pick()

    panel.on_value_change(build)
    coordinates.on_value_change(show_pick)
    return panel
