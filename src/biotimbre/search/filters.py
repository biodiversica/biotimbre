"""Metadata filters that narrow the candidate set before any vector is scored.

Filtering first is the reason metadata and vectors share one database (PLAN.md
D1): "cicadidae within 50 km" may be a few hundred windows out of tens of
thousands, and scoring only those is both faster and more correct — a top-k
computed over the whole library then filtered is a top-k of the wrong population.

Location filtering is two-stage: a bounding box in SQL, which an index can serve,
then exact great-circle distance in Python. The box is a superset of the circle,
so the refinement only ever removes rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import (
    validate_date,
    normalize_taxa,
    parse_biotimbre_id,
)

#: Mean Earth radius, for great-circle distances.
EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance between two points, in kilometres.

    Args:
        lat1: First latitude in degrees.
        lon1: First longitude in degrees.
        lat2: Second latitude in degrees.
        lon2: Second longitude in degrees.

    Returns:
        Distance in kilometres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def _bounding_box(
    latitude: float, longitude: float, radius_km: float
) -> tuple[float, float, float, float]:
    """Latitude/longitude box enclosing a circle.

    The box is deliberately a superset of the circle: it is cheap and
    index-friendly, and the exact distance check afterwards removes the corners.

    Longitude degrees shrink with latitude, so the longitude span is divided by
    ``cos(latitude)``. Near the poles that denominator collapses, so the span
    widens to the whole world rather than dividing by ~zero.

    Args:
        latitude: Centre latitude in degrees.
        longitude: Centre longitude in degrees.
        radius_km: Radius in kilometres.

    Returns:
        ``(min_lat, max_lat, min_lon, max_lon)``.
    """
    lat_delta = math.degrees(radius_km / EARTH_RADIUS_KM)
    min_lat = max(-90.0, latitude - lat_delta)
    max_lat = min(90.0, latitude + lat_delta)

    cos_lat = math.cos(math.radians(latitude))
    if abs(cos_lat) < 1e-6:
        # A circle around a pole spans every longitude.
        return min_lat, max_lat, -180.0, 180.0

    lon_delta = math.degrees(radius_km / (EARTH_RADIUS_KM * abs(cos_lat)))
    if lon_delta >= 180.0:
        return min_lat, max_lat, -180.0, 180.0
    return min_lat, max_lat, longitude - lon_delta, longitude + lon_delta


@dataclass(frozen=True)
class SearchFilters:
    """Metadata constraints applied before scoring.

    All fields are optional; an empty instance matches the whole library.

    Attributes:
        taxa: Tags an item must carry. With ``taxa_match_all`` false (default) an
            item matching any tag qualifies.
        taxa_match_all: Require every listed tag rather than any.
        biotimbre_ids: Restrict to these biotimbres.
        date_from: Earliest recording date, ``YYYY-MM-DD`` inclusive.
        date_to: Latest recording date, ``YYYY-MM-DD`` inclusive.
        latitude: Centre latitude for a radius filter.
        longitude: Centre longitude for a radius filter.
        radius_km: Radius in kilometres. Requires latitude and longitude.
        device: Case-insensitive substring match on the device field.
        speeds: Restrict to items indexed at these speed factors.
        exclude_item_ids: Items to omit, e.g. the query's own item.
    """

    taxa: tuple[str, ...] = ()
    taxa_match_all: bool = False
    biotimbre_ids: tuple[str, ...] = ()
    date_from: str | None = None
    date_to: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    radius_km: float | None = None
    device: str | None = None
    speeds: tuple[float, ...] = ()
    exclude_item_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "taxa", normalize_taxa(self.taxa))
        object.__setattr__(
            self,
            "biotimbre_ids",
            tuple(parse_biotimbre_id(b) for b in self.biotimbre_ids),
        )
        if self.date_from is not None:
            object.__setattr__(self, "date_from", validate_date(self.date_from))
        if self.date_to is not None:
            object.__setattr__(self, "date_to", validate_date(self.date_to))
        if (
            self.date_from is not None
            and self.date_to is not None
            and self.date_from > self.date_to
        ):
            raise ValidationError(
                f"date_from ({self.date_from}) is after date_to ({self.date_to}).",
                "date_range",
            )

        location_parts = (self.latitude, self.longitude, self.radius_km)
        given = [p is not None for p in location_parts]
        if any(given) and not all(given):
            raise ValidationError(
                "A location filter needs latitude, longitude and radius_km "
                "together.",
                "location",
            )
        if self.radius_km is not None and self.radius_km <= 0:
            raise ValidationError(
                f"radius_km must be positive, got {self.radius_km}.", "radius_km"
            )

    @property
    def is_empty(self) -> bool:
        """True when nothing is constrained, so the whole library qualifies."""
        return (
            not self.taxa
            and not self.biotimbre_ids
            and self.date_from is None
            and self.date_to is None
            and self.radius_km is None
            and self.device is None
            and not self.speeds
            and not self.exclude_item_ids
        )

    @property
    def has_location(self) -> bool:
        """True when a radius filter is configured."""
        return self.radius_km is not None


def build_candidate_query(filters: SearchFilters) -> tuple[str, list[object]]:
    """Build the SQL that selects candidate items.

    Location is handled as a bounding box here; the exact circle is applied by
    :func:`refine_by_distance` afterwards.

    Args:
        filters: The constraints to apply.

    Returns:
        A SELECT statement and its parameters. The statement returns
        ``item_id``, ``latitude`` and ``longitude``.
    """
    clauses: list[str] = []
    params: list[object] = []

    if filters.taxa:
        placeholders = ", ".join("?" * len(filters.taxa))
        if filters.taxa_match_all:
            clauses.append(
                f"(SELECT COUNT(DISTINCT tag) FROM item_taxa t "
                f" WHERE t.item_id = i.item_id AND t.tag IN ({placeholders})) = ?"
            )
            params.extend(filters.taxa)
            params.append(len(filters.taxa))
        else:
            clauses.append(
                f"EXISTS (SELECT 1 FROM item_taxa t WHERE t.item_id = i.item_id "
                f"AND t.tag IN ({placeholders}))"
            )
            params.extend(filters.taxa)

    if filters.biotimbre_ids:
        placeholders = ", ".join("?" * len(filters.biotimbre_ids))
        clauses.append(f"i.biotimbre_id IN ({placeholders})")
        params.extend(filters.biotimbre_ids)

    # Lexicographic comparison is chronological because dates are stored
    # zero-padded (see domain.item.validate_date).
    if filters.date_from is not None:
        clauses.append("i.date >= ?")
        params.append(filters.date_from)
    if filters.date_to is not None:
        clauses.append("i.date <= ?")
        params.append(filters.date_to)

    if filters.has_location:
        min_lat, max_lat, min_lon, max_lon = _bounding_box(
            float(filters.latitude), float(filters.longitude), float(filters.radius_km)
        )
        clauses.append("i.latitude BETWEEN ? AND ?")
        params.extend([min_lat, max_lat])
        if (min_lon, max_lon) != (-180.0, 180.0):
            if min_lon < -180.0 or max_lon > 180.0:
                # The box crosses the antimeridian, so the longitude range wraps
                # and has to be expressed as two spans.
                clauses.append("(i.longitude >= ? OR i.longitude <= ?)")
                params.extend(
                    [
                        min_lon + 360.0 if min_lon < -180.0 else min_lon,
                        max_lon - 360.0 if max_lon > 180.0 else max_lon,
                    ]
                )
            else:
                clauses.append("i.longitude BETWEEN ? AND ?")
                params.extend([min_lon, max_lon])

    if filters.device is not None:
        clauses.append("i.device IS NOT NULL AND LOWER(i.device) LIKE ?")
        params.append(f"%{filters.device.strip().lower()}%")

    if filters.speeds:
        placeholders = ", ".join("?" * len(filters.speeds))
        clauses.append(f"i.speed IN ({placeholders})")
        params.extend([float(s) for s in filters.speeds])

    if filters.exclude_item_ids:
        placeholders = ", ".join("?" * len(filters.exclude_item_ids))
        clauses.append(f"i.item_id NOT IN ({placeholders})")
        params.extend([int(i) for i in filters.exclude_item_ids])

    sql = "SELECT i.item_id, i.latitude, i.longitude FROM items i"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY i.item_id"
    return sql, params


def refine_by_distance(
    rows: list[tuple[int, float, float]], filters: SearchFilters
) -> tuple[list[int], dict[int, float]]:
    """Apply the exact circle to bounding-box survivors.

    Args:
        rows: ``(item_id, latitude, longitude)`` from the candidate query.
        filters: The constraints, for the circle's centre and radius.

    Returns:
        Surviving item IDs, and their distances in kilometres from the centre.
        The distance map is empty when no location filter is set.
    """
    if not filters.has_location:
        return [row[0] for row in rows], {}

    centre_lat = float(filters.latitude)
    centre_lon = float(filters.longitude)
    radius = float(filters.radius_km)

    kept: list[int] = []
    distances: dict[int, float] = {}
    for item_id, lat, lon in rows:
        km = haversine_km(centre_lat, centre_lon, lat, lon)
        if km <= radius:
            kept.append(item_id)
            distances[item_id] = km
    return kept, distances


def select_candidates(
    conn, filters: SearchFilters
) -> tuple[list[int] | None, dict[int, float]]:
    """Resolve the candidate item set for a search.

    Args:
        conn: An open database connection.
        filters: The constraints to apply.

    Returns:
        ``candidate_ids`` — item IDs to score, or ``None`` meaning "no filter, so
        stream everything"; and their distances from a location filter's centre.

        ``None`` and ``[]`` are deliberately different: ``None`` means unfiltered,
        while an empty list means a filter matched nothing and no vector should be
        scored at all.
    """
    if filters.is_empty:
        return None, {}

    sql, params = build_candidate_query(filters)
    rows = [
        (int(row["item_id"]), float(row["latitude"]), float(row["longitude"]))
        for row in conn.execute(sql, params)
    ]
    return refine_by_distance(rows, filters)
