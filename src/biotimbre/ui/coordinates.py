"""One input for a location, typed as ``latitude, longitude``.

A pair of coordinates is copied and pasted as one string far more often than
it is typed digit by digit — map services, GPS apps and field notes all hand
it over that way — and two boxes made every paste a job of splitting it by
hand. Those sources do not agree on a separator, so a comma, a semicolon or
plain spaces are all taken; the numbers themselves are read by
:meth:`~biotimbre.domain.item.Location.parse`, as the sidecar format's are.
"""

from __future__ import annotations

import re

from nicegui import ui

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import Location

#: Decimal places kept when coordinates are written into the input: about
#: 10 cm, finer than any recording location is known.
DECIMALS = 6

#: Between the two numbers: a comma or a semicolon, spaced or not, or spaces
#: alone. A decimal comma is not among them — ``-26,3 -48,85`` would be four
#: numbers, and is refused rather than misread.
_SEPARATOR = re.compile(r"\s*[,;]\s*|\s+")


def format_coordinates(latitude: float, longitude: float) -> str:
    """Write a pair of coordinates the way the input reads them back."""
    return f"{round(latitude, DECIMALS)}, {round(longitude, DECIMALS)}"


class CoordinatesInput(ui.input):
    """A text input holding ``latitude, longitude`` in decimal degrees."""

    def __init__(
        self, label: str, coordinates: tuple[float, float] | None = None
    ) -> None:
        """Build the input.

        Args:
            label: The input's label.
            coordinates: ``(latitude, longitude)`` to start with, or ``None``
                to start blank.
        """
        super().__init__(
            label,
            value=format_coordinates(*coordinates) if coordinates else "",
            placeholder="-26.304412, -48.85",
        )

    def location(self) -> Location | None:
        """The typed location, read strictly.

        Returns:
            The location, or ``None`` if the input is blank.

        Raises:
            ValidationError: If the text is not two separated numbers, or
                either is out of range.
        """
        text = (self.value or "").strip()
        if not text:
            return None
        parts = _SEPARATOR.split(text)
        if len(parts) != 2:
            raise ValidationError(
                f"{text!r} must be 'latitude, longitude'.", "location"
            )
        return Location.parse(tuple(parts))

    @property
    def coordinates(self) -> tuple[float, float] | None:
        """``(latitude, longitude)``, or ``None`` if blank or not yet valid.

        Lenient, for following what is being typed: halfway through a number
        the text is not a location yet, and that is not an error.
        """
        try:
            location = self.location()
        except ValidationError:
            return None
        return None if location is None else (location.latitude, location.longitude)

    @coordinates.setter
    def coordinates(self, value: tuple[float, float] | None) -> None:
        self.value = "" if value is None else format_coordinates(*value)
