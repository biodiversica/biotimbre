"""Exception hierarchy.

All library errors derive from :class:`BiotimbreError`, so a frontend can catch
that one type and show the message to the user. Messages are written to be
user-facing: they name the offending field and value.
"""

from __future__ import annotations


class BiotimbreError(Exception):
    """Base class for every error raised by this library."""


class ValidationError(BiotimbreError):
    """A supplied value is not acceptable.

    Attributes:
        field: Name of the offending field, or ``None`` for whole-record errors.
    """

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field

    def __str__(self) -> str:
        base = super().__str__()
        return f"{self.field}: {base}" if self.field else base


class NotFoundError(BiotimbreError):
    """A referenced record does not exist."""


class DuplicateError(BiotimbreError):
    """A record would collide with one that already exists."""


class EmbeddingError(BiotimbreError):
    """Embedding computation or storage failed."""
