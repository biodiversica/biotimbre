"""Domain types and validation for the biotimbre library."""

from biotimbre.domain.errors import (
    BiotimbreError,
    DuplicateAudioError,
    DuplicateError,
    EmbeddingError,
    NotFoundError,
    ValidationError,
)
from biotimbre.domain.item import (
    FilteredBand,
    ItemDraft,
    LibraryItem,
    Location,
    SampleWindow,
    generate_biotimbre_id,
    normalize_taxa,
    parse_biotimbre_id,
)

__all__ = [
    "BiotimbreError",
    "DuplicateAudioError",
    "DuplicateError",
    "EmbeddingError",
    "FilteredBand",
    "ItemDraft",
    "LibraryItem",
    "Location",
    "NotFoundError",
    "SampleWindow",
    "ValidationError",
    "generate_biotimbre_id",
    "normalize_taxa",
    "parse_biotimbre_id",
]
