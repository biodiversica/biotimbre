"""Processing a recording before it is stored: trim, filter, normalise.

The add form in the interface does this with a spectrogram to drag over; this
module is the same work for the command line and for sidecar files, where the
choices arrive as numbers. Both write the stored copy with
:func:`~biotimbre.embeddings.audio.write_processed` and
:func:`~biotimbre.embeddings.audio.normalise_file`, so a recording processed
either way is stored the same.

Processing changes what the stored audio *is*, so it changes what its metadata
must say. :meth:`Processing.describe` makes that correction, by the rules the
interface follows:

* A trim moves where the sample sits in the full recording. ``sample_window``
  describes the file as given; after a trim of ``a..b`` seconds it becomes
  ``start + a .. start + b``, or just ``a .. b`` when the file was the full
  recording (no ``sample_window`` given).
* A filter applied here is filtering "already applied" by the time the sample
  is stored: ``filtered_band`` narrows to the tighter of what was recorded and
  what was applied, and ``filter_type``/``filter_order`` say Butterworth and the
  order used, unless the metadata already names them.
* Normalisation adds one line to ``notes`` saying so and by how much, in the
  words the interface uses.
"""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from biotimbre.domain.errors import ValidationError
from biotimbre.domain.item import FilteredBand, ItemDraft, SampleWindow
from biotimbre.embeddings import audio as audio_io

#: Butterworth order used when none is given, as in the add form.
DEFAULT_FILTER_ORDER = 4

#: Highest filter order accepted. Beyond it a Butterworth filter in
#: second-order sections still works, but nobody needs it and a typo of 40 for
#: 4 should be caught.
MAX_FILTER_ORDER = 16

#: The type recorded for a filter applied here.
FILTER_TYPE = "Butterworth"

#: Slack when checking a trim against the file's length: a duration read from
#: a header and a span typed to two decimals rarely agree to the sample.
_TRIM_EPSILON = 0.01


@dataclass(frozen=True)
class Processing:
    """What to do to a recording before storing it.

    Attributes:
        trim: ``(start, end)`` in seconds within the file: keep only this part.
        highpass: Cutoff in Hz of a zero-phase Butterworth highpass to apply.
        lowpass: Cutoff in Hz of a zero-phase Butterworth lowpass to apply.
        filter_order: Order of that filter.
        normalise: Scale the stored copy so its peak is at -1 dBFS.
    """

    trim: tuple[float, float] | None = None
    highpass: float | None = None
    lowpass: float | None = None
    filter_order: int = DEFAULT_FILTER_ORDER
    normalise: bool = False

    def __post_init__(self) -> None:
        if self.trim is not None:
            start, end = (float(v) for v in self.trim)
            if start < 0:
                raise ValidationError("A trim cannot start before 0 s.", "trim")
            if end <= start:
                raise ValidationError(
                    f"A trim must end after it starts, got {start:g}-{end:g} s.",
                    "trim",
                )
            object.__setattr__(self, "trim", (start, end))
        for name in ("highpass", "lowpass"):
            value = getattr(self, name)
            if value is not None and not float(value) > 0:
                raise ValidationError(f"The {name} cutoff must be positive.", name)
        if (
            self.highpass is not None
            and self.lowpass is not None
            and self.highpass >= self.lowpass
        ):
            raise ValidationError(
                f"The highpass cutoff ({self.highpass:g} Hz) must be below the "
                f"lowpass one ({self.lowpass:g} Hz).",
                "filter",
            )
        if isinstance(self.filter_order, bool) or not (
            1 <= int(self.filter_order) <= MAX_FILTER_ORDER
        ):
            raise ValidationError(
                f"The filter order must be between 1 and {MAX_FILTER_ORDER}.",
                "filter_order",
            )

    @property
    def is_filtered(self) -> bool:
        """Whether a cutoff is to be applied."""
        return self.highpass is not None or self.lowpass is not None

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to do."""
        return self.trim is None and not self.is_filtered and not self.normalise

    def describe(self, draft: ItemDraft, gain_db: float | None) -> ItemDraft:
        """Correct ``draft`` to describe the processed copy instead of the file.

        Args:
            draft: Metadata of the file as given.
            gain_db: What normalisation applied, or ``None`` if it did not run
                (asked for, but the audio was silent).

        Returns:
            The metadata to store the processed copy with.
        """
        changes: dict[str, object] = {}
        if self.trim is not None:
            start, end = self.trim
            offset = draft.sample_window.start if draft.sample_window else 0.0
            changes["sample_window"] = SampleWindow(
                round(offset + start, 3), round(offset + end, 3)
            )
        if self.is_filtered:
            band = draft.filtered_band or FilteredBand()
            low = _tighter(band.low, self.highpass, max)
            high = _tighter(band.high, self.lowpass, min)
            changes["filtered_band"] = FilteredBand(low, high)
            if not draft.filter_type:
                changes["filter_type"] = FILTER_TYPE
                changes["filter_order"] = int(self.filter_order)
            elif draft.filter_order is None:
                changes["filter_order"] = int(self.filter_order)
        if self.normalise and gain_db is not None:
            line = (
                f"Normalised to peak at {audio_io.NORMALISED_PEAK_DBFS:g} dBFS "
                f"(gain {gain_db:+.1f} dB)."
            )
            changes["notes"] = f"{draft.notes}\n{line}" if draft.notes else line
        return dataclasses.replace(draft, **changes)


def _tighter(recorded: float | None, applied: float | None, pick) -> float | None:
    if recorded is None:
        return applied
    if applied is None:
        return recorded
    return pick(recorded, applied)


@contextmanager
def processed(
    source: str | Path, draft: ItemDraft, processing: Processing
) -> Iterator[tuple[Path, ItemDraft]]:
    """Write the processed copy of a recording, for as long as it is needed.

    Args:
        source: The recording as given.
        draft: Its metadata, as given.
        processing: What to do to it. Empty processing yields the source and
            draft untouched, and writes nothing.

    Yields:
        The file to store and the metadata to store it with. The file is
        removed when the block ends: the library keeps its own copy.

    Raises:
        ValidationError: If the trim runs past the end of the file, or a cutoff
            is unusable at its sample rate.
    """
    source = Path(source)
    if processing.is_empty:
        yield source, draft
        return

    if processing.trim is not None:
        duration = audio_io.read_properties(source).duration_s
        if processing.trim[1] > duration + _TRIM_EPSILON:
            raise ValidationError(
                f"The trim ends at {processing.trim[1]:g} s, but {source.name} "
                f"is only {duration:.2f} s long.",
                "trim",
            )

    with tempfile.TemporaryDirectory(prefix="biotimbre-processing-") as directory:
        # The name, not just the format: it is what an import reports, and what
        # an error about this file should call it.
        target = Path(directory) / source.name
        start, end = processing.trim if processing.trim else (None, None)
        audio_io.write_processed(
            source,
            target,
            start_time=start,
            end_time=end,
            low_hz=processing.highpass,
            high_hz=processing.lowpass,
            order=int(processing.filter_order),
        )
        gain = audio_io.normalise_file(target) if processing.normalise else None
        yield target, processing.describe(draft, gain)
