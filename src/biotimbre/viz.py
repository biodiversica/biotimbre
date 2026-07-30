"""Spectrogram rendering for the interface.

Two properties matter more than looking pretty:

**Computed at the file's native rate.** Not from the model-resampled audio. A
sample whose energy sits at 35 kHz is inaudible *and* invisible in anything
resampled to 32 kHz — and being able to see what you cannot hear is the entire
reason the search page shows spectrograms rather than waveforms. The frequency
axis therefore runs to the file's own Nyquist, and the model limits are drawn on
top so it is obvious which part of the signal the embeddings actually saw.

**Time-accurate.** The image maps linearly from x to seconds, so a segment picker
beneath it lines up and a matched window can be marked at the right place.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biotimbre.domain.errors import ValidationError

#: Rendered size. Wide enough to resolve a few hundred milliseconds per pixel on
#: a long file, short enough to sit in a result row.
DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 260

#: Smaller size for result rows, where many images share one page.
THUMBNAIL_WIDTH = 450
THUMBNAIL_HEIGHT = 130

#: Palette size. See :func:`_encode_png` — beyond about 32 the file stops
#: shrinking, and 64 leaves the intensity ramp smooth.
PALETTE_COLOURS = 64

#: Dynamic range shown, in dB below the loudest bin. Bioacoustic recordings carry
#: a lot of low-level noise; 70 dB keeps faint calls visible without rendering
#: the noise floor as signal.
DEFAULT_DYNAMIC_RANGE_DB = 70.0

#: FFT size at 32 kHz, scaled for other rates so frequency resolution stays
#: roughly constant regardless of the file's sample rate.
_REFERENCE_NFFT = 1024
_REFERENCE_RATE = 32_000

#: Decoding cap. A spectrogram of an hour-long file cannot show useful detail at
#: 900 px anyway, and reading it all would stall the page.
MAX_RENDER_SECONDS = 600.0


@dataclass(frozen=True)
class Spectrogram:
    """A rendered spectrogram and the axes needed to interpret it.

    Attributes:
        png: PNG bytes, ready for a data URI or an HTTP response.
        width: Image width in pixels.
        height: Image height in pixels.
        start_time: Seconds into the file at the left edge.
        end_time: Seconds into the file at the right edge.
        max_frequency: Frequency at the top edge, in Hz (the file's Nyquist).
        sample_rate: Native sample rate of the source audio.
    """

    png: bytes
    width: int
    height: int
    start_time: float
    end_time: float
    max_frequency: float
    sample_rate: int

    @property
    def duration(self) -> float:
        """Seconds spanned by the image."""
        return self.end_time - self.start_time

    def time_to_fraction(self, seconds: float) -> float:
        """Map a time in seconds to a 0–1 position across the image.

        Used to place a segment selection or a matched-window marker.

        Args:
            seconds: Time within the file.

        Returns:
            Clamped fraction of the image width.
        """
        if self.duration <= 0:
            return 0.0
        return min(1.0, max(0.0, (seconds - self.start_time) / self.duration))

    def frequency_to_fraction(self, hz: float) -> float:
        """Map a frequency to a 0–1 position from the bottom of the image.

        Used to draw a model's Nyquist limit across the spectrogram.

        Args:
            hz: Frequency in Hz.

        Returns:
            Clamped fraction of the image height.
        """
        if self.max_frequency <= 0:
            return 0.0
        return min(1.0, max(0.0, hz / self.max_frequency))

    def data_uri(self) -> str:
        """The image as a ``data:`` URI, for embedding directly in the page."""
        import base64

        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


def _nfft_for(sample_rate: int) -> int:
    """FFT size giving roughly constant frequency resolution across rates."""
    scaled = _REFERENCE_NFFT * max(1.0, sample_rate / _REFERENCE_RATE)
    # Round to a power of two, bounded so a 192 kHz file stays tractable.
    power = int(round(np.log2(scaled)))
    return int(2 ** min(14, max(8, power)))


def _magnitude_db(
    samples: np.ndarray, sample_rate: int, width: int, height: int
) -> np.ndarray:
    """Compute a dB-scaled magnitude spectrogram resampled to the image grid.

    Args:
        samples: 1-D audio.
        sample_rate: Native rate in Hz.
        width: Target columns.
        height: Target rows.

    Returns:
        A ``(height, width)`` array in dB, with row 0 the lowest frequency.
    """
    from scipy.signal import spectrogram as scipy_spectrogram

    nfft = _nfft_for(sample_rate)
    # Enough hop to produce at least `width` columns, so the image is not
    # upsampled from fewer frames than it has pixels.
    hop = max(1, samples.size // max(1, width))
    nperseg = min(nfft, max(64, samples.size))
    noverlap = max(0, nperseg - hop) if nperseg > hop else 0

    _, _, power = scipy_spectrogram(
        samples,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=max(nperseg, nfft),
        scaling="spectrum",
        mode="magnitude",
    )

    decibels = 20.0 * np.log10(np.maximum(power, 1e-10))

    # Resample onto the pixel grid by index selection: cheap, and avoids
    # smearing narrow-band calls the way interpolation would.
    rows = np.linspace(0, decibels.shape[0] - 1, height).round().astype(int)
    cols = np.linspace(0, decibels.shape[1] - 1, width).round().astype(int)
    return decibels[np.ix_(rows, cols)]


def _levels(
    decibels: np.ndarray, dynamic_range_db: float
) -> tuple[float, float]:
    """Choose the dB window to display, from the data rather than from the peak.

    A fixed window below the peak does not work on field recordings. Measured on
    the example samples, the *median* bin sits only 27 dB (GUYVIR) and 37 dB
    (ZAMTYM) below the loudest one, so a 70 dB window maps the noise floor to 60 %
    brightness and the image saturates into a wash with no visible structure.

    Instead the floor comes from a percentile, which puts the noise floor at black
    whatever its absolute level, and the top comes from a high percentile so a
    single hot bin cannot compress everything else. ``dynamic_range_db`` becomes a
    *cap* on the window, preventing a nearly-silent file from being stretched into
    amplified noise.

    A consequence worth knowing: scaling is per-image, so two spectrograms are not
    comparable in absolute level. That is the right trade for visual
    identification, and is what audio editors do by default.

    Args:
        decibels: ``(height, width)`` dB array.
        dynamic_range_db: Largest window to display, in dB.

    Returns:
        ``(floor, top)`` in dB.
    """
    if decibels.size == 0:
        return -1.0, 0.0
    top = float(np.percentile(decibels, 99.5))
    floor = float(np.percentile(decibels, 50.0))
    # Never show more range than requested, and never a degenerate window.
    floor = max(floor, top - dynamic_range_db)
    if top - floor < 1.0:
        floor = top - 1.0
    return floor, top


def _colourise(decibels: np.ndarray, dynamic_range_db: float) -> np.ndarray:
    """Map dB values to RGB using a perceptually monotonic dark-to-bright ramp.

    A monotonic ramp matters: with a non-monotonic palette, two very different
    intensities can render as similar colours and a faint call becomes ambiguous.

    Args:
        decibels: ``(height, width)`` dB array.
        dynamic_range_db: Largest dB window to display. See :func:`_levels`.

    Returns:
        A ``(height, width, 3)`` uint8 array.
    """
    floor, top = _levels(decibels, dynamic_range_db)
    normalised = np.clip((decibels - floor) / max(1e-6, top - floor), 0.0, 1.0)

    # Dark blue -> magenta -> orange -> pale yellow. Brightness rises throughout,
    # so intensity reads correctly even in greyscale or for colour-blind viewers.
    stops = np.array(
        [
            [12, 8, 38],
            [64, 12, 96],
            [148, 26, 96],
            [222, 82, 52],
            [252, 168, 60],
            [255, 244, 200],
        ],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, len(stops))
    flat = normalised.ravel()
    rgb = np.stack(
        [np.interp(flat, positions, stops[:, channel]) for channel in range(3)],
        axis=-1,
    )
    return rgb.reshape(*normalised.shape, 3).round().astype(np.uint8)


def _encode_png(rgb: np.ndarray, colours: int = PALETTE_COLOURS) -> bytes:
    """Encode an RGB array as a palette PNG.

    Palette mode rather than truecolour, because a noisy spectrogram is
    high-entropy and barely compresses as RGB. Measured on a real sample at
    900x260: 495 KB truecolour versus 115 KB palettised — and a search page shows
    one per result row. The saving comes from storing one byte per pixel instead
    of three, so it is lossless in practice at this palette size; beyond about 32
    colours the file size stops changing.

    Args:
        rgb: ``(height, width, 3)`` uint8 array, row 0 at the top.
        colours: Palette size.

    Returns:
        PNG bytes.
    """
    from PIL import Image

    image = Image.fromarray(rgb, mode="RGB").quantize(
        colors=colours, method=Image.Quantize.FASTOCTREE
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_spectrogram(
    path: str | Path,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    dynamic_range_db: float = DEFAULT_DYNAMIC_RANGE_DB,
) -> Spectrogram:
    """Render an audio file, or part of it, as a spectrogram image.

    Args:
        path: Audio file to render.
        start_time: Start of the span to render. ``None`` starts at the beginning.
        end_time: End of the span. ``None`` runs to the end, capped at
            :data:`MAX_RENDER_SECONDS`.
        width: Image width in pixels.
        height: Image height in pixels.
        dynamic_range_db: dB below the peak to display.

    Returns:
        The rendered :class:`Spectrogram`.

    Raises:
        ValidationError: If the audio cannot be read or the span is empty.
    """
    from biotimbre.embeddings.audio import load_mono, read_properties

    if width < 16 or height < 16:
        raise ValidationError(
            f"Spectrogram must be at least 16x16, got {width}x{height}.", "spectrogram"
        )

    properties = read_properties(path)
    span_start = 0.0 if start_time is None else max(0.0, float(start_time))
    span_end = (
        min(properties.duration_s, span_start + MAX_RENDER_SECONDS)
        if end_time is None
        else min(float(end_time), properties.duration_s)
    )
    if span_end <= span_start:
        raise ValidationError(
            f"Nothing to render between {span_start} s and {span_end} s.",
            "spectrogram",
        )

    # Channels are averaged, matching what the embedding pipeline sees, so the
    # picture corresponds to what the models were given.
    samples, sample_rate = load_mono(
        path, start_time=span_start, end_time=span_end
    )
    if samples.size == 0:
        raise ValidationError("Selected span contains no audio.", "spectrogram")

    decibels = _magnitude_db(samples, sample_rate, width, height)
    # Flip so low frequencies sit at the bottom of the image, as convention has it.
    rgb = _colourise(decibels, dynamic_range_db)[::-1]

    return Spectrogram(
        png=_encode_png(rgb),
        width=width,
        height=height,
        start_time=span_start,
        end_time=span_end,
        max_frequency=sample_rate / 2.0,
        sample_rate=sample_rate,
    )
