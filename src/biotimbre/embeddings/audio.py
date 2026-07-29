"""Audio loading, speed shifting, resampling and windowing.

The speed shift (PLAN.md D2b) is a reinterpretation of the source rate, not a
separate DSP stage: playing a file at speed ``s`` means its effective sample rate
is ``sr_native * s``, so we resample from that rate instead of from the native
one. Frequencies scale by ``s``, duration by ``1/s``, and there is still only one
resampling step — no compounded interpolation loss.

Window times are produced in the *shifted* time base and must be converted back
to the sample's own time base before storage, which
:func:`window_times_in_sample_domain` does. Skipping that conversion would make
the player's matched-window marker point at the wrong region of every
speed-shifted item.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from biotimbre.domain.errors import ValidationError

#: Cap on the resampling ratio's denominator. A speed like 0.3333 would
#: otherwise produce an enormous polyphase filter; 1000 keeps the rate error
#: below 0.1% while bounding the filter length.
_MAX_RESAMPLE_DENOMINATOR = 1000

#: Read audio in chunks when hashing, so a large file is not held twice.
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class AudioProperties:
    """Physical properties of a stored audio file.

    Attributes:
        duration_s: Length in seconds at the file's native rate.
        sample_rate: Native sample rate in Hz.
        channels: Channel count as stored.
        n_frames: Length in frames.
    """

    duration_s: float
    sample_rate: int
    channels: int
    n_frames: int


def read_properties(path: str | Path) -> AudioProperties:
    """Read an audio file's properties without decoding its samples.

    Args:
        path: Path to the audio file.

    Returns:
        The file's :class:`AudioProperties`.

    Raises:
        ValidationError: If the file cannot be read as audio, or is empty.
    """
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ValidationError(
            f"Could not read {Path(path).name} as audio: {exc}", "audio"
        ) from exc

    if info.frames <= 0:
        raise ValidationError(f"{Path(path).name} contains no audio.", "audio")

    return AudioProperties(
        duration_s=float(info.frames) / float(info.samplerate),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        n_frames=int(info.frames),
    )


def file_sha256(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes, read incrementally.

    Args:
        path: Path to hash.

    Returns:
        64-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at its native rate.

    Multi-channel input is averaged rather than having one channel picked, so no
    part of a stereo recording's content is discarded.

    Args:
        path: Path to the audio file.

    Returns:
        The samples as 1-D float32, and the native sample rate.

    Raises:
        ValidationError: If the file cannot be decoded or holds no audio.
    """
    import soundfile as sf

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValidationError(
            f"Could not decode {Path(path).name}: {exc}", "audio"
        ) from exc

    if data.size == 0:
        raise ValidationError(f"{Path(path).name} contains no audio.", "audio")

    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate)


def effective_sample_rate(native_rate: int, speed: float) -> float:
    """Sample rate the audio is treated as having under a speed shift.

    Args:
        native_rate: The file's real sample rate in Hz.
        speed: Speed factor; ``<1`` slows down, ``>1`` speeds up.

    Returns:
        ``native_rate * speed``.

    Raises:
        ValidationError: If ``speed`` is not positive.
    """
    if speed <= 0:
        raise ValidationError(f"Speed {speed} must be positive.", "speed")
    return float(native_rate) * float(speed)


def resample(audio: np.ndarray, from_rate: float, to_rate: int) -> np.ndarray:
    """Resample mono audio to a target rate.

    The ratio is derived with :class:`fractions.Fraction` rather than an integer
    GCD, because a speed shift makes ``from_rate`` fractional (e.g. 32000 × 0.25
    is fine, but 32000 × 0.3 is not an integer).

    Args:
        audio: 1-D float32 samples.
        from_rate: Current (possibly fractional) rate in Hz.
        to_rate: Target rate in Hz.

    Returns:
        Resampled 1-D float32 samples. Returned unchanged when the rates already
        match.

    Raises:
        ValidationError: If either rate is not positive.
    """
    if from_rate <= 0 or to_rate <= 0:
        raise ValidationError(
            f"Cannot resample from {from_rate} Hz to {to_rate} Hz.", "audio"
        )
    if abs(from_rate - to_rate) < 1e-9:
        return audio

    from scipy.signal import resample_poly

    ratio = Fraction(to_rate / from_rate).limit_denominator(
        _MAX_RESAMPLE_DENOMINATOR
    )
    up, down = ratio.numerator, ratio.denominator
    if up <= 0 or down <= 0:
        raise ValidationError(
            f"Resampling ratio {to_rate}/{from_rate} is degenerate.", "audio"
        )
    return np.ascontiguousarray(
        resample_poly(audio, up, down), dtype=np.float32
    )


def frame(
    audio: np.ndarray, window_samples: int, stride_samples: int
) -> np.ndarray:
    """Split audio into fixed-length windows.

    A signal shorter than one window is zero-padded up to a single window rather
    than dropped. A trailing partial hop is *not* padded into an extra window, so
    5 s of audio under a 5 s window yields exactly one window at any overlap.

    Args:
        audio: 1-D float32 samples, already at the model's rate.
        window_samples: Window length in samples.
        stride_samples: Hop between window starts, in samples.

    Returns:
        An ``(n_windows, window_samples)`` float32 array.

    Raises:
        ValidationError: If the audio is empty or the geometry is invalid.
    """
    if window_samples <= 0 or stride_samples <= 0:
        raise ValidationError(
            f"Window geometry is invalid: window={window_samples}, "
            f"stride={stride_samples}.",
            "audio",
        )
    if audio.size == 0:
        raise ValidationError("Cannot window empty audio.", "audio")

    if audio.size < window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[: audio.size] = audio
        return padded.reshape(1, window_samples)

    excess = audio.size - window_samples
    n_windows = 1 + -(-excess // stride_samples)

    windows = np.zeros((n_windows, window_samples), dtype=np.float32)
    for index in range(n_windows):
        start = index * stride_samples
        chunk = audio[start : start + window_samples]
        windows[index, : chunk.size] = chunk
    return windows


def window_times_in_sample_domain(
    n_windows: int,
    window_samples: int,
    stride_samples: int,
    model_rate: int,
    speed: float,
) -> list[tuple[float, float]]:
    """Window spans in the sample's own time base, undoing the speed shift.

    The extractor works in the shifted domain, where a window starts at
    ``index * stride / model_rate`` seconds. At speed ``s`` the shifted timeline
    is ``1/s`` times longer than the real sample, so multiplying by ``s`` maps
    back to real seconds within the stored file. This is what makes the matched
    window marker land on the right region during playback (PLAN.md D2b.2).

    Note that a zero-padded final window reports an ``end_time`` beyond the
    sample's duration — BirdNET's third window on a 5 s file ends at 6.0 s,
    because that is genuinely the span the model saw (2 s of audio, 1 s of
    padding). The true span is stored rather than clamped, so the padding stays
    visible; a UI seeking to that position must clamp to the file's duration.

    Args:
        n_windows: Number of windows produced.
        window_samples: Window length in samples at ``model_rate``.
        stride_samples: Hop between windows in samples at ``model_rate``.
        model_rate: The model's sample rate in Hz.
        speed: The speed factor that was applied.

    Returns:
        ``(start_time, end_time)`` per window, in seconds within the sample.
        ``end_time`` may exceed the sample duration for a padded final window.
    """
    window_seconds = window_samples / float(model_rate)
    stride_seconds = stride_samples / float(model_rate)
    spans = []
    for index in range(n_windows):
        shifted_start = index * stride_seconds
        start = shifted_start * speed
        end = (shifted_start + window_seconds) * speed
        spans.append((start, end))
    return spans


def prepare_windows(
    audio: np.ndarray,
    native_rate: int,
    *,
    model_rate: int,
    window_samples: int,
    stride_samples: int,
    speed: float = 1.0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Turn native-rate mono audio into model-ready windows and their times.

    Applies the speed shift, resamples once to the model's rate, frames, and
    converts the window times back to the sample's time base.

    Args:
        audio: 1-D float32 samples at ``native_rate``.
        native_rate: The file's real sample rate in Hz.
        model_rate: Rate the model expects, in Hz.
        window_samples: Window length in samples at ``model_rate``.
        stride_samples: Hop in samples at ``model_rate``.
        speed: Speed factor to apply first. ``1.0`` is no change.

    Returns:
        An ``(n_windows, window_samples)`` float32 array, and the matching
        ``(start_time, end_time)`` spans in the sample's time base.
    """
    source_rate = effective_sample_rate(native_rate, speed)
    resampled = resample(audio, source_rate, model_rate)
    windows = frame(resampled, window_samples, stride_samples)
    spans = window_times_in_sample_domain(
        windows.shape[0], window_samples, stride_samples, model_rate, speed
    )
    return windows, spans
