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

#: Longest query segment a search will embed, in seconds.
#:
#: This library matches short vocalization samples, not whole recordings. A long
#: query is both slow to embed and analytically weak — max-over-window-pairs will
#: latch onto whichever fragment happens to match, so a minute of audio mostly
#: dilutes intent. Callers with a longer file choose an explicit segment instead.
MAX_QUERY_DURATION_S = 15.0

#: Slack for floating-point comparisons on durations.
_DURATION_EPSILON = 1e-6


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


def load_mono(
    path: str | Path,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load an audio file, or a segment of it, as mono float32 at its native rate.

    Multi-channel input is averaged rather than having one channel picked, so no
    part of a stereo recording's content is discarded.

    When a segment is given, only those frames are decoded — a long file is never
    pulled into memory in full just to take 15 s out of the middle.

    Args:
        path: Path to the audio file.
        start_time: Segment start in seconds. ``None`` starts at the beginning.
        end_time: Segment end in seconds. ``None`` runs to the end.

    Returns:
        The samples as 1-D float32, and the native sample rate.

    Raises:
        ValidationError: If the file cannot be decoded or the segment is empty.
    """
    import soundfile as sf

    read_kwargs: dict[str, object] = {}
    if start_time is not None or end_time is not None:
        properties = read_properties(path)
        rate = properties.sample_rate
        start_frame = 0 if start_time is None else int(round(start_time * rate))
        start_frame = max(0, min(start_frame, properties.n_frames))
        read_kwargs["start"] = start_frame
        if end_time is not None:
            end_frame = min(int(round(end_time * rate)), properties.n_frames)
            frames = end_frame - start_frame
            if frames <= 0:
                raise ValidationError(
                    f"Segment {start_time}-{end_time} s of {Path(path).name} "
                    f"contains no audio.",
                    "audio",
                )
            read_kwargs["frames"] = frames

    try:
        data, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=True, **read_kwargs
        )
    except Exception as exc:
        raise ValidationError(
            f"Could not decode {Path(path).name}: {exc}", "audio"
        ) from exc

    if data.size == 0:
        raise ValidationError(f"{Path(path).name} contains no audio.", "audio")

    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sample_rate)


def resolve_query_segment(
    duration_s: float,
    start_time: float | None = None,
    end_time: float | None = None,
    max_duration_s: float = MAX_QUERY_DURATION_S,
) -> tuple[float, float]:
    """Decide which part of a query file to search with, enforcing the length cap.

    A file already within the cap is used whole. A longer one requires an explicit
    segment: guessing (taking the first 15 s, say) would silently search something
    other than what the user pointed at.

    Args:
        duration_s: The file's full duration in seconds.
        start_time: Requested segment start. ``None`` means the beginning.
        end_time: Requested segment end. ``None`` means the end of the file,
            subject to the cap.
        max_duration_s: Longest segment allowed.

    Returns:
        ``(start, end)`` in seconds, at most ``max_duration_s`` apart.

    Raises:
        ValidationError: If the file is too long and no segment was given, or the
            requested segment is empty, out of range, or over the cap.
    """
    if duration_s <= 0:
        raise ValidationError("Query audio has no duration.", "audio")

    if start_time is None and end_time is None:
        if duration_s <= max_duration_s + _DURATION_EPSILON:
            return 0.0, duration_s
        raise ValidationError(
            f"Query audio is {duration_s:.1f} s, but a search is limited to "
            f"{max_duration_s:g} s. Choose a start and end time at most "
            f"{max_duration_s:g} s apart — this library matches short "
            f"vocalizations, so searching a whole recording dilutes the query "
            f"rather than improving it.",
            "query_segment",
        )

    start = 0.0 if start_time is None else float(start_time)
    if start < 0:
        raise ValidationError(
            f"start_time {start} s cannot be negative.", "start_time"
        )
    if start >= duration_s - _DURATION_EPSILON:
        raise ValidationError(
            f"start_time {start} s is at or past the end of a "
            f"{duration_s:.1f} s file.",
            "start_time",
        )

    if end_time is None:
        end = min(duration_s, start + max_duration_s)
    else:
        end = min(float(end_time), duration_s)

    if end <= start + _DURATION_EPSILON:
        raise ValidationError(
            f"end_time ({end} s) must be greater than start_time ({start} s).",
            "end_time",
        )
    if end - start > max_duration_s + _DURATION_EPSILON:
        raise ValidationError(
            f"Requested segment is {end - start:.1f} s, but a search is limited "
            f"to {max_duration_s:g} s.",
            "query_segment",
        )
    return start, end


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


def apply_filter(
    audio: np.ndarray,
    sample_rate: int,
    *,
    low_hz: float | None,
    high_hz: float | None,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter to mono audio.

    Unlike ``items.filtered_band``, which only *records* filtering someone else
    already applied before the sample was submitted, this actually runs the
    filter — used to let a search query be filtered on the fly. Cutoffs are real
    Hz values, so filtering happens here, at the audio's native rate, before any
    speed shift or resampling would change what a given Hz means.

    ``sosfiltfilt`` (forward-backward, zero-phase) is used instead of a causal
    ``lfilter`` so the filter introduces no group delay — a delay would shift
    window boundaries and misalign the matched-window marker.

    Args:
        audio: 1-D float32 samples at ``sample_rate``.
        sample_rate: The audio's sample rate in Hz.
        low_hz: Highpass cutoff, or ``None`` for none. With ``high_hz`` also set,
            the two form a bandpass.
        high_hz: Lowpass cutoff, or ``None`` for none.
        order: Butterworth filter order.

    Returns:
        Filtered 1-D float32 samples, same length as the input. Returned
        unchanged when both cutoffs are ``None``.

    Raises:
        ValidationError: If a cutoff is not strictly between 0 and the Nyquist
            frequency, if ``low_hz >= high_hz``, if ``order < 1``, or if the
            audio is too short for the resulting filter.
    """
    if low_hz is None and high_hz is None:
        return audio
    if order < 1:
        raise ValidationError(f"Filter order must be >= 1, got {order}.", "filter_order")

    nyquist = sample_rate / 2.0
    for name, value in (("Highpass cutoff", low_hz), ("Lowpass cutoff", high_hz)):
        if value is not None and not 0 < value < nyquist:
            raise ValidationError(
                f"{name} {value:g} Hz must be between 0 and the Nyquist "
                f"frequency ({nyquist:g} Hz for this {sample_rate} Hz audio).",
                "filtered_band",
            )
    if low_hz is not None and high_hz is not None and low_hz >= high_hz:
        raise ValidationError(
            f"Highpass cutoff ({low_hz:g} Hz) must be below the lowpass cutoff "
            f"({high_hz:g} Hz).",
            "filtered_band",
        )

    from scipy.signal import butter, sosfiltfilt

    if low_hz is not None and high_hz is not None:
        sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    elif low_hz is not None:
        sos = butter(order, low_hz, btype="highpass", fs=sample_rate, output="sos")
    else:
        sos = butter(order, high_hz, btype="lowpass", fs=sample_rate, output="sos")

    try:
        filtered = sosfiltfilt(sos, audio)
    except ValueError as exc:
        raise ValidationError(
            f"Audio is too short to filter at order {order}: {exc}", "filter_order"
        ) from None
    return np.ascontiguousarray(filtered, dtype=np.float32)


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
