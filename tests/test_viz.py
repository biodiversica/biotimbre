"""Spectrogram rendering: axes, ultrasonic visibility, and image size."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from biotimbre.domain.errors import ValidationError
from biotimbre.viz import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    THUMBNAIL_HEIGHT,
    THUMBNAIL_WIDTH,
    render_spectrogram,
)
from conftest import EXAMPLES_DIR

GUYVIR = EXAMPLES_DIR / "GUYVIR.wav"


@pytest.fixture
def ultrasonic(tmp_path):
    """96 kHz file: a 6 kHz tone in the first half, a 38 kHz tone in the second."""
    rate = 96_000
    t = np.arange(int(rate * 2), dtype=np.float32) / rate
    audible = np.sin(2 * np.pi * 6_000 * t) * (t < 1.0)
    ultra = np.sin(2 * np.pi * 38_000 * t) * (t >= 1.0)
    path = tmp_path / "ultra.wav"
    sf.write(path, ((audible + ultra) * 0.4).astype(np.float32), rate, subtype="FLOAT")
    return path


class TestAxes:
    def test_frequency_axis_reaches_the_files_own_nyquist(self, ultrasonic):
        # Computed at the native rate, not the model rate: a 96 kHz file must show
        # up to 48 kHz, or the ultrasonic content it exists to reveal is invisible.
        spec = render_spectrogram(ultrasonic)
        assert spec.sample_rate == 96_000
        assert spec.max_frequency == 48_000.0

    def test_example_file_axis(self):
        spec = render_spectrogram(GUYVIR)
        assert spec.max_frequency == 16_000.0

    def test_default_dimensions(self):
        spec = render_spectrogram(GUYVIR)
        assert (spec.width, spec.height) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)

    def test_time_axis_is_linear(self):
        spec = render_spectrogram(GUYVIR)
        assert spec.time_to_fraction(0.0) == pytest.approx(0.0)
        assert spec.time_to_fraction(2.5) == pytest.approx(0.5, abs=0.01)
        assert spec.time_to_fraction(5.0) == pytest.approx(1.0)

    def test_time_fraction_is_clamped(self):
        spec = render_spectrogram(GUYVIR)
        assert spec.time_to_fraction(-10.0) == 0.0
        assert spec.time_to_fraction(999.0) == 1.0

    def test_frequency_fraction_places_model_limits(self, ultrasonic):
        spec = render_spectrogram(ultrasonic)
        # Perch's 16 kHz limit sits a third of the way up a 48 kHz axis, which is
        # what makes "the model never saw the top of this signal" visible.
        assert spec.frequency_to_fraction(16_000) == pytest.approx(1 / 3, abs=0.01)
        assert spec.frequency_to_fraction(38_000) == pytest.approx(0.79, abs=0.02)

    def test_frequency_fraction_is_clamped(self):
        spec = render_spectrogram(GUYVIR)
        assert spec.frequency_to_fraction(999_999) == 1.0
        assert spec.frequency_to_fraction(-5) == 0.0


class TestSegments:
    def test_renders_a_segment(self):
        spec = render_spectrogram(GUYVIR, start_time=1.0, end_time=3.0)
        assert (spec.start_time, spec.end_time) == (1.0, 3.0)
        assert spec.duration == 2.0

    def test_segment_time_axis_is_offset(self):
        spec = render_spectrogram(GUYVIR, start_time=1.0, end_time=3.0)
        # Fractions are relative to the rendered span, so a marker drawn on a
        # segment view lands in the right place.
        assert spec.time_to_fraction(1.0) == pytest.approx(0.0)
        assert spec.time_to_fraction(2.0) == pytest.approx(0.5)

    def test_end_beyond_the_file_is_clamped(self):
        spec = render_spectrogram(GUYVIR, start_time=1.0, end_time=99.0)
        assert spec.end_time == pytest.approx(5.0)

    def test_empty_span_is_refused(self):
        with pytest.raises(ValidationError, match="Nothing to render"):
            render_spectrogram(GUYVIR, start_time=3.0, end_time=3.0)

    def test_inverted_span_is_refused(self):
        with pytest.raises(ValidationError, match="Nothing to render"):
            render_spectrogram(GUYVIR, start_time=4.0, end_time=2.0)


class TestUltrasonicVisibility:
    def test_ultrasonic_energy_appears_in_the_upper_image(self, ultrasonic):
        """The whole point: content you cannot hear must still be visible."""
        from PIL import Image
        import io

        spec = render_spectrogram(ultrasonic)
        pixels = np.asarray(
            Image.open(io.BytesIO(spec.png)).convert("L"), dtype=np.float32
        )
        height, width = pixels.shape

        # Second half of the file (38 kHz), upper part of the image.
        right = pixels[:, width // 2 :]
        upper_band = right[: int(height * 0.3)].mean()
        lower_band = right[int(height * 0.8) :].mean()

        # The 38 kHz tone sits at ~79% height, i.e. ~21% down from the top.
        assert upper_band > lower_band, (
            "ultrasonic energy should be brighter than the quiet low band"
        )

    def test_audible_and_ultrasonic_halves_differ(self, ultrasonic):
        from PIL import Image
        import io

        spec = render_spectrogram(ultrasonic)
        pixels = np.asarray(
            Image.open(io.BytesIO(spec.png)).convert("L"), dtype=np.float32
        )
        height, width = pixels.shape
        top = slice(0, int(height * 0.3))
        # Left half is the 6 kHz tone (low in the image), right half is 38 kHz.
        assert pixels[top, : width // 2].mean() < pixels[top, width // 2 :].mean()


class TestContrast:
    """The display window is derived from the data, not from the peak.

    A fixed window below the peak saturated real recordings into a featureless
    wash: GUYVIR's median bin is only 27 dB below its loudest, so a 70 dB window
    mapped the noise floor to 60 % brightness.
    """

    def _greyscale(self, path, **kwargs):
        import io

        from PIL import Image

        spec = render_spectrogram(path, **kwargs)
        return np.asarray(
            Image.open(io.BytesIO(spec.png)).convert("L"), dtype=np.float32
        )

    def test_real_recording_is_not_saturated(self):
        pixels = self._greyscale(GUYVIR)
        # A wash would push the mean brightness high with little spread. Structure
        # means plenty of dark pixels alongside bright ones.
        assert pixels.mean() < 140, f"mean brightness {pixels.mean():.0f} looks washed out"
        assert (pixels < 60).mean() > 0.2, "expected a visible dark noise floor"

    def test_real_recording_has_dynamic_range(self):
        pixels = self._greyscale(GUYVIR)
        assert pixels.std() > 25, f"std {pixels.std():.0f} is too flat to read"

    def test_amplitude_scaling_does_not_change_the_image(self, tmp_path):
        # Scaling the samples shifts every dB value by a constant, and the mapping
        # is relative, so the rendered image is unchanged. Peak-normalising the
        # audio is therefore not a fix for contrast.
        samples, rate = sf.read(GUYVIR, dtype="float32")
        quiet = tmp_path / "quiet.wav"
        loud = tmp_path / "loud.wav"
        sf.write(quiet, samples * 0.05, rate, subtype="FLOAT")
        sf.write(loud, samples / max(1e-9, float(np.abs(samples).max())), rate,
                 subtype="FLOAT")

        a = self._greyscale(quiet)
        b = self._greyscale(loud)
        # Not bit-identical (quantisation differs slightly), but visually the same.
        assert abs(a.mean() - b.mean()) < 6
        assert abs(a.std() - b.std()) < 6

    def test_window_is_capped_by_dynamic_range(self):
        from biotimbre.viz import _levels

        # A wide-spread image must not show more range than asked for.
        decibels = np.linspace(-200, 0, 1000).reshape(10, 100)
        floor, top = _levels(decibels, 30.0)
        assert top - floor == pytest.approx(30.0)

    def test_window_never_degenerates(self):
        from biotimbre.viz import _levels

        # Uniform input would otherwise give a zero-width window and divide by ~0.
        floor, top = _levels(np.full((10, 10), -40.0), 70.0)
        assert top > floor


class TestEncoding:
    def test_png_is_palettised_and_small(self):
        # Truecolour was 495 KB on this file; a search page shows one per result.
        spec = render_spectrogram(GUYVIR)
        assert spec.png.startswith(b"\x89PNG")
        assert len(spec.png) < 200_000, f"{len(spec.png) / 1024:.0f} KB is too large"

    def test_thumbnail_is_much_smaller(self):
        spec = render_spectrogram(
            GUYVIR, width=THUMBNAIL_WIDTH, height=THUMBNAIL_HEIGHT
        )
        assert len(spec.png) < 60_000

    def test_data_uri_round_trips(self):
        import base64

        spec = render_spectrogram(
            GUYVIR, width=THUMBNAIL_WIDTH, height=THUMBNAIL_HEIGHT
        )
        uri = spec.data_uri()
        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1]) == spec.png

    def test_image_dimensions_match_the_request(self):
        from PIL import Image
        import io

        spec = render_spectrogram(GUYVIR, width=320, height=64)
        image = Image.open(io.BytesIO(spec.png))
        assert image.size == (320, 64)

    def test_absurd_size_is_refused(self):
        with pytest.raises(ValidationError, match="at least"):
            render_spectrogram(GUYVIR, width=4, height=4)


class TestRobustness:
    def test_handles_silence(self, tmp_path):
        path = tmp_path / "silence.wav"
        sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000, subtype="FLOAT")
        # A log of zero must not produce NaNs or crash.
        spec = render_spectrogram(path)
        assert spec.png.startswith(b"\x89PNG")

    def test_handles_a_very_short_file(self, tmp_path):
        path = tmp_path / "blip.wav"
        sf.write(path, np.ones(200, dtype=np.float32) * 0.1, 32_000, subtype="FLOAT")
        spec = render_spectrogram(path)
        assert spec.width == DEFAULT_WIDTH

    def test_handles_stereo_by_averaging(self, tmp_path):
        rate = 32_000
        t = np.arange(rate, dtype=np.float32) / rate
        stereo = np.stack(
            [np.sin(2 * np.pi * 2000 * t), np.sin(2 * np.pi * 2000 * t)], axis=1
        ).astype(np.float32)
        path = tmp_path / "stereo.wav"
        sf.write(path, stereo * 0.4, rate, subtype="FLOAT")
        # Matches what the embedding pipeline sees, so the picture corresponds to
        # what the models were actually given.
        spec = render_spectrogram(path)
        assert spec.max_frequency == 16_000.0

    def test_rejects_non_audio(self, tmp_path):
        path = tmp_path / "nope.wav"
        path.write_bytes(b"not audio at all")
        with pytest.raises(ValidationError):
            render_spectrogram(path)
