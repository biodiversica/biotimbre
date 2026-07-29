"""Audio preprocessing: framing, resampling, speed shift, time base conversion.

These need no model files.
"""

from __future__ import annotations

import numpy as np
import pytest

from biotimbre.domain.errors import ValidationError
from biotimbre.embeddings import audio as audio_io
from conftest import EXAMPLES_DIR

GUYVIR = EXAMPLES_DIR / "GUYVIR.wav"


class TestReading:
    def test_reads_example_properties(self):
        props = audio_io.read_properties(GUYVIR)
        assert props.sample_rate == 32_000
        assert props.channels == 1
        assert props.duration_s == pytest.approx(5.0)

    def test_loads_example_as_mono_float32(self):
        samples, rate = audio_io.load_mono(GUYVIR)
        assert rate == 32_000
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert samples.size == 160_000

    def test_averages_multichannel(self, tmp_path):
        import soundfile as sf

        stereo = np.zeros((1000, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        stereo[:, 1] = -0.5
        path = tmp_path / "stereo.wav"
        # Float subtype so 16-bit quantization asymmetry (+1.0 clips to 32767
        # while -1.0 maps to -32768) does not contaminate what is being tested.
        sf.write(path, stereo, 16_000, subtype="FLOAT")
        samples, _ = audio_io.load_mono(path)
        # Averaging rather than picking channel 0 keeps both channels' content.
        assert samples.shape == (1000,)
        np.testing.assert_allclose(samples, 0.0, atol=1e-7)

    def test_rejects_non_audio(self, tmp_path):
        path = tmp_path / "not_audio.wav"
        path.write_bytes(b"definitely not a wav")
        with pytest.raises(ValidationError, match="audio"):
            audio_io.load_mono(path)

    def test_sha256_is_stable_and_matches_hashlib(self):
        import hashlib

        expected = hashlib.sha256(GUYVIR.read_bytes()).hexdigest()
        assert audio_io.file_sha256(GUYVIR) == expected


class TestEffectiveSampleRate:
    @pytest.mark.parametrize(
        "speed,expected", [(1.0, 32_000), (0.5, 16_000), (0.25, 8_000), (2.0, 64_000)]
    )
    def test_scales_by_speed(self, speed, expected):
        assert audio_io.effective_sample_rate(32_000, speed) == expected

    @pytest.mark.parametrize("speed", [0.0, -1.0])
    def test_rejects_non_positive(self, speed):
        with pytest.raises(ValidationError, match="positive"):
            audio_io.effective_sample_rate(32_000, speed)


class TestResample:
    def test_returns_input_unchanged_when_rates_match(self):
        audio = np.ones(100, dtype=np.float32)
        assert audio_io.resample(audio, 32_000, 32_000) is audio

    def test_upsamples_to_expected_length(self):
        audio = np.zeros(16_000, dtype=np.float32)
        out = audio_io.resample(audio, 16_000, 32_000)
        assert out.size == pytest.approx(32_000, rel=0.01)

    def test_handles_fractional_source_rate(self):
        # A speed shift makes the source rate fractional: 32000 x 0.3 = 9600.0,
        # but 32000 x 0.333 is not an integer. Fraction-based ratios cope where
        # an integer gcd would not.
        audio = np.zeros(10_000, dtype=np.float32)
        out = audio_io.resample(audio, 32_000 * 0.333, 32_000)
        assert out.size > 0
        assert np.isfinite(out).all()

    def test_preserves_a_tone_below_nyquist(self):
        rate = 32_000
        t = np.arange(rate, dtype=np.float32) / rate
        tone = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
        out = audio_io.resample(tone, rate, 48_000)
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.fft.rfftfreq(out.size, 1 / 48_000)[spectrum.argmax()]
        assert peak_hz == pytest.approx(1000.0, abs=5.0)

    def test_speed_shift_moves_a_tone_down(self):
        # The point of the speed shift: content above a model's Nyquist can be
        # brought into range. A 12 kHz tone at speed 0.25 lands at 3 kHz.
        rate = 32_000
        t = np.arange(rate, dtype=np.float32) / rate
        tone = np.sin(2 * np.pi * 12_000.0 * t).astype(np.float32)
        out = audio_io.resample(
            tone, audio_io.effective_sample_rate(rate, 0.25), 32_000
        )
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.fft.rfftfreq(out.size, 1 / 32_000)[spectrum.argmax()]
        assert peak_hz == pytest.approx(3000.0, abs=25.0)


class TestFrame:
    def test_exact_length_yields_one_window_at_any_overlap(self):
        # 5 s of audio under a 5 s window leaves no room for a second hop.
        audio = np.ones(160_000, dtype=np.float32)
        assert audio_io.frame(audio, 160_000, 160_000).shape == (1, 160_000)
        assert audio_io.frame(audio, 160_000, 80_000).shape == (1, 160_000)

    def test_short_audio_is_padded_to_one_window(self):
        audio = np.ones(1000, dtype=np.float32)
        windows = audio_io.frame(audio, 4000, 2000)
        assert windows.shape == (1, 4000)
        np.testing.assert_array_equal(windows[0, :1000], 1.0)
        np.testing.assert_array_equal(windows[0, 1000:], 0.0)

    @pytest.mark.parametrize(
        "n_samples,window,stride,expected",
        [
            (240_000, 144_000, 144_000, 2),  # BirdNET, 5 s, 0% overlap
            (240_000, 144_000, 72_000, 3),   # BirdNET, 5 s, 50% overlap
            (160_000, 160_000, 80_000, 1),   # Perch, 5 s, 50% overlap
            (320_000, 160_000, 80_000, 3),   # Perch, 10 s, 50% overlap
        ],
    )
    def test_window_counts(self, n_samples, window, stride, expected):
        audio = np.ones(n_samples, dtype=np.float32)
        assert audio_io.frame(audio, window, stride).shape[0] == expected

    def test_final_window_is_zero_padded(self):
        audio = np.ones(5000, dtype=np.float32)
        windows = audio_io.frame(audio, 4000, 4000)
        assert windows.shape == (2, 4000)
        np.testing.assert_array_equal(windows[1, :1000], 1.0)
        np.testing.assert_array_equal(windows[1, 1000:], 0.0)

    def test_rejects_empty_audio(self):
        with pytest.raises(ValidationError, match="empty"):
            audio_io.frame(np.zeros(0, dtype=np.float32), 100, 100)


class TestWindowTimesInSampleDomain:
    def test_identity_at_unit_speed(self):
        spans = audio_io.window_times_in_sample_domain(3, 144_000, 72_000, 48_000, 1.0)
        assert spans == [(0.0, 3.0), (1.5, 4.5), (3.0, 6.0)]

    def test_converts_back_from_shifted_domain(self):
        # At speed 0.5 the shifted timeline is twice as long as the real sample,
        # so real times are the shifted ones halved. Without this the player's
        # matched-window marker would point at the wrong region.
        spans = audio_io.window_times_in_sample_domain(3, 160_000, 80_000, 32_000, 0.5)
        assert spans == [(0.0, 2.5), (1.25, 3.75), (2.5, 5.0)]

    def test_spans_stay_within_the_real_sample(self):
        # A 5 s sample at speed 0.25 becomes 20 s of shifted audio, but every
        # span must still fall inside the real 5 s file.
        spans = audio_io.window_times_in_sample_domain(7, 160_000, 80_000, 32_000, 0.25)
        assert spans[0][0] == 0.0
        assert spans[-1][1] == pytest.approx(5.0)

    def test_padded_final_window_may_overrun_duration(self):
        # Honest rather than clamped: the model really did see 3 s starting at
        # 3 s, of which 1 s was padding. UIs clamp when seeking.
        spans = audio_io.window_times_in_sample_domain(3, 144_000, 72_000, 48_000, 1.0)
        assert spans[-1][1] > 5.0


class TestPrepareWindows:
    def test_end_to_end_on_the_example_file(self):
        samples, rate = audio_io.load_mono(GUYVIR)
        windows, spans = audio_io.prepare_windows(
            samples,
            rate,
            model_rate=48_000,
            window_samples=144_000,
            stride_samples=72_000,
        )
        # 32 kHz source resampled up to BirdNET's 48 kHz, 50% overlap.
        assert windows.shape == (3, 144_000)
        assert len(spans) == 3
        assert windows.dtype == np.float32

    def test_speed_shift_changes_window_count(self):
        samples, rate = audio_io.load_mono(GUYVIR)
        unshifted, _ = audio_io.prepare_windows(
            samples, rate, model_rate=32_000,
            window_samples=160_000, stride_samples=80_000,
        )
        slowed, spans = audio_io.prepare_windows(
            samples, rate, model_rate=32_000,
            window_samples=160_000, stride_samples=80_000, speed=0.5,
        )
        assert unshifted.shape[0] == 1
        assert slowed.shape[0] == 3
        assert spans[-1][1] == pytest.approx(5.0)
