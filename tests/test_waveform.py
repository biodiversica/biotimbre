"""Waveform envelopes and the SVG drawn from them."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from biotimbre.domain.errors import ValidationError
from biotimbre.viz import waveform_envelope
from conftest import EXAMPLES_DIR

GUYVIR = EXAMPLES_DIR / "GUYVIR.wav"


@pytest.fixture
def tone(tmp_path):
    """A 1 s sine at half scale, with one sample spiking to full."""
    rate = 16_000
    samples = (0.5 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(
        np.float32
    )
    samples[rate // 2] = 1.0
    path = tmp_path / "tone.wav"
    sf.write(path, samples, rate, subtype="FLOAT")
    return path


class TestEnvelope:
    def test_reduces_to_the_requested_columns(self):
        wave = waveform_envelope(GUYVIR, columns=300)
        assert wave.columns == 300
        assert wave.minima.shape == wave.maxima.shape == wave.rms.shape
        assert wave.duration == pytest.approx(5.0)
        assert wave.sample_rate == 32_000

    def test_keeps_a_transient_a_decimation_would_drop(self, tone):
        wave = waveform_envelope(tone, columns=100)
        # The spike is one sample in sixteen thousand; taking every nth sample
        # would almost certainly miss it, and it is exactly what a waveform is
        # looked at for.
        assert wave.peak == pytest.approx(1.0, abs=1e-6)
        assert wave.maxima.max() == pytest.approx(1.0, abs=1e-6)

    def test_rms_reports_loudness_not_extremes(self, tone):
        wave = waveform_envelope(tone, columns=100)
        # A half-scale sine reads 0.5/sqrt(2) whatever its peaks do.
        assert float(np.median(wave.rms)) == pytest.approx(0.354, abs=0.01)
        assert (wave.rms <= np.maximum(wave.maxima, -wave.minima) + 1e-6).all()

    def test_reads_only_the_requested_span(self):
        whole = waveform_envelope(GUYVIR, columns=64)
        tail = waveform_envelope(GUYVIR, start_time=2.5, end_time=5.0, columns=64)
        assert (tail.start_time, tail.end_time) == (2.5, 5.0)
        assert tail.duration == pytest.approx(2.5)
        # The second half of the file is not the whole of it.
        assert not np.allclose(whole.rms, tail.rms)

    def test_columns_never_exceed_the_samples(self, tmp_path):
        path = tmp_path / "tiny.wav"
        sf.write(path, np.zeros(50, dtype=np.float32), 16_000, subtype="FLOAT")
        # Padding with empty columns would draw a shape the signal never had.
        assert waveform_envelope(path, columns=900).columns == 50

    def test_rejects_an_empty_span(self):
        with pytest.raises(ValidationError, match="Nothing to render"):
            waveform_envelope(GUYVIR, start_time=3.0, end_time=3.0)

    def test_rejects_a_useless_column_count(self):
        with pytest.raises(ValidationError, match="at least 2 columns"):
            waveform_envelope(GUYVIR, columns=1)


class TestSvg:
    def test_draws_both_readings_and_takes_its_colours_from_the_theme(self):
        pytest.importorskip("nicegui", reason="needs the 'gui' extra")
        from biotimbre.ui.visuals import waveform_svg

        svg = waveform_svg(waveform_envelope(GUYVIR, columns=32), height=120)
        # Two bands: the peak envelope and the RMS inside it.
        assert svg.count("<polygon") == 2
        assert 'height:120px' in svg
        # A rendered image could not follow the reader's light/dark choice.
        assert "var(--bt-primary)" in svg
        assert "viewBox=\"0 0 31 2\"" in svg

    def test_scales_a_quiet_signal_to_its_own_peak(self, tmp_path):
        pytest.importorskip("nicegui", reason="needs the 'gui' extra")
        from biotimbre.ui.visuals import waveform_svg

        rate = 16_000
        quiet = (0.01 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate)).astype(
            np.float32
        )
        path = tmp_path / "quiet.wav"
        sf.write(path, quiet, rate, subtype="FLOAT")

        loud_svg = waveform_svg(waveform_envelope(GUYVIR, columns=32), height=100)
        quiet_svg = waveform_svg(waveform_envelope(path, columns=32), height=100)

        def extent(svg: str) -> float:
            points = svg.split('points="')[1].split('"')[0].split()
            ys = [float(point.split(",")[1]) for point in points]
            return max(ys) - min(ys)

        # A quiet recording is a picture of its own signal, not a flat line —
        # as the spectrogram's dB range is relative to its own loudest bin.
        assert extent(quiet_svg) == pytest.approx(extent(loud_svg), abs=0.15)

    def test_silence_does_not_divide_by_zero(self, tmp_path):
        pytest.importorskip("nicegui", reason="needs the 'gui' extra")
        from biotimbre.ui.visuals import waveform_svg

        path = tmp_path / "silence.wav"
        sf.write(path, np.zeros(8_000, dtype=np.float32), 16_000, subtype="FLOAT")
        svg = waveform_svg(waveform_envelope(path, columns=16), height=80)
        assert "nan" not in svg.lower()
