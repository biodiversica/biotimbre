"""Every stored recording is a WAV, whatever format it arrived in.

The library names its audio ``<item>.wav``, bundles carry it as
``audio.wav`` and the interface serves it as ``audio/wav``; a FLAC or MP3
stored as it came would be mislabelled in all three.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
import soundfile as sf

from biotimbre.config import Settings
from biotimbre.domain.item import ItemDraft, Location
from biotimbre.embeddings import audio as audio_io
from biotimbre.service import LibraryService
from conftest import EXAMPLES_DIR

GUYVIR_WAV = EXAMPLES_DIR / "GUYVIR.wav"


@pytest.fixture
def service(tmp_path):
    data_dir = tmp_path / "library"
    settings = Settings(
        data_dir=data_dir, db_path=data_dir / "library.db", audio_dir=data_dir / "audio"
    )
    with LibraryService(settings) as service:
        yield service


def _draft() -> ItemDraft:
    return ItemDraft(date="2020-01-12", time="09:40:21", location=Location(-26.0, -48.0))


def _store(service, path):
    item = service.add_item(path, _draft(), compute_embeddings=False).item
    return item, service.settings.absolute_audio_path(item.audio_path)


def _as(tmp_path, fmt: str, subtype: str, suffix: str):
    samples, rate = sf.read(str(GUYVIR_WAV), dtype="int16")
    path = tmp_path / f"clip{suffix}"
    sf.write(str(path), samples, rate, format=fmt, subtype=subtype)
    return path, samples


class TestConverted:
    def test_a_flac_is_stored_as_wav_sample_for_sample(self, service, tmp_path):
        path, samples = _as(tmp_path, "FLAC", "PCM_16", ".flac")
        item, stored = _store(service, path)

        info = sf.info(str(stored))
        assert (info.format, info.subtype) == ("WAV", "PCM_16")
        assert np.array_equal(sf.read(str(stored), dtype="int16")[0], samples)
        assert item.audio_sha256 == audio_io.file_sha256(stored)

    def test_a_lossy_file_is_decoded_to_float_wav(self, service, tmp_path):
        path, _ = _as(tmp_path, "MP3", "MPEG_LAYER_III", ".mp3")
        _, stored = _store(service, path)

        info = sf.info(str(stored))
        assert (info.format, info.subtype) == ("WAV", "FLOAT")
        assert info.channels == 1

    def test_the_name_does_not_decide(self, service, tmp_path):
        # An MP3 called .wav, as an iNaturalist download used to be staged.
        path, _ = _as(tmp_path, "MP3", "MPEG_LAYER_III", ".mp3")
        disguised = path.rename(tmp_path / "disguised.wav")
        _, stored = _store(service, disguised)
        assert sf.info(str(stored)).format == "WAV"

    def test_a_bundle_carries_a_real_wav(self, service, tmp_path):
        path, _ = _as(tmp_path, "FLAC", "PCM_16", ".flac")
        item, _ = _store(service, path)
        bundle = service.export_bundle([item.item_id], tmp_path / "out.btz")
        with zipfile.ZipFile(bundle) as archive:
            name = next(n for n in archive.namelist() if n.endswith("audio.wav"))
            assert archive.read(name)[:4] == b"RIFF"


class TestKept:
    def test_a_mono_wav_is_stored_as_it_came(self, service):
        item, stored = _store(service, GUYVIR_WAV)
        assert stored.read_bytes() == GUYVIR_WAV.read_bytes()
        assert item.audio_sha256 == audio_io.file_sha256(GUYVIR_WAV)

    def test_the_same_flac_twice_is_recognised(self, service, tmp_path):
        path, _ = _as(tmp_path, "FLAC", "PCM_16", ".flac")
        _store(service, path)
        again = service.add_item(path, _draft(), compute_embeddings=False)
        assert again.duplicate_of == (1,)


class TestStable:
    """Processing or converting the same recording twice gives the same file,
    so its digest recognises it the second time."""

    def test_a_float_wav_is_written_the_same_each_time(self, tmp_path):
        import time

        first, second = tmp_path / "a.wav", tmp_path / "b.wav"
        audio_io.write_processed(GUYVIR_WAV, first, low_hz=500.0)
        time.sleep(1.1)  # the PEAK chunk's timestamp is in seconds
        audio_io.write_processed(GUYVIR_WAV, second, low_hz=500.0)
        assert first.read_bytes() == second.read_bytes()
        assert b"PEAK" not in first.read_bytes()
        assert sf.info(str(first)).subtype == "FLOAT"

    def test_a_processed_recording_is_recognised_again(self, service):
        from biotimbre.processing import Processing

        processing = Processing(trim=(1.0, 4.0), highpass=500.0, normalise=True)
        service.add_item(GUYVIR_WAV, _draft(), compute_embeddings=False,
                         processing=processing)
        from biotimbre.domain.errors import DuplicateAudioError

        with pytest.raises(DuplicateAudioError):
            service.add_item(GUYVIR_WAV, _draft(), compute_embeddings=False,
                             processing=processing, skip_duplicates=True)

    def test_an_mp3_is_recognised_again(self, service, tmp_path):
        path, _ = _as(tmp_path, "MP3", "MPEG_LAYER_III", ".mp3")
        _store(service, path)
        again = service.add_item(path, _draft(), compute_embeddings=False)
        assert again.duplicate_of == (1,)
