"""Reading an observation from iNaturalist, and the recording attached to it."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

httpx = pytest.importorskip("httpx")

from biotimbre.domain.errors import (  # noqa: E402
    BiotimbreError,
    NotFoundError,
    ValidationError,
)
from biotimbre.sources import inaturalist as inat  # noqa: E402

#: One observation as the API returns it, trimmed to what is read.
PAYLOAD = {
    "results": [
        {
            "id": 400139850,
            "quality_grade": "needs_id",
            "time_observed_at": "2026-09-14T17:59:27+10:00",
            "observed_on": "2026-09-14",
            "location": "-28.8220683397,153.286555119",
            "obscured": False,
            "taxon": {"name": "Guyalna bonaerensis", "rank": "species"},
            "user": {"login": "pmont", "name": "Paul Montgomery"},
            "sounds": [
                {
                    "id": 2158429,
                    "file_url": "https://static.inaturalist.org/sounds/2158429.m4a?1",
                    "file_content_type": "audio/mp4",
                    "license_code": "cc-by-nc",
                    "attribution": "(c) Paul Montgomery, some rights reserved (CC BY-NC)",
                }
            ],
        }
    ]
}


def client_for(handler) -> httpx.Client:
    """A client whose requests are answered by ``handler``."""
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestObservationId:
    @pytest.mark.parametrize(
        "typed",
        [
            "400139850",
            " 400139850 ",
            "https://www.inaturalist.org/observations/400139850",
            "https://www.inaturalist.org/observations/400139850/",
            "https://www.inaturalist.org/observations/400139850?x=1",
        ],
    )
    def test_reads_an_id_from_a_number_or_a_link(self, typed):
        # A pasted link is as likely as a number, and names the same thing.
        assert inat.parse_observation_id(typed) == 400139850

    @pytest.mark.parametrize("typed", ["", "   ", None, "not-an-id", "abc/def"])
    def test_rejects_anything_else(self, typed):
        with pytest.raises(ValidationError):
            inat.parse_observation_id(typed)


class TestReadingAnObservation:
    def test_reads_what_a_sample_needs(self):
        observation = inat.observation_from_payload(PAYLOAD["results"][0])
        assert observation.id == 400139850
        assert observation.url.endswith("/observations/400139850")
        # Local time where it was recorded, not UTC: the library stores wall
        # clock, and +10:00 is what the recordist saw.
        assert (observation.date, observation.time) == ("2026-09-14", "17:59:27")
        assert observation.latitude == pytest.approx(-28.8220683)
        assert observation.taxon == "Guyalna bonaerensis"
        assert observation.observer == "Paul Montgomery"
        assert observation.quality_grade == "needs_id"
        assert observation.sounds[0].license_code == "cc-by-nc"

    @pytest.mark.parametrize(
        "hidden",
        [
            {"obscured": True},
            {"geoprivacy": "obscured"},
            {"geoprivacy": "private"},
            {"taxon_geoprivacy": "obscured"},
        ],
    )
    def test_withholds_coordinates_it_is_not_allowed_to_know(self, hidden):
        payload = {**PAYLOAD["results"][0], **hidden}
        observation = inat.observation_from_payload(payload)
        # The coordinates iNaturalist gives for a hidden observation name a
        # random point; recording them as fact would be a lie.
        assert observation.obscured is True
        assert observation.latitude is None
        assert observation.longitude is None

    def test_falls_back_to_the_date_when_there_is_no_time(self):
        payload = {**PAYLOAD["results"][0], "time_observed_at": None}
        observation = inat.observation_from_payload(payload)
        assert (observation.date, observation.time) == ("2026-09-14", None)

    def test_an_observation_without_sounds_reads_fine(self):
        payload = {**PAYLOAD["results"][0], "sounds": []}
        assert inat.observation_from_payload(payload).sounds == ()

    def test_credit_names_the_observation_and_the_recordist(self):
        observation = inat.observation_from_payload(PAYLOAD["results"][0])
        credit = observation.credit
        assert "400139850" in credit
        assert "Paul Montgomery" in credit


class TestFetching:
    def test_fetches_and_parses(self):
        def handler(request):
            assert request.url.path.endswith("/observations/400139850")
            return httpx.Response(200, json=PAYLOAD)

        with client_for(handler) as client:
            observation = inat.fetch_observation(400139850, client=client)
        assert observation.taxon == "Guyalna bonaerensis"

    def test_missing_observation_is_not_found(self):
        with client_for(lambda request: httpx.Response(404)) as client:
            with pytest.raises(NotFoundError, match="400139850"):
                inat.fetch_observation(400139850, client=client)

    def test_an_empty_result_is_not_found(self):
        with client_for(lambda r: httpx.Response(200, json={"results": []})) as client:
            with pytest.raises(NotFoundError):
                inat.fetch_observation(1, client=client)

    def test_an_unreachable_api_says_so(self):
        def handler(request):
            raise httpx.ConnectError("no route to host")

        with client_for(handler) as client:
            with pytest.raises(BiotimbreError, match="Could not reach iNaturalist"):
                inat.fetch_observation(1, client=client)


class TestDownloading:
    @pytest.fixture
    def wav_bytes(self, tmp_path):
        path = tmp_path / "source.wav"
        sf.write(path, np.zeros(8_000, dtype=np.float32), 16_000, subtype="FLOAT")
        return path.read_bytes()

    def test_a_readable_format_is_saved_as_it_came(self, tmp_path, wav_bytes):
        sound = inat.Sound(id=1, url="https://x/1.wav", content_type="audio/wav")
        assert sound.needs_transcoding is False

        with client_for(lambda r: httpx.Response(200, content=wav_bytes)) as client:
            written = inat.download_sound(sound, tmp_path / "q.wav", client=client)
        assert written.read_bytes() == wav_bytes

    def test_m4a_without_ffmpeg_says_what_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(inat, "ffmpeg_available", lambda: False)
        sound = inat.Sound(id=1, url="https://x/1.m4a", content_type="audio/mp4")
        # A codec error from libsndfile would tell the reader nothing they
        # could act on.
        with pytest.raises(ValidationError, match="needs ffmpeg"):
            inat.download_sound(sound, tmp_path / "q.wav")

    def test_progress_is_reported_as_it_arrives(self, tmp_path, wav_bytes):
        seen: list[tuple[int, int]] = []
        sound = inat.Sound(id=1, url="https://x/1.wav")

        def handler(request):
            return httpx.Response(
                200,
                content=wav_bytes,
                headers={"content-length": str(len(wav_bytes))},
            )

        with client_for(handler) as client:
            inat.download_sound(
                sound,
                tmp_path / "q.wav",
                client=client,
                on_progress=lambda received, total: seen.append((received, total)),
            )

        assert seen[0] == (0, len(wav_bytes))
        assert seen[-1] == (len(wav_bytes), len(wav_bytes))
        # Monotonic, so a bar built from it never goes backwards.
        assert [received for received, _ in seen] == sorted(
            received for received, _ in seen
        )

    def test_an_unmeasured_download_reports_no_total(self, tmp_path, wav_bytes):
        seen: list[tuple[int, int]] = []

        def handler(request):
            # Chunked, as a server that does not know the size would send it.
            return httpx.Response(200, content=iter([wav_bytes]))

        with client_for(handler) as client:
            inat.download_sound(
                inat.Sound(id=1, url="https://x/1.wav"),
                tmp_path / "q.wav",
                client=client,
                on_progress=lambda received, total: seen.append((received, total)),
            )

        # Zero means "unmeasured", for a caller to show as a sweep rather than
        # as a bar stuck at nothing.
        assert all(total == 0 for _, total in seen)
        assert seen[-1][0] == len(wav_bytes)

    def test_a_failed_download_leaves_no_half_file(self, tmp_path):
        target = tmp_path / "q.wav"
        with client_for(lambda r: httpx.Response(500)) as client:
            with pytest.raises(BiotimbreError):
                inat.download_sound(
                    inat.Sound(id=1, url="https://x/1.wav"), target, client=client
                )
        assert not target.exists()

    def test_a_failed_download_says_so(self, tmp_path):
        sound = inat.Sound(id=1, url="https://x/1.wav")
        with client_for(lambda r: httpx.Response(500)) as client:
            with pytest.raises(BiotimbreError, match="Could not download"):
                inat.download_sound(sound, tmp_path / "q.wav", client=client)

    @pytest.mark.skipif(
        not inat.ffmpeg_available(), reason="needs ffmpeg to transcode"
    )
    def test_m4a_is_decoded_to_a_readable_wav(self, tmp_path, wav_bytes):
        import subprocess

        source = tmp_path / "source.wav"
        source.write_bytes(wav_bytes)
        encoded = tmp_path / "source.m4a"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(source), str(encoded)],
            check=True,
            capture_output=True,
        )

        sound = inat.Sound(id=1, url="https://x/1.m4a", content_type="audio/mp4")
        with client_for(
            lambda r: httpx.Response(200, content=encoded.read_bytes())
        ) as client:
            written = inat.download_sound(sound, tmp_path / "q.wav", client=client)

        assert written.suffix == ".wav"
        # Readable by the same library everything else goes through.
        assert sf.info(str(written)).samplerate == 16_000
        assert not (tmp_path / "q.m4a").exists()
