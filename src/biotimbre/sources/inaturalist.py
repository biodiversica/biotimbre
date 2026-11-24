"""Fetching a recording, and what is known about it, from iNaturalist.

An observation carries the things a sample needs anyway — when, where, and what
it was thought to be — so giving an observation ID is a shorter and more
accurate route than typing them again. Nothing here touches the library: it
returns what iNaturalist says and a decoded file, and the caller decides what
to keep.

Two things about that audio are worth knowing before it is filed. Most of it is
AAC in an ``.m4a``, which libsndfile does not read, so decoding needs ffmpeg —
absent, the fetch fails with a message saying so rather than a codec error. And
a typical 128 kbps upload carries nothing above about 16 kHz whatever its
sample rate claims, which matters when a model listens higher than that.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx2

from biotimbre.domain.errors import BiotimbreError, NotFoundError, ValidationError

#: Where observations are read from.
API_ROOT = "https://api.inaturalist.org/v1"

#: Public page for an observation, and what goes in ``link_to_full_recording``.
OBSERVATION_URL = "https://www.inaturalist.org/observations/{id}"

#: iNaturalist asks callers to identify themselves.
USER_AGENT = "biotimbre (local research tool)"

#: Seconds to wait on the API and on a download.
TIMEOUT_S = 20.0

#: Pacing for many requests in a row, as a table import makes. iNaturalist
#: sends no rate-limit headers; these follow the limits its recommended
#: practices ask for — about one request a second (60 a minute) and under
#: 5 GB of media an hour — and can be lowered, not raised, if it asks for less.
MIN_REQUEST_INTERVAL_S = 1.0
MEDIA_BYTES_PER_HOUR = 5 * 1000**3

#: Observations read in one request: the API's largest page.
BATCH_SIZE = 200

#: Times a request told to slow down (HTTP 429 or 503) is tried again, waiting
#: what it says or twice as long each time, before giving up.
MAX_RETRIES = 6

#: Statuses that mean "slow down", not "this failed".
_THROTTLED_STATUSES = frozenset({429, 503})

#: Bytes per chunk while streaming a download. Small enough that progress
#: moves visibly on a slow line, large enough not to thrash.
_CHUNK_BYTES = 64 * 1024

#: Extensions libsndfile reads, so the download needs no help.
_NATIVE_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".w64", ".mp3"})

#: Geoprivacy values that mean the coordinates are not where the animal was.
_HIDDEN = frozenset({"obscured", "private"})


@dataclass(frozen=True)
class Sound:
    """One recording attached to an observation.

    Attributes:
        id: iNaturalist's ID for the sound.
        url: Where the file is.
        content_type: As iNaturalist reports it, e.g. ``audio/mp4``.
        license_code: Licence code, e.g. ``cc-by-nc``. ``None`` means the
            uploader reserved all rights.
        attribution: The credit line iNaturalist renders.
    """

    id: int
    url: str
    content_type: str | None = None
    license_code: str | None = None
    attribution: str | None = None

    @property
    def suffix(self) -> str:
        """Extension of the file, from its URL."""
        return Path(self.url.split("?")[0]).suffix.lower()

    @property
    def needs_transcoding(self) -> bool:
        """Whether libsndfile would refuse this file."""
        return self.suffix not in _NATIVE_SUFFIXES


@dataclass(frozen=True)
class Observation:
    """What an observation says about a recording.

    Attributes:
        id: The observation ID.
        url: Its public page.
        date: Local date at the observation, ``YYYY-MM-DD``, or ``None``.
        time: Local time, ``HH:MM:SS``, or ``None``.
        latitude: Latitude, or ``None`` when absent or obscured.
        longitude: Longitude, likewise.
        obscured: Whether iNaturalist is hiding the true location. The
            coordinates it gives then name a random point in a wide box, so
            they are withheld here rather than recorded as fact.
        taxon: Name of the taxon it is filed under, at whatever rank.
        taxon_rank: That rank, e.g. ``species`` or ``family``.
        quality_grade: ``research``, ``needs_id`` or ``casual``.
        observer: Who recorded it.
        sounds: The recordings attached, in the order iNaturalist lists them.
    """

    id: int
    url: str
    date: str | None = None
    time: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    obscured: bool = False
    taxon: str | None = None
    taxon_rank: str | None = None
    quality_grade: str | None = None
    observer: str | None = None
    sounds: tuple[Sound, ...] = ()

    @property
    def credit_prefix(self) -> str:
        """How every :meth:`credit` line for this observation starts."""
        return f"iNaturalist observation {self.id}"

    @property
    def credit(self) -> str:
        """:meth:`credit_for` the first recording."""
        return self.credit_for(self.sounds[0] if self.sounds else None)

    def credit_for(self, sound: Sound | None) -> str:
        """A one-line provenance note for one recording, for an item's ``notes``.

        For the recording actually used: two recordings of one observation can
        carry different licences, and the note is what says which applies.
        """
        parts = [self.credit_prefix]
        if sound and sound.attribution:
            parts.append(sound.attribution)
        elif sound and sound.license_code:
            parts.append(sound.license_code.upper())
        return " — ".join(parts)


class Throttled(BiotimbreError):
    """iNaturalist asked for fewer requests (HTTP 429 or 503).

    Attributes:
        retry_after: Seconds it said to wait, or ``None`` if it did not say.
    """

    def __init__(self, retry_after: float | None = None):
        super().__init__("iNaturalist asked for fewer requests.")
        self.retry_after = retry_after


def _raise_if_throttled(response) -> None:
    if response.status_code not in _THROTTLED_STATUSES:
        return
    header = response.headers.get("retry-after")
    try:
        retry_after = float(header) if header is not None else None
    except ValueError:  # an HTTP date: wait the default instead
        retry_after = None
    raise Throttled(retry_after)


class Throttle:
    """Paces many requests to iNaturalist, and backs off when it says so.

    Each request waits until :data:`MIN_REQUEST_INTERVAL_S` has passed since
    the one before; downloads also wait, when the last hour's add up to
    :data:`MEDIA_BYTES_PER_HOUR`, until enough of that hour has gone by. A
    :class:`Throttled` answer is retried after the wait it names, or after a
    doubling one.

    Args:
        interval_s: Least time between two requests.
        media_bytes_per_hour: Most media downloaded in any hour.
        on_wait: Called with ``(seconds, why)`` before any wait longer than a
            second, so a long one can be explained to whoever is watching.
        clock: Monotonic clock, for tests.
        sleep: Sleep function, for tests.
    """

    def __init__(
        self,
        *,
        interval_s: float = MIN_REQUEST_INTERVAL_S,
        media_bytes_per_hour: int = MEDIA_BYTES_PER_HOUR,
        on_wait: Callable[[float, str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.interval_s = interval_s
        self.media_bytes_per_hour = media_bytes_per_hour
        self._on_wait = on_wait
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._media: list[tuple[float, int]] = []

    def _pause(self, seconds: float, why: str) -> None:
        if seconds <= 0:
            return
        if seconds > 1.0 and self._on_wait is not None:
            self._on_wait(seconds, why)
        self._sleep(seconds)

    def _pace(self) -> None:
        if self._last is not None:
            self._pause(self.interval_s - (self._clock() - self._last), "pacing")
        self._last = self._clock()

    def _media_room(self) -> None:
        hour_ago = self._clock() - 3600.0
        self._media = [(t, n) for t, n in self._media if t > hour_ago]
        used = sum(n for _, n in self._media)
        if used >= self.media_bytes_per_hour and self._media:
            self._pause(
                self._media[0][0] - hour_ago,
                "the hourly media limit was reached",
            )
            self._media.pop(0)

    def record_media(self, received: int) -> None:
        """Count a finished download against the hourly media limit."""
        self._media.append((self._clock(), int(received)))

    def call(self, request: Callable[[], object], *, media: bool = False):
        """Make one request, paced, and again while iNaturalist says to wait.

        Args:
            request: Makes the request, raising :class:`Throttled` when told
                to slow down.
            media: Whether it downloads media, which also waits for room
                under the hourly limit.

        Returns:
            What ``request`` returns.

        Raises:
            BiotimbreError: If iNaturalist is still asking to wait after
                :data:`MAX_RETRIES` tries, or the request fails otherwise.
        """
        backoff = max(self.interval_s, 1.0) * 2
        for _attempt in range(MAX_RETRIES + 1):
            if media:
                self._media_room()
            self._pace()
            try:
                return request()
            except Throttled as exc:
                wait = exc.retry_after if exc.retry_after is not None else backoff
                self._pause(wait, "iNaturalist asked to slow down")
                backoff *= 2
        raise BiotimbreError(
            f"iNaturalist kept asking to slow down after {MAX_RETRIES} retries; "
            f"try again later."
        )


def fetch_observations(
    observation_ids, *, client: httpx2.Client | None = None, throttle: Throttle | None = None
) -> dict[int, Observation]:
    """Read many observations, :data:`BATCH_SIZE` to a request. Blocking.

    Args:
        observation_ids: The observations to read.
        client: An open client to use, for tests. ``None`` opens one.
        throttle: Paces the requests. ``None`` makes one with the defaults.

    Returns:
        The observations found, by ID. One that does not exist, or that is
        private, is simply absent: iNaturalist does not say which.

    Raises:
        BiotimbreError: If iNaturalist cannot be reached or answers oddly.
    """
    wanted = list(dict.fromkeys(int(i) for i in observation_ids))
    throttle = throttle or Throttle()
    owned = client is None
    client = client or httpx2.Client(
        timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    )
    found: dict[int, Observation] = {}
    try:
        for start in range(0, len(wanted), BATCH_SIZE):
            batch = wanted[start : start + BATCH_SIZE]

            def request(batch=batch):
                response = client.get(
                    f"{API_ROOT}/observations",
                    params={"id": ",".join(str(i) for i in batch),
                            "per_page": BATCH_SIZE},
                )
                _raise_if_throttled(response)
                response.raise_for_status()
                return response.json()

            try:
                payload = throttle.call(request)
            except httpx2.HTTPError as exc:
                raise BiotimbreError(f"Could not reach iNaturalist: {exc}") from exc
            except ValueError as exc:  # not JSON
                raise BiotimbreError(
                    f"iNaturalist answered unreadably: {exc}"
                ) from exc
            for result in payload.get("results") or []:
                observation = observation_from_payload(result)
                found[observation.id] = observation
    finally:
        if owned:
            client.close()
    return found


def parse_observation_id(value: object) -> int:
    """Read an observation ID from what a person typed.

    A pasted URL is as likely as a number, and both name the same observation.

    Args:
        value: The ID, or a link containing it.

    Returns:
        The numeric ID.

    Raises:
        ValidationError: If no ID can be found in it.
    """
    text = str(value or "").strip().rstrip("/")
    if not text:
        raise ValidationError("Give an observation ID.", "observation")
    tail = text.split("/")[-1].split("?")[0]
    if not tail.isdigit():
        raise ValidationError(
            f"{value!r} is not an observation ID or link.", "observation"
        )
    return int(tail)


def _local_datetime(payload: dict) -> tuple[str | None, str | None]:
    """Date and time as they were where the recording was made.

    ``time_observed_at`` carries the observation's own offset, so reading it as
    an aware datetime and taking its wall clock gives local time without a
    timezone database — which is what the library stores.
    """
    stamp = payload.get("time_observed_at")
    if isinstance(stamp, str) and stamp:
        try:
            moment = datetime.fromisoformat(stamp)
        except ValueError:
            moment = None
        if moment is not None:
            return moment.strftime("%Y-%m-%d"), moment.strftime("%H:%M:%S")

    observed_on = payload.get("observed_on")
    if isinstance(observed_on, str) and observed_on:
        return observed_on[:10], None
    return None, None


def _coordinates(payload: dict) -> tuple[float | None, float | None, bool]:
    """Latitude and longitude, withheld when iNaturalist is hiding them."""
    obscured = bool(payload.get("obscured")) or any(
        payload.get(key) in _HIDDEN
        for key in ("geoprivacy", "taxon_geoprivacy")
    )
    location = payload.get("location")
    if obscured or not isinstance(location, str) or "," not in location:
        return None, None, obscured
    latitude, _, longitude = location.partition(",")
    try:
        return float(latitude), float(longitude), obscured
    except ValueError:
        return None, None, obscured


def observation_from_payload(payload: dict) -> Observation:
    """Build an :class:`Observation` from one API result."""
    latitude, longitude, obscured = _coordinates(payload)
    date, time = _local_datetime(payload)
    taxon = payload.get("taxon") or {}
    user = payload.get("user") or {}

    sounds = []
    for entry in payload.get("sounds") or []:
        url = entry.get("file_url")
        if not url:
            continue
        sounds.append(
            Sound(
                id=int(entry.get("id") or 0),
                url=str(url),
                content_type=entry.get("file_content_type"),
                license_code=entry.get("license_code"),
                attribution=entry.get("attribution"),
            )
        )

    observation_id = int(payload.get("id"))
    return Observation(
        id=observation_id,
        url=OBSERVATION_URL.format(id=observation_id),
        date=date,
        time=time,
        latitude=latitude,
        longitude=longitude,
        obscured=obscured,
        taxon=taxon.get("name"),
        taxon_rank=taxon.get("rank"),
        quality_grade=payload.get("quality_grade"),
        observer=user.get("name") or user.get("login"),
        sounds=tuple(sounds),
    )


def fetch_observation(
    observation_id: int, *, client: httpx2.Client | None = None
) -> Observation:
    """Read one observation from iNaturalist. Blocking.

    Args:
        observation_id: The observation to read.
        client: An open client to use, for tests. ``None`` opens one.

    Returns:
        The :class:`Observation`.

    Raises:
        NotFoundError: If there is no such observation.
        BiotimbreError: If iNaturalist cannot be reached or answers oddly.
    """
    owned = client is None
    client = client or httpx2.Client(
        timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    )
    try:
        response = client.get(f"{API_ROOT}/observations/{observation_id}")
        if response.status_code == 404:
            raise NotFoundError(f"No iNaturalist observation {observation_id}.")
        response.raise_for_status()
        payload = response.json()
    except (NotFoundError, BiotimbreError):
        raise
    except httpx2.HTTPError as exc:
        raise BiotimbreError(f"Could not reach iNaturalist: {exc}") from exc
    except ValueError as exc:  # not JSON
        raise BiotimbreError(f"iNaturalist answered unreadably: {exc}") from exc
    finally:
        if owned:
            client.close()

    results = payload.get("results") or []
    if not results:
        raise NotFoundError(f"No iNaturalist observation {observation_id}.")
    return observation_from_payload(results[0])


def ffmpeg_available() -> bool:
    """Whether ffmpeg is on the path, for formats libsndfile cannot read."""
    return shutil.which("ffmpeg") is not None


def download_sound(
    sound: Sound,
    destination: str | Path,
    *,
    client: httpx2.Client | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download one recording, decoding it if libsndfile could not read it.

    Streamed to disk rather than held in memory: an observation can carry a
    recording of any length, and streaming is also what lets the caller watch
    it arrive.

    Args:
        sound: The recording to fetch.
        destination: Where to write it. Its suffix is replaced by one that
            says what the file holds: the recording's own when it is read as
            it is (``.mp3``, ``.flac``...), ``.wav`` when it had to be decoded.
        client: An open client to use, for tests. ``None`` opens one.
        on_progress: Called with ``(received, total)`` in bytes as the file
            arrives. ``total`` is 0 when the server does not say how big it is,
            which a caller should show as unmeasured rather than as no
            progress. Called from whatever thread runs this.

    Returns:
        The written path, whose suffix may differ from ``destination``'s.

    Raises:
        ValidationError: If the file needs ffmpeg and there is none.
        BiotimbreError: If the download or the decode fails.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    if sound.needs_transcoding and not ffmpeg_available():
        raise ValidationError(
            f"This recording is {sound.content_type or sound.suffix}, which "
            f"needs ffmpeg to read. Install ffmpeg, or use an observation "
            f"whose audio is WAV, FLAC, MP3 or OGG.",
            "audio",
        )

    # Written beside the target when a decode is coming, so ffmpeg reads a real
    # file: it needs to seek, and a pipe would rule out formats whose index
    # sits at the end.
    encoded = target.with_suffix(sound.suffix or ".bin")
    # Named for what it holds: an MP3 under a .wav name reads (libsndfile goes
    # by content), but the name is what anything else goes by.
    landing = encoded if sound.needs_transcoding else (
        target.with_suffix(sound.suffix) if sound.suffix else target
    )

    owned = client is None
    client = client or httpx2.Client(
        timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    )
    try:
        with client.stream("GET", sound.url) as response:
            _raise_if_throttled(response)
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            received = 0
            if on_progress is not None:
                on_progress(received, total)
            with open(landing, "wb") as handle:
                for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
                    handle.write(chunk)
                    received += len(chunk)
                    if on_progress is not None:
                        on_progress(received, total)
    except Throttled:
        landing.unlink(missing_ok=True)
        raise
    except httpx2.HTTPError as exc:
        landing.unlink(missing_ok=True)
        raise BiotimbreError(f"Could not download the recording: {exc}") from exc
    finally:
        if owned:
            client.close()

    if not sound.needs_transcoding:
        return landing

    decoded = target.with_suffix(".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(encoded),
             "-c:a", "pcm_f32le", str(decoded)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise BiotimbreError(
            f"ffmpeg could not decode the recording: "
            f"{exc.stderr.decode('utf-8', 'replace').strip()[:200]}"
        ) from exc
    finally:
        encoded.unlink(missing_ok=True)
    return decoded
