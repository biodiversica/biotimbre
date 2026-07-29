"""In-memory cache for query embeddings.

Search embeds the query audio at every speed present in the library. Without a
cache, changing nothing but a *filter* re-decodes the file and re-runs both models
— pure waste, and the dominant cost of a search over a small library. In a GUI
where filters are adjusted repeatedly against one query, that is the difference
between instant and sluggish.

Entries are keyed per ``(audio identity, model, speed)`` rather than per sweep, so
a filter that newly admits items at a previously unused speed computes only that
one missing entry instead of discarding the whole sweep.

Scope is deliberately one process and one session. A durable on-disk cache would
need invalidation against model files and preprocessing parameters for a saving
that only matters within a single interactive session.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biotimbre.embeddings.extractor import ExtractorPool, WindowEmbeddings
from biotimbre.models_registry import ModelSpec

#: Distinct query audios to retain. Each entry is small — a 60 s Perch query is
#: about 150 KB of vectors — so this is a generous bound in practice.
DEFAULT_MAX_ENTRIES = 16

#: Bytes read when fingerprinting audio held in memory. Hashing a whole long
#: array costs more than the inference it would save, so the fingerprint samples
#: the head and tail plus the length and rate.
_ARRAY_FINGERPRINT_BYTES = 1 << 16


@dataclass(frozen=True)
class AudioKey:
    """Identity of a query audio, cheap to compute and stable within a session.

    Attributes:
        kind: ``"path"`` or ``"array"``.
        token: Fingerprint distinguishing this audio from any other.
    """

    kind: str
    token: str

    @classmethod
    def for_path(
        cls,
        path: str | Path,
        *,
        segment: tuple[float, float] | None = None,
    ) -> "AudioKey":
        """Key a file by identity, modification state and searched segment.

        Uses size and nanosecond mtime rather than a content hash so that a cache
        hit avoids reading the file at all — which is the point, since decoding is
        a large part of what the cache exists to skip. Within a single session,
        an edit that preserves both size and nanosecond mtime is not a realistic
        concern.

        The segment is part of the key: two different 15 s windows of the same
        recording are different queries, and serving one for the other would be
        silently wrong.

        Args:
            path: Path to the query audio.
            segment: ``(start, end)`` in seconds being searched, if not the whole
                file.

        Returns:
            The key.

        Raises:
            OSError: If the file cannot be stat-ed.
        """
        resolved = Path(path).resolve()
        stat = resolved.stat()
        span = "" if segment is None else f"|{segment[0]:.6f}-{segment[1]:.6f}"
        return cls(
            kind="path",
            token=f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}{span}",
        )

    @classmethod
    def for_array(cls, samples: np.ndarray, native_rate: int) -> "AudioKey":
        """Key in-memory audio by a sampled fingerprint.

        Hashing every sample of a long recording would cost more than the
        inference the cache saves, so the length, rate and both ends are hashed.
        Two different recordings sharing all of those is not a case that arises
        from real audio.

        Args:
            samples: 1-D audio samples.
            native_rate: The audio's sample rate in Hz.

        Returns:
            The key.
        """
        flat = np.ascontiguousarray(samples, dtype=np.float32).ravel()
        digest = hashlib.sha256()
        digest.update(f"{flat.size}|{native_rate}".encode())
        edge = _ARRAY_FINGERPRINT_BYTES // 4
        digest.update(flat[:edge].tobytes())
        if flat.size > edge:
            digest.update(flat[-edge:].tobytes())
        return cls(kind="array", token=digest.hexdigest())


def _spec_fingerprint(spec: ModelSpec) -> str:
    """Preprocessing identity of a model.

    Included in the cache key so that re-registering a model with a different
    window or overlap cannot serve vectors computed under the old parameters.
    """
    return (
        f"{spec.model_id}@{spec.sample_rate}/{spec.window_duration_s}"
        f"/{spec.window_overlap}"
    )


class QueryEmbeddingCache:
    """LRU cache of query embeddings, keyed by audio, model and speed.

    Args:
        pool: Extractor pool used to compute misses.
        max_entries: Distinct query audios to retain.
    """

    def __init__(self, pool: ExtractorPool, *, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._pool = pool
        self._max_entries = max(1, int(max_entries))
        #: audio key -> {(spec fingerprint, speed): embeddings}
        self._entries: OrderedDict[AudioKey, dict[tuple[str, float], WindowEmbeddings]] = (
            OrderedDict()
        )
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        """Embeddings served from the cache."""
        return self._hits

    @property
    def misses(self) -> int:
        """Embeddings that had to be computed."""
        return self._misses

    def clear(self) -> None:
        """Drop every entry and reset the counters."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    def embed(
        self,
        key: AudioKey,
        speeds: list[float],
        *,
        samples: np.ndarray | None = None,
        native_rate: int | None = None,
        loader=None,
    ) -> dict[tuple[str, float], WindowEmbeddings]:
        """Return query embeddings for every ``(model, speed)`` pair.

        Audio is obtained only if something is actually missing, so a fully
        cached query never touches the file.

        Args:
            key: Identity of the query audio.
            speeds: Speeds to provide embeddings at.
            samples: Already-decoded audio, when the caller has it.
            native_rate: Sample rate of ``samples``.
            loader: Zero-argument callable returning ``(samples, native_rate)``,
                used only on a miss. Lets the caller defer decoding.

        Returns:
            Mapping of ``(model_id, speed)`` to embeddings.

        Raises:
            ValueError: If a miss occurs and no audio or loader was supplied.
        """
        cached = self._entries.get(key)
        if cached is None:
            cached = {}

        wanted: list[tuple[str, str, float]] = []
        for model_id in self._pool.model_ids:
            fingerprint = _spec_fingerprint(self._pool.get(model_id).spec)
            for speed in speeds:
                if (fingerprint, float(speed)) not in cached:
                    wanted.append((model_id, fingerprint, float(speed)))

        if wanted:
            if samples is None:
                if loader is None:
                    raise ValueError(
                        "Query audio is not cached and no samples or loader was "
                        "supplied."
                    )
                samples, native_rate = loader()
            for model_id, fingerprint, speed in wanted:
                cached[(fingerprint, speed)] = self._pool.get(model_id).embed_array(
                    samples, native_rate, speed=speed
                )
                self._misses += 1

        self._entries[key] = cached
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

        result: dict[tuple[str, float], WindowEmbeddings] = {}
        for model_id in self._pool.model_ids:
            fingerprint = _spec_fingerprint(self._pool.get(model_id).spec)
            for speed in speeds:
                entry = cached[(fingerprint, float(speed))]
                result[(model_id, float(speed))] = entry
                if (model_id, fingerprint, float(speed)) not in wanted:
                    self._hits += 1
        return result
